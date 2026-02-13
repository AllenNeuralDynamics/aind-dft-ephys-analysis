from __future__ import annotations

import os
import gc
import traceback
import multiprocessing as mp
import concurrent.futures as cf
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401

from general_utils import smart_read_csv
from behavior_utils import find_trials, get_fitted_model_names
from nwb_utils import NWBUtils
from create_psth import load_zarr
from check_monotonic import summarize_monotonic_unit_df_by_latent_quantile


def _set_worker_env_single_thread() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _try_close(obj: Any) -> None:
    if obj is None:
        return
    try:
        close = getattr(obj, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def _as_1d_float_array_maybe_object(x: Any) -> np.ndarray:
    """
    Robustly convert a behavior CSV latent column into a 1D float array.

    Handles:
      - numeric 1D arrays
      - object dtype where each element is a scalar OR a list/array (cell contains list)
        e.g. dtype: object, first element: [-0.2, 0.31, ...]
    """
    arr = np.asarray(x)

    # If object dtype and looks like "one row containing a list", unwrap it.
    if arr.dtype == object:
        if arr.size == 1 and isinstance(arr.flat[0], (list, tuple, np.ndarray)):
            arr = np.asarray(arr.flat[0])
        else:
            # common case: object array of scalars -> try to cast directly
            try:
                arr = arr.astype(float)
            except Exception:
                # fallback: if it's an object array of lists (rare), flatten
                if arr.size > 0 and isinstance(arr.flat[0], (list, tuple, np.ndarray)):
                    arr = np.asarray(arr.flat[0])
                else:
                    raise

    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D latent array, got shape={arr.shape}")
    return arr.astype(float, copy=False)


def _tag_float(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def _window_tag(w: Tuple[float, float]) -> str:
    a, b = w
    return f"win_{_tag_float(a)}__{_tag_float(b)}"


def _latent_tag(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:180]


def run_one_session(
    session_name: str,
    *,
    psth_dir: Path,
    results_dir: Path,
    out_dir: Path,
    latent_names: Sequence[str],
    activity_windows: Sequence[Tuple[float, float]],
    zarr_suffix: str,
    n_bins: int,
    binning="equal",
    quantile_stat: str,
    ci: str,
    monotonic_tol: float,
    activity_min_threshold=1,
    align_to_event: str,
    overwrite=True,
    save_format: str,
) -> Dict[str, Any]:
    """
    This function MUST live in an importable module for multiprocessing+spawn.
    """
    _set_worker_env_single_thread()

    io = None
    psth = None

    try:
        zarr_path = psth_dir / f"{session_name}{zarr_suffix}"
        beh_csv = results_dir / f"behavior_summary-{session_name}.csv"

        if not zarr_path.exists():
            return {"session": session_name, "ok": False, "reason": f"Missing zarr: {zarr_path}"}
        if not beh_csv.exists():
            return {"session": session_name, "ok": False, "reason": f"Missing behavior csv: {beh_csv}"}

        models = get_fitted_model_names(session_name=session_name)

        # Load NWB (combined) and keep the NWB object to pass into the monotonic summarizer
        nwb_data, io = NWBUtils.combine_nwb(session_name=session_name)

        psth = load_zarr(str(zarr_path))
        beh = smart_read_csv(str(beh_csv))

        # Trials: response trials (indices into go-cue aligned arrays)
        trials = np.asarray(find_trials(nwb_data, "response"))

        per_latent: List[Dict[str, Any]] = []

        for latent_name in latent_names:
            if latent_name not in beh.columns:
                per_latent.append(
                    {"latent_name": latent_name, "ok": False, "reason": f"Missing latent column: {latent_name}"}
                )
                continue

            latent_values = _as_1d_float_array_maybe_object(beh[latent_name].to_numpy())

            if len(latent_values) != len(trials):
                per_latent.append(
                    {
                        "latent_name": latent_name,
                        "ok": False,
                        "reason": f"Length mismatch: latent_values={len(latent_values)} vs trials={len(trials)}",
                    }
                )
                continue

            latent_out_root = out_dir / _latent_tag(latent_name)

            per_window: List[Dict[str, Any]] = []
            for w in activity_windows:
                out_subdir = latent_out_root / _window_tag(w)
                out_subdir.mkdir(parents=True, exist_ok=True)

                unit_df, fp = summarize_monotonic_unit_df_by_latent_quantile(
                    source=psth,
                    latent_values=latent_values,
                    latent_trial_ids=trials,
                    activity_window=w,
                    n_bins=n_bins,
                    binning=binning,
                    quantile_stat=quantile_stat,
                    ci=ci,
                    monotonic_tol=monotonic_tol,
                    activity_min_threshold=activity_min_threshold,
                    align_to_event=align_to_event,
                    session_name=session_name,
                    latent_name=latent_name,
                    # NEW: provide NWB so brain_region/ccf_location can be attached
                    nwb_data=nwb_data,
                    save_path=str(out_subdir),
                    overwrite=overwrite,
                    save_format=save_format,
                )

                per_window.append(
                    {
                        "activity_window": tuple(w),
                        "ok": True,
                        "out_dir": str(out_subdir),
                        "out_file": str(fp) if fp is not None else None,
                        "n_units": int(unit_df.shape[0]) if hasattr(unit_df, "shape") else None,
                    }
                )

                plt.close("all")

            per_latent.append({"latent_name": latent_name, "ok": True, "per_window": per_window})

            del latent_values
            gc.collect()

        return {"session": session_name, "ok": True, "models": models, "per_latent": per_latent}

    except Exception as e:
        return {
            "session": session_name,
            "ok": False,
            "reason": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }

    finally:
        _try_close(io)
        try:
            store = getattr(psth, "store", None)
            _try_close(store)
        except Exception:
            pass
        try:
            store2 = getattr(psth, "chunk_store", None)
            _try_close(store2)
        except Exception:
            pass

        plt.close("all")
        gc.collect()


def run_all_sessions_parallel(
    sessions: Sequence[str],
    *,
    psth_dir: Path,
    results_dir: Path,
    out_dir: Path,
    latent_names: Sequence[str],
    activity_windows: Sequence[Tuple[float, float]],
    max_workers: int = 2,
    zarr_suffix: str = "_0.2s.zarr",
    n_bins: int = 4,
    binning: str = "quantile",
    quantile_stat: str = "mean",
    ci: str = "sem",
    monotonic_tol: float = 0.0,
    activity_min_threshold: float = 0.0,
    align_to_event: str = "go_cue",
    overwrite: bool = True,
    save_format: str = "csv",
) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")

    total = len(sessions)
    print(f"Total sessions: {total}")
    print(f"Using max_workers={max_workers} (spawn)")
    print(f"LATENT_NAMES={list(latent_names)}")
    print(f"ACTIVITY_WINDOWS={list(activity_windows)}")

    results: List[Dict[str, Any]] = []
    done = 0
    ok = 0
    fail = 0

    with cf.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futs = [
            ex.submit(
                run_one_session,
                s,
                psth_dir=psth_dir,
                results_dir=results_dir,
                out_dir=out_dir,
                latent_names=latent_names,
                activity_windows=activity_windows,
                zarr_suffix=zarr_suffix,
                n_bins=n_bins,
                binning=binning,
                quantile_stat=quantile_stat,
                ci=ci,
                monotonic_tol=monotonic_tol,
                activity_min_threshold=activity_min_threshold,
                align_to_event=align_to_event,
                overwrite=overwrite,
                save_format=save_format,
            )
            for s in sessions
        ]

        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1

            if r.get("ok", False):
                ok += 1
                print(f"[{done:>4}/{total}] OK   {r['session']}")
            else:
                fail += 1
                print(f"[{done:>4}/{total}] FAIL {r['session']}  reason={r.get('reason')}")

    print(f"\nDone. OK={ok}, FAIL={fail}")
    return results
