from __future__ import annotations

import os
import gc
import time
import math
import traceback
import multiprocessing as mp
from pathlib import Path
import concurrent.futures as cf
from typing import Dict, Any, List, Optional, Tuple

from nwb_utils import NWBUtils
from model_fitting import fit_compare_to_threshold_model_different_learning_rate


# -----------------------------
# Config
# -----------------------------
NWB_ROOT = Path("/root/capsule/scratch/general_behavior")
OUT_ROOT = Path("/root/capsule/scratch/CTT_grid_json_resetmode_threshold")

RESET_MODE = "threshold"

RESET_ON_SWITCH_OPTIONS = [False, True]
STAY_BIAS_OPTIONS = [False, True]
SIDE_BIAS_OPTIONS = [False, True]
FIT_SEPARATE_LR_OPTIONS = [False, True]
ADAPTIVE_THRESHOLD_OPTIONS = [False, True]
FIT_SEPARATE_TH_LR_OPTIONS = [False, True]
TIE_THRESHOLD_LRS_TO_VALUE_LRS = False

# Processes
MAX_WORKERS = 12

# Batch multiple files per submitted task to reduce overhead
FILES_PER_TASK = 3

# Heartbeat interval (main process)
HEARTBEAT_EVERY_SEC = 20


# -----------------------------
# Helpers
# -----------------------------
def find_all_nwbs_two_level(root: Path) -> List[Path]:
    """Find NWB files under root/animal_id/*.nwb."""
    nwb_files: List[Path] = []
    for animal_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        nwb_files.extend(sorted(animal_dir.glob("*.nwb")))
    return nwb_files


def extract_animal_id(nwb_path: Path) -> str:
    return nwb_path.parent.name


def build_param_grid() -> List[Dict[str, Any]]:
    """Build the grid of model configs (reset_mode fixed to 'threshold')."""
    grid: List[Dict[str, Any]] = []
    for reset_on_switch in RESET_ON_SWITCH_OPTIONS:
        for include_stay_bias in STAY_BIAS_OPTIONS:
            for include_side_bias in SIDE_BIAS_OPTIONS:
                for fit_separate_learning_rates in FIT_SEPARATE_LR_OPTIONS:
                    for adaptive_threshold in ADAPTIVE_THRESHOLD_OPTIONS:
                        if not adaptive_threshold:
                            grid.append(
                                dict(
                                    reset_on_switch=reset_on_switch,
                                    include_stay_bias=include_stay_bias,
                                    include_side_bias=include_side_bias,
                                    fit_separate_learning_rates=fit_separate_learning_rates,
                                    adaptive_threshold=False,
                                    fit_separate_threshold_learning_rates=False,
                                    tie_threshold_lrs_to_value_lrs=False,
                                )
                            )
                        else:
                            if TIE_THRESHOLD_LRS_TO_VALUE_LRS:
                                grid.append(
                                    dict(
                                        reset_on_switch=reset_on_switch,
                                        include_stay_bias=include_stay_bias,
                                        include_side_bias=include_side_bias,
                                        fit_separate_learning_rates=fit_separate_learning_rates,
                                        adaptive_threshold=True,
                                        fit_separate_threshold_learning_rates=False,  # ignored when tied
                                        tie_threshold_lrs_to_value_lrs=True,
                                    )
                                )
                            else:
                                for fit_separate_threshold_learning_rates in FIT_SEPARATE_TH_LR_OPTIONS:
                                    grid.append(
                                        dict(
                                            reset_on_switch=reset_on_switch,
                                            include_stay_bias=include_stay_bias,
                                            include_side_bias=include_side_bias,
                                            fit_separate_learning_rates=fit_separate_learning_rates,
                                            adaptive_threshold=True,
                                            fit_separate_threshold_learning_rates=fit_separate_threshold_learning_rates,
                                            tie_threshold_lrs_to_value_lrs=False,
                                        )
                                    )
    return grid


PARAM_GRID = build_param_grid()


def make_model_name(cfg: Dict[str, Any]) -> str:
    """Unique model_name encoding all flags so JSON outputs do not collide."""
    reset_tag = f"reset{int(cfg['reset_on_switch'])}"
    stay_tag = f"stayB{int(cfg['include_stay_bias'])}"
    side_tag = f"sideB{int(cfg['include_side_bias'])}"
    sepLR_tag = f"sepLR{int(cfg['fit_separate_learning_rates'])}"
    adapt_tag = f"adaptTh{int(cfg['adaptive_threshold'])}"
    tie_tag = f"tieThLR{int(cfg['tie_threshold_lrs_to_value_lrs'])}"
    sepEta_tag = f"sepEta{int(cfg['fit_separate_threshold_learning_rates'])}" if cfg["adaptive_threshold"] else "sepEtaNA"
    return f"CTT_threshold_{reset_tag}_{stay_tag}_{side_tag}_{sepLR_tag}_{adapt_tag}_{sepEta_tag}_{tie_tag}"


def expected_json_path(out_dir: Path, model_name: str) -> Path:
    """
    Fast-path skip: MUST match your fitter’s output naming.
    If your fitter uses a different filename, adjust this function.
    """
    return out_dir / f"{model_name}.json"


def chunk_list(xs: List[Path], k: int) -> List[List[Path]]:
    return [xs[i : i + k] for i in range(0, len(xs), k)]


def _fmt_seconds(sec: Optional[float]) -> str:
    if sec is None or not math.isfinite(sec) or sec < 0:
        return "ETA: ?"
    sec_int = int(round(sec))
    h = sec_int // 3600
    m = (sec_int % 3600) // 60
    s = sec_int % 60
    if h > 0:
        return f"ETA: {h:d}h {m:02d}m {s:02d}s"
    return f"ETA: {m:d}m {s:02d}s"


# -----------------------------
# Worker (must be top-level in a .py file!)
# -----------------------------
def run_one_file(nwb_path_str: str) -> Dict[str, Any]:
    """
    Worker: loads one NWB once, fits all model variants, writes JSONs.
    Returns lightweight stats for the main process.
    """
    nwb_path = Path(nwb_path_str)
    animal_id = extract_animal_id(nwb_path)

    out_dir = OUT_ROOT / str(animal_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    status: Dict[str, Any] = {
        "file": str(nwb_path),
        "animal_id": animal_id,
        "ok": True,
        "errors": [],
        "n_attempted": 0,
        "n_skipped_existing": 0,
        "n_fit_ok": 0,
        "n_fit_fail": 0,
        "n_models_total": len(PARAM_GRID),
    }

    nwb_data = None
    try:
        nwb_data = NWBUtils.read_behavior_nwb(nwb_full_path=str(nwb_path))
    except Exception as e:
        status["ok"] = False
        status["errors"].append({"stage": "read_nwb", "error": repr(e)})
        return status

    try:
        for cfg in PARAM_GRID:
            status["n_attempted"] += 1
            model_name = make_model_name(cfg)

            # FAST SKIP
            jp = expected_json_path(out_dir, model_name)
            if jp.exists():
                status["n_skipped_existing"] += 1
                continue

            try:
                fit_out = fit_compare_to_threshold_model_different_learning_rate(
                    nwb_data,
                    model_name=model_name,
                    reset_on_switch=cfg["reset_on_switch"],
                    include_stay_bias=cfg["include_stay_bias"],
                    include_side_bias=cfg["include_side_bias"],
                    reset_mode=RESET_MODE,
                    reset_value_fixed=0.0,  # unused for threshold
                    fit_separate_learning_rates=cfg["fit_separate_learning_rates"],
                    adaptive_threshold=cfg["adaptive_threshold"],
                    fit_separate_threshold_learning_rates=cfg["fit_separate_threshold_learning_rates"],
                    tie_threshold_lrs_to_value_lrs=cfg["tie_threshold_lrs_to_value_lrs"],
                    save_results=True,
                    save_folder=out_dir,
                    overwrite=False,
                )
                if fit_out is None:
                    status["n_skipped_existing"] += 1
                else:
                    status["n_fit_ok"] += 1
                    del fit_out
            except Exception:
                status["ok"] = False
                status["n_fit_fail"] += 1
                # Keep per-file error list small to reduce IPC payload
                if len(status["errors"]) < 3:
                    status["errors"].append(
                        {"stage": "fit_exception", "model_name": model_name, "traceback": traceback.format_exc()}
                    )

        return status

    finally:
        if nwb_data is not None:
            if hasattr(nwb_data, "close") and callable(getattr(nwb_data, "close")):
                try:
                    nwb_data.close()
                except Exception:
                    pass
            del nwb_data
        gc.collect()


def run_file_batch(nwb_path_strs: List[str]) -> Dict[str, Any]:
    """
    Worker: run multiple NWB files sequentially (amortizes overhead).
    """
    out = {
        "files_ok": 0,
        "files_fail": 0,
        "models_total": 0,
        "models_done": 0,  # attempted includes skipped+fit_ok+fit_fail
        "models_fit_ok": 0,
        "models_skipped": 0,
        "models_fit_fail": 0,
        "sample_errors": [],
    }

    for p in nwb_path_strs:
        res = run_one_file(p)

        out["models_total"] += int(res.get("n_models_total", 0))
        out["models_done"] += int(res.get("n_attempted", 0))
        out["models_fit_ok"] += int(res.get("n_fit_ok", 0))
        out["models_skipped"] += int(res.get("n_skipped_existing", 0))
        out["models_fit_fail"] += int(res.get("n_fit_fail", 0))

        if res.get("ok", False):
            out["files_ok"] += 1
        else:
            out["files_fail"] += 1
            if res.get("errors") and len(out["sample_errors"]) < 5:
                out["sample_errors"].append(
                    {"file": res.get("file", ""), "errors": res.get("errors", [])}
                )

    return out


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    nwb_files = find_all_nwbs_two_level(NWB_ROOT)
    total_files = len(nwb_files)
    total_models = total_files * len(PARAM_GRID)

    print(f"Found {total_files} NWB files under: {NWB_ROOT}")
    print(f"Saving JSON results under: {OUT_ROOT}")
    print(f"Grid size per session: {len(PARAM_GRID)}")
    print(f"Total models: {total_models}")

    if total_files == 0:
        return

    max_workers = min(MAX_WORKERS, os.cpu_count() or 1, total_files)
    print(f"Running with {max_workers} processes (spawn)")
    print(f"FILES_PER_TASK={FILES_PER_TASK}")

    # IMPORTANT: Use spawn context explicitly (NWB/HDF5 safer than fork)
    ctx = mp.get_context("spawn")

    # Submit batches as lists of strings (more picklable than Path in some envs)
    batches = chunk_list([str(p) for p in nwb_files], max(1, FILES_PER_TASK))
    print(f"Submitting {len(batches)} tasks")

    models_done = 0
    files_done = 0
    files_ok = 0
    files_fail = 0

    t0 = time.time()
    last_hb = 0.0

    # If numpy/scipy uses MKL/OpenBLAS, avoid oversubscription
    # You can also set these in your shell before running:
    # OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    sample_errors: List[Dict[str, Any]] = []

    with cf.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futs = [ex.submit(run_file_batch, b) for b in batches]

        for fut in cf.as_completed(futs):
            r = fut.result()

            files_ok += int(r.get("files_ok", 0))
            files_fail += int(r.get("files_fail", 0))
            batch_models_done = int(r.get("models_done", 0))
            batch_models_total = int(r.get("models_total", 0))

            # For ETA, "models_total" is fixed, but we count progress by attempted models.
            models_done += batch_models_done
            files_done = files_ok + files_fail

            if r.get("sample_errors") and len(sample_errors) < 10:
                sample_errors.extend(r["sample_errors"])

            now = time.time()
            if (now - last_hb) >= HEARTBEAT_EVERY_SEC:
                elapsed = now - t0
                rate = (models_done / elapsed) if elapsed > 0 else 0.0
                remaining = max(0, total_models - models_done)
                eta_sec = (remaining / rate) if rate > 0 else None

                e = int(round(elapsed))
                eh, em, es = e // 3600, (e % 3600) // 60, e % 60
                elapsed_str = f"{eh:d}h {em:02d}m {es:02d}s" if eh > 0 else f"{em:d}m {es:02d}s"
                pct = 100.0 * models_done / max(1, total_models)

                print(
                    f"[HEARTBEAT] Elapsed: {elapsed_str} | "
                    f"Files {files_done}/{total_files} (OK={files_ok}, FAIL={files_fail}) | "
                    f"Models {models_done}/{total_models} ({pct:5.1f}%) {_fmt_seconds(eta_sec)}"
                )
                last_hb = now

    print(f"Done. Files OK={files_ok}, FAIL={files_fail}. Models attempted={models_done}/{total_models}")

    if sample_errors:
        print("\nSample errors (first few):")
        for item in sample_errors[:5]:
            print(" -", item.get("file", ""))
            for e in item.get("errors", [])[:1]:
                print("    ", e.get("stage", ""), str(e.get("error", ""))[:200])


if __name__ == "__main__":
    main()
