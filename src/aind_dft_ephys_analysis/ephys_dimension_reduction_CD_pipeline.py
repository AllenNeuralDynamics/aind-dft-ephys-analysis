"""
High-level helpers built on top of :mod:`ephys_dimension_reduction_CD` and
:mod:`ephys_dimension_reduction_CD_visualization` so that the
``ephys_dimension_reduction_CD.ipynb`` notebook only needs configuration and a
few function calls.

Three groups of helpers:

1. **Build**

   - :func:`region_label`                — canonical filename label for a
     brain-region group.
   - :func:`cd_save_path`                — build the on-disk path for a CD
     zarr from session/region/trial-types/time-window.
   - :func:`build_cd_for_session`        — for one session, run
     :func:`coding_direction_from_psth` over every (region-group × time-window)
     combination.
   - :func:`build_cd_dataset`            — drive :func:`build_cd_for_session`
     over many sessions and return the list of failures.

2. **Aggregate**

   - :func:`iter_cd_files`               — yield ``(zarr_path, session_name)``
     for CD zarrs matching a glob/suffix.
   - :func:`load_cd_session`             — load one CD zarr + matching
     behavior CSV; return a tidy :class:`CDSessionData` object.
   - :func:`aggregate_cd_sessions`       — concatenate raw A/B + LR/RL switch
     subsets across sessions into a :class:`CDAggregate`.

3. **Visualize / analyze**

   - :func:`plot_cd_session`             — average + (optional) per-trial +
     window-distribution plots for a single session.
   - :func:`plot_cd_aggregate`           — A-vs-B and LR-vs-RL plots over
     pooled sessions (train, test, train∪test).
   - :func:`compute_choice_prob_vs_activity` and
     :func:`plot_choice_prob_vs_activity` — choice-probability-vs-projection
     analysis (with quantile/uniform binning and SD/SEM error bars).
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from ephys_dimension_reduction_CD import coding_direction_from_psth
from ephys_dimension_reduction_CD_visualization import (
    plot_cd_window_distribution,
    plot_cd_projection,
)


# ---------------------------------------------------------------------------
# 1. BUILD
# ---------------------------------------------------------------------------

_SESSION_RE = re.compile(r"^CD_(?P<session>.+?)_RG_")


def region_label(region_group: Sequence[str]) -> Tuple[str, str]:
    """
    Build canonical ``(label_for_filename, label_for_print)`` from a region group.

    Empty group → ``("RG_ALL", "ALL")``.
    """
    if not region_group:
        return "RG_ALL", "ALL"
    safe = [r.replace("/", "_") for r in region_group]
    return "RG_" + "_".join(safe), ", ".join(region_group)


def cd_save_path(
    cd_root: str | Path,
    session: str,
    region_lbl: str,
    trial_types: Sequence[str],
    time_window: Sequence[float],
    align: Optional[str] = None,
) -> Path:
    """Return the standard CD zarr path for a session/region/trial-types/window.

    If ``align`` is given, ``_ALIGN_{align}`` is appended just before ``.zarr``.
    The session-extraction regex (``^CD_(?P<session>.+?)_RG_``) is unaffected.
    """
    tw0, tw1 = time_window
    align_suffix = f"_ALIGN_{align}" if align else ""
    return (
        Path(cd_root)
        / f"CD_{session}_{region_lbl}_{trial_types[0]}_{trial_types[1]}_TW_{tw0}_{tw1}{align_suffix}.zarr"
    )


def _clean_ids(series_value: Any) -> np.ndarray:
    """Convert ``behavior_summary[col][0]`` to a clean ``int`` array (drops NaN)."""
    arr = np.asarray(series_value[0])
    if arr.dtype.kind == "f":
        arr = arr[~np.isnan(arr)]
    return arr.astype(int)


def _write_pipeline_attrs(zarr_path: str | Path, **extra_attrs: Any) -> None:
    """
    Merge ``extra_attrs`` into the root ``.zattrs`` of an existing zarr store.

    Used to persist pipeline-level params (binsize, region_group, trial_types,
    ...) that ``coding_direction_from_psth`` does not know about.
    """
    import json

    zattrs_path = Path(zarr_path) / ".zattrs"
    if not zattrs_path.exists():
        return  # not a top-level zarr we can amend
    try:
        with open(zattrs_path, "r") as f:
            attrs = json.load(f)
    except Exception:  # noqa: BLE001
        attrs = {}
    pipeline_attrs = {k: v for k, v in extra_attrs.items() if v is not None}
    attrs["pipeline"] = pipeline_attrs
    with open(zattrs_path, "w") as f:
        json.dump(attrs, f)


def build_cd_for_session(
    *,
    session: str,
    psth_root: str | Path,
    behavior_root: str | Path,
    cd_root: str | Path,
    metadata: Optional[pd.DataFrame] = None,
    binsize: str = "0.1",
    align: str = "go_cue",
    brain_regions_groups: Sequence[Sequence[str]] = ((),),
    time_windows: Sequence[Sequence[float]] = ((-1.0, 0.0),),
    trial_types: Sequence[str] = ("right_choice_trials", "left_choice_trials"),
    min_units_num: int = 30,
    projection_time_window: Optional[Tuple[float, float]] = None,
    two_fold_cv: bool = True,
    norm_mode: str = "divide_sqrtN",
    random_state: int = 0,
    overwrite: bool = True,
    verbose: bool = True,
) -> List[Tuple[str, str, str, str]]:
    """
    Build CD zarrs for a single session across every region-group × time-window.

    Returns a list of failures: ``[(session, region_label, scope, error_msg)]``.

    ``metadata`` may be ``None`` or empty:
      * If ``None``, every region group is treated as "all units" (no filter
        per region), and the min-units check is skipped.
      * If a DataFrame is provided, it must contain the columns
        ``sorted_session_name``, ``brain_region``, and ``unit_index``. Empty
        DataFrames (or sessions absent from the table) yield zero units and
        each region group is skipped.
    """
    metadata_empty = metadata is None or len(metadata) == 0
    from create_psth import load_zarr
    from general_utils import smart_read_csv

    psth_root = Path(psth_root)
    behavior_root = Path(behavior_root)
    cd_root = Path(cd_root)
    cd_root.mkdir(parents=True, exist_ok=True)

    failures: List[Tuple[str, str, str, str]] = []

    try:
        psth_path = psth_root / f"{session}_{binsize}s.zarr"
        behavior_path = behavior_root / f"behavior_summary-{session}.csv"
        psth_da = load_zarr(str(psth_path))
        df = smart_read_csv(str(behavior_path))

        typeA_ids = _clean_ids(df[trial_types[0]])
        typeB_ids = _clean_ids(df[trial_types[1]])
    except Exception as e:  # noqa: BLE001
        failures.append((session, "load_failure", "session-load", str(e)))
        if verbose:
            print(f"❌ ERROR loading session {session}: {e}")
        return failures

    for region_group in brain_regions_groups:
        region_lbl, region_print = region_label(region_group)
        if metadata_empty:
            unit_indices = None  # use all units
        else:
            if region_group:
                mask = (
                    (metadata["sorted_session_name"] == session)
                    & (metadata["brain_region"].isin(region_group))
                )
            else:
                mask = metadata["sorted_session_name"] == session
            unit_indices = metadata.loc[mask, "unit_index"].to_numpy()

        n_units = "ALL" if unit_indices is None else len(unit_indices)
        if unit_indices is not None and len(unit_indices) < min_units_num:
            if verbose:
                print(
                    f"  Skip region {region_lbl} ({region_print}): "
                    f"{len(unit_indices)} units < min {min_units_num}"
                )
            continue
        if verbose:
            print(f"  Region {region_lbl} ({region_print}): {n_units} units")

        for time_window in time_windows:
            tw0, tw1 = time_window
            save_path = cd_save_path(
                cd_root, session, region_lbl, trial_types, time_window, align=align,
            )
            try:
                out = coding_direction_from_psth(
                    psth_da=psth_da,
                    trial_ids_typeA=typeA_ids,
                    trial_ids_typeB=typeB_ids,
                    align=align,
                    time_window=tuple(time_window),
                    projection_time_window=projection_time_window,
                    random_state=random_state,
                    two_fold_cv=two_fold_cv,
                    norm_mode=norm_mode,
                    save_path=str(save_path),
                    save_format="zarr",
                    overwrite=overwrite,
                    unit_ids=unit_indices,
                )
                _write_pipeline_attrs(
                    save_path,
                    session=session,
                    align=align,
                    binsize=binsize,
                    region_label=region_lbl,
                    region_group=list(region_group),
                    trial_types=list(trial_types),
                    time_window=list(time_window),
                    projection_time_window=projection_time_window,
                    two_fold_cv=bool(two_fold_cv),
                    norm_mode=norm_mode,
                    min_units_num=int(min_units_num),
                    random_state=int(random_state),
                    n_units=(int(len(unit_indices)) if unit_indices is not None else -1),
                    unit_ids=(unit_indices.tolist() if unit_indices is not None else None),
                )
                if verbose:
                    print(
                        f"    ✔ Finished time window {time_window}: "
                        f"{out.get('saved_to', str(save_path))}"
                    )
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"    ❌ Error in time window {time_window}: {e}")
                failures.append(
                    (session, region_lbl, f"time_window={time_window}", str(e))
                )
    return failures


def build_cd_dataset(
    *,
    sessions: Iterable[str],
    psth_root: str | Path,
    behavior_root: str | Path,
    cd_root: str | Path,
    metadata: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> List[Tuple[str, ...]]:
    """
    Drive :func:`build_cd_for_session` over many sessions.

    All ``**kwargs`` are forwarded to :func:`build_cd_for_session` (so e.g.
    ``binsize``, ``align``, ``brain_regions_groups``, ``time_windows``,
    ``trial_types``, ``min_units_num`` are accepted).

    Returns the accumulated failure list. Prints a summary at the end.
    """
    failed: List[Tuple[str, ...]] = []
    sessions = list(sessions)
    for session in sessions:
        print("\n==============================")
        print(f"Session: {session}")
        print("==============================")
        failed.extend(
            build_cd_for_session(
                session=session,
                psth_root=psth_root,
                behavior_root=behavior_root,
                cd_root=cd_root,
                metadata=metadata,
                **kwargs,
            )
        )

    print("\n==============================")
    print("All sessions done.")
    print("==============================")
    if failed:
        print("Failed items:")
        for f in failed:
            print(" -", f)
    else:
        print("No errors 🎉")
    return failed


# ---------------------------------------------------------------------------
# 2. AGGREGATE
# ---------------------------------------------------------------------------

@dataclass
class CDSessionData:
    """All projection traces / trial IDs / switch subsets for one CD zarr."""

    session: str
    time: np.ndarray
    dt: float
    # raw projections and trial ids
    proj_train_A: np.ndarray
    proj_train_B: np.ndarray
    proj_test_A: np.ndarray
    proj_test_B: np.ndarray
    trial_id_train_A: np.ndarray
    trial_id_train_B: np.ndarray
    trial_id_test_A: np.ndarray
    trial_id_test_B: np.ndarray
    # switch subsets (None if behavior CSV lacks the column or no matches)
    switch_LR_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    switch_RL_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    # convenience: switch subsets per train/test × A/B
    subsets: Dict[str, np.ndarray] = field(default_factory=dict)

    def counts(self) -> Dict[str, int]:
        """Trial counts for each split × class × switch type."""
        return {k: int(v.shape[0]) for k, v in self.subsets.items()}


def iter_cd_files(
    cd_root: str | Path,
    required_suffix: Optional[str] = None,
) -> Iterator[Tuple[str, str]]:
    """
    Yield ``(zarr_path, session_name)`` pairs for CD zarrs in ``cd_root``.

    If ``required_suffix`` is given, only files whose basename contains that
    substring are yielded.
    """
    for zpath in sorted(glob.glob(os.path.join(str(cd_root), "CD_*.zarr"))):
        zname = os.path.basename(zpath)
        if required_suffix and required_suffix not in zname:
            continue
        m = _SESSION_RE.match(zname)
        if not m:
            continue
        yield zpath, m.group("session")


def _select_by_ids(proj: np.ndarray, trial_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    if proj.size == 0 or target_ids.size == 0:
        return np.empty((0, proj.shape[1] if proj.ndim == 2 else 0), dtype=proj.dtype)
    mask = np.isin(trial_ids, target_ids)
    return proj[mask]


def load_cd_session(
    zarr_path: str | Path,
    behavior_csv: str | Path,
) -> CDSessionData:
    """Load a CD zarr + behavior CSV and pre-compute LR/RL switch subsets."""
    from general_utils import smart_read_csv

    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    df = smart_read_csv(str(behavior_csv))

    time = ds["time"].values
    dt = float(np.mean(np.diff(time))) if len(time) > 1 else float("nan")

    def _ids(col: str) -> np.ndarray:
        return np.asarray(ds[col].values, dtype=int)

    def _trace(col: str) -> np.ndarray:
        return ds[col].values

    m = _SESSION_RE.match(os.path.basename(str(zarr_path)))
    session = m.group("session") if m else os.path.basename(str(zarr_path))

    sess = CDSessionData(
        session=session,
        time=time,
        dt=dt,
        proj_train_A=_trace("projection_trace_train_A"),
        proj_train_B=_trace("projection_trace_train_B"),
        proj_test_A=_trace("projection_trace_test_A"),
        proj_test_B=_trace("projection_trace_test_B"),
        trial_id_train_A=_ids("trial_id_train_A"),
        trial_id_train_B=_ids("trial_id_train_B"),
        trial_id_test_A=_ids("trial_id_test_A"),
        trial_id_test_B=_ids("trial_id_test_B"),
    )

    def _col_ids(col: str) -> np.ndarray:
        if col in df.columns:
            try:
                return np.asarray(df[col][0], dtype=int)
            except Exception:  # noqa: BLE001
                return np.empty(0, dtype=int)
        return np.empty(0, dtype=int)

    sess.switch_LR_ids = _col_ids("switch_LR_trials")
    sess.switch_RL_ids = _col_ids("switch_RL_trials")

    for split_name, proj, ids in (
        ("train_A", sess.proj_train_A, sess.trial_id_train_A),
        ("train_B", sess.proj_train_B, sess.trial_id_train_B),
        ("test_A", sess.proj_test_A, sess.trial_id_test_A),
        ("test_B", sess.proj_test_B, sess.trial_id_test_B),
    ):
        sess.subsets[f"{split_name}_LR"] = _select_by_ids(proj, ids, sess.switch_LR_ids)
        sess.subsets[f"{split_name}_RL"] = _select_by_ids(proj, ids, sess.switch_RL_ids)
    return sess


@dataclass
class CDAggregate:
    """Concatenated projection traces across sessions."""

    time: Optional[np.ndarray]
    dt: float
    # raw pools
    train_A: Optional[np.ndarray]
    train_B: Optional[np.ndarray]
    test_A: Optional[np.ndarray]
    test_B: Optional[np.ndarray]
    # switch-subset pools
    subsets: Dict[str, np.ndarray]
    # convenience pooled LR/RL across A∪B
    pooled: Dict[str, np.ndarray]
    # per-session counts table
    counts_df: pd.DataFrame


def _cat_or_none(arrays: Sequence[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    arrs = [a for a in arrays if isinstance(a, np.ndarray) and a.ndim == 2 and a.size > 0]
    if not arrs:
        return None
    return np.concatenate(arrs, axis=0)


def aggregate_cd_sessions(
    sessions_data: Sequence[CDSessionData],
) -> CDAggregate:
    """Concatenate raw and switch-filtered projections across sessions."""
    if not sessions_data:
        return CDAggregate(
            time=None, dt=float("nan"),
            train_A=None, train_B=None, test_A=None, test_B=None,
            subsets={}, pooled={}, counts_df=pd.DataFrame(),
        )

    time = sessions_data[0].time
    dt = sessions_data[0].dt
    for s in sessions_data[1:]:
        if s.time.shape != time.shape or not np.allclose(s.time, time):
            print(
                "[warn] time axis differs across sessions; pooling assumes identical length."
            )
            break

    train_A = _cat_or_none([s.proj_train_A for s in sessions_data])
    train_B = _cat_or_none([s.proj_train_B for s in sessions_data])
    test_A = _cat_or_none([s.proj_test_A for s in sessions_data])
    test_B = _cat_or_none([s.proj_test_B for s in sessions_data])

    subset_keys = [
        "train_A_LR", "train_A_RL", "train_B_LR", "train_B_RL",
        "test_A_LR",  "test_A_RL",  "test_B_LR",  "test_B_RL",
    ]
    subsets = {
        k: _cat_or_none([s.subsets.get(k) for s in sessions_data])
        for k in subset_keys
    }

    pooled = {
        "train_LR": _cat_or_none([subsets.get("train_A_LR"), subsets.get("train_B_LR")]),
        "train_RL": _cat_or_none([subsets.get("train_A_RL"), subsets.get("train_B_RL")]),
        "test_LR":  _cat_or_none([subsets.get("test_A_LR"),  subsets.get("test_B_LR")]),
        "test_RL":  _cat_or_none([subsets.get("test_A_RL"),  subsets.get("test_B_RL")]),
    }
    pooled["all_LR"] = _cat_or_none([pooled["train_LR"], pooled["test_LR"]])
    pooled["all_RL"] = _cat_or_none([pooled["train_RL"], pooled["test_RL"]])

    counts_df = pd.DataFrame(
        [
            {"session": s.session, **{f"{k}_n": v for k, v in s.counts().items()}}
            for s in sessions_data
        ]
    )

    return CDAggregate(
        time=time, dt=dt,
        train_A=train_A, train_B=train_B, test_A=test_A, test_B=test_B,
        subsets=subsets, pooled=pooled, counts_df=counts_df,
    )


# ---------------------------------------------------------------------------
# 3. VISUALIZE / ANALYZE
# ---------------------------------------------------------------------------

def compute_per_trial_event_offsets(
    session_name: str,
    *,
    event_start: str = "trial_start",
    event_end: str = "go_cue",
    align: Optional[str] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Per-trial ``(t_event_start, t_event_end)`` offsets relative to ``align``.

    Loads the session NWB once and returns
    ``{trial_index: (start_offset, end_offset)}``.

    Only events whose ``extract_event_timestamps`` result is one entry per
    trial (indexed by trial index) are supported — e.g. ``"trial_start"``,
    ``"trial_end"``, ``"go_cue"``. If ``align`` is ``None`` it defaults to
    ``event_start`` (so the start offset is exactly 0).

    Trials with NaN times are dropped.
    """
    from nwb_utils import NWBUtils
    from behavior_utils import extract_event_timestamps

    nwb_data = NWBUtils.read_ophys_or_behavior_nwb(session_name=session_name)
    if nwb_data is None:
        raise FileNotFoundError(f"Could not load NWB for session '{session_name}'.")

    if align is None:
        align = event_start
    try:
        t_start = np.asarray(extract_event_timestamps(nwb_data, event_start), dtype=float)
        t_end = np.asarray(extract_event_timestamps(nwb_data, event_end), dtype=float)
        t_align = np.asarray(extract_event_timestamps(nwb_data, align), dtype=float)
    finally:
        try:
            nwb_data.io.close()
        except Exception:  # noqa: BLE001
            pass

    n = min(len(t_start), len(t_end), len(t_align))
    out: Dict[int, Tuple[float, float]] = {}
    for i in range(n):
        s = t_start[i] - t_align[i]
        e = t_end[i] - t_align[i]
        if np.isfinite(s) and np.isfinite(e):
            out[int(i)] = (float(s), float(e))
    return out


def _mask_trace_per_trial(
    trace: np.ndarray,
    trial_ids: np.ndarray,
    time: np.ndarray,
    window_map: Dict[int, Tuple[float, float]],
    *,
    smooth_seconds: Optional[float] = None,
    dt: Optional[float] = None,
    smooth_mode: str = "gaussian",
) -> np.ndarray:
    """Return a copy of ``trace`` with per-trial out-of-window samples set to NaN.

    Trials missing from ``window_map`` are dropped (set to all-NaN).

    If ``smooth_seconds`` is given (and > 0), each trial is smoothed *within*
    its own valid window first, then masked. This avoids the NaN propagation
    of ``gaussian_filter1d`` / ``uniform_filter1d`` on full traces (which
    would otherwise wipe out the entire row).
    """
    if trace.ndim != 2 or trace.size == 0:
        return trace
    out = np.full_like(trace, np.nan, dtype=float)

    do_smooth = smooth_seconds is not None and smooth_seconds > 0
    if do_smooth:
        from scipy.ndimage import gaussian_filter1d, uniform_filter1d

        kernel_pts = max(1, int(round(smooth_seconds / (dt or 1.0))))

    for i, tid in enumerate(trial_ids):
        win = window_map.get(int(tid))
        if win is None:
            continue
        t0, t1 = win
        mask = (time >= t0) & (time <= t1)
        if not np.any(mask):
            continue
        seg = trace[i, mask].astype(float, copy=True)
        if do_smooth and seg.size > 1:
            if smooth_mode == "gaussian":
                seg = gaussian_filter1d(seg, sigma=kernel_pts, mode="nearest")
            else:
                seg = uniform_filter1d(seg, size=kernel_pts, mode="nearest")
        out[i, mask] = seg
    return out


def plot_cd_session(
    sess: CDSessionData,
    *,
    distribution_window: Tuple[float, float] = (0.3, 2.0),
    smooth_gauss: float = 0.1,
    smooth_moving_window: int = 5,
    plot_single_trial: bool = True,
    restrict_window_per_trial: Optional[Dict[int, Tuple[float, float]]] = None,
    restrict_events: Optional[Tuple[str, str]] = None,
    restrict_align: Optional[str] = None,
) -> None:
    """Run the standard 3-panel CD plot for a single session (train projections).

    Parameters
    ----------
    restrict_window_per_trial : dict[int, (float, float)], optional
        Per-trial ``(t0, t1)`` window (seconds, relative to the CD's align
        event). Samples outside each trial's window are set to NaN before
        plotting, so trials with shorter ITIs contribute only where they
        actually have data. Build via :func:`compute_per_trial_event_offsets`.
    restrict_events : (event_start, event_end), optional
        Convenience shortcut: if given (and ``restrict_window_per_trial`` is
        not), call :func:`compute_per_trial_event_offsets` automatically.
        Default ``("trial_start", "go_cue")`` would limit each trial to its
        own ITI.
    restrict_align : str, optional
        Forwarded to :func:`compute_per_trial_event_offsets` when
        ``restrict_events`` is used. Defaults to ``event_start`` there.
    """
    proj_A = sess.proj_train_A
    proj_B = sess.proj_train_B
    title_suffix = ""

    if restrict_window_per_trial is None and restrict_events is not None:
        ev_start, ev_end = restrict_events
        restrict_window_per_trial = compute_per_trial_event_offsets(
            sess.session,
            event_start=ev_start,
            event_end=ev_end,
            align=restrict_align,
        )

    if restrict_window_per_trial is not None:
        proj_A = _mask_trace_per_trial(
            sess.proj_train_A, sess.trial_id_train_A, sess.time, restrict_window_per_trial,
            smooth_seconds=smooth_gauss, dt=sess.dt, smooth_mode="gaussian",
        )
        proj_B = _mask_trace_per_trial(
            sess.proj_train_B, sess.trial_id_train_B, sess.time, restrict_window_per_trial,
            smooth_seconds=smooth_gauss, dt=sess.dt, smooth_mode="gaussian",
        )
        if restrict_events is not None:
            title_suffix = f" [{restrict_events[0]}→{restrict_events[1]}]"
        else:
            title_suffix = " [per-trial window]"

    # When per-trial restriction is on, smoothing is already done per-trial
    # within each valid window (NaN-safe). Disable the plotter's smoothing
    # so it does not re-smooth and propagate NaNs across the trace.
    proj_smooth_gauss = None if restrict_window_per_trial is not None else smooth_gauss
    proj_smooth_moving = None if restrict_window_per_trial is not None else smooth_moving_window

    plot_cd_projection(
        sess.time,
        proj_A, proj_B,
        average=True,
        smooth=proj_smooth_gauss, dt=sess.dt, smooth_mode="gaussian",
        title=f"[{sess.session}] Coding Direction Projection (Smoothed){title_suffix}",
    )
    if plot_single_trial:
        plot_cd_projection(
            sess.time,
            proj_A, proj_B,
            average=False,
            smooth=proj_smooth_moving, smooth_mode="moving",
            title=f"[{sess.session}] Single-Trial CD Projections (Smoothed){title_suffix}",
        )
    plot_cd_window_distribution(
        sess.time, proj_A, proj_B,
        window=distribution_window,
        kind="hist", bins=40, hist_overlay=True,
        title=f"[{sess.session}] Train set{title_suffix}",
    )


def _plot_pair(
    time: np.ndarray, dt: float, A: Optional[np.ndarray], B: Optional[np.ndarray],
    *, title_prefix: str, distribution_window: Tuple[float, float],
    smooth_gauss: float,
) -> None:
    if A is None or B is None:
        return
    plot_cd_projection(
        time, A, B,
        average=True,
        smooth=smooth_gauss, dt=dt, smooth_mode="gaussian",
        title=f"{title_prefix} (Smoothed mean±SEM)",
    )
    plot_cd_window_distribution(
        time, A, B,
        window=distribution_window,
        kind="hist", bins=40, hist_overlay=True,
        title=f"{title_prefix}, Window [{distribution_window[0]},{distribution_window[1]}]s",
    )


def plot_cd_aggregate(
    agg: CDAggregate,
    *,
    distribution_window: Tuple[float, float] = (-1.0, 0.0),
    smooth_gauss: float = 0.1,
    show: bool = True,
) -> None:
    """Average + window-distribution plots for A-vs-B and LR-vs-RL pooled across sessions."""
    if agg.time is None:
        print("[viz] No aggregated data; skip plots.")
        return

    # A vs B (train, test)
    _plot_pair(
        agg.time, agg.dt, agg.train_A, agg.train_B,
        title_prefix="[All Sessions] TRAIN: A vs B",
        distribution_window=distribution_window, smooth_gauss=smooth_gauss,
    )
    _plot_pair(
        agg.time, agg.dt, agg.test_A, agg.test_B,
        title_prefix="[All Sessions] TEST: A vs B",
        distribution_window=distribution_window, smooth_gauss=smooth_gauss,
    )

    # LR vs RL (all, train, test)
    _plot_pair(
        agg.time, agg.dt, agg.pooled.get("all_LR"), agg.pooled.get("all_RL"),
        title_prefix="[All Sessions] LR vs RL",
        distribution_window=distribution_window, smooth_gauss=smooth_gauss,
    )
    _plot_pair(
        agg.time, agg.dt, agg.pooled.get("train_LR"), agg.pooled.get("train_RL"),
        title_prefix="[All Sessions] TRAIN: LR vs RL",
        distribution_window=distribution_window, smooth_gauss=smooth_gauss,
    )
    _plot_pair(
        agg.time, agg.dt, agg.pooled.get("test_LR"), agg.pooled.get("test_RL"),
        title_prefix="[All Sessions] TEST: LR vs RL",
        distribution_window=distribution_window, smooth_gauss=smooth_gauss,
    )

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Choice-probability vs averaged projection
# ---------------------------------------------------------------------------

def _build_choice_vector(trial_ids: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """1.0 = right, 0.0 = left, NaN otherwise, aligned to ``trial_ids``."""
    left = set(np.asarray(df["left_choice_trials"][0], dtype=int).tolist()) \
        if "left_choice_trials" in df.columns else set()
    right = set(np.asarray(df["right_choice_trials"][0], dtype=int).tolist()) \
        if "right_choice_trials" in df.columns else set()
    out = np.full(trial_ids.shape[0], np.nan, dtype=float)
    for i, tid in enumerate(trial_ids):
        if tid in right:
            out[i] = 1.0
        elif tid in left:
            out[i] = 0.0
    return out


def _avg_in_window(X: np.ndarray, tmask: np.ndarray) -> np.ndarray:
    if X.ndim != 2 or X.size == 0:
        return np.empty(0, dtype=float)
    if not np.any(tmask):
        raise ValueError("activity_avg_window selects no samples.")
    return np.nanmean(X[:, tmask], axis=1)


@dataclass
class ChoiceProbBins:
    centers: np.ndarray
    means: np.ndarray
    errs: np.ndarray
    counts: np.ndarray
    edges: np.ndarray
    method: str


def compute_choice_prob_vs_activity(
    sessions_data: Sequence[CDSessionData],
    behavior_csvs: Sequence[str | Path],
    *,
    activity_avg_window: Tuple[float, float] = (-1.0, 0.0),
    n_bins: int = 20,
    binning_method: str = "quantile",
    error_mode: str = "SEM",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Collect ``(Y_avg, choice)`` pairs across sessions for each split.

    Returns ``{"train": (X, Y), "test": (X, Y), "all": (X, Y)}`` where each
    pair is a 1-D array of averaged projections and matching ``0/1`` choices.
    Choice/NaN trials are dropped.
    """
    if len(sessions_data) != len(behavior_csvs):
        raise ValueError("sessions_data and behavior_csvs must have the same length.")
    from general_utils import smart_read_csv

    accum = {"train": ([], []), "test": ([], [])}

    for sess, csv in zip(sessions_data, behavior_csvs):
        df = smart_read_csv(str(csv))
        t0, t1 = activity_avg_window
        if t0 > t1:
            raise ValueError("activity_avg_window must satisfy t0 <= t1")
        tmask = (sess.time >= t0) & (sess.time <= t1)

        for split, proj, ids in (
            ("train", sess.proj_train_A, sess.trial_id_train_A),
            ("train", sess.proj_train_B, sess.trial_id_train_B),
            ("test", sess.proj_test_A, sess.trial_id_test_A),
            ("test", sess.proj_test_B, sess.trial_id_test_B),
        ):
            Yavg = _avg_in_window(proj, tmask)
            if Yavg.size == 0:
                continue
            choice = _build_choice_vector(ids, df)
            valid = ~np.isnan(choice)
            if valid.any():
                accum[split][0].append(Yavg[valid])
                accum[split][1].append(choice[valid])

    def _cat(split: str) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = accum[split]
        if not xs:
            return np.empty(0), np.empty(0)
        return np.concatenate(xs), np.concatenate(ys)

    X_train, Y_train = _cat("train")
    X_test, Y_test = _cat("test")
    X_all = np.concatenate([X_train, X_test]) if X_train.size or X_test.size else np.empty(0)
    Y_all = np.concatenate([Y_train, Y_test]) if Y_train.size or Y_test.size else np.empty(0)
    return {
        "train": (X_train, Y_train),
        "test": (X_test, Y_test),
        "all": (X_all, Y_all),
    }


def _bin_activity(
    x: np.ndarray, y: np.ndarray, *, n_bins: int, method: str, error_mode: str
) -> ChoiceProbBins:
    if method == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(x, qs))
        if edges.size < 3:
            edges = np.linspace(float(np.min(x)), float(np.max(x)), n_bins + 1)
            method_used = "uniform (fallback)"
        else:
            method_used = "quantile"
    else:
        edges = np.linspace(float(np.min(x)), float(np.max(x)), n_bins + 1)
        method_used = "uniform"

    inds = np.digitize(x, edges, right=False) - 1
    inds = np.clip(inds, 0, len(edges) - 2)

    centers, means, errs, counts = [], [], [], []
    for b in range(len(edges) - 1):
        m = inds == b
        cnt = int(m.sum())
        counts.append(cnt)
        if cnt > 0:
            p = float(np.mean(y[m]))
            sd = float(np.std(y[m], ddof=1)) if cnt > 1 else 0.0
            err = sd if error_mode == "SD" else sd / np.sqrt(cnt)
            c = float(np.mean(x[m]))
        else:
            p = np.nan
            err = np.nan
            c = float(0.5 * (edges[b] + edges[b + 1]))
        means.append(p)
        errs.append(err)
        centers.append(c)
    return ChoiceProbBins(
        centers=np.asarray(centers),
        means=np.asarray(means),
        errs=np.asarray(errs),
        counts=np.asarray(counts),
        edges=edges,
        method=method_used,
    )


def plot_choice_prob_vs_activity(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    activity_avg_window: Tuple[float, float] = (-1.0, 0.0),
    n_bins: int = 20,
    binning_method: str = "quantile",
    error_mode: str = "SEM",
    splits: Sequence[str] = ("train", "test", "all"),
    show: bool = True,
) -> Dict[str, ChoiceProbBins]:
    """
    Plot right-choice probability vs averaged projection for each split.

    ``data`` is the dict returned by :func:`compute_choice_prob_vs_activity`.
    Returns a mapping ``split -> ChoiceProbBins`` for downstream inspection.
    """
    results: Dict[str, ChoiceProbBins] = {}
    for split in splits:
        if split not in data:
            continue
        X, Y = data[split]
        if X.size == 0:
            print(f"[choice-prob] No data for {split}; skip.")
            continue
        bins = _bin_activity(
            X, Y, n_bins=n_bins, method=binning_method, error_mode=error_mode
        )
        results[split] = bins
        print(
            f"[choice-prob] {split}: {bins.method} binning, counts={bins.counts.tolist()}"
        )
        plt.figure(figsize=(6, 4))
        plt.errorbar(
            bins.centers, bins.means, yerr=bins.errs,
            fmt="-o", elinewidth=1, capsize=3,
        )
        plt.xlabel(
            f"Averaged projection in [{activity_avg_window[0]:.3f}, "
            f"{activity_avg_window[1]:.3f}] s"
        )
        ylabel_suffix = "SD" if error_mode == "SD" else "SEM"
        plt.ylabel(f"Right-choice probability (mean ± {ylabel_suffix})")
        plt.title(
            f"Right-choice probability vs averaged activity — {split.upper()}"
        )
        plt.grid(True)
        plt.ylim(0, 1)
        plt.tight_layout()
    if show:
        plt.show()
    return results
