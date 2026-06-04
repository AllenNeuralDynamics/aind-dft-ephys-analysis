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
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from ephys_dimension_reduction_CD import (
    coding_direction_from_psth,
    common_action_axis_from_psth,
)
from ephys_dimension_reduction_CD_visualization import (
    plot_cd_window_distribution,
    plot_cd_projection,
    plot_cd_heatmap,
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
    *,
    axis_mode: str = "cd",
    early_time_window: Optional[Sequence[float]] = None,
    late_time_window: Optional[Sequence[float]] = None,
) -> Path:
    """Return the standard CD zarr path for a session/region/trial-types/window.

    If ``align`` is given, ``_ALIGN_{align}`` is appended just before ``.zarr``.
    For ``axis_mode='common_action'`` (trial-type-agnostic temporal contrast),
    the early/late windows are encoded in the filename via
    ``_AXIS_common_action_E_{e0}_{e1}_L_{l0}_{l1}`` so common-action zarrs do
    not collide with classic CD zarrs.
    The session-extraction regex (``^CD_(?P<session>.+?)_RG_``) is unaffected.
    """
    tw0, tw1 = time_window
    align_suffix = f"_ALIGN_{align}" if align else ""
    if axis_mode == "common_action":
        if early_time_window is None or late_time_window is None:
            raise ValueError(
                "early_time_window and late_time_window are required for axis_mode='common_action'."
            )
        e0, e1 = early_time_window
        l0, l1 = late_time_window
        axis_suffix = f"_AXIS_common_action_E_{e0}_{e1}_L_{l0}_{l1}"
    else:
        axis_suffix = ""
    return (
        Path(cd_root)
        / f"CD_{session}_{region_lbl}_{trial_types[0]}_{trial_types[1]}_TW_{tw0}_{tw1}{align_suffix}{axis_suffix}.zarr"
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
    zscore_units: bool = False,
    random_state: int = 0,
    overwrite: bool = True,
    verbose: bool = True,
    # ---- common-action axis options ----
    axis_mode: Literal["cd", "common_action"] = "cd",
    early_time_window: Optional[Tuple[float, float]] = None,
    late_time_window: Optional[Tuple[float, float]] = None,
    restrict_events: Optional[Tuple[str, str]] = None,
    restrict_align: Optional[str] = None,
) -> List[Tuple[str, str, str, str]]:
    """
    Build CD (or common-action) zarrs for one session across every
    region-group × time-window combo.

    Parameters (axis-related)
    -------------------------
    axis_mode : {'cd', 'common_action'}
        - ``'cd'`` (default): classic class-discriminating coding direction
          fit on ``time_window`` (uses ``trial_types`` A vs B).
        - ``'common_action'``: trial-type-agnostic temporal contrast.
          Axis = unit-norm of ``mean(R_late) - mean(R_early)`` across all
          eligible trials. ``time_windows`` are still iterated and used only
          as a bookkeeping/file-naming dimension (set to e.g. ``[(0,0)]`` if
          you do not care).
    early_time_window, late_time_window : (float, float)
        Required when ``axis_mode='common_action'``. Both relative to ``align``.
    restrict_events : (str, str) or None
        Optional ``(event_start, event_end)`` pair. When given alongside
        ``axis_mode='common_action'``, trials whose per-trial
        ``[event_start, event_end]`` interval (offsets relative to
        ``restrict_align``) does **not** fully contain both early and late
        windows are excluded from the axis fit and from projection outputs.
    restrict_align : str or None
        Event used as the per-trial zero for ``restrict_events``. Defaults
        to ``align`` if not given. (Window comparisons are converted to the
        PSTH ``align`` frame internally.)

    Returns
    -------
    list[tuple]
        Failures ``[(session, region_label, scope, error_msg)]``.

    Notes
    -----
    ``metadata`` may be ``None`` or empty:
      * If ``None``, every region group is treated as "all units" (no filter
        per region), and the min-units check is skipped.
      * If a DataFrame is provided, it must contain the columns
        ``sorted_session_name``, ``brain_region``, and ``unit_index``.
    """
    metadata_empty = metadata is None or len(metadata) == 0
    from create_psth import load_zarr
    from general_utils import smart_read_csv

    psth_root = Path(psth_root)
    behavior_root = Path(behavior_root)
    cd_root = Path(cd_root)
    cd_root.mkdir(parents=True, exist_ok=True)

    failures: List[Tuple[str, str, str, str]] = []

    if axis_mode == "common_action":
        if early_time_window is None or late_time_window is None:
            raise ValueError(
                "axis_mode='common_action' requires early_time_window and late_time_window."
            )

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

    # Per-trial restrict-events offsets (relative to PSTH ``align``).
    # Only needed when restrict_events is provided. Computed once per session.
    eligible_trial_ids: Optional[np.ndarray] = None
    if axis_mode == "common_action" and restrict_events is not None:
        try:
            offsets = compute_per_trial_event_offsets(
                session,
                event_start=restrict_events[0],
                event_end=restrict_events[1],
                align=align,
            )
            need_lo = float(min(early_time_window[0], late_time_window[0]))
            need_hi = float(max(early_time_window[1], late_time_window[1]))
            eligible = [
                tid for tid, (s, e) in offsets.items()
                if s <= need_lo and e >= need_hi
            ]
            eligible_trial_ids = np.asarray(sorted(eligible), dtype=int)
            if verbose:
                print(
                    f"  restrict_events={restrict_events} (align={restrict_align or align}): "
                    f"{len(eligible_trial_ids)}/{len(offsets)} trials cover "
                    f"[{need_lo}, {need_hi}]"
                )
        except Exception as e:  # noqa: BLE001
            failures.append((session, "restrict_events", "offset-load", str(e)))
            if verbose:
                print(f"❌ ERROR computing restrict_events offsets for {session}: {e}")
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
                axis_mode=axis_mode,
                early_time_window=early_time_window,
                late_time_window=late_time_window,
            )
            try:
                if axis_mode == "common_action":
                    out = common_action_axis_from_psth(
                        psth_da=psth_da,
                        align=align,
                        early_time_window=tuple(early_time_window),
                        late_time_window=tuple(late_time_window),
                        projection_time_window=projection_time_window,
                        eligible_trial_ids=eligible_trial_ids,
                        trial_ids_typeA=typeA_ids,
                        trial_ids_typeB=typeB_ids,
                        random_state=random_state,
                        two_fold_cv=two_fold_cv,
                        norm_mode=norm_mode,
                        zscore_units=zscore_units,
                        save_path=str(save_path),
                        save_format="zarr",
                        overwrite=overwrite,
                        unit_ids=unit_indices,
                    )
                    extra_attrs: Dict[str, Any] = dict(
                        axis_mode="common_action",
                        early_time_window=list(early_time_window),
                        late_time_window=list(late_time_window),
                        restrict_events=(list(restrict_events) if restrict_events else None),
                        restrict_align=restrict_align,
                        n_eligible=int(out.get("n_eligible", -1)),
                    )
                else:
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
                        zscore_units=zscore_units,
                        save_path=str(save_path),
                        save_format="zarr",
                        overwrite=overwrite,
                        unit_ids=unit_indices,
                    )
                    extra_attrs = dict(axis_mode="cd")
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
                    zscore_units=bool(zscore_units),
                    min_units_num=int(min_units_num),
                    random_state=int(random_state),
                    n_units=(int(len(unit_indices)) if unit_indices is not None else -1),
                    unit_ids=(unit_indices.tolist() if unit_indices is not None else None),
                    **extra_attrs,
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
    # class names (from build-time trial_types); falls back to ("Type A","Type B")
    trial_types: Tuple[str, str] = ("Type A", "Type B")
    # ALL-trials projections onto the final CD axis (may be empty for old zarrs)
    proj_all_trials: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    trial_id_all_trials: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    # Unbiased version: A/B entries replaced with their test-fold projection.
    # Falls back to proj_all_trials when the zarr predates this field.
    proj_unbiased_all_trials: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    # Raw behavior DataFrame (single-row per session) for trial-type lookups
    behavior_df: Optional[pd.DataFrame] = None
    # Build-time align event (e.g. 'go_cue'); used by heatmap re-alignment.
    build_align: Optional[str] = None

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

    # Try to recover class labels from pipeline attrs written at build time.
    tt: Tuple[str, str] = ("Type A", "Type B")
    build_align: Optional[str] = None
    pipeline_attrs = ds.attrs.get("pipeline")
    if isinstance(pipeline_attrs, str):
        try:
            pipeline_attrs = json.loads(pipeline_attrs)
        except Exception:  # noqa: BLE001
            pipeline_attrs = None
    if isinstance(pipeline_attrs, dict):
        tt_list = pipeline_attrs.get("trial_types")
        if isinstance(tt_list, (list, tuple)) and len(tt_list) >= 2:
            tt = (str(tt_list[0]), str(tt_list[1]))
        ba = pipeline_attrs.get("align")
        if isinstance(ba, str):
            build_align = ba

    # All-trials projections (optional; present only in newer zarrs)
    if "projection_trace_all_trials" in ds.data_vars:
        proj_all = _trace("projection_trace_all_trials")
        trial_id_all = _ids("trial_id_all") if "trial_id_all" in ds.coords else np.arange(
            proj_all.shape[0], dtype=int
        )
    else:
        proj_all = np.empty((0, len(time)))
        trial_id_all = np.empty(0, dtype=int)

    # Unbiased all-trials projections (A/B replaced by test-fold). Older zarrs
    # may lack this var — fall back to the in-sample all-trials projection.
    if "projection_trace_unbiased_all_trials" in ds.data_vars:
        proj_unbiased = _trace("projection_trace_unbiased_all_trials")
    else:
        proj_unbiased = proj_all

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
        trial_types=tt,
        proj_all_trials=proj_all,
        trial_id_all_trials=trial_id_all,
        proj_unbiased_all_trials=proj_unbiased,
        behavior_df=df,
        build_align=build_align,
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


def compute_per_trial_align_shifts(
    session_name: str,
    *,
    from_align: str,
    to_align: str,
) -> Dict[int, float]:
    """Per-trial shift (seconds) to re-align data from ``from_align`` to ``to_align``.

    Returns ``{trial_index: t_from_align[i] - t_to_align[i]}`` so that a sample
    stored at offset ``t`` relative to ``from_align`` corresponds to offset
    ``t + shift[i]`` relative to ``to_align``. Trials with NaN times are dropped.
    """
    from nwb_utils import NWBUtils
    from behavior_utils import extract_event_timestamps

    nwb_data = NWBUtils.read_ophys_or_behavior_nwb(session_name=session_name)
    if nwb_data is None:
        raise FileNotFoundError(f"Could not load NWB for session '{session_name}'.")
    try:
        t_from = np.asarray(extract_event_timestamps(nwb_data, from_align), dtype=float)
        t_to = np.asarray(extract_event_timestamps(nwb_data, to_align), dtype=float)
    finally:
        try:
            nwb_data.io.close()
        except Exception:  # noqa: BLE001
            pass

    n = min(len(t_from), len(t_to))
    out: Dict[int, float] = {}
    for i in range(n):
        s = t_from[i] - t_to[i]
        if np.isfinite(s):
            out[int(i)] = float(s)
    return out


def _realign_traces(
    trace: np.ndarray,
    trial_ids: np.ndarray,
    dt: float,
    shifts: Dict[int, float],
) -> np.ndarray:
    """Per-trial roll of ``trace`` rows by ``round(shift[id]/dt)`` bins.

    Positive shift moves samples to later columns; vacated cells are filled
    with NaN (no wrap-around). Trials missing from ``shifts`` become all-NaN.
    """
    if trace.ndim != 2 or trace.size == 0 or not np.isfinite(dt) or dt <= 0:
        return trace
    out = np.full_like(trace, np.nan, dtype=float)
    n_cols = trace.shape[1]
    for i, tid in enumerate(trial_ids):
        s = shifts.get(int(tid))
        if s is None or not np.isfinite(s):
            continue
        k = int(round(s / dt))
        row = trace[i].astype(float, copy=False)
        if k == 0:
            out[i] = row
        elif k > 0:
            if k < n_cols:
                out[i, k:] = row[:n_cols - k]
        else:  # k < 0
            kk = -k
            if kk < n_cols:
                out[i, :n_cols - kk] = row[kk:]
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


def _pick_proj_all(
    sess: CDSessionData,
    projection_source: Literal["unbiased", "all"] = "unbiased",
) -> np.ndarray:
    """Return the all-trials projection trace selected by ``projection_source``.

    - ``'unbiased'`` (default) returns ``sess.proj_unbiased_all_trials`` when
      available (A/B entries replaced by their test-fold projection),
      otherwise falls back to ``sess.proj_all_trials``.
    - ``'all'`` returns the in-sample ``sess.proj_all_trials`` (final CD axis
      fit on all A∪B trials; A/B entries are biased toward separation).
    """
    if projection_source not in ("unbiased", "all"):
        raise ValueError(
            f"projection_source must be 'unbiased' or 'all', got {projection_source!r}."
        )
    if projection_source == "unbiased":
        pu = getattr(sess, "proj_unbiased_all_trials", None)
        if pu is not None and pu.size > 0:
            return pu
    return sess.proj_all_trials


def plot_cd_session(
    sess: CDSessionData,
    *,
    split: Literal["train", "test"] = "train",
    trial_types: Optional[Sequence[str]] = None,
    distribution_window: Tuple[float, float] = (0.3, 2.0),
    smooth_gauss: float = 0.1,
    smooth_moving_window: int = 5,
    plot_single_trial: bool = True,
    show_single_trial_mean: bool = False,
    random_sample_trial_N: Optional[int] = None,
    random_sample_seed: Optional[int] = 0,
    restrict_window_per_trial: Optional[Dict[int, Tuple[float, float]]] = None,
    restrict_events: Optional[Tuple[str, str]] = None,
    restrict_align: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    projection_source: Literal["unbiased", "all"] = "unbiased",
) -> None:
    """Run the standard 3-panel CD plot for a single session.

    Parameters
    ----------
    split : {'train', 'test'}, default 'train'
        Which projections to plot when ``trial_types`` is None.
        - 'train': in-sample projections (each trial projected onto the CD
          axis fit on the same half it belonged to). Larger A–B separation.
        - 'test' : cross-validated projections (each trial projected onto the
          axis fit on the *other* half). Unbiased estimate of separability.
    trial_types : sequence of str, optional
        If provided, projections are pulled from ``sess.proj_all_trials`` and
        selected by trial-ID lookup into ``sess.behavior_df`` for each named
        column. ``split`` is ignored in this mode (the final CD axis is the
        same for all trials).
        - One name → only that group is plotted (single-class display: the
          "B" series is empty).
        - Two names → first is A, second is B.
        Requires the zarr to contain ``projection_trace_all_trials`` and the
        behavior CSV to contain the requested column(s).
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
    xlim : (float, float), optional
        x-axis limits (seconds) for the projection plots. If ``None`` and
        ``restrict_events`` is given, auto-zooms to the central 95% of the
        per-trial windows so very long trials don't blow up the axis.
    """
    # ----- Build (raw_A, raw_B, ids_A, ids_B, split_lbl, name_A, name_B) -----
    if trial_types is not None:
        proj_all_arr = _pick_proj_all(sess, projection_source)
        if proj_all_arr.size == 0:
            raise ValueError(
                f"[{sess.session}] proj_all_trials is empty; rebuild CD zarr "
                "to include projection_trace_all_trials."
            )
        if sess.behavior_df is None:
            raise ValueError(f"[{sess.session}] behavior_df missing; cannot resolve trial_types.")
        tt_list = [trial_types] if isinstance(trial_types, str) else list(trial_types)
        if len(tt_list) not in (1, 2):
            raise ValueError("trial_types must contain 1 or 2 column names.")

        def _ids_from_df(col: str) -> np.ndarray:
            if col not in sess.behavior_df.columns:
                raise KeyError(
                    f"[{sess.session}] column {col!r} not found in behavior CSV."
                )
            try:
                return np.asarray(sess.behavior_df[col].iloc[0], dtype=int).ravel()
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"[{sess.session}] could not parse {col!r}: {e}") from e

        def _select(col: str) -> Tuple[np.ndarray, np.ndarray]:
            tids = _ids_from_df(col)
            mask = np.isin(sess.trial_id_all_trials, tids)
            return proj_all_arr[mask], sess.trial_id_all_trials[mask]

        raw_A, ids_A = _select(tt_list[0])
        name_A = tt_list[0]
        if len(tt_list) == 2:
            raw_B, ids_B = _select(tt_list[1])
            name_B = tt_list[1]
        else:
            raw_B = np.empty((0, proj_all_arr.shape[1]), dtype=proj_all_arr.dtype)
            ids_B = np.empty(0, dtype=int)
            name_B = ""
        split_lbl = f"All-trials CD ({projection_source})"
    else:
        if split == "train":
            raw_A, raw_B = sess.proj_train_A, sess.proj_train_B
            ids_A, ids_B = sess.trial_id_train_A, sess.trial_id_train_B
            split_lbl = "Train"
        elif split == "test":
            raw_A, raw_B = sess.proj_test_A, sess.proj_test_B
            ids_A, ids_B = sess.trial_id_test_A, sess.trial_id_test_B
            split_lbl = "Test (CV)"
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        name_A, name_B = sess.trial_types

    proj_A = raw_A
    proj_B = raw_B
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
            raw_A, ids_A, sess.time, restrict_window_per_trial,
            smooth_seconds=smooth_gauss, dt=sess.dt, smooth_mode="gaussian",
        ) if raw_A.size else raw_A
        proj_B = _mask_trace_per_trial(
            raw_B, ids_B, sess.time, restrict_window_per_trial,
            smooth_seconds=smooth_gauss, dt=sess.dt, smooth_mode="gaussian",
        ) if raw_B.size else raw_B
        if restrict_events is not None:
            title_suffix = f" [{restrict_events[0]}→{restrict_events[1]}]"
        else:
            title_suffix = " [per-trial window]"

    # When per-trial restriction is on, smoothing is already done per-trial
    # within each valid window (NaN-safe). Disable the plotter's smoothing
    # so it does not re-smooth and propagate NaNs across the trace.
    proj_smooth_gauss = None if restrict_window_per_trial is not None else smooth_gauss
    proj_smooth_moving = None if restrict_window_per_trial is not None else smooth_moving_window

    # Auto-zoom xlim to the central 95% of per-trial windows when restricting.
    if xlim is None and restrict_window_per_trial:
        starts = np.array([w[0] for w in restrict_window_per_trial.values()])
        ends = np.array([w[1] for w in restrict_window_per_trial.values()])
        xlim = (float(np.quantile(starts, 0.025)), float(np.quantile(ends, 0.975)))

    n_A = int(raw_A.shape[0]) if raw_A.ndim == 2 else 0
    n_B = int(raw_B.shape[0]) if raw_B.ndim == 2 else 0
    labels = (
        f"{name_A} (n={n_A})",
        f"{name_B} (n={n_B})" if name_B else "(none)",
    )

    plot_cd_projection(
        sess.time,
        proj_A, proj_B,
        average=True,
        smooth=proj_smooth_gauss, dt=sess.dt, smooth_mode="gaussian",
        xlim=xlim,
        labels=labels,
        title=f"[{sess.session}] {split_lbl} CD Projection (Smoothed){title_suffix}",
    )
    if plot_single_trial:
        if random_sample_trial_N is not None and random_sample_trial_N > 0:
            rng = np.random.default_rng(random_sample_seed)

            def _sample(arr: np.ndarray) -> np.ndarray:
                if arr.ndim != 2 or arr.shape[0] <= random_sample_trial_N:
                    return arr
                sel = rng.choice(arr.shape[0], size=random_sample_trial_N, replace=False)
                return arr[np.sort(sel)]

            proj_A_st = _sample(proj_A)
            proj_B_st = _sample(proj_B)
            n_A_st = int(proj_A_st.shape[0]) if proj_A_st.ndim == 2 else 0
            n_B_st = int(proj_B_st.shape[0]) if proj_B_st.ndim == 2 else 0
            labels_st = (
                f"{name_A} (n={n_A_st}/{n_A})",
                f"{name_B} (n={n_B_st}/{n_B})" if name_B else "(none)",
            )
        else:
            proj_A_st, proj_B_st = proj_A, proj_B
            labels_st = labels
        plot_cd_projection(
            sess.time,
            proj_A_st, proj_B_st,
            average=False,
            show_mean=show_single_trial_mean,
            smooth=proj_smooth_moving, smooth_mode="moving",
            xlim=xlim,
            labels=labels_st,
            title=f"[{sess.session}] {split_lbl} Single-Trial CD Projections (Smoothed){title_suffix}",
        )
    plot_cd_window_distribution(
        sess.time, proj_A, proj_B,
        window=distribution_window,
        kind="hist", bins=40, hist_overlay=True,
        labels=labels,
        title=f"[{sess.session}] {split_lbl} set{title_suffix}",
    )


def plot_cd_session_heatmap(
    sess: CDSessionData,
    *,
    split: Literal["train", "test"] = "train",
    trial_types: Optional[Sequence[str]] = None,
    smooth_gauss: float = 0.1,
    smooth_mode: Literal["gaussian", "moving"] = "gaussian",
    sort_by: Optional[Literal["mean", "peak_time", "peak_value", "window_length", "none"]] = "mean",
    sort_window: Optional[Tuple[float, float]] = None,
    sort_ascending: bool = False,
    random_sample_trial_N: Optional[int] = None,
    random_sample_seed: Optional[int] = 0,
    restrict_window_per_trial: Optional[Dict[int, Tuple[float, float]]] = None,
    restrict_events: Optional[Tuple[str, str]] = None,
    restrict_align: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vrange_quantile: float = 0.99,
    symmetric_colorbar: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    threshold: Optional[float] = None,
) -> None:
    """Single-trial CD-projection heatmap for one session.

    Mirrors :func:`plot_cd_session` for trial selection (``split`` or
    ``trial_types``), per-trial windowing (``restrict_events`` /
    ``restrict_window_per_trial``), smoothing, and optional random
    subsampling, but renders per-trial traces as a 2-D heatmap (one row per
    trial). Each class becomes its own panel; rows are sorted within each
    panel according to ``sort_by``.

    ``threshold`` : if given (>0), any sample whose absolute value is below
    ``threshold`` is set to 0 before plotting (NaNs are preserved).
    """
    # ----- Trial selection (mirrors plot_cd_session) -----
    if trial_types is not None:
        if sess.proj_all_trials.size == 0:
            raise ValueError(
                f"[{sess.session}] proj_all_trials is empty; rebuild CD zarr "
                "to include projection_trace_all_trials."
            )
        if sess.behavior_df is None:
            raise ValueError(f"[{sess.session}] behavior_df missing; cannot resolve trial_types.")
        tt_list = [trial_types] if isinstance(trial_types, str) else list(trial_types)
        if len(tt_list) not in (1, 2):
            raise ValueError("trial_types must contain 1 or 2 column names.")

        def _ids_from_df(col: str) -> np.ndarray:
            if col not in sess.behavior_df.columns:
                raise KeyError(
                    f"[{sess.session}] column {col!r} not found in behavior CSV."
                )
            try:
                return np.asarray(sess.behavior_df[col].iloc[0], dtype=int).ravel()
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"[{sess.session}] could not parse {col!r}: {e}") from e

        def _select(col: str) -> Tuple[np.ndarray, np.ndarray]:
            tids = _ids_from_df(col)
            mask = np.isin(sess.trial_id_all_trials, tids)
            return sess.proj_all_trials[mask], sess.trial_id_all_trials[mask]

        raw_A, ids_A = _select(tt_list[0])
        name_A = tt_list[0]
        if len(tt_list) == 2:
            raw_B, ids_B = _select(tt_list[1])
            name_B = tt_list[1]
        else:
            raw_B = np.empty((0, sess.proj_all_trials.shape[1]), dtype=sess.proj_all_trials.dtype)
            ids_B = np.empty(0, dtype=int)
            name_B = ""
        split_lbl = "All-trials CD"
    else:
        if split == "train":
            raw_A, raw_B = sess.proj_train_A, sess.proj_train_B
            ids_A, ids_B = sess.trial_id_train_A, sess.trial_id_train_B
            split_lbl = "Train"
        elif split == "test":
            raw_A, raw_B = sess.proj_test_A, sess.proj_test_B
            ids_A, ids_B = sess.trial_id_test_A, sess.trial_id_test_B
            split_lbl = "Test (CV)"
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        name_A, name_B = sess.trial_types

    proj_A = raw_A
    proj_B = raw_B
    title_suffix = ""

    # ----- Per-trial re-alignment from build-time align to restrict_align -----
    # The CD zarr stores traces with time relative to the build-time `align`
    # event (e.g. 'go_cue'). When the caller wants to view per-trial windows
    # relative to a different event (e.g. 'trial_start'), each row must be
    # shifted by the per-trial offset between the two events. Missing samples
    # outside the stored PSTH range become NaN (no wrap-around).
    realigned = False
    if (
        restrict_align is not None
        and sess.build_align is not None
        and restrict_align != sess.build_align
    ):
        try:
            shifts = compute_per_trial_align_shifts(
                sess.session,
                from_align=sess.build_align,
                to_align=restrict_align,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[warn] could not re-align {sess.session} "
                f"({sess.build_align}->{restrict_align}): {e}"
            )
            shifts = {}
        if shifts:
            proj_A = _realign_traces(raw_A, ids_A, sess.dt, shifts) if raw_A.size else raw_A
            proj_B = _realign_traces(raw_B, ids_B, sess.dt, shifts) if raw_B.size else raw_B
            raw_A = proj_A
            raw_B = proj_B
            realigned = True

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
            raw_A, ids_A, sess.time, restrict_window_per_trial,
            smooth_seconds=smooth_gauss, dt=sess.dt, smooth_mode=smooth_mode,
        ) if raw_A.size else raw_A
        proj_B = _mask_trace_per_trial(
            raw_B, ids_B, sess.time, restrict_window_per_trial,
            smooth_seconds=smooth_gauss, dt=sess.dt, smooth_mode=smooth_mode,
        ) if raw_B.size else raw_B
        if restrict_events is not None:
            title_suffix = f" [{restrict_events[0]}\u2192{restrict_events[1]}]"
        else:
            title_suffix = " [per-trial window]"
    if realigned:
        title_suffix += f" (re-aligned to {restrict_align})"

    # If per-trial restriction handled smoothing, skip it inside the heatmap.
    hm_smooth = None if restrict_window_per_trial is not None else smooth_gauss

    if xlim is None and restrict_window_per_trial:
        starts = np.array([w[0] for w in restrict_window_per_trial.values()])
        ends = np.array([w[1] for w in restrict_window_per_trial.values()])
        xlim = (float(np.quantile(starts, 0.025)), float(np.quantile(ends, 0.975)))

    # Optional random subsampling per class (keep ids paired with traces).
    if random_sample_trial_N is not None and random_sample_trial_N > 0:
        rng = np.random.default_rng(random_sample_seed)

        def _sample(arr: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if arr.ndim != 2 or arr.shape[0] <= random_sample_trial_N:
                return arr, ids
            sel = rng.choice(arr.shape[0], size=random_sample_trial_N, replace=False)
            sel = np.sort(sel)
            return arr[sel], ids[sel]

        proj_A, ids_A = _sample(proj_A, ids_A)
        proj_B, ids_B = _sample(proj_B, ids_B)

    # Sort rows by per-trial restrict-window length (descending by default).
    # When chosen, we reorder here and tell plot_cd_heatmap not to re-sort.
    hm_sort_by: Optional[str] = sort_by
    if sort_by == "window_length":
        if restrict_window_per_trial is None:
            raise ValueError(
                "sort_by='window_length' requires restrict_events or "
                "restrict_window_per_trial."
            )

        def _len_order(ids: np.ndarray) -> np.ndarray:
            if ids.size == 0:
                return np.arange(0, dtype=int)
            lens = np.array([
                (restrict_window_per_trial[int(t)][1] - restrict_window_per_trial[int(t)][0])
                if int(t) in restrict_window_per_trial else np.nan
                for t in ids
            ], dtype=float)
            # Push NaNs to the bottom regardless of direction.
            order = np.argsort(np.where(np.isnan(lens), -np.inf if sort_ascending else np.inf, lens))
            if not sort_ascending:
                order = order[::-1]
            return order

        if isinstance(proj_A, np.ndarray) and proj_A.ndim == 2 and proj_A.size:
            o = _len_order(ids_A)
            proj_A = proj_A[o]
            ids_A = ids_A[o]
        if isinstance(proj_B, np.ndarray) and proj_B.ndim == 2 and proj_B.size:
            o = _len_order(ids_B)
            proj_B = proj_B[o]
            ids_B = ids_B[o]
        hm_sort_by = "none"

    # Apply threshold: zero out samples whose |value| < threshold (keep NaNs).
    if threshold is not None and threshold > 0:
        def _apply_threshold(arr: np.ndarray) -> np.ndarray:
            if not isinstance(arr, np.ndarray) or arr.size == 0:
                return arr
            out = arr.astype(float, copy=True)
            mask = np.isfinite(out) & (np.abs(out) < float(threshold))
            out[mask] = 0.0
            return out

        proj_A = _apply_threshold(proj_A)
        proj_B = _apply_threshold(proj_B)

    plot_cd_heatmap(
        sess.time,
        proj_A,
        proj_B if (isinstance(proj_B, np.ndarray) and proj_B.size) else None,
        smooth=hm_smooth,
        smooth_mode=smooth_mode,
        dt=sess.dt,
        sort_by=hm_sort_by,
        sort_window=sort_window,
        sort_ascending=sort_ascending,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        vrange_quantile=vrange_quantile,
        symmetric_colorbar=symmetric_colorbar,
        labels=(name_A, name_B or "(none)"),
        figsize=figsize,
        xlim=xlim,
        title=f"[{sess.session}] {split_lbl} Single-Trial CD Heatmap{title_suffix}",
    )


def plot_cd_session_bumps(
    sess: CDSessionData,
    *,
    split: Literal["train", "test"] = "train",
    trial_types: Optional[Sequence[str]] = None,
    search_window: Tuple[float, float] = (0.0, 3.0),
    polarity: Literal["pos", "neg", "both"] = "both",
    smooth_sigma_sec: Optional[float] = 0.1,
    min_amplitude: Optional[float] = None,
    min_prominence: Optional[float] = 1.0,
    min_width_sec: Optional[float] = 0.1,
    max_per_trial: int = 5,
    random_sample_trial_N: Optional[int] = None,
    random_sample_seed: Optional[int] = 0,
    restrict_window_per_trial: Optional[Dict[int, Tuple[float, float]]] = None,
    restrict_events: Optional[Tuple[str, str]] = None,
    restrict_align: Optional[str] = None,
    shape_window: Tuple[float, float] = (-1.0, 2.0),
    baseline_window: Optional[Tuple[float, float]] = (-1.0, -0.3),
    baseline_stat: Literal["median", "mean"] = "median",
    density_bin_width_sec: Optional[float] = 0.25,
    xlim: Optional[Tuple[float, float]] = None,
    cmap: str = "RdBu_r",
    vrange_quantile: float = 0.99,
    return_df: bool = False,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Detect and visualize transient bumps in per-trial CD projections.

    Mirrors :func:`plot_cd_session_heatmap` for trial selection
    (``split`` / ``trial_types``), per-trial re-alignment
    (``restrict_align`` vs build-time align), per-trial masking
    (``restrict_events`` / ``restrict_window_per_trial``), and optional
    random subsampling. Each class becomes one bump-detection figure plus
    a printed median±IQR summary.

    Parameters
    ----------
    search_window : (t0, t1), default (0, 3)
        Time interval (seconds, in the same frame as the displayed traces,
        i.e. after re-alignment) within which to search for the bump.
    polarity, smooth_sigma_sec, min_amplitude, min_prominence,
    min_width_sec, max_per_trial
        Forwarded to :func:`ephys_dimension_reduction_CD_bump.detect_bumps`.
    shape_window : (t0, t1)
        Window (relative to each detected peak) used for the peak-aligned
        mean ± SEM shape plot.
    return_df : bool
        If True, return ``{class_name: bumps_df}`` (in addition to plotting).
    """
    from ephys_dimension_reduction_CD_bump import (
        detect_bumps, plot_bumps, summarize_bumps, plot_inter_peak_intervals,
        plot_bump_density,
    )

    # ----- Trial selection (mirrors plot_cd_session_heatmap) -----
    if trial_types is not None:
        if sess.proj_all_trials.size == 0:
            raise ValueError(
                f"[{sess.session}] proj_all_trials is empty; rebuild CD zarr "
                "to include projection_trace_all_trials."
            )
        if sess.behavior_df is None:
            raise ValueError(f"[{sess.session}] behavior_df missing; cannot resolve trial_types.")
        tt_list = [trial_types] if isinstance(trial_types, str) else list(trial_types)
        if len(tt_list) not in (1, 2):
            raise ValueError("trial_types must contain 1 or 2 column names.")

        def _ids_from_df(col: str) -> np.ndarray:
            if col not in sess.behavior_df.columns:
                raise KeyError(
                    f"[{sess.session}] column {col!r} not found in behavior CSV."
                )
            try:
                return np.asarray(sess.behavior_df[col].iloc[0], dtype=int).ravel()
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"[{sess.session}] could not parse {col!r}: {e}") from e

        def _select(col: str) -> Tuple[np.ndarray, np.ndarray]:
            tids = _ids_from_df(col)
            mask = np.isin(sess.trial_id_all_trials, tids)
            return sess.proj_all_trials[mask], sess.trial_id_all_trials[mask]

        raw_A, ids_A = _select(tt_list[0])
        name_A = tt_list[0]
        if len(tt_list) == 2:
            raw_B, ids_B = _select(tt_list[1])
            name_B = tt_list[1]
        else:
            raw_B = np.empty((0, sess.proj_all_trials.shape[1]), dtype=sess.proj_all_trials.dtype)
            ids_B = np.empty(0, dtype=int)
            name_B = ""
        split_lbl = "All-trials CD"
    else:
        if split == "train":
            raw_A, raw_B = sess.proj_train_A, sess.proj_train_B
            ids_A, ids_B = sess.trial_id_train_A, sess.trial_id_train_B
            split_lbl = "Train"
        elif split == "test":
            raw_A, raw_B = sess.proj_test_A, sess.proj_test_B
            ids_A, ids_B = sess.trial_id_test_A, sess.trial_id_test_B
            split_lbl = "Test (CV)"
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        name_A, name_B = sess.trial_types

    proj_A = raw_A
    proj_B = raw_B
    title_suffix = ""

    # ----- Per-trial re-alignment (same logic as plot_cd_session_heatmap) -----
    realigned = False
    if (
        restrict_align is not None
        and sess.build_align is not None
        and restrict_align != sess.build_align
    ):
        try:
            shifts = compute_per_trial_align_shifts(
                sess.session,
                from_align=sess.build_align,
                to_align=restrict_align,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[warn] could not re-align {sess.session} "
                f"({sess.build_align}->{restrict_align}): {e}"
            )
            shifts = {}
        if shifts:
            proj_A = _realign_traces(raw_A, ids_A, sess.dt, shifts) if raw_A.size else raw_A
            proj_B = _realign_traces(raw_B, ids_B, sess.dt, shifts) if raw_B.size else raw_B
            raw_A = proj_A
            raw_B = proj_B
            realigned = True

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
            raw_A, ids_A, sess.time, restrict_window_per_trial,
            smooth_seconds=None, dt=sess.dt, smooth_mode="gaussian",
        ) if raw_A.size else raw_A
        proj_B = _mask_trace_per_trial(
            raw_B, ids_B, sess.time, restrict_window_per_trial,
            smooth_seconds=None, dt=sess.dt, smooth_mode="gaussian",
        ) if raw_B.size else raw_B
        if restrict_events is not None:
            title_suffix = f" [{restrict_events[0]}\u2192{restrict_events[1]}]"
        else:
            title_suffix = " [per-trial window]"
    if realigned:
        title_suffix += f" (re-aligned to {restrict_align})"

    if xlim is None and restrict_window_per_trial:
        starts = np.array([w[0] for w in restrict_window_per_trial.values()])
        ends = np.array([w[1] for w in restrict_window_per_trial.values()])
        xlim = (float(np.quantile(starts, 0.025)), float(np.quantile(ends, 0.975)))

    # Optional random subsampling per class (keep ids paired with traces).
    if random_sample_trial_N is not None and random_sample_trial_N > 0:
        rng = np.random.default_rng(random_sample_seed)

        def _sample(arr: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if arr.ndim != 2 or arr.shape[0] <= random_sample_trial_N:
                return arr, ids
            sel = np.sort(rng.choice(arr.shape[0], size=random_sample_trial_N, replace=False))
            return arr[sel], ids[sel]

        proj_A, ids_A = _sample(proj_A, ids_A)
        proj_B, ids_B = _sample(proj_B, ids_B)

    results: Dict[str, pd.DataFrame] = {}
    for proj, ids, name in (
        (proj_A, ids_A, name_A),
        (proj_B, ids_B, name_B),
    ):
        if not (isinstance(proj, np.ndarray) and proj.ndim == 2 and proj.size and name):
            continue
        df = detect_bumps(
            proj, sess.time,
            trial_ids=ids, dt=sess.dt,
            search_window=search_window,
            polarity=polarity,
            smooth_sigma_sec=smooth_sigma_sec,
            min_amplitude=min_amplitude,
            min_prominence=min_prominence,
            min_width_sec=min_width_sec,
            max_per_trial=max_per_trial,
        )
        summary = summarize_bumps(df)
        n_trials = int(proj.shape[0])
        n_trials_with_bump = int(df["trial_index"].nunique()) if not df.empty else 0
        print(
            f"\n[{sess.session}] {name}: {n_trials_with_bump}/{n_trials} "
            f"trials with bump in {search_window} s"
        )
        if not summary.empty:
            print(summary.to_string(index=False))
        plot_bumps(
            proj, sess.time, df,
            dt=sess.dt,
            cmap=cmap,
            vrange_quantile=vrange_quantile,
            xlim=xlim,
            shape_window=shape_window,
            baseline_window=baseline_window,
            baseline_stat=baseline_stat,
            title=(
                f"[{sess.session}] {name} (n={n_trials}) — "
                f"{split_lbl} bumps{title_suffix}"
            ),
        )
        plot_inter_peak_intervals(
            df,
            title=f"[{sess.session}] {name} — IPI distribution{title_suffix}",
        )
        if density_bin_width_sec is not None and density_bin_width_sec > 0:
            plot_bump_density(
                proj, sess.time, df,
                bin_width_sec=density_bin_width_sec,
                xlim=xlim,
                title=f"[{sess.session}] {name} — bump density{title_suffix}",
            )
        results[name] = df

    return results if return_df else None


# ---------------------------------------------------------------------------
# Scatter: average CD projection vs P(right) per trial
# ---------------------------------------------------------------------------

def _reconstruct_animal_response_from_df(
    df: pd.DataFrame, n_trials: int
) -> np.ndarray:
    """Build an animal_response vector (0=left, 1=right, 2=no-response) of
    length ``n_trials`` from the trial-type columns stored by
    ``generate_behavior_summary``.

    Trials not appearing in any of the three columns default to 2 (no
    response), which is the safe choice for sliding-window P(right).
    """
    resp = np.full(int(n_trials), 2, dtype=int)

    def _ids(col: str) -> np.ndarray:
        if col not in df.columns:
            return np.empty(0, dtype=int)
        try:
            return np.asarray(df[col].iloc[0], dtype=int).ravel()
        except Exception:  # noqa: BLE001
            return np.empty(0, dtype=int)

    left_ids = _ids("left_choice_trials")
    right_ids = _ids("right_choice_trials")
    if left_ids.size:
        left_ids = left_ids[(left_ids >= 0) & (left_ids < n_trials)]
        resp[left_ids] = 0
    if right_ids.size:
        right_ids = right_ids[(right_ids >= 0) & (right_ids < n_trials)]
        resp[right_ids] = 1
    return resp


def plot_cd_projection_vs_choice_probability(
    sess: CDSessionData,
    *,
    window: Tuple[float, float],
    p_right: Optional[np.ndarray] = None,
    p_right_window: int = 10,
    p_right_column: Optional[str] = None,
    highlight_trial_types: Sequence[str] = (),
    restrict_events: Optional[Tuple[str, str]] = None,
    restrict_align: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (6.0, 4.5),
    base_color: str = "0.6",
    base_alpha: float = 0.45,
    highlight_alpha: float = 0.9,
    point_size: float = 28.0,
    show_correlation: bool = True,
    title: Optional[str] = None,
    projection_source: Literal["unbiased", "all"] = "unbiased",
) -> Dict[str, np.ndarray]:
    """Per-trial scatter of CD projection (averaged over ``window``) vs P(right).

    Each dot is one trial taken from ``sess.proj_all_trials``. By default the
    x-axis (P(right)) is computed on the fly via
    :func:`compute_sliding_choice_probability` (causal window of length
    ``p_right_window``, no-response trials excluded) using ``animal_response``
    reconstructed from ``sess.behavior_df``. Override by passing either:

    - ``p_right`` : 1-D array indexed by absolute trial id (same convention as
      the ids stored in ``sess.trial_id_all_trials``), or
    - ``p_right_column`` : a column name in ``sess.behavior_df`` whose value
      is such an array (e.g. a model-fitted ``right_choice_probability``).
      If the column holds a responded-trial-only array, its values are
      remapped onto absolute trial ids using ``right_choice_trials`` +
      ``left_choice_trials``.

    Trials whose ids appear in any column listed in ``highlight_trial_types``
    are re-drawn on top with a distinct color per type.

    Parameters
    ----------
    window : (t0, t1)
        Time window (seconds, in the projection's align frame) over which the
        per-trial projection is averaged. Set ``restrict_events`` to mask out
        samples outside each trial's event interval before averaging.
    p_right_window : int, default 10
        Window length used when computing P(right) on the fly.
    restrict_events : (event_start, event_end), optional
        Per-trial masking via :func:`compute_per_trial_event_offsets`.
    restrict_align : str, optional
        Forwarded to :func:`compute_per_trial_event_offsets`.

    Returns
    -------
    dict
        ``{'trial_id', 'p_right', 'proj_mean', 'highlight_masks'}``.
    """
    if sess.proj_all_trials.size == 0:
        raise ValueError(
            f"[{sess.session}] proj_all_trials is empty; rebuild CD zarr to "
            "include projection_trace_all_trials."
        )
    if sess.behavior_df is None:
        raise ValueError(f"[{sess.session}] behavior_df missing.")

    proj_all = _pick_proj_all(sess, projection_source)
    ids_all = np.asarray(sess.trial_id_all_trials, dtype=int)
    time = sess.time
    df = sess.behavior_df

    # ----- Per-trial event masking (optional) -----
    if restrict_events is not None:
        ev_start, ev_end = restrict_events
        per_trial_win = compute_per_trial_event_offsets(
            sess.session,
            event_start=ev_start,
            event_end=ev_end,
            align=restrict_align,
        )
        proj_masked = _mask_trace_per_trial(
            proj_all, ids_all, time, per_trial_win,
            smooth_seconds=0.0, dt=sess.dt, smooth_mode="gaussian",
        )
    else:
        proj_masked = proj_all

    # ----- Per-trial projection mean over `window` -----
    t0, t1 = float(window[0]), float(window[1])
    time_mask = (time >= t0) & (time <= t1)
    if not time_mask.any():
        raise ValueError(f"window {window} does not overlap session time axis.")
    sub = proj_masked[:, time_mask]
    with np.errstate(all="ignore"):
        proj_mean = np.nanmean(sub, axis=1)

    # ----- Build P(right) aligned to absolute trial ids -----
    # Determine session-wide n_trials from behavior_df: the maximum id across
    # the trial-type columns + 1 (plus any id we see in proj ids).
    def _col_ids(col: str) -> np.ndarray:
        if col not in df.columns:
            return np.empty(0, dtype=int)
        try:
            return np.asarray(df[col].iloc[0], dtype=int).ravel()
        except Exception:  # noqa: BLE001
            return np.empty(0, dtype=int)

    all_known_ids = [ids_all]
    for c in ("left_choice_trials", "right_choice_trials", "no_response_trials"):
        all_known_ids.append(_col_ids(c))
    n_trials = int(max(int(arr.max()) for arr in all_known_ids if arr.size) + 1)

    if p_right is not None:
        p_arr = np.asarray(p_right, dtype=float).ravel()
        if p_arr.size < n_trials:
            p_arr = np.concatenate([p_arr, np.full(n_trials - p_arr.size, np.nan)])
    elif p_right_column is not None:
        if p_right_column not in df.columns:
            raise KeyError(
                f"[{sess.session}] column {p_right_column!r} not in behavior_df."
            )
        raw = np.asarray(df[p_right_column].iloc[0], dtype=float).ravel()
        if raw.size == n_trials:
            p_arr = raw
        else:
            # Treat as responded-trial-only series. Map by sorted union of
            # left+right choice ids (the temporal order of responded trials).
            responded_ids = np.sort(
                np.concatenate([_col_ids("left_choice_trials"),
                                _col_ids("right_choice_trials")])
            )
            p_arr = np.full(n_trials, np.nan, dtype=float)
            n = min(raw.size, responded_ids.size)
            p_arr[responded_ids[:n]] = raw[:n]
    else:
        # Reconstruct animal_response and compute sliding P(right) ourselves.
        # Lazy import to avoid circular module loads at top of file.
        from behavior_utils import compute_sliding_choice_probability

        resp_vec = _reconstruct_animal_response_from_df(df, n_trials)
        out = compute_sliding_choice_probability(
            resp_vec,
            window=int(p_right_window),
            step=1,
            min_periods=1,
            causal=True,
            side="right",
            exclude_value=2,
        )
        p_arr = np.asarray(out["choice_prob"], dtype=float)

    # Look up P(right) for each plotted trial id
    valid_id_mask = (ids_all >= 0) & (ids_all < n_trials)
    p_per_trial = np.full(ids_all.shape, np.nan, dtype=float)
    p_per_trial[valid_id_mask] = p_arr[ids_all[valid_id_mask]]

    # ----- Plot -----
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    finite = np.isfinite(p_per_trial) & np.isfinite(proj_mean)
    ax.scatter(
        p_per_trial[finite], proj_mean[finite],
        s=point_size, c=base_color, alpha=base_alpha,
        edgecolors="none", label=f"all trials (n={int(finite.sum())})",
    )

    # Highlight overlays
    highlight_masks: Dict[str, np.ndarray] = {}
    if highlight_trial_types:
        cmap = plt.get_cmap("tab10")
        for i, tt in enumerate(highlight_trial_types):
            hl_ids = _col_ids(tt)
            if hl_ids.size == 0:
                print(f"[{sess.session}] highlight {tt!r}: no trials / column missing.")
                highlight_masks[tt] = np.zeros(ids_all.shape, dtype=bool)
                continue
            mask = np.isin(ids_all, hl_ids) & finite
            highlight_masks[tt] = mask
            if not mask.any():
                continue
            color = cmap(i % 10)
            ax.scatter(
                p_per_trial[mask], proj_mean[mask],
                s=point_size * 1.4, c=[color], alpha=highlight_alpha,
                edgecolors="black", linewidths=0.6,
                label=f"{tt} (n={int(mask.sum())})",
            )

    # Optional correlation annotation
    if show_correlation and finite.sum() >= 3:
        r = float(np.corrcoef(p_per_trial[finite], proj_mean[finite])[0, 1])
        ax.text(
            0.02, 0.98, f"r = {r:.2f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.8),
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("P(right) (sliding, causal)" if p_right_column is None and p_right is None
                  else (p_right_column or "P(right)"))
    ax.set_ylabel(f"mean CD projection in [{t0:g}, {t1:g}] s")
    if title is None:
        title = (
            f"[{sess.session}] CD projection vs P(right)"
            + (f"  (window {p_right_window})" if p_right is None and p_right_column is None else "")
        )
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    return {
        "trial_id": ids_all,
        "p_right": p_per_trial,
        "proj_mean": proj_mean,
        "highlight_masks": highlight_masks,
    }


def plot_cd_projection_box_by_choice_probability(
    sess: CDSessionData,
    *,
    window: Tuple[float, float],
    p_right: Optional[np.ndarray] = None,
    p_right_window: int = 10,
    p_right_column: Optional[str] = None,
    bins: Optional[Sequence[float]] = None,
    min_count: int = 1,
    connect: Literal["median", "mean", "none"] = "median",
    show_scatter: bool = True,
    scatter_jitter: float = 0.015,
    highlight_trial_types: Sequence[str] = (),
    highlight_colors: Optional[Sequence[str]] = None,
    highlight_point_size: float = 36.0,
    highlight_alpha: float = 0.45,
    restrict_events: Optional[Tuple[str, str]] = None,
    restrict_align: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (7.0, 4.5),
    box_color: str = "#4C72B0",
    line_color: Optional[str] = None,
    point_size: float = 12.0,
    point_alpha: float = 0.35,
    title: Optional[str] = None,
    projection_source: Literal["unbiased", "all"] = "unbiased",
) -> Dict[str, Any]:
    """Box-plot of per-trial CD projection (averaged over ``window``) grouped
    by P(right), with a line connecting the per-bin median/mean.

    Parameters
    ----------
    bins : sequence of float, optional
        Bin **edges** for grouping P(right). If ``None`` (default), the
        function groups by the discrete unique values of P(right) — appropriate
        because causal sliding windows produce values on a ``k / window`` grid.
    min_count : int, default 1
        Bins/values with fewer than this many trials are dropped.
    connect : {'median', 'mean', 'none'}
        Statistic plotted as a connecting line across boxes. ``'none'`` shows
        boxes only.
    show_scatter : bool, default True
        Overlay individual trial dots (small, jittered along x).

    All other parameters mirror
    :func:`plot_cd_projection_vs_choice_probability`.
    """
    # Reuse the scatter helper to compute proj_mean and p_right per trial,
    # but suppress its figure by passing a throwaway axes.
    _fig_tmp, _ax_tmp = plt.subplots(figsize=(1, 1))
    res = plot_cd_projection_vs_choice_probability(
        sess,
        window=window,
        p_right=p_right,
        p_right_window=p_right_window,
        p_right_column=p_right_column,
        highlight_trial_types=tuple(highlight_trial_types),
        restrict_events=restrict_events,
        restrict_align=restrict_align,
        ax=_ax_tmp,
        show_correlation=False,
        projection_source=projection_source,
    )
    plt.close(_fig_tmp)

    p = np.asarray(res["p_right"], dtype=float)
    y = np.asarray(res["proj_mean"], dtype=float)
    hl_masks_full = res.get("highlight_masks", {}) or {}
    finite = np.isfinite(p) & np.isfinite(y)
    p, y = p[finite], y[finite]
    hl_masks = {name: np.asarray(m, dtype=bool)[finite] for name, m in hl_masks_full.items()}
    if p.size == 0:
        raise ValueError(f"[{sess.session}] no finite (P(right), projection) pairs to plot.")

    if title is None:
        title = (
            f"[{sess.session}] CD projection by P(right)"
            + (f"  (window {p_right_window})" if p_right is None and p_right_column is None else "")
        )
    xlabel = (
        "P(right) (sliding, causal)" if p_right_column is None and p_right is None
        else (p_right_column or "P(right)")
    )
    return _render_pright_boxplot(
        p=p,
        y=y,
        hl_masks=hl_masks,
        bins=bins,
        min_count=min_count,
        connect=connect,
        show_scatter=show_scatter,
        scatter_jitter=scatter_jitter,
        highlight_colors=highlight_colors,
        highlight_point_size=highlight_point_size,
        highlight_alpha=highlight_alpha,
        ax=ax,
        figsize=figsize,
        box_color=box_color,
        line_color=line_color,
        point_size=point_size,
        point_alpha=point_alpha,
        title=title,
        xlabel=xlabel,
        ylabel=f"mean CD projection in [{window[0]:g}, {window[1]:g}] s",
        empty_msg_prefix=f"[{sess.session}]",
    )


def _render_pright_boxplot(
    *,
    p: np.ndarray,
    y: np.ndarray,
    hl_masks: Dict[str, np.ndarray],
    bins: Optional[Sequence[float]],
    min_count: int,
    connect: Literal["median", "mean", "none"],
    show_scatter: bool,
    scatter_jitter: float,
    highlight_colors: Optional[Sequence[str]],
    highlight_point_size: float,
    highlight_alpha: float,
    ax: Optional[plt.Axes],
    figsize: Tuple[float, float],
    box_color: str,
    line_color: Optional[str],
    point_size: float,
    point_alpha: float,
    title: str,
    xlabel: str,
    ylabel: str,
    empty_msg_prefix: str = "",
) -> Dict[str, Any]:
    """Shared box-plot renderer used by per-session and combined entry points."""

    # ----- Group trials -----
    if bins is None:
        # Use discrete grid k/p_right_window. Round to 6 decimals to suppress
        # floating-point artefacts in the unique() call.
        keys_round = np.round(p, 6)
        unique_keys = np.unique(keys_round)
        groups = [(k, y[keys_round == k]) for k in unique_keys]
        # x-position is the value itself; label too.
        positions = np.array([k for k, _ in groups], dtype=float)
        labels = [f"{k:.2f}" for k in positions]
    else:
        edges = np.asarray(bins, dtype=float)
        # np.digitize: 1..len(edges)-1 means bin i covers [edges[i-1], edges[i])
        idx = np.digitize(p, edges, right=False)
        idx = np.clip(idx, 1, len(edges) - 1)
        groups = []
        positions = []
        labels = []
        for i in range(1, len(edges)):
            sel = (idx == i)
            if sel.sum() == 0:
                continue
            center = 0.5 * (edges[i - 1] + edges[i])
            groups.append((center, y[sel]))
            positions.append(center)
            labels.append(f"[{edges[i-1]:.2f},{edges[i]:.2f})")
        positions = np.asarray(positions, dtype=float)

    # Drop sparse groups
    kept = [(pos, vals, lbl)
            for (pos, (_, vals)), lbl in zip(zip(positions, groups), labels)
            if vals.size >= min_count]
    if not kept:
        raise ValueError(
            f"{empty_msg_prefix} no P(right) groups have >= {min_count} trials."
        )
    positions = np.array([k[0] for k in kept], dtype=float)
    data = [k[1] for k in kept]
    labels = [k[2] for k in kept]
    counts = [v.size for v in data]

    # Auto box width: a fraction of the smallest neighbor gap, capped.
    if positions.size > 1:
        gaps = np.diff(np.sort(positions))
        gap = float(np.min(gaps[gaps > 0])) if np.any(gaps > 0) else 0.05
    else:
        gap = 0.05
    box_width = max(min(0.6 * gap, 0.06), 0.01)

    # ----- Plot -----
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if show_scatter:
        rng = np.random.default_rng(0)
        for pos, vals in zip(positions, data):
            jit = rng.uniform(-scatter_jitter, scatter_jitter, size=vals.size)
            ax.scatter(
                pos + jit, vals,
                s=point_size, c=box_color, alpha=point_alpha,
                edgecolors="none", zorder=1,
            )

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        zorder=2,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(box_color)
        patch.set_alpha(0.4)
        patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.4)
    for whisker in bp["whiskers"]:
        whisker.set_color("black")
    for cap in bp["caps"]:
        cap.set_color("black")

    # Connecting line
    if connect != "none":
        stats = (
            np.array([np.median(v) for v in data]) if connect == "median"
            else np.array([np.mean(v) for v in data])
        )
        order = np.argsort(positions)
        line_c = line_color if line_color is not None else box_color
        ax.plot(
            positions[order], stats[order],
            color=line_c, lw=1.8, marker="o", ms=4, zorder=3,
            label=f"{connect} per bin",
        )

    # Highlighted trial-type overlays (use the same per-trial values, plotted
    # on top of boxes with distinct colors and a small x-jitter).
    if hl_masks:
        default_palette = ["#C44E52", "#55A868", "#8172B3", "#CCB974", "#64B5CD", "#E377C2"]
        palette = list(highlight_colors) if highlight_colors is not None else default_palette
        rng_hl = np.random.default_rng(1)
        # For each highlighted trial, locate its bin position to jitter around.
        if bins is None:
            keys_round_all = np.round(p, 6)
            pos_lookup = {k: pos for k, pos in zip(np.unique(keys_round_all), np.sort(np.unique(keys_round_all)))}
            # positions are the unique values themselves
            def _bin_pos(pv: float) -> Optional[float]:
                k = float(np.round(pv, 6))
                return float(k) if k in pos_lookup else None
        else:
            edges = np.asarray(bins, dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            kept_pos_set = set(np.round(positions, 6).tolist())
            def _bin_pos(pv: float) -> Optional[float]:
                i = int(np.clip(np.digitize([pv], edges, right=False)[0], 1, len(edges) - 1))
                c = float(centers[i - 1])
                return c if round(c, 6) in kept_pos_set else None

        for j, (name, mask) in enumerate(hl_masks.items()):
            if not mask.any():
                continue
            color = palette[j % len(palette)]
            xs, ys = [], []
            for pv, yv in zip(p[mask], y[mask]):
                bp_pos = _bin_pos(float(pv))
                if bp_pos is None:
                    continue
                xs.append(bp_pos + rng_hl.uniform(-scatter_jitter, scatter_jitter))
                ys.append(yv)
            if xs:
                ax.scatter(
                    xs, ys,
                    s=highlight_point_size, c=color, alpha=highlight_alpha,
                    edgecolors="white", linewidths=0.5, zorder=4,
                    label=f"{name} (n={int(mask.sum())})",
                )

    # Count annotations above the top whisker of each box
    y_top = ax.get_ylim()[1]
    for pos, vals in zip(positions, data):
        ax.text(
            pos, y_top, f"n={vals.size}",
            ha="center", va="bottom", fontsize=7, color="0.3",
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if connect != "none" or hl_masks:
        ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()

    return {
        "positions": positions,
        "data": data,
        "counts": counts,
        "labels": labels,
        "highlight_masks": hl_masks,
    }


def plot_cd_projection_box_by_choice_probability_combined(
    sessions_data: Sequence[CDSessionData],
    *,
    window: Tuple[float, float],
    p_right_window: int = 10,
    p_right_column: Optional[str] = None,
    bins: Optional[Sequence[float]] = None,
    min_count: int = 1,
    connect: Literal["median", "mean", "none"] = "median",
    show_scatter: bool = True,
    scatter_jitter: float = 0.015,
    highlight_trial_types: Sequence[str] = (),
    highlight_colors: Optional[Sequence[str]] = None,
    highlight_point_size: float = 36.0,
    highlight_alpha: float = 0.45,
    restrict_events: Optional[Tuple[str, str]] = None,
    restrict_align: Optional[str] = None,
    per_session_zscore: bool = False,
    session_normalize: Literal["none", "zscore", "demean", "robust"] = "none",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (7.0, 4.5),
    box_color: str = "#4C72B0",
    line_color: Optional[str] = None,
    point_size: float = 12.0,
    point_alpha: float = 0.35,
    title: Optional[str] = None,
    projection_source: Literal["unbiased", "all"] = "unbiased",
) -> Dict[str, Any]:
    """Pool CD-projection-vs-P(right) box plots across multiple sessions.

    Per session, the per-trial (P(right), CD-projection) pairs are computed
    the same way as :func:`plot_cd_projection_box_by_choice_probability`
    (via the scatter helper) and then concatenated. The pooled values are
    grouped and rendered with the same box-plot renderer.

    Parameters
    ----------
    sessions_data : sequence of CDSessionData
        Pre-loaded session objects (from :func:`load_cd_session`). Sessions
        that raise during per-trial computation (missing behavior column,
        empty proj_all_trials, etc.) are skipped with a warning.
    per_session_zscore : bool, default False
        Deprecated convenience alias for ``session_normalize='zscore'``.
    session_normalize : {'none','zscore','demean','robust'}, default 'none'
        Per-session normalization applied to the CD projection before
        pooling. Useful when sessions have different baselines / scales
        (e.g. different unit counts) and pooling raw values flattens the
        across-session trend.
          - 'none'   : pool raw projections (original behavior).
          - 'demean' : subtract per-session mean. Preserves within-session
            spread; usually the best choice when scales are comparable.
          - 'zscore' : (y - mean) / std per session. Equalizes magnitude
            across sessions; may down-weight sessions with strong CD
            signal.
          - 'robust' : (y - median) / IQR per session. Like z-score but
            insensitive to outliers.

    All other parameters mirror
    :func:`plot_cd_projection_box_by_choice_probability`.
    """
    if not sessions_data:
        raise ValueError("sessions_data is empty.")

    ps: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    hl_accum: Dict[str, List[np.ndarray]] = {name: [] for name in highlight_trial_types}
    used: List[str] = []

    for sess in sessions_data:
        try:
            _fig_tmp, _ax_tmp = plt.subplots(figsize=(1, 1))
            res = plot_cd_projection_vs_choice_probability(
                sess,
                window=window,
                p_right_window=p_right_window,
                p_right_column=p_right_column,
                highlight_trial_types=tuple(highlight_trial_types),
                restrict_events=restrict_events,
                restrict_align=restrict_align,
                ax=_ax_tmp,
                show_correlation=False,
                projection_source=projection_source,
            )
            plt.close(_fig_tmp)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {sess.session}: {e}")
            continue

        p_i = np.asarray(res["p_right"], dtype=float)
        y_i = np.asarray(res["proj_mean"], dtype=float)
        finite = np.isfinite(p_i) & np.isfinite(y_i)
        p_i, y_i = p_i[finite], y_i[finite]
        if p_i.size == 0:
            continue
        if per_session_zscore:
            session_normalize = "zscore"
        if session_normalize == "zscore":
            mu = float(np.nanmean(y_i))
            sd = float(np.nanstd(y_i)) + 1e-12
            y_i = (y_i - mu) / sd
        elif session_normalize == "demean":
            y_i = y_i - float(np.nanmean(y_i))
        elif session_normalize == "robust":
            med = float(np.nanmedian(y_i))
            q75, q25 = np.nanpercentile(y_i, [75, 25])
            iqr = float(q75 - q25) + 1e-12
            y_i = (y_i - med) / iqr
        ps.append(p_i)
        ys.append(y_i)
        masks_full = res.get("highlight_masks", {}) or {}
        for name in highlight_trial_types:
            m_full = np.asarray(masks_full.get(name, np.zeros(finite.size, dtype=bool)), dtype=bool)
            hl_accum[name].append(m_full[finite])
        used.append(sess.session)

    if not ps:
        raise ValueError("No usable sessions produced (P(right), projection) pairs.")

    p = np.concatenate(ps)
    y = np.concatenate(ys)
    hl_masks = {name: np.concatenate(masks) for name, masks in hl_accum.items() if masks}

    norm_tag = {
        "none": "",
        "zscore": "  [per-session z]",
        "demean": "  [per-session demean]",
        "robust": "  [per-session robust z]",
    }[session_normalize]
    if title is None:
        title = (
            f"Combined ({len(used)} sessions) — CD projection by P(right)"
            + (f"  (window {p_right_window})" if p_right_column is None else "")
            + norm_tag
        )
    xlabel = (
        "P(right) (sliding, causal)" if p_right_column is None
        else (p_right_column or "P(right)")
    )
    ylabel_unit = {
        "none": f"mean CD projection in [{window[0]:g}, {window[1]:g}] s",
        "zscore": f"per-session z(CD projection) in [{window[0]:g}, {window[1]:g}] s",
        "demean": f"per-session demeaned CD projection in [{window[0]:g}, {window[1]:g}] s",
        "robust": f"per-session robust-z(CD projection) in [{window[0]:g}, {window[1]:g}] s",
    }
    ylabel = ylabel_unit[session_normalize]
    out = _render_pright_boxplot(
        p=p,
        y=y,
        hl_masks=hl_masks,
        bins=bins,
        min_count=min_count,
        connect=connect,
        show_scatter=show_scatter,
        scatter_jitter=scatter_jitter,
        highlight_colors=highlight_colors,
        highlight_point_size=highlight_point_size,
        highlight_alpha=highlight_alpha,
        ax=ax,
        figsize=figsize,
        box_color=box_color,
        line_color=line_color,
        point_size=point_size,
        point_alpha=point_alpha,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        empty_msg_prefix="[combined]",
    )
    out["sessions_used"] = used
    return out


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


def _select_all_trials_for_type(
    sess: CDSessionData, col: str,
) -> np.ndarray:
    """Select rows of ``sess.proj_all_trials`` whose trial IDs are in ``df[col][0]``."""
    if sess.proj_all_trials.size == 0 or sess.behavior_df is None:
        return np.empty((0, len(sess.time)), dtype=float)
    if col not in sess.behavior_df.columns:
        return np.empty((0, len(sess.time)), dtype=float)
    try:
        tids = np.asarray(sess.behavior_df[col].iloc[0], dtype=int).ravel()
    except Exception:  # noqa: BLE001
        return np.empty((0, len(sess.time)), dtype=float)
    mask = np.isin(sess.trial_id_all_trials, tids)
    return sess.proj_all_trials[mask]


def plot_cd_aggregate(
    agg: CDAggregate,
    *,
    sessions_data: Optional[Sequence[CDSessionData]] = None,
    trial_types: Optional[Sequence[str]] = None,
    distribution_window: Tuple[float, float] = (-1.0, 0.0),
    smooth_gauss: float = 0.1,
    show: bool = True,
) -> None:
    """Average + window-distribution plots pooled across sessions.

    Parameters
    ----------
    agg : CDAggregate
        Aggregate built by :func:`aggregate_cd_sessions`. Used when
        ``trial_types`` is None.
    sessions_data, trial_types : optional
        If both provided, projections for the requested behavior columns are
        pulled from each session's ``proj_all_trials`` (the all-trials
        projection onto the final CD axis) and concatenated. ``agg`` is then
        used only for the time axis.
        - One name → single-class display (no comparison plot).
        - Two names → first vs second (e.g. ``["right_choice_trials","left_choice_trials"]``).
    """
    if agg.time is None:
        print("[viz] No aggregated data; skip plots.")
        return

    # ---- Trial-types mode: pool from per-session proj_all_trials ----
    if trial_types is not None:
        if sessions_data is None:
            raise ValueError(
                "plot_cd_aggregate: sessions_data must be provided when trial_types is set."
            )
        tt_list = [trial_types] if isinstance(trial_types, str) else list(trial_types)
        if len(tt_list) not in (1, 2):
            raise ValueError("trial_types must contain 1 or 2 column names.")

        A_parts = [_select_all_trials_for_type(s, tt_list[0]) for s in sessions_data]
        A_all = _cat_or_none(A_parts)
        n_A = int(A_all.shape[0]) if A_all is not None else 0

        if len(tt_list) == 2:
            B_parts = [_select_all_trials_for_type(s, tt_list[1]) for s in sessions_data]
            B_all = _cat_or_none(B_parts)
            n_B = int(B_all.shape[0]) if B_all is not None else 0
            if A_all is None or B_all is None:
                print(f"[viz] Empty pool for {tt_list[0]} or {tt_list[1]}; skip plot.")
                return
            labels = (f"{tt_list[0]} (n={n_A})", f"{tt_list[1]} (n={n_B})")
            plot_cd_projection(
                agg.time, A_all, B_all,
                average=True,
                smooth=smooth_gauss, dt=agg.dt, smooth_mode="gaussian",
                labels=labels,
                title=f"[All Sessions] {tt_list[0]} vs {tt_list[1]} (mean±CI)",
            )
            plot_cd_window_distribution(
                agg.time, A_all, B_all,
                window=distribution_window, kind="hist", bins=40, hist_overlay=True,
                labels=labels,
                title=(
                    f"[All Sessions] {tt_list[0]} vs {tt_list[1]}, "
                    f"Window [{distribution_window[0]},{distribution_window[1]}]s"
                ),
            )
        else:
            if A_all is None:
                print(f"[viz] Empty pool for {tt_list[0]}; skip plot.")
                return
            # Single group: pass A as both A and B with empty B handled by plotter
            empty_B = np.empty((0, A_all.shape[1]), dtype=A_all.dtype)
            plot_cd_projection(
                agg.time, A_all, empty_B,
                average=True,
                smooth=smooth_gauss, dt=agg.dt, smooth_mode="gaussian",
                labels=(f"{tt_list[0]} (n={n_A})", "(none)"),
                title=f"[All Sessions] {tt_list[0]} (mean±CI)",
            )

        if show:
            plt.show()
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
