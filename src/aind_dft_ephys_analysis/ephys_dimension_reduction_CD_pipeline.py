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


# ---------------------------------------------------------------------------
# 1b. ACTION/TRANSITION 4-axis CD helpers (Prev × Upcoming 2x2 design)
# ---------------------------------------------------------------------------

#: Names of the 4 cells in the (previous choice × upcoming choice) 2x2 design.
#: These must already be present as ``{name}_trials`` columns in the per-session
#: behavior summary CSV (written by :func:`behavior_utils.find_trials`).
ACTION_CELLS: Tuple[str, str, str, str] = (
    "L_L",          # prev L, up L  (stay)
    "switch_LR",    # prev L, up R  (switch)
    "switch_RL",    # prev R, up L  (switch)
    "R_R",          # prev R, up R  (stay)
)

#: Definition of the four contrasts. Each entry maps ``axis_name`` to the two
#: cell unions that form class A / class B for that axis. After per-session
#: subsampling to ``n = min(|cell|)``, every axis below is automatically
#: balanced w.r.t. the orthogonal factors.
ACTION_AXES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    # Prev-L (L_L + switch_LR) vs Prev-R (switch_RL + R_R)
    "prev_choice":   (("L_L", "switch_LR"), ("switch_RL", "R_R")),
    # Up-L (L_L + switch_RL) vs Up-R (switch_LR + R_R)
    "up_choice":     (("L_L", "switch_RL"), ("switch_LR", "R_R")),
    # Switch (switch_LR + switch_RL) vs Stay (L_L + R_R)
    "switch_stay":   (("switch_LR", "switch_RL"), ("L_L", "R_R")),
    # Switch direction: LR vs RL (already balanced after subsampling).
    "switch_dir":    (("switch_LR",), ("switch_RL",)),
}

# ---------------------------------------------------------------------------
# Previous-trial reward design (independent 2-cell decoder)
# ---------------------------------------------------------------------------
# A separate, single-axis design that contrasts trials whose *previous* trial
# was rewarded vs. unrewarded. Cells are computed on the fly from the NWB
# ``rewarded_historyL/R`` columns (shifted by 1 trial), then per-bin balanced
# the same way the action axes are.

PREV_REWARD_CELLS: Tuple[str, str] = ("prev_rewarded", "prev_unrewarded")

PREV_REWARD_AXES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "prev_reward": (("prev_rewarded",), ("prev_unrewarded",)),
}


def compute_prev_reward_cells(session_name: str) -> Dict[str, np.ndarray]:
    """Return ``{'prev_rewarded': ids, 'prev_unrewarded': ids}`` for one session.

    Cell membership uses the previous trial's reward outcome
    (``rewarded_historyL[i-1] OR rewarded_historyR[i-1]``). Trial 0 is
    excluded (no previous trial).
    """
    from nwb_utils import NWBUtils

    nwb_data = NWBUtils.read_ophys_or_behavior_nwb(session_name=session_name)
    if nwb_data is None:
        raise FileNotFoundError(f"Could not load NWB for session '{session_name}'.")
    try:
        rL = np.asarray(nwb_data.trials["rewarded_historyL"][:], dtype=bool)
        rR = np.asarray(nwb_data.trials["rewarded_historyR"][:], dtype=bool)
    finally:
        try:
            nwb_data.io.close()
        except Exception:  # noqa: BLE001
            pass
    rewarded = np.logical_or(rL, rR)
    idx = np.arange(1, len(rewarded), dtype=int)
    prev_rew = rewarded[:-1]
    return {
        "prev_rewarded":   idx[prev_rew],
        "prev_unrewarded": idx[~prev_rew],
    }


def _balance_action_cells(
    behavior_csv: str | Path,
    *,
    seed: int = 0,
    suffix: str = "balanced",
    n_per_cell: Optional[int] = None,
    overwrite_columns: bool = True,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Subsample the four action-transition cells to a balanced count and
    persist them (plus axis unions) as new ``*_{suffix}_trials`` columns in
    the per-session behavior summary CSV.

    Parameters
    ----------
    behavior_csv : path
        Per-session behavior summary CSV (``behavior_summary-{session}.csv``)
        produced by :func:`behavior_utils.generate_behavior_summary`. Must
        contain the four base columns ``L_L_trials``, ``switch_LR_trials``,
        ``switch_RL_trials``, ``R_R_trials``.
    seed : int, default 0
        RNG seed for the per-cell subsample.
    suffix : str, default ``"balanced"``
        Column-name suffix. New columns will be e.g.
        ``L_L_{suffix}_trials`` and ``prev_L_{suffix}_trials``.
    n_per_cell : int, optional
        If given, force the per-cell sample size. Otherwise uses
        ``min(|cell|)`` across the four cells (the largest count where
        every cell still has enough trials).
    overwrite_columns : bool, default True
        If False and the target columns already exist, the function reads
        them from the CSV without re-sampling and returns the existing
        names. Useful when chaining repeated calls.

    Returns
    -------
    tuple
        ``(axis_columns, counts)`` where:

        - ``axis_columns`` maps each axis in :data:`ACTION_AXES` to a
          ``(colA, colB)`` pair — these are the column names you'd pass
          as ``trial_types=`` to :func:`build_cd_dataset`.
        - ``counts`` maps cell name → balanced sample size (also includes
          ``"n_per_cell"`` and ``"n_min_raw"``).
    """
    from general_utils import smart_read_csv

    behavior_csv = Path(behavior_csv)
    df = smart_read_csv(str(behavior_csv))

    # Read the raw cell IDs.
    raw_ids: Dict[str, np.ndarray] = {}
    for cell in ACTION_CELLS:
        col = f"{cell}_trials"
        if col not in df.columns:
            raise KeyError(
                f"{behavior_csv.name}: missing column {col!r}; "
                f"regenerate behavior summary."
            )
        ids = np.asarray(df[col].iloc[0], dtype=int).ravel()
        raw_ids[cell] = np.unique(ids[ids >= 0])

    n_min_raw = int(min(v.size for v in raw_ids.values()))
    if n_per_cell is None:
        n_per_cell = n_min_raw
    if n_per_cell <= 0:
        raise ValueError(
            f"{behavior_csv.name}: cannot balance — at least one of "
            f"{ACTION_CELLS} is empty (sizes: "
            f"{ {k: int(v.size) for k, v in raw_ids.items()} })."
        )

    # Subsample each cell deterministically.
    rng = np.random.default_rng(int(seed))
    balanced_ids: Dict[str, np.ndarray] = {}
    for cell in ACTION_CELLS:
        pool = raw_ids[cell]
        if pool.size <= n_per_cell:
            balanced_ids[cell] = np.sort(pool)
        else:
            pick = rng.choice(pool, size=int(n_per_cell), replace=False)
            balanced_ids[cell] = np.sort(pick.astype(int))

    # Build axis unions.
    axis_unions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for axis, (cells_a, cells_b) in ACTION_AXES.items():
        ids_a = np.unique(np.concatenate([balanced_ids[c] for c in cells_a]))
        ids_b = np.unique(np.concatenate([balanced_ids[c] for c in cells_b]))
        axis_unions[axis] = (ids_a.astype(int), ids_b.astype(int))

    # Build column-name maps.
    cell_cols: Dict[str, str] = {c: f"{c}_{suffix}_trials" for c in ACTION_CELLS}
    axis_a_name = {
        "prev_choice": "prev_L",
        "up_choice":   "up_L",
        "switch_stay": "switch",
        "switch_dir":  "switch_LR",
    }
    axis_b_name = {
        "prev_choice": "prev_R",
        "up_choice":   "up_R",
        "switch_stay": "stay",
        "switch_dir":  "switch_RL",
    }
    axis_columns: Dict[str, Tuple[str, str]] = {
        axis: (
            f"{axis_a_name[axis]}_{suffix}_trials",
            f"{axis_b_name[axis]}_{suffix}_trials",
        )
        for axis in ACTION_AXES
    }

    # Decide whether to skip (re-read existing columns).
    target_cols = set(cell_cols.values()) | {c for pair in axis_columns.values() for c in pair}
    if (not overwrite_columns) and target_cols.issubset(df.columns):
        counts = {c: int(np.asarray(df[cell_cols[c]].iloc[0], dtype=int).size) for c in ACTION_CELLS}
        counts["n_per_cell"] = int(min(counts.values())) if counts else 0
        counts["n_min_raw"] = n_min_raw
        return axis_columns, counts

    # Write columns.
    for cell, col in cell_cols.items():
        df[col] = [balanced_ids[cell].tolist()]
    for axis, (colA, colB) in axis_columns.items():
        ids_a, ids_b = axis_unions[axis]
        df[colA] = [ids_a.tolist()]
        df[colB] = [ids_b.tolist()]

    df.to_csv(behavior_csv, index=False)

    counts = {c: int(balanced_ids[c].size) for c in ACTION_CELLS}
    counts["n_per_cell"] = int(n_per_cell)
    counts["n_min_raw"] = n_min_raw
    return axis_columns, counts


def build_action_transition_cds(
    *,
    sessions: Iterable[str],
    psth_root: str | Path,
    behavior_root: str | Path,
    cd_root: str | Path,
    metadata: Optional[pd.DataFrame] = None,
    seed: int = 0,
    suffix: str = "balanced",
    n_per_cell: Optional[int] = None,
    axes: Sequence[str] = ("prev_choice", "up_choice", "switch_stay", "switch_dir"),
    overwrite_columns: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> Tuple[List[Tuple[str, ...]], Dict[str, Dict[str, int]]]:
    """Build four balanced CD axes per session for the Prev x Upcoming design.

    For every session this:
      1. Reads ``behavior_summary-{session}.csv``.
      2. Subsamples ``L_L``, ``switch_LR``, ``switch_RL``, ``R_R`` to a common
         per-cell size ``n = min(...)`` (with the supplied ``seed``) so that
         every contrast is balanced w.r.t. the orthogonal factor.
      3. Persists those balanced lists plus the four axis unions
         (``prev_L_balanced_trials``, ..., ``stay_balanced_trials``) as new
         columns in the same CSV so downstream loaders can reach them by
         name.
      4. Calls :func:`build_cd_dataset` once per axis (``axes`` argument)
         using the corresponding union columns as ``trial_types``.

    Parameters
    ----------
    axes : sequence of str
        Subset of :data:`ACTION_AXES` keys to actually fit. Default = all
        four. Each produces its own CD zarr per session/region/window.
    seed, suffix, n_per_cell, overwrite_columns
        Forwarded to :func:`_balance_action_cells`.
    **kwargs
        Forwarded to :func:`build_cd_dataset` (e.g. ``align``, ``binsize``,
        ``brain_regions_groups``, ``time_windows``, ``min_units_num``,
        ``norm_mode``, ``random_state``, ``overwrite``, ...).

    Returns
    -------
    tuple
        ``(failed, counts_per_session)`` where ``failed`` is the merged
        failure list across all axes (same shape as
        :func:`build_cd_dataset`) and ``counts_per_session`` maps session
        name to the per-cell balanced counts.
    """
    behavior_root = Path(behavior_root)
    sessions = list(sessions)
    failed_all: List[Tuple[str, ...]] = []
    counts_per_session: Dict[str, Dict[str, int]] = {}
    axis_columns_per_session: Dict[str, Dict[str, Tuple[str, str]]] = {}

    # ----- 1) Balance each session -----
    sessions_for_axes: List[str] = []
    for session in sessions:
        beh_path = behavior_root / f"behavior_summary-{session}.csv"
        if not beh_path.exists():
            print(f"[skip] {session}: behavior CSV not found at {beh_path}")
            continue
        try:
            axis_columns, counts = _balance_action_cells(
                beh_path,
                seed=seed,
                suffix=suffix,
                n_per_cell=n_per_cell,
                overwrite_columns=overwrite_columns,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {session}: balancing failed: {e}")
            failed_all.append((session, "action_axes", "balance", str(e)))
            continue
        counts_per_session[session] = counts
        axis_columns_per_session[session] = axis_columns
        sessions_for_axes.append(session)
        if verbose:
            cell_n = {c: counts[c] for c in ACTION_CELLS}
            print(
                f"[{session}] balanced cells (seed={seed}): {cell_n} "
                f"(min raw = {counts['n_min_raw']}, per-cell = {counts['n_per_cell']})"
            )

    if not sessions_for_axes:
        print("\nNo usable sessions after balancing.")
        return failed_all, counts_per_session

    # ----- 2) For each axis, build a CD zarr across the kept sessions -----
    for axis in axes:
        if axis not in ACTION_AXES:
            raise ValueError(
                f"Unknown axis {axis!r}; expected one of {list(ACTION_AXES.keys())}."
            )
        # All sessions use the same column-name pair for a given axis (built
        # deterministically by `_balance_action_cells`).
        ref_cols = axis_columns_per_session[sessions_for_axes[0]][axis]
        if verbose:
            print(
                f"\n>>> Building axis {axis!r}: A={ref_cols[0]} | B={ref_cols[1]} "
                f"({len(sessions_for_axes)} sessions)"
            )
        axis_failed = build_cd_dataset(
            sessions=sessions_for_axes,
            psth_root=psth_root,
            behavior_root=behavior_root,
            cd_root=cd_root,
            metadata=metadata,
            trial_types=ref_cols,
            **kwargs,
        )
        failed_all.extend(axis_failed)

    return failed_all, counts_per_session


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
    ``"trial_end"``, ``"go_cue"``, ``"previous_trial_go_cue"``,
    ``"previous_trial_start"``, ``"previous_trial_end"``. If ``align`` is
    ``None`` it defaults to ``event_start`` (so the start offset is exactly 0).

    Trials with NaN times (including trial 0 when any ``previous_trial_*``
    event is used) are dropped automatically.
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
    vrange_quantile: float = 0.98,
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
    vrange_quantile: float = 0.98,
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
# Single-trial dwell metrics (persistency index)
# ---------------------------------------------------------------------------

def _dwell_metrics_from_trace(
    trace: np.ndarray,
    time: np.ndarray,
    *,
    t0: float,
    t1: float,
    center: float,
    class_sign: int,
    deadband: float = 0.0,
) -> Dict[str, float]:
    """Per-trial dwell summary for one 1-D projection trace.

    ``trace``/``time`` are equal-length 1-D arrays; NaN samples are treated
    as missing and excluded from every count. Only samples whose ``time`` is
    in ``[t0, t1]`` contribute. ``center`` is subtracted before scoring; a
    sample is considered "on the class side" when its (centered) sign equals
    ``class_sign`` (``+1`` for class A, ``-1`` for class B, ``0`` to disable
    the on-class metrics). A non-zero ``deadband`` zeros out any sample with
    ``abs(centered) < deadband`` (i.e. it is treated as ambiguous / not on
    either side).

    Returns a dict of scalar metrics; if no valid samples fall in the
    window, every value is NaN (except counts, which are 0).
    """
    if trace.size == 0 or time.size == 0:
        nan = float("nan")
        return {
            "valid_duration_sec": 0.0, "n_valid_samples": 0,
            "mean_projection": nan, "median_projection": nan,
            "mean_abs_projection": nan,
            "frac_pos": nan, "frac_neg": nan, "frac_zero": nan,
            "mean_sign": nan, "n_sign_changes": 0,
            "longest_run_pos_sec": nan, "longest_run_neg_sec": nan,
            "frac_on_class": nan, "frac_off_class": nan,
            "mean_signed_distance_on_class": nan,
            "longest_run_on_class_sec": nan, "longest_run_off_class_sec": nan,
            "persistency_index": nan,
        }

    sel = (time >= t0) & (time <= t1)
    x = trace[sel].astype(float, copy=False)
    t = time[sel]
    valid = np.isfinite(x)
    if not np.any(valid):
        return _dwell_metrics_from_trace(
            np.empty(0, dtype=float), np.empty(0, dtype=float),
            t0=t0, t1=t1, center=center, class_sign=class_sign,
            deadband=deadband,
        )
    x = x[valid] - float(center)
    t = t[valid]

    # Sample spacing (use a robust median diff; falls back to total / N).
    if t.size >= 2:
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            dt = float((t[-1] - t[0]) / max(t.size - 1, 1))
    else:
        dt = 0.0
    valid_dur = float(dt * x.size)

    # Sign per sample with optional deadband.
    s = np.sign(x)
    if deadband > 0:
        s = np.where(np.abs(x) < float(deadband), 0.0, s)

    n_pos = int(np.sum(s > 0))
    n_neg = int(np.sum(s < 0))
    n_zero = int(np.sum(s == 0))
    n = int(s.size)

    def _longest_run(mask: np.ndarray) -> int:
        if mask.size == 0 or not np.any(mask):
            return 0
        best = cur = 0
        for v in mask:
            if v:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        return int(best)

    longest_pos = _longest_run(s > 0)
    longest_neg = _longest_run(s < 0)

    # Zero-crossings: count transitions between strictly opposite signs,
    # treating zeros (deadband or exact 0) as transparent.
    nz = s[s != 0]
    if nz.size >= 2:
        n_sign_changes = int(np.sum(np.diff(np.sign(nz)) != 0))
    else:
        n_sign_changes = 0

    if class_sign == 0:
        frac_on = float("nan")
        frac_off = float("nan")
        longest_on = float("nan")
        longest_off = float("nan")
        mean_on_distance = float("nan")
        persistency = float("nan")
    else:
        cs = int(np.sign(class_sign))
        on_mask = s == cs
        off_mask = s == -cs
        frac_on = float(np.sum(on_mask) / n) if n else float("nan")
        frac_off = float(np.sum(off_mask) / n) if n else float("nan")
        longest_on = float(_longest_run(on_mask) * dt)
        longest_off = float(_longest_run(off_mask) * dt)
        mean_on_distance = float(cs * np.mean(x))
        # Persistency index in [-1, +1]: +1 if always on own side, -1 if
        # always off; 0 if equally split. Computed from non-zero samples
        # so deadband samples don't pull the score toward 0.
        non_zero = n_pos + n_neg
        if non_zero > 0:
            persistency = float(cs * (n_pos - n_neg) / non_zero)
        else:
            persistency = float("nan")

    return {
        "valid_duration_sec": valid_dur,
        "n_valid_samples": n,
        "mean_projection": float(np.mean(x)),
        "median_projection": float(np.median(x)),
        "mean_abs_projection": float(np.mean(np.abs(x))),
        "frac_pos": float(n_pos / n) if n else float("nan"),
        "frac_neg": float(n_neg / n) if n else float("nan"),
        "frac_zero": float(n_zero / n) if n else float("nan"),
        "mean_sign": float(np.mean(s)) if n else float("nan"),
        "n_sign_changes": n_sign_changes,
        "longest_run_pos_sec": float(longest_pos * dt),
        "longest_run_neg_sec": float(longest_neg * dt),
        "frac_on_class": frac_on,
        "frac_off_class": frac_off,
        "mean_signed_distance_on_class": mean_on_distance,
        "longest_run_on_class_sec": longest_on,
        "longest_run_off_class_sec": longest_off,
        "persistency_index": persistency,
    }


def compute_cd_dwell_metrics(
    sess: CDSessionData,
    *,
    trial_types: Optional[Sequence[str]] = None,
    split: Literal["train", "test"] = "test",
    time_window: Tuple[float, float] = (0.0, 5.0),
    restrict_events: Optional[Tuple[str, str]] = None,
    restrict_align: Optional[str] = None,
    restrict_window_per_trial: Optional[Dict[int, Tuple[float, float]]] = None,
    smooth_gauss: float = 0.1,
    smooth_mode: Literal["gaussian", "moving"] = "gaussian",
    projection_source: Literal["unbiased", "all"] = "unbiased",
    center: float | Literal["cross_class_median"] = 0.0,
    deadband: float = 0.0,
    class_signs: Tuple[int, int] = (1, -1),
) -> pd.DataFrame:
    """Per-trial dwell / persistency metrics for one CD session.

    For each selected trial, compute a battery of scalar metrics that
    summarize how persistently the single-trial projection stays on its
    class side within ``time_window``. The headline metric is
    ``persistency_index`` ``in [-1, +1]``:

    - ``+1`` â†’ every (non-deadband) sample is on the trial's own-class side
    - ``-1`` â†’ every (non-deadband) sample is on the other class's side
    - ``0``  â†’ time on either side is equal

    Trial selection, per-trial re-alignment, and per-trial masking mirror
    :func:`plot_cd_session_heatmap`. When ``trial_types`` is given, the
    final all-trials projection (``projection_source``) is used; otherwise
    the train/test CV traces are used (``split``).

    Parameters
    ----------
    time_window
        ``(t0, t1)`` seconds, in the time frame of the trace shown to the
        scorer (i.e. after any re-alignment from ``restrict_align``). The
        intersection with each trial's per-trial valid window is used.
    center
        Origin subtracted from each sample before sign-scoring. Either a
        float (default ``0``) or ``"cross_class_median"`` (use the median of
        every (trial, time) sample across both classes in ``time_window`` as
        a data-driven decision boundary).
    deadband
        Samples with ``abs(value - center) < deadband`` are treated as
        ambiguous and contribute to neither side (still counted in
        ``valid_duration_sec`` and in ``frac_zero``).
    class_signs
        Two-tuple ``(sign_A, sign_B)`` mapping the class index (``0`` for the
        first ``trial_types`` entry / class A, ``1`` for the second / class B)
        to its expected projection-side sign (``+1`` / ``-1``). Default
        ``(+1, -1)``.

    Returns
    -------
    pd.DataFrame
        Long-form, one row per trial. Columns include ``session``,
        ``class_name``, ``class_index``, ``class_sign``, ``trial_id``, plus
        the scalar metrics produced by :func:`_dwell_metrics_from_trace`.
    """
    # --- Trial selection ---------------------------------------------------
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
    else:
        if split == "train":
            raw_A, raw_B = sess.proj_train_A, sess.proj_train_B
            ids_A, ids_B = sess.trial_id_train_A, sess.trial_id_train_B
        elif split == "test":
            raw_A, raw_B = sess.proj_test_A, sess.proj_test_B
            ids_A, ids_B = sess.trial_id_test_A, sess.trial_id_test_B
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        name_A, name_B = sess.trial_types

    proj_A = raw_A
    proj_B = raw_B

    # --- Per-trial re-alignment (build_align -> restrict_align) ------------
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

    # --- Per-trial eligibility masking + smoothing -------------------------
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
    elif smooth_gauss is not None and smooth_gauss > 0:
        # No per-trial mask: smooth the whole row (rows that are all-NaN
        # remain all-NaN, so this is safe for sparse data too).
        from scipy.ndimage import gaussian_filter1d, uniform_filter1d
        kp = max(1, int(round(float(smooth_gauss) / (sess.dt or 1.0))))

        def _smooth_rows(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            out = arr.astype(float, copy=True)
            for i in range(out.shape[0]):
                row = out[i]
                valid = np.isfinite(row)
                if not np.any(valid):
                    continue
                seg = row[valid]
                if seg.size > 1:
                    if smooth_mode == "gaussian":
                        seg = gaussian_filter1d(seg, sigma=kp, mode="nearest")
                    else:
                        seg = uniform_filter1d(seg, size=kp, mode="nearest")
                out[i, valid] = seg
            return out

        proj_A = _smooth_rows(proj_A)
        proj_B = _smooth_rows(proj_B)

    # --- Center selection --------------------------------------------------
    if isinstance(center, str):
        if center != "cross_class_median":
            raise ValueError(
                f"center must be a float or 'cross_class_median', got {center!r}"
            )
        # Pool every (trial, time) sample inside time_window across both classes.
        tmask = (sess.time >= float(time_window[0])) & (sess.time <= float(time_window[1]))
        pool = []
        if proj_A.size and proj_A.ndim == 2:
            pool.append(proj_A[:, tmask].ravel())
        if proj_B.size and proj_B.ndim == 2:
            pool.append(proj_B[:, tmask].ravel())
        if pool:
            all_pool = np.concatenate(pool)
            finite = all_pool[np.isfinite(all_pool)]
            center_val = float(np.median(finite)) if finite.size else 0.0
        else:
            center_val = 0.0
    else:
        center_val = float(center)

    # --- Per-trial dwell metrics ------------------------------------------
    sign_A, sign_B = int(class_signs[0]), int(class_signs[1])
    t0, t1 = float(time_window[0]), float(time_window[1])

    rows: List[Dict[str, Any]] = []

    def _emit(proj: np.ndarray, ids: np.ndarray, name: str, class_idx: int,
              class_sign: int) -> None:
        if proj.size == 0 or proj.ndim != 2:
            return
        for i, tid in enumerate(ids):
            m = _dwell_metrics_from_trace(
                proj[i], sess.time,
                t0=t0, t1=t1,
                center=center_val,
                class_sign=class_sign,
                deadband=float(deadband),
            )
            m.update({
                "session": sess.session,
                "trial_id": int(tid),
                "class_name": name,
                "class_index": class_idx,
                "class_sign": class_sign,
                "center": center_val,
                "deadband": float(deadband),
                "t_start": t0,
                "t_end": t1,
            })
            rows.append(m)

    _emit(proj_A, ids_A, name_A, 0, sign_A)
    if name_B:
        _emit(proj_B, ids_B, name_B, 1, sign_B)

    if not rows:
        return pd.DataFrame()

    cols_first = [
        "session", "trial_id", "class_name", "class_index", "class_sign",
        "t_start", "t_end", "center", "deadband",
    ]
    df = pd.DataFrame(rows)
    rest = [c for c in df.columns if c not in cols_first]
    return df[cols_first + rest]


def compute_cd_dwell_metrics_multi(
    sessions_data: Iterable[CDSessionData],
    **kwargs: Any,
) -> pd.DataFrame:
    """Run :func:`compute_cd_dwell_metrics` on many sessions; concat rows.

    Sessions that raise are skipped with a printed warning. Returns a long
    DataFrame keyed by ``session`` + ``trial_id``; empty if nothing succeeded.
    """
    frames: List[pd.DataFrame] = []
    for sess in sessions_data:
        try:
            df = compute_cd_dwell_metrics(sess, **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {getattr(sess, 'session', '?')}: {e}")
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_persistency_vs_iti(
    dwell_df: pd.DataFrame,
    *,
    bin_edges: Optional[Sequence[float]] = None,
    n_bins: int = 10,
    upper_quantile: float = 0.98,
    min_count_for_violin: int = 5,
    jitter: float = 0.02,
    scatter_alpha: float = 0.25,
    scatter_size: float = 8.0,
    rng_seed: int = 0,
    figsize: Tuple[float, float] = (12, 4),
    title: Optional[str] = None,
    print_stats: bool = True,
) -> None:
    """Plot ``persistency_index`` vs ITI length from a dwell-metrics frame.

    Two panels:

    - Left: per-trial scatter (lightly x-jittered) and per-bin median trace,
      one color per ``class_name``.
    - Right: pooled-class violin of ``persistency_index`` within ITI bins.

    Also prints the pooled and per-class Spearman correlations when
    ``print_stats=True``.

    Parameters
    ----------
    dwell_df
        Long-form DataFrame returned by :func:`compute_cd_dwell_metrics` /
        :func:`compute_cd_dwell_metrics_multi`. Must contain
        ``persistency_index``, ``valid_duration_sec``, ``class_name``, and
        ``session``.
    bin_edges
        Explicit ITI bin edges (seconds). If ``None``, ``n_bins`` evenly
        spaced bins are built between 0 and the ``upper_quantile`` of
        ``valid_duration_sec``.
    n_bins
        Number of bins when ``bin_edges`` is not provided.
    upper_quantile
        Upper quantile of ``valid_duration_sec`` used as the top bin edge
        when ``bin_edges`` is not provided.
    min_count_for_violin
        Skip violin bodies for bins with fewer than this many trials.
    jitter, scatter_alpha, scatter_size
        Cosmetic scatter knobs.
    rng_seed
        Seed for the x-jitter RNG (kept deterministic for reproducibility).
    figsize
        Matplotlib figure size in inches.
    title
        Optional override for the suptitle.
    print_stats
        Print Spearman correlations (pooled + per-class) before plotting.
    """
    from scipy.stats import spearmanr

    if dwell_df is None or dwell_df.empty:
        print("No dwell metrics computed.")
        return

    required = {"persistency_index", "valid_duration_sec", "class_name", "session"}
    missing = required - set(dwell_df.columns)
    if missing:
        raise KeyError(
            f"dwell_df is missing required column(s): {sorted(missing)}"
        )

    df = dwell_df.dropna(subset=["persistency_index", "valid_duration_sec"]).copy()
    if df.empty:
        print("No dwell metrics after dropping NaNs.")
        return

    classes = list(df["class_name"].unique())
    colors = {c: f"C{i}" for i, c in enumerate(classes)}

    if bin_edges is None:
        iti_hi = float(np.quantile(df["valid_duration_sec"], float(upper_quantile)))
        edges = np.linspace(0, max(iti_hi, 1.0), int(n_bins) + 1)
    else:
        edges = np.asarray(bin_edges, dtype=float)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    df["iti_bin"] = pd.cut(df["valid_duration_sec"], edges, include_lowest=True)

    rho_all, p_all = spearmanr(df["valid_duration_sec"], df["persistency_index"])
    if print_stats:
        print(f"Spearman rho (pooled): {rho_all:+.3f}  (p={p_all:.2e},  n={len(df)})")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Left: scatter + per-bin median per class ---
    ax = axes[0]
    rng = np.random.default_rng(rng_seed)
    for c in classes:
        sub = df[df["class_name"] == c]
        jit = rng.uniform(-float(jitter), float(jitter), size=len(sub))
        ax.scatter(
            sub["valid_duration_sec"] + jit,
            sub["persistency_index"],
            s=float(scatter_size), alpha=float(scatter_alpha),
            color=colors[c], edgecolor="none",
        )
        med = (
            sub.groupby("iti_bin", observed=True)["persistency_index"]
            .median().reindex(df["iti_bin"].cat.categories)
        )
        ax.plot(bin_centers, med.values, color=colors[c], lw=2,
                marker="o", label=f"{c} (median)")
        if print_stats:
            rho, p = spearmanr(sub["valid_duration_sec"], sub["persistency_index"])
            print(f"  {c}: Spearman rho = {rho:+.3f}  (p={p:.2e},  n={len(sub)})")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlim(0, edges[-1])
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("ITI length (valid_duration_sec, s)")
    ax.set_ylabel("persistency_index")
    ax.set_title("Persistency vs ITI — per class")
    ax.legend(loc="lower right", fontsize=9)

    # --- Right: pooled violin per ITI bin ---
    ax = axes[1]
    data_by_bin = [
        df.loc[df["iti_bin"] == lvl, "persistency_index"].values
        for lvl in df["iti_bin"].cat.categories
    ]
    valid = [(c, d) for c, d in zip(bin_centers, data_by_bin)
             if d.size > int(min_count_for_violin)]
    if valid:
        vcenters, vdata = zip(*valid)
        parts = ax.violinplot(
            vdata, positions=vcenters,
            widths=0.9 * (edges[1] - edges[0]),
            showmedians=True, showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_alpha(0.4)
            body.set_facecolor("steelblue")
        for c, d in valid:
            ax.text(c, 1.03, f"n={d.size}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlim(0, edges[-1])
    ax.set_ylim(-1.05, 1.10)
    ax.set_xlabel("ITI length bin center (s)")
    ax.set_ylabel("persistency_index (pooled classes)")
    ax.set_title("Persistency by ITI bin")

    if title is None:
        title = (
            f"Persistency vs ITI — {len(classes)} classes, "
            f"{df['session'].nunique()} sessions"
        )
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Single-trial transient-vs-persistent metrics
# ---------------------------------------------------------------------------

@dataclass
class _PreparedCDClasses:
    """Output of :func:`_prepare_cd_session_for_scoring`."""
    proj_A: np.ndarray
    ids_A: np.ndarray
    name_A: str
    sign_A: int
    proj_B: np.ndarray
    ids_B: np.ndarray
    name_B: str
    sign_B: int
    center_val: float


def _prepare_cd_session_for_scoring(
    sess: CDSessionData,
    *,
    trial_types: Optional[Sequence[str]] = None,
    split: Literal["train", "test"] = "test",
    time_window: Tuple[float, float] = (0.0, 5.0),
    restrict_events: Optional[Tuple[str, str]] = None,
    restrict_align: Optional[str] = None,
    restrict_window_per_trial: Optional[Dict[int, Tuple[float, float]]] = None,
    smooth_gauss: float = 0.1,
    smooth_mode: Literal["gaussian", "moving"] = "gaussian",
    projection_source: Literal["unbiased", "all"] = "unbiased",
    center: float | Literal["cross_class_median"] = 0.0,
    class_signs: Tuple[int, int] = (1, -1),
) -> _PreparedCDClasses:
    """Prepare per-class projection traces for any single-trial scorer.

    Encapsulates the trial-prep used by :func:`compute_cd_dwell_metrics`
    (trial selection, per-trial re-alignment, per-trial masking +
    smoothing, and center selection) so that other per-trial metric
    functions (e.g. :func:`compute_cd_transient_metrics`) can reuse the
    same logic and produce rows that join row-for-row with the dwell
    output on ``(session, trial_id, class_index)``.
    """
    # --- Trial selection ---------------------------------------------------
    if trial_types is not None:
        proj_all_arr = _pick_proj_all(sess, projection_source)
        if proj_all_arr.size == 0:
            raise ValueError(
                f"[{sess.session}] proj_all_trials is empty; rebuild CD zarr "
                "to include projection_trace_all_trials."
            )
        if sess.behavior_df is None:
            raise ValueError(
                f"[{sess.session}] behavior_df missing; cannot resolve trial_types."
            )
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
    else:
        if split == "train":
            raw_A, raw_B = sess.proj_train_A, sess.proj_train_B
            ids_A, ids_B = sess.trial_id_train_A, sess.trial_id_train_B
        elif split == "test":
            raw_A, raw_B = sess.proj_test_A, sess.proj_test_B
            ids_A, ids_B = sess.trial_id_test_A, sess.trial_id_test_B
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        name_A, name_B = sess.trial_types

    proj_A = raw_A
    proj_B = raw_B

    # --- Per-trial re-alignment (build_align -> restrict_align) ------------
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

    # --- Per-trial eligibility masking + smoothing -------------------------
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
    elif smooth_gauss is not None and smooth_gauss > 0:
        from scipy.ndimage import gaussian_filter1d, uniform_filter1d
        kp = max(1, int(round(float(smooth_gauss) / (sess.dt or 1.0))))

        def _smooth_rows(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            out = arr.astype(float, copy=True)
            for i in range(out.shape[0]):
                row = out[i]
                valid = np.isfinite(row)
                if not np.any(valid):
                    continue
                seg = row[valid]
                if seg.size > 1:
                    if smooth_mode == "gaussian":
                        seg = gaussian_filter1d(seg, sigma=kp, mode="nearest")
                    else:
                        seg = uniform_filter1d(seg, size=kp, mode="nearest")
                out[i, valid] = seg
            return out

        proj_A = _smooth_rows(proj_A)
        proj_B = _smooth_rows(proj_B)

    # --- Center selection --------------------------------------------------
    if isinstance(center, str):
        if center != "cross_class_median":
            raise ValueError(
                f"center must be a float or 'cross_class_median', got {center!r}"
            )
        tmask = (sess.time >= float(time_window[0])) & (sess.time <= float(time_window[1]))
        pool = []
        if proj_A.size and proj_A.ndim == 2:
            pool.append(proj_A[:, tmask].ravel())
        if proj_B.size and proj_B.ndim == 2:
            pool.append(proj_B[:, tmask].ravel())
        if pool:
            all_pool = np.concatenate(pool)
            finite = all_pool[np.isfinite(all_pool)]
            center_val = float(np.median(finite)) if finite.size else 0.0
        else:
            center_val = 0.0
    else:
        center_val = float(center)

    sign_A, sign_B = int(class_signs[0]), int(class_signs[1])
    return _PreparedCDClasses(
        proj_A=proj_A, ids_A=ids_A, name_A=name_A, sign_A=sign_A,
        proj_B=proj_B, ids_B=ids_B, name_B=name_B, sign_B=sign_B,
        center_val=center_val,
    )


def _transient_metrics_from_trace(
    trace: np.ndarray,
    time: np.ndarray,
    *,
    t0: float,
    t1: float,
    center: float,
    class_sign: int,
    signal_mode: Literal["abs", "rectified_on_class", "centered_squared"] = "abs",
    peak_height_frac: float = 0.5,
    min_peak_distance_sec: float = 0.05,
) -> Dict[str, float]:
    """Per-trial transient-vs-persistent metrics for one 1-D trace.

    Returns a dict of scalar metrics computed on a non-negative signal
    ``s(t)`` derived from the centered trace ``x(t) = trace - center``:

    - **L_eff_sec**: effective lifetime,
      :math:`L_\\text{eff} = (\\sum s^2)^2 / \\sum s^4 \\cdot dt` (seconds).
      Flat plateau of length T → L_eff = T. Delta-like spike of width δ
      → L_eff ≈ δ.
    - **L_eff_frac**: ``L_eff_sec / valid_duration_sec``, clipped to
      ``(0, 1]``. 1 = fully persistent, → 0 = single transient bump.
    - **temporal_gini**: Gini concentration of |x|. 0 = uniform across
      time (persistent), 1 = all mass at one sample. Scale-invariant.
    - **peak_to_mean**: ``max(s) / mean(s)`` (≥ 1). Large = bumpy,
      ≈ 1 = flat.
    - **dominant_bump_fwhm_sec**: FWHM (seconds) of the tallest peak in
      ``s`` at half its height.
    - **n_bumps**: number of peaks in ``s`` above
      ``peak_height_frac * max(s)``, with minimum spacing
      ``min_peak_distance_sec``.

    Parameters
    ----------
    signal_mode
        How to convert the centered trace into the non-negative scoring
        signal ``s``:
        - ``"abs"`` (default): ``s = |x|`` (energy on either side).
        - ``"rectified_on_class"``: ``s = max(0, class_sign * x)`` — only
          on-class excursions contribute. ``class_sign=0`` falls back to
          ``"abs"``.
        - ``"centered_squared"``: ``s = x**2`` (emphasises big excursions).
    """
    nan = float("nan")
    blank = {
        "valid_duration_sec": 0.0, "n_valid_samples": 0,
        "L_eff_sec": nan, "L_eff_frac": nan,
        "temporal_gini": nan,
        "peak_to_mean": nan,
        "dominant_bump_fwhm_sec": nan,
        "n_bumps": 0,
    }
    if trace.size == 0 or time.size == 0:
        return dict(blank)

    sel = (time >= t0) & (time <= t1)
    x = trace[sel].astype(float, copy=False)
    t = time[sel]
    valid = np.isfinite(x)
    if not np.any(valid):
        return dict(blank)
    x = x[valid] - float(center)
    t = t[valid]

    if t.size >= 2:
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            dt = float((t[-1] - t[0]) / max(t.size - 1, 1))
    else:
        dt = 0.0
    valid_dur = float(dt * x.size)

    if signal_mode == "abs":
        s = np.abs(x)
    elif signal_mode == "rectified_on_class":
        cs = int(np.sign(class_sign)) if class_sign != 0 else 0
        s = np.maximum(0.0, cs * x) if cs != 0 else np.abs(x)
    elif signal_mode == "centered_squared":
        s = x * x
    else:
        raise ValueError(
            "signal_mode must be 'abs' | 'rectified_on_class' | 'centered_squared', "
            f"got {signal_mode!r}"
        )

    s_max = float(np.max(s)) if s.size else 0.0
    if not np.isfinite(s_max) or s_max <= 0.0:
        out = dict(blank)
        out["valid_duration_sec"] = valid_dur
        out["n_valid_samples"] = int(s.size)
        return out

    # --- 1. Effective lifetime --------------------------------------------
    s2 = s * s
    s4 = s2 * s2
    num = float(s2.sum()) ** 2
    den = float(s4.sum())
    if den > 0 and dt > 0:
        L_eff_sec = num / den * dt
        if valid_dur > 0:
            L_eff_frac = min(L_eff_sec / valid_dur, 1.0)
        else:
            L_eff_frac = nan
    else:
        L_eff_sec = nan
        L_eff_frac = nan

    # --- 2. Temporal Gini of |x| ------------------------------------------
    abs_x = np.abs(x)
    abs_total = float(abs_x.sum())
    if abs_total > 0 and abs_x.size > 0:
        sorted_abs = np.sort(abs_x)
        n_g = sorted_abs.size
        idx = np.arange(1, n_g + 1, dtype=float)
        gini = float(
            (2.0 * (idx * sorted_abs).sum()) / (n_g * abs_total)
            - (n_g + 1.0) / n_g
        )
    else:
        gini = nan

    # --- 3. Peak-to-mean ratio --------------------------------------------
    mean_s = float(np.mean(s)) if s.size else 0.0
    pmr = float(s_max / mean_s) if mean_s > 0 else nan

    # --- 4 & 5. Peak finding ----------------------------------------------
    from scipy.signal import find_peaks, peak_widths
    height_thr = float(peak_height_frac) * s_max
    distance = max(1, int(round(float(min_peak_distance_sec) / dt))) if dt > 0 else 1
    try:
        peaks, _props = find_peaks(s, height=height_thr, distance=distance)
    except Exception:  # noqa: BLE001
        peaks = np.empty(0, dtype=int)

    n_bumps = int(peaks.size)
    if peaks.size > 0 and dt > 0:
        tallest = int(peaks[int(np.argmax(s[peaks]))])
        try:
            widths, _, _, _ = peak_widths(s, [tallest], rel_height=0.5)
            fwhm = float(widths[0]) * dt
        except Exception:  # noqa: BLE001
            fwhm = nan
    else:
        fwhm = nan

    return {
        "valid_duration_sec": valid_dur,
        "n_valid_samples": int(s.size),
        "L_eff_sec": float(L_eff_sec),
        "L_eff_frac": float(L_eff_frac),
        "temporal_gini": float(gini),
        "peak_to_mean": float(pmr),
        "dominant_bump_fwhm_sec": float(fwhm),
        "n_bumps": int(n_bumps),
    }


def compute_cd_transient_metrics(
    sess: CDSessionData,
    *,
    trial_types: Optional[Sequence[str]] = None,
    split: Literal["train", "test"] = "test",
    time_window: Tuple[float, float] = (0.0, 5.0),
    restrict_events: Optional[Tuple[str, str]] = None,
    restrict_align: Optional[str] = None,
    restrict_window_per_trial: Optional[Dict[int, Tuple[float, float]]] = None,
    smooth_gauss: float = 0.1,
    smooth_mode: Literal["gaussian", "moving"] = "gaussian",
    projection_source: Literal["unbiased", "all"] = "unbiased",
    center: float | Literal["cross_class_median"] = 0.0,
    class_signs: Tuple[int, int] = (1, -1),
    signal_mode: Literal["abs", "rectified_on_class", "centered_squared"] = "abs",
    peak_height_frac: float = 0.5,
    min_peak_distance_sec: float = 0.05,
) -> pd.DataFrame:
    """Per-trial transient-vs-persistent metrics for one CD session.

    Computes five scalar metrics per trial that quantify how *temporally
    concentrated* the single-trial CD projection is. Complements the
    sign-based :func:`compute_cd_dwell_metrics`. Trial selection,
    re-alignment, masking, smoothing and centering match that function,
    so output rows join row-for-row with the dwell output on
    ``(session, trial_id, class_index)``.

    Metric definitions (see :func:`_transient_metrics_from_trace`):

    - ``L_eff_sec``, ``L_eff_frac`` — effective lifetime (s) and its
      fraction of the per-trial valid window. **``L_eff_frac → 1``
      = persistent, ``L_eff_frac → 0`` = transient.**
    - ``temporal_gini`` — concentration of ``|x|`` over time
      (0 = spread, 1 = single-sample spike).
    - ``peak_to_mean`` — ``max(s) / mean(s)`` (≥ 1).
    - ``dominant_bump_fwhm_sec`` — physical width of the tallest peak.
    - ``n_bumps`` — peaks above ``peak_height_frac * max(s)``.

    Pairs naturally with :func:`plot_transient_vs_persistency` for the
    standard 2-D scatter ``persistency_index × L_eff_frac``.
    """
    prep = _prepare_cd_session_for_scoring(
        sess,
        trial_types=trial_types,
        split=split,
        time_window=time_window,
        restrict_events=restrict_events,
        restrict_align=restrict_align,
        restrict_window_per_trial=restrict_window_per_trial,
        smooth_gauss=smooth_gauss,
        smooth_mode=smooth_mode,
        projection_source=projection_source,
        center=center,
        class_signs=class_signs,
    )

    t0, t1 = float(time_window[0]), float(time_window[1])
    rows: List[Dict[str, Any]] = []

    def _emit(proj: np.ndarray, ids: np.ndarray, name: str, class_idx: int,
              class_sign: int) -> None:
        if proj.size == 0 or proj.ndim != 2 or not name:
            return
        for i, tid in enumerate(ids):
            m = _transient_metrics_from_trace(
                proj[i], sess.time,
                t0=t0, t1=t1,
                center=prep.center_val,
                class_sign=class_sign,
                signal_mode=signal_mode,
                peak_height_frac=float(peak_height_frac),
                min_peak_distance_sec=float(min_peak_distance_sec),
            )
            m.update({
                "session": sess.session,
                "trial_id": int(tid),
                "class_name": name,
                "class_index": class_idx,
                "class_sign": class_sign,
                "center": prep.center_val,
                "t_start": t0,
                "t_end": t1,
                "signal_mode": signal_mode,
            })
            rows.append(m)

    _emit(prep.proj_A, prep.ids_A, prep.name_A, 0, prep.sign_A)
    _emit(prep.proj_B, prep.ids_B, prep.name_B, 1, prep.sign_B)

    if not rows:
        return pd.DataFrame()

    cols_first = [
        "session", "trial_id", "class_name", "class_index", "class_sign",
        "t_start", "t_end", "center", "signal_mode",
    ]
    df = pd.DataFrame(rows)
    rest = [c for c in df.columns if c not in cols_first]
    return df[cols_first + rest]


def compute_cd_transient_metrics_multi(
    sessions_data: Iterable[CDSessionData],
    **kwargs: Any,
) -> pd.DataFrame:
    """Run :func:`compute_cd_transient_metrics` on many sessions; concat rows.

    Sessions that raise are skipped with a printed warning. Returns an
    empty DataFrame if nothing succeeded.
    """
    frames: List[pd.DataFrame] = []
    for sess in sessions_data:
        try:
            df = compute_cd_transient_metrics(sess, **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {getattr(sess, 'session', '?')}: {e}")
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_transient_vs_persistency(
    transient_df: pd.DataFrame,
    dwell_df: Optional[pd.DataFrame] = None,
    *,
    x: str = "persistency_index",
    y: str = "L_eff_frac",
    hue: str = "class_name",
    alpha: float = 0.4,
    point_size: float = 18.0,
    figsize: Tuple[float, float] = (7, 6),
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> None:
    """Scatter of a transient metric against a persistency metric.

    Defaults to ``persistency_index`` (x) vs ``L_eff_frac`` (y). If
    ``dwell_df`` is given, it is inner-joined to ``transient_df`` on
    ``(session, trial_id, class_index)`` so the standard pair works
    without manual merging.

    Reading guide (for the default axes):

    - **Top-right** (high persistency, high ``L_eff_frac``)
      ⇒ genuinely **persistent** (plateau on the correct side).
    - **Top-left** (high persistency, low ``L_eff_frac``)
      ⇒ **transient** bump but on the correct side.
    - **Bottom-right** (low persistency, high ``L_eff_frac``)
      ⇒ spread out but oscillating across the boundary.
    - **Bottom-left** (low persistency, low ``L_eff_frac``)
      ⇒ short off-class spike with little useful encoding.
    """
    df = transient_df
    if dwell_df is not None and not dwell_df.empty:
        merge_keys = ["session", "trial_id", "class_index"]
        right_cols = [
            c for c in dwell_df.columns
            if c in merge_keys or c not in transient_df.columns
        ]
        df = pd.merge(
            transient_df, dwell_df[right_cols],
            on=merge_keys, how="inner",
        )

    if df is None or df.empty:
        print("Nothing to plot.")
        return
    missing = [c for c in (x, y) if c not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing from the joined frame: {missing}")

    plot_df = df.dropna(subset=[x, y]).copy()
    if plot_df.empty:
        print("Nothing to plot after dropping NaNs.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    classes = list(plot_df[hue].unique()) if hue in plot_df.columns else [None]
    colors = {c: f"C{i}" for i, c in enumerate(classes)}

    for c in classes:
        sub = plot_df if c is None else plot_df[plot_df[hue] == c]
        if sub.empty:
            continue
        ax.scatter(
            sub[x], sub[y],
            s=float(point_size), alpha=float(alpha),
            color=colors.get(c, "C0"),
            edgecolor="none",
            label=(f"{c} (n={len(sub)})" if c is not None else f"n={len(sub)}"),
        )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Reference lines for the canonical (persistency_index × L_eff_frac) view.
    if x == "persistency_index":
        ax.axvline(0.0, color="k", lw=0.6, alpha=0.5)
    if y == "L_eff_frac":
        ax.axhline(0.5, color="k", lw=0.6, alpha=0.5, ls="--")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    n_sess = plot_df["session"].nunique() if "session" in plot_df.columns else "?"
    ax.set_title(title or f"{y} vs {x}  ({len(plot_df)} trials, {n_sess} sessions)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Per-bump event extraction (long-form: one row per detected peak)
# ---------------------------------------------------------------------------

def _bump_events_from_trace(
    trace: np.ndarray,
    time: np.ndarray,
    *,
    t0: float,
    t1: float,
    center: float,
    class_sign: int,
    signal_mode: Literal["abs", "rectified_on_class", "centered_squared"] = "abs",
    peak_height_frac: float = 0.5,
    min_peak_distance_sec: float = 0.05,
) -> List[Dict[str, float]]:
    """Detect bumps in one trial's CD trace; return one dict per bump.

    Mirrors the masking / smoothing / signal-mode conventions used by
    :func:`_transient_metrics_from_trace`, so per-bump rows produced
    here are consistent with the per-trial transient metrics (in
    particular, the same peak set drives ``n_bumps``).

    Each returned dict has:
    ``bump_idx`` (0-based within the trial), ``peak_time_sec``
    (time coordinate in the same frame as ``time``), ``peak_amplitude``
    (value of the non-negative scoring signal ``s`` at the peak),
    ``peak_amplitude_signed`` (value of the centered trace
    ``x = trace - center`` at the same sample, sign preserved),
    ``peak_fwhm_sec`` (per-bump FWHM at half-height, not just the
    dominant peak), ``inter_bump_sec`` (NaN for the first bump in a
    trial, otherwise ``peak_time_sec`` minus the previous bump's
    ``peak_time_sec``), ``n_bumps_in_trial``, ``valid_duration_sec``.
    """
    out: List[Dict[str, float]] = []
    if trace.size == 0 or time.size == 0:
        return out

    sel = (time >= t0) & (time <= t1)
    x_raw = trace[sel].astype(float, copy=False)
    t_raw = time[sel]
    valid = np.isfinite(x_raw)
    if not np.any(valid):
        return out
    x = x_raw[valid] - float(center)
    t = t_raw[valid]

    if t.size >= 2:
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            dt = float((t[-1] - t[0]) / max(t.size - 1, 1))
    else:
        dt = 0.0
    valid_dur = float(dt * x.size)

    if signal_mode == "abs":
        s = np.abs(x)
    elif signal_mode == "rectified_on_class":
        cs = int(np.sign(class_sign)) if class_sign != 0 else 0
        s = np.maximum(0.0, cs * x) if cs != 0 else np.abs(x)
    elif signal_mode == "centered_squared":
        s = x * x
    else:
        raise ValueError(
            "signal_mode must be 'abs' | 'rectified_on_class' | "
            f"'centered_squared', got {signal_mode!r}"
        )

    s_max = float(np.max(s)) if s.size else 0.0
    if not np.isfinite(s_max) or s_max <= 0.0:
        return out

    from scipy.signal import find_peaks, peak_widths
    height_thr = float(peak_height_frac) * s_max
    distance = max(1, int(round(float(min_peak_distance_sec) / dt))) if dt > 0 else 1
    try:
        peaks, _props = find_peaks(s, height=height_thr, distance=distance)
    except Exception:  # noqa: BLE001
        peaks = np.empty(0, dtype=int)
    if peaks.size == 0:
        return out

    # Per-peak FWHM at half-height of the global signal (matches the
    # convention used for the "dominant" bump in the per-trial metrics).
    try:
        widths, _, _, _ = peak_widths(s, peaks, rel_height=0.5)
    except Exception:  # noqa: BLE001
        widths = np.full(peaks.size, np.nan, dtype=float)

    peak_times = t[peaks]
    peak_amps = s[peaks]
    peak_amps_signed = x[peaks]
    inter_bump = np.full(peaks.size, np.nan, dtype=float)
    if peaks.size >= 2:
        inter_bump[1:] = np.diff(peak_times)

    n_bumps = int(peaks.size)
    for i, _ in enumerate(peaks):
        out.append({
            "bump_idx": int(i),
            "peak_time_sec": float(peak_times[i]),
            "peak_amplitude": float(peak_amps[i]),
            "peak_amplitude_signed": float(peak_amps_signed[i]),
            "peak_fwhm_sec": float(widths[i]) * dt if dt > 0 else float("nan"),
            "inter_bump_sec": float(inter_bump[i]),
            "n_bumps_in_trial": n_bumps,
            "valid_duration_sec": valid_dur,
        })
    return out


def compute_cd_bump_events(
    sess: CDSessionData,
    *,
    trial_types: Optional[Sequence[str]] = None,
    split: Literal["train", "test"] = "test",
    time_window: Tuple[float, float] = (0.0, 5.0),
    restrict_events: Optional[Tuple[str, str]] = None,
    restrict_align: Optional[str] = None,
    restrict_window_per_trial: Optional[Dict[int, Tuple[float, float]]] = None,
    smooth_gauss: float = 0.1,
    smooth_mode: Literal["gaussian", "moving"] = "gaussian",
    projection_source: Literal["unbiased", "all"] = "unbiased",
    center: float | Literal["cross_class_median"] = 0.0,
    class_signs: Tuple[int, int] = (1, -1),
    signal_mode: Literal["abs", "rectified_on_class", "centered_squared"] = "abs",
    peak_height_frac: float = 0.5,
    min_peak_distance_sec: float = 0.05,
) -> pd.DataFrame:
    """Long-form per-bump events for one CD session.

    Returns a DataFrame with **one row per detected bump** (so trials
    with three bumps contribute three rows). Useful for diagnostics
    that aggregate across bumps rather than across trials:

    - **median inter-bump interval vs ITI** (Poisson-like burst test)
    - **per-bump amplitude distribution** binned by ITI
      (stereotyped-shape test)
    - **bump rate (= n_bumps / iti_sec)** vs ITI

    All prep parameters (``trial_types``, ``time_window``,
    ``restrict_events`` / ``restrict_align``, ``smooth_gauss``,
    ``projection_source``, ``center``, ``class_signs``, ``signal_mode``,
    ``peak_height_frac``, ``min_peak_distance_sec``) match
    :func:`compute_cd_transient_metrics`, so the per-trial bump count
    in this frame equals the ``n_bumps`` column in the corresponding
    per-trial frame.
    """
    prep = _prepare_cd_session_for_scoring(
        sess,
        trial_types=trial_types,
        split=split,
        time_window=time_window,
        restrict_events=restrict_events,
        restrict_align=restrict_align,
        restrict_window_per_trial=restrict_window_per_trial,
        smooth_gauss=smooth_gauss,
        smooth_mode=smooth_mode,
        projection_source=projection_source,
        center=center,
        class_signs=class_signs,
    )

    t0, t1 = float(time_window[0]), float(time_window[1])
    rows: List[Dict[str, Any]] = []

    def _emit(proj: np.ndarray, ids: np.ndarray, name: str, class_idx: int,
              class_sign: int) -> None:
        if proj.size == 0 or proj.ndim != 2 or not name:
            return
        for i, tid in enumerate(ids):
            events = _bump_events_from_trace(
                proj[i], sess.time,
                t0=t0, t1=t1,
                center=prep.center_val,
                class_sign=class_sign,
                signal_mode=signal_mode,
                peak_height_frac=float(peak_height_frac),
                min_peak_distance_sec=float(min_peak_distance_sec),
            )
            for ev in events:
                ev.update({
                    "session": sess.session,
                    "trial_id": int(tid),
                    "class_name": name,
                    "class_index": class_idx,
                    "class_sign": class_sign,
                    "center": prep.center_val,
                    "t_start": t0,
                    "t_end": t1,
                    "signal_mode": signal_mode,
                })
                rows.append(ev)

    _emit(prep.proj_A, prep.ids_A, prep.name_A, 0, prep.sign_A)
    _emit(prep.proj_B, prep.ids_B, prep.name_B, 1, prep.sign_B)

    if not rows:
        return pd.DataFrame()

    cols_first = [
        "session", "trial_id", "class_name", "class_index", "class_sign",
        "bump_idx", "n_bumps_in_trial",
        "peak_time_sec", "peak_amplitude", "peak_amplitude_signed",
        "peak_fwhm_sec", "inter_bump_sec",
        "valid_duration_sec",
        "t_start", "t_end", "center", "signal_mode",
    ]
    df = pd.DataFrame(rows)
    rest = [c for c in df.columns if c not in cols_first]
    return df[cols_first + rest]


def compute_cd_bump_events_multi(
    sessions_data: Iterable[CDSessionData],
    **kwargs: Any,
) -> pd.DataFrame:
    """Run :func:`compute_cd_bump_events` on many sessions; concat rows."""
    frames: List[pd.DataFrame] = []
    for sess in sessions_data:
        try:
            df = compute_cd_bump_events(sess, **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {getattr(sess, 'session', '?')}: {e}")
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_bump_diagnostics(
    bump_df: pd.DataFrame,
    *,
    covariate: str = "iti_sec",
    hue: Optional[str] = "class_name",
    n_bins: int = 8,
    bin_mode: Literal["quantile", "uniform"] = "quantile",
    log_x: bool = True,
    amp_log_y: bool = False,
    figsize: Tuple[float, float] = (15, 4.2),
    title: Optional[str] = None,
) -> None:
    """Three-panel "fixed-shape burst" diagnostic vs a trial-level covariate.

    Expects ``bump_df`` to be a long-form per-bump frame (output of
    :func:`compute_cd_bump_events_multi`) that has been passed through
    :func:`attach_trial_covariates` to add the trial-level covariate
    column (default ``iti_sec``). Three subplots:

    1. **Bump rate (Hz) vs covariate** -- one point per *trial*
       (``n_bumps_in_trial / iti_sec``). Flat ⇒ time-invariant burst
       generation.
    2. **Median inter-bump interval vs covariate** -- one point per
       *bump* (excluding the first bump of each trial). Flat ⇒
       Poisson-like timing.
    3. **Per-bump amplitude distribution by covariate bin** -- boxplots
       of ``peak_amplitude`` across covariate bins, one panel per
       ``hue`` group. Invariant ⇒ stereotyped bump shape.
    """
    if bump_df is None or bump_df.empty:
        print("Nothing to plot.")
        return
    if covariate not in bump_df.columns:
        raise KeyError(
            f"covariate {covariate!r} not in bump_df; pass bump_df through "
            "attach_trial_covariates first."
        )

    df = bump_df.dropna(subset=[covariate]).copy()
    if df.empty:
        print(f"No rows with finite {covariate}.")
        return

    classes = list(df[hue].unique()) if (hue and hue in df.columns) else [None]
    colors = {c: f"C{i}" for i, c in enumerate(classes)}

    # Build bin edges from the bump-level covariate values
    cov_all = df[covariate].to_numpy(dtype=float)
    if bin_mode == "quantile":
        q = np.linspace(0.0, 1.0, int(n_bins) + 1)
        edges = np.unique(np.nanquantile(cov_all, q))
    else:
        lo, hi = float(np.nanmin(cov_all)), float(np.nanmax(cov_all))
        edges = np.linspace(lo, hi, int(n_bins) + 1)
    if edges.size < 2:
        print("Not enough unique covariate values to bin.")
        return
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ---- Panel 1: bump rate (one point per trial) ------------------------
    ax = axes[0]
    trial_keys = ["session", "trial_id", "class_index"]
    if hue and hue in df.columns:
        trial_keys = trial_keys + [hue]
    trial_df = (
        df.drop_duplicates(subset=["session", "trial_id", "class_index"])
        .loc[:, trial_keys + ["n_bumps_in_trial", covariate]]
        .copy()
    )
    trial_df = trial_df[trial_df[covariate] > 0]
    trial_df["bump_rate_hz"] = trial_df["n_bumps_in_trial"] / trial_df[covariate]

    for c in classes:
        sub = trial_df if c is None else trial_df[trial_df[hue] == c]
        if sub.empty:
            continue
        # Per-bin median of rate
        bin_idx = np.clip(np.digitize(sub[covariate].to_numpy(dtype=float),
                                      edges[1:-1], right=True),
                          0, edges.size - 2)
        meds = np.full(edges.size - 1, np.nan, dtype=float)
        q25 = np.full(edges.size - 1, np.nan, dtype=float)
        q75 = np.full(edges.size - 1, np.nan, dtype=float)
        for b in range(edges.size - 1):
            vals = sub["bump_rate_hz"].to_numpy(dtype=float)[bin_idx == b]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                meds[b] = float(np.median(vals))
                q25[b]  = float(np.quantile(vals, 0.25))
                q75[b]  = float(np.quantile(vals, 0.75))
        ax.scatter(sub[covariate], sub["bump_rate_hz"],
                   s=8, alpha=0.15, color=colors.get(c, "C0"), edgecolor="none")
        finite = np.isfinite(meds)
        ax.plot(centers[finite], meds[finite], "-o",
                color=colors.get(c, "C0"),
                label=(f"{c} (n_trials={len(sub)})" if c is not None else None))
        ax.fill_between(centers[finite], q25[finite], q75[finite],
                        color=colors.get(c, "C0"), alpha=0.15)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(covariate)
    ax.set_ylabel("bump_rate_hz  (n_bumps / iti_sec)")
    ax.set_title("Bump rate vs trial length")
    if hue:
        ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: inter-bump interval (one point per bump, excl. 1st) ----
    ax = axes[1]
    ibi_df = df.dropna(subset=["inter_bump_sec"]).copy()
    for c in classes:
        sub = ibi_df if c is None else ibi_df[ibi_df[hue] == c]
        if sub.empty:
            continue
        bin_idx = np.clip(np.digitize(sub[covariate].to_numpy(dtype=float),
                                      edges[1:-1], right=True),
                          0, edges.size - 2)
        meds = np.full(edges.size - 1, np.nan, dtype=float)
        q25 = np.full(edges.size - 1, np.nan, dtype=float)
        q75 = np.full(edges.size - 1, np.nan, dtype=float)
        for b in range(edges.size - 1):
            vals = sub["inter_bump_sec"].to_numpy(dtype=float)[bin_idx == b]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                meds[b] = float(np.median(vals))
                q25[b]  = float(np.quantile(vals, 0.25))
                q75[b]  = float(np.quantile(vals, 0.75))
        ax.scatter(sub[covariate], sub["inter_bump_sec"],
                   s=6, alpha=0.10, color=colors.get(c, "C0"), edgecolor="none")
        finite = np.isfinite(meds)
        ax.plot(centers[finite], meds[finite], "-o",
                color=colors.get(c, "C0"),
                label=(f"{c} (n_bumps={len(sub)})" if c is not None else None))
        ax.fill_between(centers[finite], q25[finite], q75[finite],
                        color=colors.get(c, "C0"), alpha=0.15)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(covariate)
    ax.set_ylabel("inter_bump_sec")
    ax.set_title("Inter-bump interval vs trial length")
    if hue:
        ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: per-bump amplitude distribution by covariate bin -------
    ax = axes[2]
    for c in classes:
        sub = df if c is None else df[df[hue] == c]
        if sub.empty:
            continue
        bin_idx = np.clip(np.digitize(sub[covariate].to_numpy(dtype=float),
                                      edges[1:-1], right=True),
                          0, edges.size - 2)
        meds = np.full(edges.size - 1, np.nan, dtype=float)
        q25 = np.full(edges.size - 1, np.nan, dtype=float)
        q75 = np.full(edges.size - 1, np.nan, dtype=float)
        for b in range(edges.size - 1):
            vals = sub["peak_amplitude"].to_numpy(dtype=float)[bin_idx == b]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                meds[b] = float(np.median(vals))
                q25[b]  = float(np.quantile(vals, 0.25))
                q75[b]  = float(np.quantile(vals, 0.75))
        ax.scatter(sub[covariate], sub["peak_amplitude"],
                   s=6, alpha=0.10, color=colors.get(c, "C0"), edgecolor="none")
        finite = np.isfinite(meds)
        ax.plot(centers[finite], meds[finite], "-o",
                color=colors.get(c, "C0"),
                label=(f"{c} (n_bumps={len(sub)})" if c is not None else None))
        ax.fill_between(centers[finite], q25[finite], q75[finite],
                        color=colors.get(c, "C0"), alpha=0.15)
    if log_x:
        ax.set_xscale("log")
    if amp_log_y:
        ax.set_yscale("log")
    ax.set_xlabel(covariate)
    ax.set_ylabel("peak_amplitude")
    ax.set_title("Per-bump amplitude vs trial length")
    if hue:
        ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Trial-level covariates (ITI, P(right)) + generic metric-vs-covariate plot
# ---------------------------------------------------------------------------

def attach_trial_covariates(
    metric_df: pd.DataFrame,
    sessions_data: Iterable[CDSessionData],
    *,
    restrict_events: Tuple[str, str] = ("trial_start", "go_cue"),
    restrict_align: Optional[str] = "trial_start",
    p_right_window: int = 10,
    p_right_column: Optional[str] = None,
) -> pd.DataFrame:
    """Add per-trial ``iti_sec`` and ``p_right`` columns to a metric DataFrame.

    For every ``session`` represented in ``metric_df``, looks up:

    - ``iti_sec`` = ``event_end_offset - event_start_offset`` (seconds),
      computed via :func:`compute_per_trial_event_offsets` from the session's
      NWB. Defaults align ``trial_start → go_cue``, so this is the standard
      inter-trial-interval length. Unlike ``valid_duration_sec`` (which is
      clipped to the scoring window), ``iti_sec`` reflects the raw
      behaviorally-defined window length, including trials longer than the
      scoring window.

    - ``p_right`` = per-trial sliding-window P(right). If ``p_right_column``
      is given, that column is read from ``sess.behavior_df`` directly;
      otherwise P(right) is recomputed via
      :func:`behavior_utils.compute_sliding_choice_probability` on the
      reconstructed ``animal_response`` vector (causal window of length
      ``p_right_window``, no-response trials excluded).

    Returns a copy of ``metric_df`` with the two new columns added. Rows
    whose ``(session, trial_id)`` cannot be resolved get NaN.
    """
    if metric_df is None or metric_df.empty:
        return metric_df.copy() if metric_df is not None else pd.DataFrame()
    if "session" not in metric_df.columns or "trial_id" not in metric_df.columns:
        raise KeyError("metric_df must contain 'session' and 'trial_id' columns.")

    sessions_by_name: Dict[str, CDSessionData] = {
        s.session: s for s in sessions_data
    }

    out = metric_df.copy()
    out["iti_sec"] = np.nan
    out["p_right"] = np.nan

    for sess_name, sub in metric_df.groupby("session", observed=True):
        sess = sessions_by_name.get(sess_name)
        if sess is None:
            print(f"[warn] {sess_name}: not in sessions_data; skipped covariates.")
            continue

        # ---- ITI per trial -------------------------------------------------
        try:
            offsets = compute_per_trial_event_offsets(
                sess_name,
                event_start=restrict_events[0],
                event_end=restrict_events[1],
                align=restrict_align,
            )
            iti_map = {tid: float(end - start) for tid, (start, end) in offsets.items()}
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {sess_name}: could not compute ITI ({e})")
            iti_map = {}

        # ---- P(right) per trial -------------------------------------------
        p_arr: Optional[np.ndarray] = None
        try:
            df_beh = sess.behavior_df
            if df_beh is None:
                raise ValueError("behavior_df is None")

            ids_all = (
                np.asarray(sess.trial_id_all_trials, dtype=int).ravel()
                if sess.trial_id_all_trials is not None
                else np.empty(0, dtype=int)
            )

            def _ids_from_col(col: str) -> np.ndarray:
                if col not in df_beh.columns:
                    return np.empty(0, dtype=int)
                try:
                    return np.asarray(df_beh[col].iloc[0], dtype=int).ravel()
                except Exception:  # noqa: BLE001
                    return np.empty(0, dtype=int)

            id_pool = [ids_all]
            for c in ("left_choice_trials", "right_choice_trials",
                      "no_response_trials"):
                id_pool.append(_ids_from_col(c))
            n_trials = int(max(int(arr.max()) for arr in id_pool if arr.size) + 1)

            if p_right_column is not None:
                if p_right_column not in df_beh.columns:
                    raise KeyError(
                        f"column {p_right_column!r} not in behavior_df"
                    )
                raw = np.asarray(df_beh[p_right_column].iloc[0], dtype=float).ravel()
                if raw.size == n_trials:
                    p_arr = raw
                else:
                    responded_ids = np.sort(np.concatenate([
                        _ids_from_col("left_choice_trials"),
                        _ids_from_col("right_choice_trials"),
                    ]))
                    p_arr = np.full(n_trials, np.nan, dtype=float)
                    n = min(raw.size, responded_ids.size)
                    p_arr[responded_ids[:n]] = raw[:n]
            else:
                from behavior_utils import compute_sliding_choice_probability
                resp_vec = _reconstruct_animal_response_from_df(df_beh, n_trials)
                cp_out = compute_sliding_choice_probability(
                    resp_vec,
                    window=int(p_right_window),
                    step=1,
                    min_periods=1,
                    causal=True,
                    side="right",
                    exclude_value=2,
                )
                p_arr = np.asarray(cp_out["choice_prob"], dtype=float)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {sess_name}: could not compute P(right) ({e})")

        # ---- write back per row -------------------------------------------
        for idx in sub.index:
            tid = int(out.at[idx, "trial_id"])
            if tid in iti_map:
                out.at[idx, "iti_sec"] = iti_map[tid]
            if p_arr is not None and 0 <= tid < p_arr.size:
                out.at[idx, "p_right"] = float(p_arr[tid])

    return out


def plot_metrics_vs_covariate(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    covariate: str,
    hue: Optional[str] = "class_name",
    n_bins: int = 10,
    bin_edges: Optional[Sequence[float]] = None,
    bin_mode: Literal["quantile", "uniform"] = "quantile",
    show_scatter: bool = True,
    scatter_alpha: float = 0.2,
    scatter_size: float = 8.0,
    agg: Literal["median", "mean"] = "median",
    show_iqr: bool = True,
    log_x: bool = False,
    x_clip: Optional[Tuple[float, float]] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> None:
    """Grid of per-trial metric vs. continuous covariate.

    One subplot per metric in ``metrics``. Each subplot shows:

    - Light scatter of every (covariate, metric) point coloured by ``hue``.
    - A per-class binned aggregate (median or mean) joined as a line.
    - Optional shaded IQR band per class.

    Parameters
    ----------
    df
        Long DataFrame with columns ``covariate``, every name in
        ``metrics``, and (optionally) ``hue``. Typically the output of
        :func:`attach_trial_covariates` applied to a merged dwell + transient
        frame.
    metrics
        Column names to plot. Missing columns are silently dropped.
    covariate
        The x-axis column (e.g. ``"iti_sec"``, ``"p_right"``,
        ``"valid_duration_sec"``).
    n_bins, bin_edges, bin_mode
        Binning of the covariate for the aggregate line.
        - ``bin_mode="quantile"`` → equal-population bins (good for skewed
          covariates like ITI).
        - ``bin_mode="uniform"`` → equal-width bins (natural for P(right)).
        Pass explicit ``bin_edges`` to override both.
    x_clip, log_x
        Clip x to a range or render on a log scale (useful for ITIs with
        a long tail).
    """
    if df is None or df.empty:
        print("Nothing to plot.")
        return
    if covariate not in df.columns:
        raise KeyError(f"covariate {covariate!r} not in DataFrame.")
    metrics_present = [m for m in metrics if m in df.columns]
    if not metrics_present:
        raise ValueError("None of the requested metrics are in the DataFrame.")

    work = df.dropna(subset=[covariate]).copy()
    if x_clip is not None:
        work = work[(work[covariate] >= x_clip[0]) & (work[covariate] <= x_clip[1])]
    if work.empty:
        print("Nothing to plot after dropping NaNs / clipping x.")
        return

    # --- bin edges -------------------------------------------------------
    if bin_edges is not None:
        edges = np.asarray(bin_edges, dtype=float)
    elif bin_mode == "quantile":
        qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
        edges = np.unique(np.quantile(work[covariate], qs))
        if edges.size < 2:
            edges = np.array([
                float(work[covariate].min()),
                float(work[covariate].max()) + 1e-9,
            ])
    elif bin_mode == "uniform":
        lo = float(work[covariate].min())
        hi = float(work[covariate].max())
        edges = np.linspace(lo, hi + max(hi - lo, 1.0) * 1e-6, int(n_bins) + 1)
    else:
        raise ValueError("bin_mode must be 'quantile' or 'uniform'.")

    work["_bin"] = pd.cut(work[covariate], edges, include_lowest=True)

    # --- subplot grid ----------------------------------------------------
    n_panels = len(metrics_present)
    if ncols is None:
        ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (4.6 * ncols, 3.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    classes = (list(work[hue].dropna().unique())
               if hue and hue in work.columns else [None])
    colors = {c: f"C{i}" for i, c in enumerate(classes)}

    for k, m in enumerate(metrics_present):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        sub_all = work.dropna(subset=[m])
        if sub_all.empty:
            ax.set_visible(False)
            continue

        for cl in classes:
            sub = sub_all if cl is None else sub_all[sub_all[hue] == cl]
            if sub.empty:
                continue
            color = colors.get(cl, "C0")

            if show_scatter:
                ax.scatter(
                    sub[covariate], sub[m],
                    s=float(scatter_size), alpha=float(scatter_alpha),
                    color=color, edgecolor="none",
                )

            grp = sub.groupby("_bin", observed=True)[m]
            center_y = grp.mean() if agg == "mean" else grp.median()
            counts = grp.count()
            xs = np.array([(iv.left + iv.right) / 2.0
                            for iv in center_y.index], dtype=float)
            valid = counts.values > 0

            if show_iqr:
                lo_y = grp.quantile(0.25).reindex(center_y.index)
                hi_y = grp.quantile(0.75).reindex(center_y.index)
                ax.fill_between(
                    xs[valid], lo_y.values[valid], hi_y.values[valid],
                    color=color, alpha=0.15,
                )

            ax.plot(
                xs[valid], center_y.values[valid],
                color=color, lw=1.8, marker="o", ms=4,
                label=(f"{cl} (n={int(counts.sum())})" if cl is not None
                       else f"n={int(counts.sum())}"),
            )

        ax.set_xlabel(covariate)
        ax.set_ylabel(m)
        if log_x:
            ax.set_xscale("log")
        ax.grid(alpha=0.2)
        if k == 0:
            ax.legend(loc="best", fontsize=8)

    # Hide unused panels.
    for k in range(n_panels, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].set_visible(False)

    if title is None:
        n_sess = df["session"].nunique() if "session" in df.columns else "?"
        title = (f"Metrics vs {covariate}  "
                 f"({len(work)} trials, {n_sess} sessions, "
                 f"{agg} ± IQR per bin)")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    plt.show()


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
    aggregate: Literal["trials", "session_medians", "session_means"] = "trials",
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
    aggregate : {'trials','session_medians','session_means'}, default 'trials'
        How each P(right) bin is summarized.
          - 'trials'         : pool every trial across sessions, one box per
            bin (original behavior). Bin counts can be dominated by sessions
            with many trials.
          - 'session_medians': compute per-session per-bin median first, then
            box-plot those session medians within each bin. Each session
            contributes one observation per bin (or zero if it has no trials
            there). Recommended for cross-session trend visualization
            because it removes within-session sample-size imbalance.
          - 'session_means'  : same as 'session_medians' but uses the mean.

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

    # ----- Optional per-session-per-bin aggregation (cleaner cross-session trend) -----
    if aggregate in ("session_medians", "session_means"):
        if bins is None:
            raise ValueError(
                "aggregate='session_medians'/'session_means' requires explicit `bins` edges."
            )
        reducer = np.nanmedian if aggregate == "session_medians" else np.nanmean
        edges_a = np.asarray(bins, dtype=float)
        centers_a = 0.5 * (edges_a[:-1] + edges_a[1:])
        agg_p: List[float] = []
        agg_y: List[float] = []
        for p_i, y_i in zip(ps, ys):
            idx_i = np.digitize(p_i, edges_a, right=False)
            idx_i = np.clip(idx_i, 1, len(edges_a) - 1)
            for k in range(1, len(edges_a)):
                sel = (idx_i == k)
                if sel.sum() == 0:
                    continue
                agg_p.append(float(centers_a[k - 1]))
                agg_y.append(float(reducer(y_i[sel])))
        p = np.asarray(agg_p, dtype=float)
        y = np.asarray(agg_y, dtype=float)
        hl_masks = {}  # highlights don't survive this aggregation
    else:
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
        agg_tag = {
            "trials": "",
            "session_medians": "  [session medians]",
            "session_means": "  [session means]",
        }[aggregate]
        title = (
            f"Combined ({len(used)} sessions) — CD projection by P(right)"
            + (f"  (window {p_right_window})" if p_right_column is None else "")
            + norm_tag
            + agg_tag
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


# ---------------------------------------------------------------------------
# 4. DECODER (1D threshold classifier on CD projection)
# ---------------------------------------------------------------------------

def _window_mean(
    trace: np.ndarray,
    time: np.ndarray,
    window: Tuple[float, float],
) -> np.ndarray:
    """Average a (n_trials, n_time) trace over a time window. NaNs ignored."""
    if trace.size == 0:
        return np.empty(0, dtype=float)
    mask = (time >= window[0]) & (time <= window[1])
    if not np.any(mask):
        raise ValueError(
            f"window {window} does not overlap trace time range "
            f"[{time[0]:.3f}, {time[-1]:.3f}]."
        )
    seg = trace[:, mask]
    with np.errstate(invalid="ignore"):
        return np.nanmean(seg, axis=1)


def _fit_threshold(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    method: Literal["midpoint", "optimal"] = "optimal",
) -> Tuple[float, int]:
    """Find a 1D decision boundary separating class A from class B.

    Returns
    -------
    (boundary, sign) : (float, int)
        Decision rule: predict A iff ``sign * (score - boundary) > 0``.
        ``sign = +1`` when mean(A) > mean(B), else ``-1``.
    """
    if scores_a.size == 0 or scores_b.size == 0:
        raise ValueError("Both classes need at least one trial to fit a threshold.")
    ma = float(np.nanmean(scores_a))
    mb = float(np.nanmean(scores_b))
    sign = 1 if ma >= mb else -1
    if method == "midpoint":
        return 0.5 * (ma + mb), sign
    if method == "optimal":
        # Scan all midpoints between consecutive sorted unique scores; pick the
        # one with the highest balanced accuracy on (A, B).
        all_scores = np.concatenate([scores_a, scores_b])
        all_scores = all_scores[~np.isnan(all_scores)]
        if all_scores.size == 0:
            return 0.5 * (ma + mb), sign
        srt = np.sort(np.unique(all_scores))
        if srt.size < 2:
            return float(srt[0]), sign
        cands = 0.5 * (srt[:-1] + srt[1:])
        # also consider the data extremes
        cands = np.concatenate(([srt[0] - 1e-9, srt[-1] + 1e-9], cands))
        best_bal = -np.inf
        best_t = 0.5 * (ma + mb)
        for t in cands:
            if sign > 0:
                tpr = np.mean(scores_a > t)
                tnr = np.mean(scores_b <= t)
            else:
                tpr = np.mean(scores_a < t)
                tnr = np.mean(scores_b >= t)
            bal = 0.5 * (tpr + tnr)
            if bal > best_bal:
                best_bal = bal
                best_t = float(t)
        return best_t, sign
    raise ValueError(f"Unknown threshold method: {method!r}")


def _roc_auc_1d(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    sign: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve & AUC for 1D scores. Positive class = A.

    Higher ``sign * score`` should indicate class A.
    """
    sa = sign * scores_a
    sb = sign * scores_b
    sa = sa[~np.isnan(sa)]
    sb = sb[~np.isnan(sb)]
    if sa.size == 0 or sb.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    # Mann-Whitney U based AUC
    all_s = np.concatenate([sa, sb])
    labels = np.concatenate([np.ones_like(sa), np.zeros_like(sb)])
    order = np.argsort(all_s, kind="mergesort")
    s_sorted = all_s[order]
    l_sorted = labels[order]
    # rank with ties handled
    ranks = np.empty_like(s_sorted, dtype=float)
    i = 0
    n = len(s_sorted)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos = sa.size
    n_neg = sb.size
    sum_ranks_pos = ranks[l_sorted == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    # Build ROC curve by sweeping thresholds from high to low (predict A when
    # score > threshold).
    thresh_order = np.argsort(-all_s, kind="mergesort")
    l_desc = labels[thresh_order]
    tps = np.cumsum(l_desc == 1)
    fps = np.cumsum(l_desc == 0)
    tpr = np.concatenate(([0.0], tps / max(n_pos, 1)))
    fpr = np.concatenate(([0.0], fps / max(n_neg, 1)))
    return fpr, tpr, float(auc)


@dataclass
class CDDecoderResult:
    """Outcome of a 1D threshold decoder on a CD projection."""
    session: str
    trial_types: Tuple[str, str]
    window: Tuple[float, float]
    method: str
    boundary: float
    sign: int                   # +1 if class A has higher score, else -1
    train_scores_a: np.ndarray
    train_scores_b: np.ndarray
    test_scores_a: np.ndarray
    test_scores_b: np.ndarray
    train_accuracy: float
    test_accuracy: float
    train_balanced_accuracy: float
    test_balanced_accuracy: float
    test_confusion: np.ndarray  # 2x2 ints; rows = true [A,B], cols = pred [A,B]
    test_auc: float
    test_fpr: np.ndarray
    test_tpr: np.ndarray


def _predict(scores: np.ndarray, boundary: float, sign: int) -> np.ndarray:
    """Return 1 for predicted-A, 0 for predicted-B."""
    return ((sign * (scores - boundary)) > 0).astype(int)


def decode_cd_session(
    sess: CDSessionData,
    *,
    window: Tuple[float, float],
    method: Literal["midpoint", "optimal"] = "optimal",
) -> CDDecoderResult:
    """Fit a 1D decision boundary on the **train** projections of a CD session
    and evaluate it on the held-out **test** projections.

    Parameters
    ----------
    sess
        Output of :func:`load_cd_session`.
    window
        Time window (s) over which to average the projection trace to obtain
        one scalar feature per trial.
    method
        ``"midpoint"`` uses 0.5*(mean(A)+mean(B)); ``"optimal"`` scans all
        candidate thresholds and picks the one with the highest balanced
        accuracy on the training projections.
    """
    train_a = _window_mean(sess.proj_train_A, sess.time, window)
    train_b = _window_mean(sess.proj_train_B, sess.time, window)
    test_a = _window_mean(sess.proj_test_A, sess.time, window)
    test_b = _window_mean(sess.proj_test_B, sess.time, window)

    boundary, sign = _fit_threshold(train_a, train_b, method=method)

    # Train metrics
    pa_tr = _predict(train_a, boundary, sign)
    pb_tr = _predict(train_b, boundary, sign)
    train_acc = (pa_tr.sum() + (1 - pb_tr).sum()) / max(pa_tr.size + pb_tr.size, 1)
    train_bal = 0.5 * (
        (pa_tr.mean() if pa_tr.size else 0.0)
        + ((1 - pb_tr).mean() if pb_tr.size else 0.0)
    )

    # Test metrics
    pa_te = _predict(test_a, boundary, sign)
    pb_te = _predict(test_b, boundary, sign)
    test_acc = (pa_te.sum() + (1 - pb_te).sum()) / max(pa_te.size + pb_te.size, 1)
    test_bal = 0.5 * (
        (pa_te.mean() if pa_te.size else 0.0)
        + ((1 - pb_te).mean() if pb_te.size else 0.0)
    )

    # Confusion matrix on test
    conf = np.array(
        [
            [int(pa_te.sum()), int((1 - pa_te).sum())],
            [int(pb_te.sum()), int((1 - pb_te).sum())],
        ],
        dtype=int,
    )

    fpr, tpr, auc = _roc_auc_1d(test_a, test_b, sign)

    return CDDecoderResult(
        session=sess.session,
        trial_types=sess.trial_types,
        window=tuple(window),
        method=str(method),
        boundary=float(boundary),
        sign=int(sign),
        train_scores_a=train_a,
        train_scores_b=train_b,
        test_scores_a=test_a,
        test_scores_b=test_b,
        train_accuracy=float(train_acc),
        test_accuracy=float(test_acc),
        train_balanced_accuracy=float(train_bal),
        test_balanced_accuracy=float(test_bal),
        test_confusion=conf,
        test_auc=float(auc),
        test_fpr=fpr,
        test_tpr=tpr,
    )


def decode_cd_session_over_time(
    sess: CDSessionData,
    *,
    bin_window: float = 0.2,
    step: Optional[float] = None,
    method: Literal["midpoint", "optimal"] = "optimal",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a small window across the trial and decode at each step.

    Returns
    -------
    (centers, test_accuracy, test_auc) : tuple of 1D arrays
        Length = number of windows. ``centers`` are the window midpoints (s).
    """
    if step is None:
        step = max(sess.dt, bin_window / 4)
    t0 = float(sess.time[0]) + bin_window / 2.0
    t1 = float(sess.time[-1]) - bin_window / 2.0
    if t1 <= t0:
        raise ValueError(
            f"bin_window {bin_window} too large for trace span "
            f"[{sess.time[0]:.3f}, {sess.time[-1]:.3f}]."
        )
    centers = np.arange(t0, t1 + 1e-9, step)
    accs = np.empty_like(centers)
    aucs = np.empty_like(centers)
    for i, c in enumerate(centers):
        win = (c - bin_window / 2.0, c + bin_window / 2.0)
        try:
            r = decode_cd_session(sess, window=win, method=method)
        except Exception:  # noqa: BLE001
            accs[i] = np.nan
            aucs[i] = np.nan
            continue
        accs[i] = r.test_accuracy
        aucs[i] = r.test_auc
    return centers, accs, aucs


def plot_cd_decoder(
    sess: CDSessionData,
    *,
    window: Tuple[float, float],
    method: Literal["midpoint", "optimal"] = "optimal",
    show_time_curve: bool = True,
    time_curve_bin: float = 0.2,
    time_curve_step: Optional[float] = None,
    bins: int = 30,
    figsize: Tuple[float, float] = (15, 9),
    show: bool = True,
) -> CDDecoderResult:
    """Decode a CD session and produce a 2x2 (or 2x3) summary figure:

    1. Train histogram + boundary
    2. Test histogram + boundary (predictions colored)
    3. ROC curve (test)
    4. Confusion-matrix annotation + accuracy bars
    5. (Optional) accuracy / AUC vs time

    Returns the :class:`CDDecoderResult`.
    """
    r = decode_cd_session(sess, window=window, method=method)

    have_time = show_time_curve and sess.time.size > 1
    if have_time:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(figsize[0] * 2 / 3, figsize[1]))

    label_a, label_b = r.trial_types
    color_a, color_b = "tab:blue", "tab:orange"

    # --- (1) Train histogram ---
    ax = axes[0, 0]
    train_all = np.concatenate([r.train_scores_a, r.train_scores_b])
    if train_all.size:
        edges = np.histogram_bin_edges(train_all, bins=bins)
        ax.hist(r.train_scores_a, bins=edges, color=color_a, alpha=0.6, label=label_a)
        ax.hist(r.train_scores_b, bins=edges, color=color_b, alpha=0.6, label=label_b)
    ax.axvline(r.boundary, color="k", linestyle="--", lw=1.5, label=f"boundary={r.boundary:.3f}")
    ax.set_xlabel("CD projection (train)")
    ax.set_ylabel("Trials")
    ax.set_title(
        f"Train — acc={r.train_accuracy:.2%}, bal={r.train_balanced_accuracy:.2%}"
    )
    ax.legend(fontsize=8)

    # --- (2) Test histogram ---
    ax = axes[0, 1]
    test_all = np.concatenate([r.test_scores_a, r.test_scores_b])
    if test_all.size:
        edges = np.histogram_bin_edges(test_all, bins=bins)
        ax.hist(r.test_scores_a, bins=edges, color=color_a, alpha=0.6, label=label_a)
        ax.hist(r.test_scores_b, bins=edges, color=color_b, alpha=0.6, label=label_b)
    ax.axvline(r.boundary, color="k", linestyle="--", lw=1.5)
    ax.set_xlabel("CD projection (test)")
    ax.set_ylabel("Trials")
    ax.set_title(
        f"Test — acc={r.test_accuracy:.2%}, bal={r.test_balanced_accuracy:.2%}, "
        f"AUC={r.test_auc:.3f}"
    )
    ax.legend(fontsize=8)

    # --- (3) ROC ---
    ax = axes[1, 0]
    ax.plot(r.test_fpr, r.test_tpr, "-", color="tab:purple", lw=2,
            label=f"AUC = {r.test_auc:.3f}")
    ax.plot([0, 1], [0, 1], ":", color="gray", lw=1)
    ax.set_xlabel(f"False positive rate ({label_b} called {label_a})")
    ax.set_ylabel(f"True positive rate ({label_a} called {label_a})")
    ax.set_title("ROC (test)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (4) Confusion + summary bars ---
    ax = axes[1, 1]
    cm = r.test_confusion
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=[f"pred {label_a}", f"pred {label_b}"])
    ax.set_yticks([0, 1], labels=[f"true {label_a}", f"true {label_b}"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white",
                    fontsize=12, fontweight="bold")
    ax.set_title(
        f"Confusion (test)\nmethod={r.method}, window={window}"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- (5) Decoding over time ---
    if have_time:
        centers, accs, aucs = decode_cd_session_over_time(
            sess, bin_window=time_curve_bin, step=time_curve_step, method=method,
        )
        # Top-right (1,2): accuracy
        ax = axes[0, 2]
        ax.plot(centers, accs, "-", color="tab:green", lw=2, label="accuracy")
        ax.axhline(0.5, color="gray", linestyle=":", lw=1)
        ax.axvspan(window[0], window[1], color="orange", alpha=0.15,
                   label=f"summary window")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"Accuracy vs time ({time_curve_bin}s bin)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom-right (1,2): AUC
        ax = axes[1, 2]
        ax.plot(centers, aucs, "-", color="tab:purple", lw=2, label="AUC")
        ax.axhline(0.5, color="gray", linestyle=":", lw=1)
        ax.axvspan(window[0], window[1], color="orange", alpha=0.15)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Test AUC")
        ax.set_ylim(0, 1)
        ax.set_title("AUC vs time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"CD decoder — {sess.session}\n"
        f"{label_a} vs {label_b}",
        y=1.02,
    )
    fig.tight_layout()
    if show:
        plt.show()
    return r


# ---------------------------------------------------------------------------
# 5. TIME-RESOLVED 4-axis DECODER (CD re-fit per bin, with eligibility &
#    per-bin re-balancing)
# ---------------------------------------------------------------------------

def _balance_cells_from_ids(
    cell_ids: Dict[str, np.ndarray],
    eligible: Optional[np.ndarray],
    *,
    n_per_cell: Optional[int],
    seed: int,
) -> Tuple[Dict[str, np.ndarray], int]:
    """In-memory variant of :func:`_balance_action_cells`.

    Intersects each cell with ``eligible`` (if given), then subsamples each to
    a common ``n = min(...)`` (or ``n_per_cell`` if forced).
    Returns ``(balanced_ids, n_used_per_cell)``. Returns empty arrays and 0
    if any cell ends up empty.
    """
    filt: Dict[str, np.ndarray] = {}
    for cell, ids in cell_ids.items():
        ids = np.asarray(ids, dtype=int).ravel()
        ids = ids[ids >= 0]
        if eligible is not None:
            ids = np.intersect1d(ids, eligible, assume_unique=False)
        filt[cell] = np.unique(ids)

    sizes = {c: int(v.size) for c, v in filt.items()}
    if any(s == 0 for s in sizes.values()):
        return {c: np.empty(0, dtype=int) for c in filt}, 0

    n_min = min(sizes.values())
    if n_per_cell is not None:
        n = min(int(n_per_cell), n_min)
    else:
        n = n_min
    if n <= 0:
        return {c: np.empty(0, dtype=int) for c in filt}, 0

    rng = np.random.default_rng(int(seed))
    out: Dict[str, np.ndarray] = {}
    for cell, pool in filt.items():
        if pool.size <= n:
            out[cell] = np.sort(pool)
        else:
            pick = rng.choice(pool, size=int(n), replace=False)
            out[cell] = np.sort(pick.astype(int))
    return out, int(n)


def decode_action_axes_over_time(
    *,
    sessions: Iterable[str],
    psth_root: str | Path,
    behavior_root: str | Path,
    metadata: Optional[pd.DataFrame] = None,
    binsize: str = "0.1",
    align: str = "go_cue",
    bin_centers: Optional[np.ndarray] = None,
    t_start: float = -1.5,
    t_end: float = 0.5,
    bin_step: float = 0.1,
    bin_window: float = 0.2,
    restrict_events: Optional[Tuple[str, str]] = ("trial_start", "go_cue"),
    restrict_align: Optional[str] = None,
    axes: Sequence[str] = ("prev_choice", "up_choice", "switch_stay", "switch_dir"),
    region_group: Sequence[str] = (),
    min_units_num: int = 30,
    n_per_cell: Optional[int] = None,
    seed: int = 0,
    norm_mode: str = "divide_sqrtN",
    zscore_units: bool = False,
    decoder_method: Literal["midpoint", "optimal"] = "optimal",
    two_fold_cv: bool = True,
    random_state: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Time-resolved 4-axis CD decoder.

    For every session and every time bin:

    1. Identify trials whose ``[restrict_events[0], restrict_events[1]]``
       interval (offsets w.r.t. ``restrict_align`` or ``align``) covers the
       full bin ``[c - bin_window/2, c + bin_window/2]``.
    2. Intersect the 4 action cells (``L_L``, ``switch_LR``, ``switch_RL``,
       ``R_R``) with the eligible set and subsample each to a common count
       ``n = min(...)`` (deterministic given ``seed``). This balances every
       axis w.r.t. the orthogonal factors at *this specific bin*.
    3. For each requested axis, fit a fresh CD axis on the balanced cell
       unions using only that bin's window, decode the held-out test trials
       with a 1D threshold, and store accuracy + AUC.

    Parameters
    ----------
    bin_centers : array-like or None
        Explicit bin centers (s). When ``None`` they are built from
        ``t_start``, ``t_end``, ``bin_step``.
    bin_window : float
        Width of the CD-fit + decoding window centered on each bin center.
    restrict_events : (start_event, end_event) or None
        If given, restricts each bin to trials whose per-trial event interval
        fully covers it. Set to ``None`` to skip eligibility filtering.
        Both events must be one-per-trial (e.g. ``"trial_start"``,
        ``"trial_end"``, ``"go_cue"``, ``"previous_trial_go_cue"``,
        ``"previous_trial_start"``, ``"previous_trial_end"``). Useful pairs
        include ``("trial_start", "go_cue")`` (current ITI) and
        ``("previous_trial_go_cue", "go_cue")`` (full prev-to-current span,
        which automatically drops trial 0).
    n_per_cell : int or None
        If set, forces the per-cell sample size (capped by the available
        eligible count). Otherwise uses ``min(...)`` per bin.
    seed : int
        Master seed for the per-cell subsampling RNG. The actual RNG seed
        used at bin ``k`` is ``seed + k`` so each bin gets a fresh draw
        while still being reproducible.

    Returns
    -------
    pandas.DataFrame
        One row per (session, axis, bin_center) with columns:
        ``session``, ``axis``, ``center``, ``n_per_cell``, ``n_eligible``,
        ``train_acc``, ``test_acc``, ``train_bal_acc``, ``test_bal_acc``,
        ``test_auc``, ``boundary``, ``sign``.
    """
    from create_psth import load_zarr
    from general_utils import smart_read_csv

    for ax in axes:
        if ax not in ACTION_AXES:
            raise ValueError(f"Unknown axis {ax!r}; expected one of {list(ACTION_AXES.keys())}")

    if bin_centers is None:
        # Use linspace so multiples of bin_step land exactly (avoids floating
        # drift from arange that can cost edge bins via eligibility checks).
        n_bins = int(round((float(t_end) - float(t_start)) / float(bin_step))) + 1
        n_bins = max(n_bins, 1)
        bin_centers = np.linspace(float(t_start), float(t_start) + (n_bins - 1) * float(bin_step), n_bins)
    bin_centers = np.asarray(bin_centers, dtype=float)

    psth_root = Path(psth_root)
    behavior_root = Path(behavior_root)
    sessions = list(sessions)
    region_lbl, region_print = region_label(region_group)
    rows: List[Dict[str, Any]] = []

    metadata_empty = metadata is None or len(metadata) == 0

    for session in sessions:
        if verbose:
            print(f"\n=== {session} (region={region_lbl}) ===")
        try:
            psth_path = psth_root / f"{session}_{binsize}s.zarr"
            beh_path = behavior_root / f"behavior_summary-{session}.csv"
            psth_da = load_zarr(str(psth_path))
            df = smart_read_csv(str(beh_path))
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] load failed: {e}")
            continue

        # Unit selection
        if metadata_empty or not region_group:
            unit_ids = None
            n_units_used = "ALL"
        else:
            mask = (
                (metadata["sorted_session_name"] == session)
                & (metadata["brain_region"].isin(region_group))
            )
            unit_ids = metadata.loc[mask, "unit_index"].to_numpy()
            if len(unit_ids) < min_units_num:
                print(f"  [skip] only {len(unit_ids)} units < {min_units_num}")
                continue
            n_units_used = len(unit_ids)
        if verbose:
            print(f"  units={n_units_used}")

        # Raw cell IDs from the per-session behavior CSV.
        cell_ids: Dict[str, np.ndarray] = {}
        try:
            for cell in ACTION_CELLS:
                col = f"{cell}_trials"
                if col not in df.columns:
                    raise KeyError(f"missing column {col!r} in {beh_path.name}")
                cell_ids[cell] = np.asarray(df[col].iloc[0], dtype=int).ravel()
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] cell ID load failed: {e}")
            continue

        # Per-trial event offsets for the eligibility test (load once).
        # Offsets are always returned in the PSTH ``align`` frame so they can
        # be compared directly to the bin window. ``restrict_align`` is kept
        # for backward-compat but ignored when it differs from ``align``.
        offsets: Optional[Dict[int, Tuple[float, float]]] = None
        if restrict_events is not None:
            if restrict_align is not None and restrict_align != align:
                if verbose:
                    print(
                        f"  [info] restrict_align={restrict_align!r} differs from "
                        f"PSTH align={align!r}; using PSTH align for eligibility frame."
                    )
            try:
                offsets = compute_per_trial_event_offsets(
                    session,
                    event_start=restrict_events[0],
                    event_end=restrict_events[1],
                    align=align,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] restrict_events offsets failed ({e}); skipping eligibility")
                offsets = None

        # --- per-bin loop ---
        for k, center in enumerate(bin_centers):
            win = (float(center) - bin_window / 2.0, float(center) + bin_window / 2.0)

            if offsets is not None:
                # Tiny tolerance so bins that touch the eligibility edge
                # exactly (e.g. window ends at go_cue = 0.0) aren't lost to
                # floating-point drift coming from np.arange grids.
                eps = 1e-6
                eligible = np.array(
                    [tid for tid, (s, e) in offsets.items()
                     if s <= win[0] + eps and e >= win[1] - eps],
                    dtype=int,
                )
            else:
                eligible = None

            balanced, n_used = _balance_cells_from_ids(
                cell_ids, eligible, n_per_cell=n_per_cell, seed=int(seed) + int(k),
            )
            n_elig = int(eligible.size) if eligible is not None else -1

            if n_used == 0:
                for ax in axes:
                    rows.append({
                        "session": session, "axis": ax, "center": float(center),
                        "n_per_cell": 0, "n_eligible": n_elig,
                        "train_acc": np.nan, "test_acc": np.nan,
                        "train_bal_acc": np.nan, "test_bal_acc": np.nan,
                        "test_auc": np.nan, "boundary": np.nan, "sign": 0,
                    })
                continue

            for ax_name in axes:
                cells_a, cells_b = ACTION_AXES[ax_name]
                a_ids = np.unique(np.concatenate([balanced[c] for c in cells_a]))
                b_ids = np.unique(np.concatenate([balanced[c] for c in cells_b]))
                if a_ids.size == 0 or b_ids.size == 0:
                    rows.append({
                        "session": session, "axis": ax_name, "center": float(center),
                        "n_per_cell": n_used, "n_eligible": n_elig,
                        "train_acc": np.nan, "test_acc": np.nan,
                        "train_bal_acc": np.nan, "test_bal_acc": np.nan,
                        "test_auc": np.nan, "boundary": np.nan, "sign": 0,
                    })
                    continue
                try:
                    out = coding_direction_from_psth(
                        psth_da=psth_da,
                        trial_ids_typeA=a_ids,
                        trial_ids_typeB=b_ids,
                        align=align,
                        time_window=win,
                        projection_time_window=win,
                        random_state=int(random_state),
                        two_fold_cv=bool(two_fold_cv),
                        norm_mode=norm_mode,
                        zscore_units=zscore_units,
                        save_path=None,
                        unit_ids=unit_ids,
                    )
                except Exception as e:  # noqa: BLE001
                    if verbose:
                        print(f"    [bin {center:+.2f}s axis={ax_name}] CD fit failed: {e}")
                    rows.append({
                        "session": session, "axis": ax_name, "center": float(center),
                        "n_per_cell": n_used, "n_eligible": n_elig,
                        "train_acc": np.nan, "test_acc": np.nan,
                        "train_bal_acc": np.nan, "test_bal_acc": np.nan,
                        "test_auc": np.nan, "boundary": np.nan, "sign": 0,
                    })
                    continue

                tr_a = np.asarray(out["projection_train_A"], dtype=float).ravel()
                tr_b = np.asarray(out["projection_train_B"], dtype=float).ravel()
                te_a = np.asarray(out["projection_test_A"], dtype=float).ravel()
                te_b = np.asarray(out["projection_test_B"], dtype=float).ravel()
                if tr_a.size == 0 or tr_b.size == 0:
                    rows.append({
                        "session": session, "axis": ax_name, "center": float(center),
                        "n_per_cell": n_used, "n_eligible": n_elig,
                        "train_acc": np.nan, "test_acc": np.nan,
                        "train_bal_acc": np.nan, "test_bal_acc": np.nan,
                        "test_auc": np.nan, "boundary": np.nan, "sign": 0,
                    })
                    continue

                boundary, sign = _fit_threshold(tr_a, tr_b, method=decoder_method)
                pa_tr = _predict(tr_a, boundary, sign)
                pb_tr = _predict(tr_b, boundary, sign)
                train_acc = (pa_tr.sum() + (1 - pb_tr).sum()) / max(pa_tr.size + pb_tr.size, 1)
                train_bal = 0.5 * (
                    (pa_tr.mean() if pa_tr.size else 0.0)
                    + ((1 - pb_tr).mean() if pb_tr.size else 0.0)
                )
                if te_a.size and te_b.size:
                    pa_te = _predict(te_a, boundary, sign)
                    pb_te = _predict(te_b, boundary, sign)
                    test_acc = (pa_te.sum() + (1 - pb_te).sum()) / max(pa_te.size + pb_te.size, 1)
                    test_bal = 0.5 * (
                        (pa_te.mean() if pa_te.size else 0.0)
                        + ((1 - pb_te).mean() if pb_te.size else 0.0)
                    )
                    _, _, auc = _roc_auc_1d(te_a, te_b, sign)
                else:
                    test_acc = np.nan
                    test_bal = np.nan
                    auc = np.nan

                rows.append({
                    "session": session, "axis": ax_name, "center": float(center),
                    "n_per_cell": n_used, "n_eligible": n_elig,
                    "train_acc": float(train_acc),
                    "test_acc": float(test_acc),
                    "train_bal_acc": float(train_bal),
                    "test_bal_acc": float(test_bal),
                    "test_auc": float(auc),
                    "boundary": float(boundary),
                    "sign": int(sign),
                })

            if verbose and (k % max(1, len(bin_centers) // 10) == 0):
                print(f"    bin {center:+.3f}s done (n_per_cell={n_used}, eligible={n_elig})")

    return pd.DataFrame(rows)


def decode_prev_reward_over_time(
    *,
    sessions: Iterable[str],
    psth_root: str | Path,
    metadata: Optional[pd.DataFrame] = None,
    binsize: str = "0.1",
    align: str = "go_cue",
    bin_centers: Optional[np.ndarray] = None,
    t_start: float = -1.5,
    t_end: float = 0.5,
    bin_step: float = 0.1,
    bin_window: float = 0.2,
    restrict_events: Optional[Tuple[str, str]] = ("trial_start", "go_cue"),
    restrict_align: Optional[str] = None,
    region_group: Sequence[str] = (),
    min_units_num: int = 30,
    n_per_cell: Optional[int] = None,
    seed: int = 0,
    norm_mode: str = "divide_sqrtN",
    zscore_units: bool = False,
    decoder_method: Literal["midpoint", "optimal"] = "optimal",
    two_fold_cv: bool = True,
    random_state: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Time-resolved CD decoder for the **previous-trial reward** axis.

    Single-axis, 2-cell design (``prev_rewarded`` vs ``prev_unrewarded``).
    Trial membership is computed from the NWB ``rewarded_historyL/R`` columns
    shifted by one trial (see :func:`compute_prev_reward_cells`).

    Per bin, the two cells are intersected with the eligible set (trials
    whose ``[restrict_events[0], restrict_events[1]]`` interval covers the
    bin) and subsampled to ``n = min(...)`` so the decoder is balanced
    across reward outcomes at every bin.

    Returns a DataFrame with the same columns as
    :func:`decode_action_axes_over_time` (axis is always ``"prev_reward"``).
    """
    from create_psth import load_zarr

    if bin_centers is None:
        n_bins = int(round((float(t_end) - float(t_start)) / float(bin_step))) + 1
        n_bins = max(n_bins, 1)
        bin_centers = np.linspace(
            float(t_start), float(t_start) + (n_bins - 1) * float(bin_step), n_bins
        )
    bin_centers = np.asarray(bin_centers, dtype=float)

    psth_root = Path(psth_root)
    sessions = list(sessions)
    rows: List[Dict[str, object]] = []

    for session in sessions:
        if verbose:
            print(f"\n=========== Session: {session} ===========")
        zarr_path = psth_root / f"{session}_{binsize}s.zarr"
        if not zarr_path.exists():
            print(f"  [skip] PSTH zarr missing: {zarr_path}")
            continue
        try:
            psth_da = load_zarr(str(zarr_path))
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] failed to load {zarr_path.name}: {e}")
            continue

        # Unit selection (same convention as the action decoder).
        unit_ids: Optional[np.ndarray] = None
        n_units_used = int(psth_da.sizes.get("unit", 0))
        if metadata is not None and len(region_group):
            mask = (
                (metadata["sorted_session_name"] == session)
                & (metadata["brain_region"].isin(region_group))
            )
            unit_ids = metadata.loc[mask, "unit_index"].to_numpy()
            if len(unit_ids) < min_units_num:
                print(f"  [skip] only {len(unit_ids)} units < {min_units_num}")
                continue
            n_units_used = len(unit_ids)
        if verbose:
            print(f"  units={n_units_used}")

        # Prev-reward cell membership (from NWB).
        try:
            cell_ids = compute_prev_reward_cells(session)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] prev-reward cell construction failed: {e}")
            continue

        # Eligibility offsets in the PSTH align frame.
        offsets: Optional[Dict[int, Tuple[float, float]]] = None
        if restrict_events is not None:
            if restrict_align is not None and restrict_align != align and verbose:
                print(
                    f"  [info] restrict_align={restrict_align!r} differs from "
                    f"PSTH align={align!r}; using PSTH align for eligibility frame."
                )
            try:
                offsets = compute_per_trial_event_offsets(
                    session,
                    event_start=restrict_events[0],
                    event_end=restrict_events[1],
                    align=align,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] restrict_events offsets failed ({e}); skipping eligibility")
                offsets = None

        for k, center in enumerate(bin_centers):
            win = (float(center) - bin_window / 2.0, float(center) + bin_window / 2.0)

            if offsets is not None:
                eps = 1e-6
                eligible = np.array(
                    [tid for tid, (s, e) in offsets.items()
                     if s <= win[0] + eps and e >= win[1] - eps],
                    dtype=int,
                )
            else:
                eligible = None

            balanced, n_used = _balance_cells_from_ids(
                cell_ids, eligible, n_per_cell=n_per_cell, seed=int(seed) + int(k),
            )
            n_elig = int(eligible.size) if eligible is not None else -1

            base_row = {
                "session": session, "axis": "prev_reward", "center": float(center),
                "n_per_cell": n_used, "n_eligible": n_elig,
                "train_acc": np.nan, "test_acc": np.nan,
                "train_bal_acc": np.nan, "test_bal_acc": np.nan,
                "test_auc": np.nan, "boundary": np.nan, "sign": 0,
            }
            if n_used == 0:
                rows.append({**base_row, "n_per_cell": 0})
                continue

            a_ids = balanced["prev_rewarded"]
            b_ids = balanced["prev_unrewarded"]
            if a_ids.size == 0 or b_ids.size == 0:
                rows.append(base_row)
                continue
            try:
                out = coding_direction_from_psth(
                    psth_da=psth_da,
                    trial_ids_typeA=a_ids,
                    trial_ids_typeB=b_ids,
                    align=align,
                    time_window=win,
                    projection_time_window=win,
                    random_state=int(random_state),
                    two_fold_cv=bool(two_fold_cv),
                    norm_mode=norm_mode,
                    zscore_units=zscore_units,
                    save_path=None,
                    unit_ids=unit_ids,
                )
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"    [bin {center:+.2f}s prev_reward] CD fit failed: {e}")
                rows.append(base_row)
                continue

            tr_a = np.asarray(out["projection_train_A"], dtype=float).ravel()
            tr_b = np.asarray(out["projection_train_B"], dtype=float).ravel()
            te_a = np.asarray(out["projection_test_A"], dtype=float).ravel()
            te_b = np.asarray(out["projection_test_B"], dtype=float).ravel()
            if tr_a.size == 0 or tr_b.size == 0:
                rows.append(base_row)
                continue

            boundary, sign = _fit_threshold(tr_a, tr_b, method=decoder_method)
            pa_tr = _predict(tr_a, boundary, sign)
            pb_tr = _predict(tr_b, boundary, sign)
            train_acc = (pa_tr.sum() + (1 - pb_tr).sum()) / max(pa_tr.size + pb_tr.size, 1)
            train_bal = 0.5 * (
                (pa_tr.mean() if pa_tr.size else 0.0)
                + ((1 - pb_tr).mean() if pb_tr.size else 0.0)
            )
            if te_a.size and te_b.size:
                pa_te = _predict(te_a, boundary, sign)
                pb_te = _predict(te_b, boundary, sign)
                test_acc = (pa_te.sum() + (1 - pb_te).sum()) / max(pa_te.size + pb_te.size, 1)
                test_bal = 0.5 * (
                    (pa_te.mean() if pa_te.size else 0.0)
                    + ((1 - pb_te).mean() if pb_te.size else 0.0)
                )
                _, _, auc = _roc_auc_1d(te_a, te_b, sign)
            else:
                test_acc = np.nan
                test_bal = np.nan
                auc = np.nan

            rows.append({
                "session": session, "axis": "prev_reward", "center": float(center),
                "n_per_cell": n_used, "n_eligible": n_elig,
                "train_acc": float(train_acc),
                "test_acc": float(test_acc),
                "train_bal_acc": float(train_bal),
                "test_bal_acc": float(test_bal),
                "test_auc": float(auc),
                "boundary": float(boundary),
                "sign": int(sign),
            })

            if verbose and (k % max(1, len(bin_centers) // 10) == 0):
                print(f"    bin {center:+.3f}s done (n_per_cell={n_used}, eligible={n_elig})")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CD axis stability across non-overlapping time windows
# ---------------------------------------------------------------------------

def _make_nonoverlapping_windows(
    t_start: float, t_end: float, bin_width: float
) -> List[Tuple[float, float]]:
    """Return a list of contiguous non-overlapping ``(t0, t1)`` windows.

    Windows tile ``[t_start, t_end]`` with width ``bin_width``. The last
    window is truncated if it does not fit exactly.
    """
    if bin_width <= 0:
        raise ValueError(f"bin_width must be > 0, got {bin_width!r}")
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be > t_start ({t_start})")
    n = int(np.floor((float(t_end) - float(t_start)) / float(bin_width)))
    edges = [float(t_start) + i * float(bin_width) for i in range(n + 1)]
    if edges[-1] < float(t_end) - 1e-9:
        edges.append(float(t_end))
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def compute_cd_stability(
    *,
    session: str,
    psth_root: str | Path,
    behavior_root: str | Path,
    trial_types: Tuple[str, str],
    align: str = "go_cue",
    binsize: str = "0.1",
    time_windows: Optional[Sequence[Tuple[float, float]]] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    bin_width: Optional[float] = None,
    metadata: Optional[pd.DataFrame] = None,
    region_group: Sequence[str] = (),
    min_units_num: int = 30,
    restrict_events: Optional[Tuple[str, str]] = None,
    restrict_align: Optional[str] = None,
    use_common_trials: bool = False,
    norm_mode: str = "divide_sqrtN",
    zscore_units: bool = False,
    random_state: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute CD axes in non-overlapping time windows + their cosine matrix.

    For one session we refit the coding-direction axis (class A vs class B,
    defined by ``trial_types``) inside every requested window, then take the
    pairwise cosine similarity between the unit-space axis vectors. Each axis
    is unit-norm (set by :func:`_compute_cd_axis`), so cosine = dot product.

    Provide either an explicit ``time_windows`` list of ``(t0, t1)`` tuples,
    or ``t_start`` / ``t_end`` / ``bin_width`` (windows are then tiled
    contiguously, no overlap).

    ``restrict_events`` : optional ``(event_start, event_end)`` pair. When
    given, per window the CD axis is fit only on trials whose per-trial
    ``[event_start, event_end]`` interval (offsets in the PSTH ``align``
    frame; ``restrict_align`` is accepted for symmetry but ignored when it
    differs from ``align``) fully covers that window. A small floating-point
    tolerance (1e-6) is applied on each edge. The per-window ``n_typeA`` /
    ``n_typeB`` reflect the *eligible* trials actually used.

    ``use_common_trials`` : if ``True``, intersect the per-window eligible
    trial sets and refit every CD axis on the *same* (common) trial set.
    This isolates temporal variation in the population code from variation
    driven by changing trial composition. Has no effect when
    ``restrict_events is None`` (every trial is then eligible everywhere).

    Returns
    -------
    dict
        ``session``       : session name
        ``time_windows``  : ``np.ndarray`` of shape ``(n_windows, 2)``
        ``centers``       : window centers (s, ``np.ndarray``)
        ``axes``          : ``(n_windows, n_units)`` matrix of unit-norm CDs
        ``cosine``        : ``(n_windows, n_windows)`` cosine-similarity matrix
        ``unit_ids``      : unit indices used (or None for ALL units)
        ``n_typeA``, ``n_typeB`` : trial counts going into each per-window fit
        ``trial_types``   : the input ``trial_types`` tuple
        ``align``, ``binsize`` : echoed from inputs
    """
    from create_psth import load_zarr
    from general_utils import smart_read_csv

    # Resolve window list ----------------------------------------------------
    if time_windows is None:
        if t_start is None or t_end is None or bin_width is None:
            raise ValueError(
                "Provide either `time_windows` or all of `t_start`/`t_end`/`bin_width`."
            )
        windows = _make_nonoverlapping_windows(float(t_start), float(t_end), float(bin_width))
    else:
        windows = [(float(a), float(b)) for (a, b) in time_windows]
    if not windows:
        raise ValueError("No time windows produced; check inputs.")

    psth_path = Path(psth_root) / f"{session}_{binsize}s.zarr"
    beh_path = Path(behavior_root) / f"behavior_summary-{session}.csv"
    if not psth_path.exists():
        raise FileNotFoundError(f"PSTH zarr not found: {psth_path}")
    if not beh_path.exists():
        raise FileNotFoundError(f"Behavior CSV not found: {beh_path}")

    psth_da = load_zarr(str(psth_path))
    df = smart_read_csv(str(beh_path))

    if trial_types[0] not in df.columns or trial_types[1] not in df.columns:
        raise KeyError(
            f"Behavior CSV missing trial_types columns: {trial_types}"
        )
    typeA_ids = _clean_ids(df[trial_types[0]])
    typeB_ids = _clean_ids(df[trial_types[1]])
    if typeA_ids.size == 0 or typeB_ids.size == 0:
        raise ValueError(
            f"Empty trial set: |A|={typeA_ids.size}, |B|={typeB_ids.size}"
        )

    # Unit selection (same convention as decode_action_axes_over_time) ------
    unit_ids: Optional[np.ndarray] = None
    if metadata is not None and len(region_group):
        mask = (
            (metadata["sorted_session_name"] == session)
            & (metadata["brain_region"].isin(region_group))
        )
        unit_ids = metadata.loc[mask, "unit_index"].to_numpy()
        if len(unit_ids) < min_units_num:
            raise ValueError(
                f"Only {len(unit_ids)} units < min_units_num={min_units_num}"
            )

    if verbose:
        print(
            f"[{session}] CD stability: {len(windows)} windows, "
            f"|A|={typeA_ids.size}, |B|={typeB_ids.size}, "
            f"units={'ALL' if unit_ids is None else len(unit_ids)}"
        )

    # Per-trial eligibility offsets (in the PSTH ``align`` frame), if any.
    offsets: Optional[Dict[int, Tuple[float, float]]] = None
    if restrict_events is not None:
        if restrict_align is not None and restrict_align != align and verbose:
            print(
                f"  [info] restrict_align={restrict_align!r} differs from "
                f"PSTH align={align!r}; using PSTH align for eligibility frame."
            )
        try:
            offsets = compute_per_trial_event_offsets(
                session,
                event_start=restrict_events[0],
                event_end=restrict_events[1],
                align=align,
            )
            if verbose:
                print(
                    f"  restrict_events={restrict_events}: "
                    f"{len(offsets)} trials have valid offsets."
                )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(
                    f"  [warn] restrict_events offsets failed ({e}); "
                    "skipping eligibility filter."
                )
            offsets = None

    # Optional: intersect eligible trials across all windows so every CD
    # axis is fit on the SAME trial set.
    common_ids: Optional[np.ndarray] = None
    if use_common_trials and offsets is not None:
        per_win_eligible: List[np.ndarray] = []
        for win in windows:
            elig = np.array(
                [tid for tid, (s, e) in offsets.items()
                 if s <= win[0] + 1e-6 and e >= win[1] - 1e-6],
                dtype=int,
            )
            per_win_eligible.append(elig)
        if per_win_eligible:
            common_ids = per_win_eligible[0]
            for elig in per_win_eligible[1:]:
                common_ids = np.intersect1d(common_ids, elig, assume_unique=False)
        else:
            common_ids = np.empty(0, dtype=int)
        if verbose:
            print(
                f"  use_common_trials=True: common eligible trials across "
                f"{len(windows)} windows = {common_ids.size}"
            )
        if common_ids.size == 0:
            raise RuntimeError(
                f"[{session}] use_common_trials=True but no trial is eligible "
                f"in every window. Narrow t_start/t_end/bin_width or disable."
            )

    # Fit one CD axis per window --------------------------------------------
    axes: List[np.ndarray] = []
    used_windows: List[Tuple[float, float]] = []
    per_window_counts: List[Tuple[int, int, int]] = []  # (n_eligible, n_A, n_B)
    eps = 1e-6
    for win in windows:
        if common_ids is not None:
            # Same trial set for every window.
            a_ids = np.intersect1d(typeA_ids, common_ids, assume_unique=False)
            b_ids = np.intersect1d(typeB_ids, common_ids, assume_unique=False)
            n_elig = int(common_ids.size)
            if a_ids.size == 0 or b_ids.size == 0:
                if verbose:
                    print(
                        f"  [skip] window {win}: common-trial |A|={a_ids.size}, "
                        f"|B|={b_ids.size}"
                    )
                continue
        elif offsets is not None:
            eligible = np.array(
                [tid for tid, (s, e) in offsets.items()
                 if s <= win[0] + eps and e >= win[1] - eps],
                dtype=int,
            )
            a_ids = np.intersect1d(typeA_ids, eligible, assume_unique=False)
            b_ids = np.intersect1d(typeB_ids, eligible, assume_unique=False)
            n_elig = int(eligible.size)
            if a_ids.size == 0 or b_ids.size == 0:
                if verbose:
                    print(
                        f"  [skip] window {win}: |A_eligible|={a_ids.size}, "
                        f"|B_eligible|={b_ids.size}"
                    )
                continue
        else:
            a_ids = typeA_ids
            b_ids = typeB_ids
            n_elig = -1

        try:
            out = coding_direction_from_psth(
                psth_da=psth_da,
                trial_ids_typeA=a_ids,
                trial_ids_typeB=b_ids,
                align=align,
                time_window=win,
                projection_time_window=win,
                random_state=int(random_state),
                two_fold_cv=False,
                norm_mode=norm_mode,
                zscore_units=zscore_units,
                save_path=None,
                unit_ids=unit_ids,
            )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [warn] window {win} CD fit failed: {e}")
            continue
        w = np.asarray(out["final_all"]["axis_w"], dtype=float).ravel()
        if w.size == 0:
            continue
        axes.append(w)
        used_windows.append(win)
        per_window_counts.append((n_elig, int(a_ids.size), int(b_ids.size)))

    if not axes:
        raise RuntimeError(f"[{session}] No CD axes successfully fitted.")

    axes_arr = np.vstack(axes)                         # (n_windows, n_units)
    used = np.asarray(used_windows, dtype=float)       # (n_windows, 2)
    centers = used.mean(axis=1)
    counts_arr = np.asarray(per_window_counts, dtype=int)  # (n_windows, 3)

    # Cosine similarity (axes are already unit-norm; renormalize defensively).
    norms = np.linalg.norm(axes_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    axes_unit = axes_arr / norms
    cosine = axes_unit @ axes_unit.T                   # (n_windows, n_windows)

    return {
        "session": session,
        "time_windows": used,
        "centers": centers,
        "axes": axes_arr,
        "cosine": cosine,
        "unit_ids": unit_ids,
        "n_typeA": int(typeA_ids.size),
        "n_typeB": int(typeB_ids.size),
        "per_window_n_eligible": counts_arr[:, 0] if counts_arr.size else np.empty(0, int),
        "per_window_n_A":        counts_arr[:, 1] if counts_arr.size else np.empty(0, int),
        "per_window_n_B":        counts_arr[:, 2] if counts_arr.size else np.empty(0, int),
        "trial_types": tuple(trial_types),
        "align": align,
        "binsize": binsize,
        "restrict_events": (tuple(restrict_events) if restrict_events else None),
        "restrict_align": restrict_align,
        "use_common_trials": bool(use_common_trials),
        "common_trial_ids": (None if common_ids is None else common_ids.copy()),
    }


def compute_cd_stability_multi(
    *,
    sessions: Iterable[str],
    psth_root: str | Path,
    behavior_root: str | Path,
    trial_types: Tuple[str, str],
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Run :func:`compute_cd_stability` for many sessions; skip failures."""
    out: Dict[str, Dict[str, Any]] = {}
    for session in sessions:
        try:
            out[session] = compute_cd_stability(
                session=session,
                psth_root=psth_root,
                behavior_root=behavior_root,
                trial_types=trial_types,
                **kwargs,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {session}: {e}")
    return out


def compute_action_cd_stability(
    *,
    session: str,
    psth_root: str | Path,
    behavior_root: str | Path,
    align: str = "trial_start",
    binsize: str = "0.1",
    time_windows: Optional[Sequence[Tuple[float, float]]] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    bin_width: Optional[float] = None,
    axes: Sequence[str] = ("prev_choice", "up_choice", "switch_stay", "switch_dir"),
    metadata: Optional[pd.DataFrame] = None,
    region_group: Sequence[str] = (),
    min_units_num: int = 30,
    restrict_events: Optional[Tuple[str, str]] = None,
    restrict_align: Optional[str] = None,
    n_per_cell: Optional[int] = None,
    use_common_trials: bool = False,
    norm_mode: str = "divide_sqrtN",
    zscore_units: bool = False,
    seed: int = 0,
    random_state: int = 0,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """CD-axis stability across non-overlapping windows for the action axes.

    Counterpart of :func:`compute_cd_stability` for the 4-cell (prev-choice
    x upcoming-choice) design used by :func:`decode_action_axes_over_time`.
    For each window:

    1. Find trials whose ``[restrict_events[0], restrict_events[1]]`` offsets
       (in the PSTH ``align`` frame) cover the window.
    2. Intersect with each of the 4 :data:`ACTION_CELLS` and subsample to a
       common ``n = min(...)`` (or ``n_per_cell`` if forced) so each axis is
       balanced w.r.t. the orthogonal factors at this bin.
    3. For each requested axis (``prev_choice``, ``up_choice``,
       ``switch_stay``, ``switch_dir``), union the balanced cells and fit a
       fresh unit-norm CD axis on the window's data.

    ``use_common_trials`` : if ``True``, the eligible-trial sets are first
    intersected across all windows, then balancing is performed *once* on
    that common set and reused for every window. This isolates temporal
    variation in the population code from variation driven by changing trial
    composition.

    Returns
    -------
    dict
        ``{axis_name: stab_dict}`` where each ``stab_dict`` has the same
        schema as :func:`compute_cd_stability` (with ``trial_types`` set to
        the per-axis cell tuples).
    """
    from create_psth import load_zarr
    from general_utils import smart_read_csv

    # --- Resolve windows ---------------------------------------------------
    if time_windows is None:
        if t_start is None or t_end is None or bin_width is None:
            raise ValueError(
                "Provide either `time_windows` or all of `t_start`/`t_end`/`bin_width`."
            )
        windows = _make_nonoverlapping_windows(float(t_start), float(t_end), float(bin_width))
    else:
        windows = [(float(a), float(b)) for (a, b) in time_windows]
    if not windows:
        raise ValueError("No time windows produced; check inputs.")

    # --- Validate axes -----------------------------------------------------
    bad = [a for a in axes if a not in ACTION_AXES]
    if bad:
        raise ValueError(f"Unknown axis names {bad}; expected subset of {list(ACTION_AXES.keys())}")

    # --- Load PSTH + behavior CSV -----------------------------------------
    psth_path = Path(psth_root) / f"{session}_{binsize}s.zarr"
    beh_path = Path(behavior_root) / f"behavior_summary-{session}.csv"
    if not psth_path.exists():
        raise FileNotFoundError(f"PSTH zarr not found: {psth_path}")
    if not beh_path.exists():
        raise FileNotFoundError(f"Behavior CSV not found: {beh_path}")

    psth_da = load_zarr(str(psth_path))
    df = smart_read_csv(str(beh_path))

    cell_ids: Dict[str, np.ndarray] = {}
    for cell in ACTION_CELLS:
        col = f"{cell}_trials"
        if col not in df.columns:
            raise KeyError(f"Missing column {col!r} in {beh_path.name}")
        cell_ids[cell] = np.asarray(df[col].iloc[0], dtype=int).ravel()

    # --- Unit selection ---------------------------------------------------
    unit_ids: Optional[np.ndarray] = None
    if metadata is not None and len(region_group):
        mask = (
            (metadata["sorted_session_name"] == session)
            & (metadata["brain_region"].isin(region_group))
        )
        unit_ids = metadata.loc[mask, "unit_index"].to_numpy()
        if len(unit_ids) < min_units_num:
            raise ValueError(
                f"Only {len(unit_ids)} units < min_units_num={min_units_num}"
            )

    if verbose:
        print(
            f"[{session}] action CD stability: {len(windows)} windows, "
            f"axes={list(axes)}, units={'ALL' if unit_ids is None else len(unit_ids)}"
        )

    # --- Per-trial eligibility offsets (PSTH align frame) -----------------
    offsets: Optional[Dict[int, Tuple[float, float]]] = None
    if restrict_events is not None:
        if restrict_align is not None and restrict_align != align and verbose:
            print(
                f"  [info] restrict_align={restrict_align!r} differs from "
                f"PSTH align={align!r}; using PSTH align for eligibility frame."
            )
        try:
            offsets = compute_per_trial_event_offsets(
                session,
                event_start=restrict_events[0],
                event_end=restrict_events[1],
                align=align,
            )
            if verbose:
                print(
                    f"  restrict_events={restrict_events}: "
                    f"{len(offsets)} trials have valid offsets."
                )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(
                    f"  [warn] restrict_events offsets failed ({e}); "
                    "skipping eligibility filter."
                )
            offsets = None

    # --- Optional: shared eligible trial set across all windows ------------
    common_ids: Optional[np.ndarray] = None
    common_balanced: Optional[Dict[str, np.ndarray]] = None
    common_n_used: int = 0
    if use_common_trials and offsets is not None:
        per_win_eligible: List[np.ndarray] = []
        for win in windows:
            elig = np.array(
                [tid for tid, (s, e) in offsets.items()
                 if s <= win[0] + 1e-6 and e >= win[1] - 1e-6],
                dtype=int,
            )
            per_win_eligible.append(elig)
        common_ids = per_win_eligible[0]
        for elig in per_win_eligible[1:]:
            common_ids = np.intersect1d(common_ids, elig, assume_unique=False)
        if verbose:
            print(
                f"  use_common_trials=True: common eligible trials across "
                f"{len(windows)} windows = {common_ids.size}"
            )
        if common_ids.size == 0:
            raise RuntimeError(
                f"[{session}] use_common_trials=True but no trial is eligible "
                f"in every window. Narrow t_start/t_end/bin_width or disable."
            )
        # Balance once on the common set; reuse for every window.
        common_balanced, common_n_used = _balance_cells_from_ids(
            cell_ids, common_ids, n_per_cell=n_per_cell, seed=int(seed),
        )
        if common_n_used == 0:
            raise RuntimeError(
                f"[{session}] use_common_trials=True but common-trial balance "
                f"yielded 0 trials per cell."
            )
        if verbose:
            print(f"  common balanced: n_per_cell={common_n_used}")

    # --- Per-window per-axis CD fit ---------------------------------------
    # axes_per: axis -> list of (window, axis_vector, n_eligible, n_per_cell, n_A, n_B)
    axes_per: Dict[str, List[Tuple[Tuple[float, float], np.ndarray, int, int, int, int]]] = {
        ax: [] for ax in axes
    }

    eps = 1e-6
    for k, win in enumerate(windows):
        if common_balanced is not None:
            balanced = common_balanced
            n_used = common_n_used
            n_elig = int(common_ids.size) if common_ids is not None else -1
        else:
            if offsets is not None:
                eligible = np.array(
                    [tid for tid, (s, e) in offsets.items()
                     if s <= win[0] + eps and e >= win[1] - eps],
                    dtype=int,
                )
                n_elig = int(eligible.size)
            else:
                eligible = None
                n_elig = -1
            balanced, n_used = _balance_cells_from_ids(
                cell_ids, eligible, n_per_cell=n_per_cell, seed=int(seed) + int(k),
            )
            if n_used == 0:
                if verbose:
                    print(f"  [skip] window {win}: empty after balancing")
                continue

        for ax_name in axes:
            cells_a, cells_b = ACTION_AXES[ax_name]
            a_ids = np.unique(np.concatenate([balanced[c] for c in cells_a]))
            b_ids = np.unique(np.concatenate([balanced[c] for c in cells_b]))
            if a_ids.size == 0 or b_ids.size == 0:
                continue
            try:
                out = coding_direction_from_psth(
                    psth_da=psth_da,
                    trial_ids_typeA=a_ids,
                    trial_ids_typeB=b_ids,
                    align=align,
                    time_window=win,
                    projection_time_window=win,
                    random_state=int(random_state),
                    two_fold_cv=False,
                    norm_mode=norm_mode,
                    zscore_units=zscore_units,
                    save_path=None,
                    unit_ids=unit_ids,
                )
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"  [warn] window {win} axis={ax_name} CD fit failed: {e}")
                continue
            w = np.asarray(out["final_all"]["axis_w"], dtype=float).ravel()
            if w.size == 0:
                continue
            axes_per[ax_name].append(
                (win, w, n_elig, int(n_used), int(a_ids.size), int(b_ids.size))
            )

    # --- Assemble per-axis stab dicts -------------------------------------
    results: Dict[str, Dict[str, Any]] = {}
    for ax_name in axes:
        recs = axes_per[ax_name]
        if not recs:
            if verbose:
                print(f"  [warn] axis={ax_name}: no CD axes successfully fitted")
            continue
        used = np.asarray([r[0] for r in recs], dtype=float)        # (n_windows, 2)
        axes_arr = np.vstack([r[1] for r in recs])                  # (n_windows, n_units)
        centers = used.mean(axis=1)
        n_elig_arr = np.asarray([r[2] for r in recs], dtype=int)
        n_pc_arr   = np.asarray([r[3] for r in recs], dtype=int)
        n_a_arr    = np.asarray([r[4] for r in recs], dtype=int)
        n_b_arr    = np.asarray([r[5] for r in recs], dtype=int)
        norms = np.linalg.norm(axes_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        axes_unit = axes_arr / norms
        cosine = axes_unit @ axes_unit.T
        cells_a, cells_b = ACTION_AXES[ax_name]
        results[ax_name] = {
            "session": session,
            "axis": ax_name,
            "time_windows": used,
            "centers": centers,
            "axes": axes_arr,
            "cosine": cosine,
            "unit_ids": unit_ids,
            "n_typeA": int(sum(cell_ids[c].size for c in cells_a)),
            "n_typeB": int(sum(cell_ids[c].size for c in cells_b)),
            "per_window_n_eligible": n_elig_arr,
            "per_window_n_per_cell": n_pc_arr,
            "per_window_n_A": n_a_arr,
            "per_window_n_B": n_b_arr,
            "trial_types": (tuple(cells_a), tuple(cells_b)),
            "align": align,
            "binsize": binsize,
            "restrict_events": (tuple(restrict_events) if restrict_events else None),
            "restrict_align": restrict_align,
            "use_common_trials": bool(use_common_trials),
            "common_trial_ids": (None if common_ids is None else common_ids.copy()),
            "n_per_cell": n_per_cell,
        }
    if not results:
        raise RuntimeError(f"[{session}] No action CD axes successfully fitted.")
    return results


def compute_action_cd_stability_multi(
    *,
    sessions: Iterable[str],
    psth_root: str | Path,
    behavior_root: str | Path,
    axes: Sequence[str] = ("prev_choice", "up_choice", "switch_stay", "switch_dir"),
    **kwargs: Any,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run :func:`compute_action_cd_stability` for many sessions.

    Returns a nested dict keyed first by axis name, then by session, so each
    axis's per-session collection can be passed directly to
    :func:`plot_cd_stability`.
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {ax: {} for ax in axes}
    for session in sessions:
        try:
            per_axis = compute_action_cd_stability(
                session=session,
                psth_root=psth_root,
                behavior_root=behavior_root,
                axes=axes,
                **kwargs,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {session}: {e}")
            continue
        for ax_name, stab in per_axis.items():
            out[ax_name][session] = stab
    # Drop axes that ended up empty across all sessions.
    return {ax: d for ax, d in out.items() if d}


def plot_cd_stability(
    stability: Dict[str, Any] | Sequence[Dict[str, Any]] | Dict[str, Dict[str, Any]],
    *,
    aggregate: bool = False,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annotate_centers: bool = True,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Heatmap(s) of pairwise CD cosine similarity across time windows.

    Accepts either:

    - a single stability ``dict`` (one heatmap),
    - a sequence/dict of stability dicts (one heatmap per session), or
    - a sequence/dict of stability dicts with ``aggregate=True`` -> a single
      heatmap of the **mean cosine across sessions** (sessions must share
      the same window grid).
    """
    # Normalize input -------------------------------------------------------
    if isinstance(stability, dict) and "cosine" in stability and "centers" in stability:
        stabs: List[Dict[str, Any]] = [stability]
        single_input = True
    elif isinstance(stability, dict):
        stabs = list(stability.values())
        single_input = False
    else:
        stabs = list(stability)
        single_input = False
    if not stabs:
        print("[plot_cd_stability] no stability dicts provided.")
        return None

    def _draw(ax: plt.Axes, cos: np.ndarray, centers: np.ndarray, panel_title: str) -> Any:
        n = cos.shape[0]
        extent = (-0.5, n - 0.5, n - 0.5, -0.5)
        im = ax.imshow(cos, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
                        aspect="equal", origin="upper", interpolation="nearest")
        if annotate_centers:
            # Show window-center labels every few ticks.
            step = max(1, n // 12)
            ticks = np.arange(0, n, step)
            labels = [f"{centers[i]:+.2f}" for i in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels, fontsize=8)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel("Window center (s)")
        ax.set_ylabel("Window center (s)")
        ax.set_title(panel_title, fontsize=10)
        return im

    # Aggregate across sessions --------------------------------------------
    if aggregate:
        # Build the union of all window centers (rounded for robust hashing),
        # then for each session map its cosine matrix into that full grid
        # (NaN where a window was skipped). nanmean across sessions handles
        # the gaps. This makes aggregation robust to per-session eligibility
        # filtering producing different (shorter) window lists.
        round_decimals = 6
        center_sets = []
        for s in stabs:
            center_sets.append(np.round(np.asarray(s["centers"], dtype=float), round_decimals))
        all_centers = np.unique(np.concatenate(center_sets))
        n_full = all_centers.size
        cubes = np.full((len(stabs), n_full, n_full), np.nan, dtype=float)
        for k, s in enumerate(stabs):
            sc = np.round(np.asarray(s["centers"], dtype=float), round_decimals)
            cos = np.asarray(s["cosine"], dtype=float)
            # Index of each session center in the full grid
            idx = np.searchsorted(all_centers, sc)
            # Defensive: ensure exact matches (np.unique was applied above)
            if not np.all(all_centers[idx] == sc):
                raise RuntimeError(
                    "Failed to align session centers to the union grid; "
                    "this should not happen."
                )
            cubes[k][np.ix_(idx, idx)] = cos
        with np.errstate(invalid="ignore"):
            cos_mean = np.nanmean(cubes, axis=0)
        # Coverage per cell: how many sessions contribute to each pair.
        coverage = np.sum(np.isfinite(cubes), axis=0)
        centers = all_centers

        if np.any(coverage == 0):
            print(
                f"[plot_cd_stability] {int((coverage == 0).sum())} cells have "
                "no session coverage and will appear blank."
            )

        fig, ax = plt.subplots(figsize=figsize or (6, 5))
        ttl = title or (
            f"CD stability (cosine) — mean across {len(stabs)} sessions"
        )
        im = _draw(ax, cos_mean, centers, ttl)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
        fig.tight_layout()
        if show:
            plt.show()
            return None
        return fig

    # One panel per session -------------------------------------------------
    n_sess = len(stabs)
    if single_input:
        n_cols, n_rows = 1, 1
    else:
        n_cols = min(3, n_sess)
        n_rows = int(np.ceil(n_sess / n_cols))
    if figsize is None:
        figsize = (5.0 * n_cols, 4.5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    for k, s in enumerate(stabs):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        ttl = title if (single_input and title) else f"{s.get('session', '?')}"
        im = _draw(ax, s["cosine"], s["centers"], ttl)
        # Per-panel colorbar (snug next to its own heatmap).
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="cosine similarity")
    # Hide unused axes
    for k in range(n_sess, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].set_visible(False)
    if title and not single_input:
        fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show()
        return None
    return fig


def plot_action_decoding_over_time(
    df: pd.DataFrame,
    *,
    metric: Literal["test_bal_acc", "test_acc", "test_auc"] = "test_bal_acc",
    axes: Optional[Sequence[str]] = None,
    aggregate: Literal["sessions", "mean_sem"] = "mean_sem",
    show_n: bool = True,
    figsize: Tuple[float, float] = (11, 5),
    chance: float = 0.5,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot time-resolved decoder performance for the 4 action axes.

    Parameters
    ----------
    df
        Output of :func:`decode_action_axes_over_time`.
    metric
        Which column to plot vs time.
    axes
        Subset of axes to draw (default: all axes present in ``df``).
    aggregate
        ``"mean_sem"`` plots mean ± SEM across sessions per axis.
        ``"sessions"`` plots one thin line per session and overlays the mean.
    show_n
        If True, plots ``min(n_per_cell)`` across sessions on a second
        y-axis so you can see when balancing starts to drop sample size.
    chance
        Horizontal reference line.
    show
        If True, calls ``plt.show()`` and returns ``None`` (avoids the
        Jupyter display hook re-rendering the returned ``Figure``). If
        False, returns the ``Figure`` for further customization.
    """
    if axes is None:
        present = list(df["axis"].unique())
        # Preserve the canonical ACTION_AXES ordering when applicable, then
        # append any other axes (e.g. "prev_reward") found in df.
        axes = [a for a in ACTION_AXES.keys() if a in present]
        axes += [a for a in present if a not in axes]
    n_axes = len(axes)

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_axes, 4)))
    fig, ax = plt.subplots(figsize=figsize)

    for i, axis_name in enumerate(axes):
        sub = df[df["axis"] == axis_name].copy()
        if sub.empty:
            continue
        color = colors[i]

        if aggregate == "sessions":
            for sess, sdf in sub.groupby("session"):
                sdf = sdf.sort_values("center")
                ax.plot(
                    sdf["center"], sdf[metric],
                    "-", color=color, alpha=0.25, lw=1,
                )

        agg = sub.groupby("center")[metric].agg(["mean", "std", "count"]).reset_index()
        agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
        ax.plot(agg["center"], agg["mean"], "-o", color=color, lw=2, ms=4,
                label=axis_name)
        ax.fill_between(
            agg["center"], agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
            color=color, alpha=0.2,
        )

    ax.axhline(chance, color="red", linestyle=":", lw=1, label=f"chance ({chance:g})")
    ax.axvline(0, color="gray", linestyle="--", lw=1, alpha=0.5)
    ax.set_xlabel("Time (s, aligned to PSTH event)")
    ax.set_ylabel({"test_bal_acc": "Test balanced accuracy",
                   "test_acc": "Test accuracy",
                   "test_auc": "Test AUC"}.get(metric, metric))
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Time-resolved CD decoder — 4 action/transition axes")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best", ncol=2)

    if show_n:
        ax2 = ax.twinx()
        nser = df.groupby("center")["n_per_cell"].agg(lambda v: int(np.nanmin(v)) if len(v) else 0)
        ax2.plot(nser.index, nser.values, "-", color="gray", alpha=0.4, lw=1)
        ax2.set_ylabel("min n_per_cell across sessions", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

    fig.tight_layout()
    if show:
        plt.show()
        return None
    return fig
