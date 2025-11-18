from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Optional, Union

import numpy as np
import pandas as pd
import zarr

from nwb_utils import NWBUtils
from behavior_utils import find_trials
from create_psth import load_psth_raster_subset

from typing import Any, Dict, Optional, Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import zarr

def _find_psth_zarr_for_session(
    session_id: str,
    psth_root: Union[str, Path] = "/root/capsule/scratch/psth",
    bin_size_label: str = "0.2s",
) -> Path:
    """
    Locate the PSTH Zarr folder for a given ephys session.

    It searches for files like:
        ecephys_{session_id}_*_{bin_size_label}.zarr

    Parameters
    ----------
    session_id : str
        Session identifier used in your pipeline, e.g. '764769_2024-12-11_18-21-49'.
    psth_root : str | Path, default "/root/capsule/scratch/psth"
        Root directory where precomputed PSTH Zarr folders are stored.
    bin_size_label : str, default "0.2s"
        Suffix that encodes the PSTH bin size in the file name.

    Returns
    -------
    Path
        Path to the matched PSTH Zarr folder.

    Raises
    ------
    FileNotFoundError
        If no matching PSTH Zarr folder is found.
    RuntimeError
        If multiple matching PSTH Zarr folders are found.
    """
    psth_root = Path(psth_root)
    pattern = f"ecephys_{session_id}_*_{bin_size_label}.zarr"
    matches = sorted(psth_root.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No PSTH Zarr found for session {session_id!r} with pattern {pattern!r}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple PSTH Zarr folders found for session {session_id!r}: {matches}. "
            f"Please refine the naming pattern or choose one explicitly."
        )
    return matches[0]


def compute_average_psth_matrix(
    unit_specs: Sequence[Dict[str, Any]],
    trial_types: Sequence[str],
    *,
    psth_root: Union[str, Path] = "/root/capsule/scratch/psth",
    align_to_event: Optional[str] = None,
    time_window: Optional[Tuple[float, float]] = None,
    baseline_window: Tuple[float, float] = (-0.5, 0.0),
    bin_size_label: str = "0.2s",
    consolidated: bool = True,
    save_zarr_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Compute average PSTH and z-scored PSTH in one matrix for multiple units and trial types.

    This function:
      - Groups units by session to avoid repeated loading of NWB and PSTH files.
      - For each session, loads a PSTH subset containing:
          (all requested units) × (all trials from the requested trial types).
      - For each (unit, trial_type), averages PSTH across trials.
      - Computes a z-scored PSTH based on a baseline time window.
      - Returns a single set of matrices and metadata.
      - Optionally saves the result into a Zarr group.

    Parameters
    ----------
    unit_specs : sequence of dict
        Each dict must contain:
            {"session_id": <str>, "unit_index": <int>}

        Example:
            [
                {"session_id": "764769_2024-12-11_18-21-49", "unit_index": 852},
                {"session_id": "764769_2024-12-11_18-21-49", "unit_index": 857},
                {"session_id": "764769_2024-12-11_18-21-49", "unit_index": 877},
            ]

        Order is preserved, and duplicate (session_id, unit_index) pairs are removed.

    trial_types : sequence of str
        Trial type names as understood by `find_trials`, e.g. ["rewarded", "unrewarded"].

    psth_root : str | Path, default "/root/capsule/scratch/psth"
        Root directory where precomputed PSTH Zarr folders are stored.

    align_to_event : str, optional
        Event name used when PSTHs were extracted (e.g. "go_cue").
        Must match the value used inside `extract_neuron_psth_to_zarr`.

    time_window : (float, float), optional
        Time window (in seconds) to slice the PSTH around the alignment event.
        If None, the full time axis from the PSTH Zarr is used.

    baseline_window : (float, float), default (-0.5, 0.0)
        Time window (in seconds, relative to the alignment event) used to compute
        baseline mean and standard deviation for z-scoring. The z-score is:
            z(t) = (mean_rate(t) - baseline_mean) / baseline_std
        where the baseline is taken across all trials and all time bins within
        the baseline_window.

    bin_size_label : str, default "0.2s"
        Suffix that encodes the PSTH bin size in the Zarr folder names and is
        used by `_find_psth_zarr_for_session`.

    consolidated : bool, default True
        Passed through to `load_psth_raster_subset` to control xarray's indexing
        behavior when reading from Zarr. Using True is usually faster.

    save_zarr_path : str | Path, optional
        If provided, the result will be saved to a Zarr group at this path.
        The group will contain datasets:
          - "time"          : 1D float array, length T
          - "trial_types"   : 1D unicode array, length K
          - "unit_session"  : 1D unicode array, length N
          - "unit_index"    : 1D int array, length N
          - "mean_rate"     : 3D float array, shape (N, K, T)
          - "zscore"        : 3D float array, shape (N, K, T)
        and group attributes with basic metadata.

    Returns
    -------
    result : dict
        Dictionary with the following keys:

        - "time" : numpy.ndarray, shape (T,)
            Shared time axis used for all PSTHs.

        - "trial_types" : list of str, length K
            Copy of the input trial_types in the resolved order.

        - "unit_table" : pandas.DataFrame with columns
            ["row_index", "session_id", "unit_index"],
            where row_index corresponds to the first dimension of the matrices.

        - "mean_rate" : numpy.ndarray, shape (N, K, T)
            Average PSTH for each unit (N) and trial type (K).

        - "zscore" : numpy.ndarray, shape (N, K, T)
            Z-scored PSTH for each unit and trial type, using the specified baseline_window.

    Raises
    ------
    FileNotFoundError
        If no matching PSTH Zarr folder is found for a requested session.
    RuntimeError
        If no PSTH data is found for any of the requested (unit, trial_type) pairs,
        or if multiple PSTH Zarr folders exist for a given session.
    ValueError
        If the time axes differ across sessions, or if the baseline_window does not
        overlap with the PSTH time axis.
    """
    # ------------------------------------------------------------------
    # 1) Deduplicate unit specs while preserving order
    # ------------------------------------------------------------------
    seen_keys: set[Tuple[str, int]] = set()
    unit_keys: list[Tuple[str, int]] = []
    for spec in unit_specs:
        sid = spec["session_id"]
        uid = int(spec["unit_index"])
        key = (sid, uid)
        if key not in seen_keys:
            seen_keys.add(key)
            unit_keys.append(key)

    trial_types = list(trial_types)

    # Group units by session so we only load once per session
    units_by_session: Dict[str, list[int]] = {}
    for sid, uid in unit_keys:
        units_by_session.setdefault(sid, []).append(uid)

    # Containers for mean PSTH and z-score per (session, unit, trial_type)
    mean_map: Dict[Tuple[str, int, str], np.ndarray] = {}
    zscore_map: Dict[Tuple[str, int, str], np.ndarray] = {}

    common_time: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 2) Loop over sessions and load PSTHs once per session
    # ------------------------------------------------------------------
    for session_id, unit_indices in units_by_session.items():
        # 2.1) Find the PSTH Zarr for this session
        psth_path: Path = _find_psth_zarr_for_session(
            session_id=session_id,
            psth_root=psth_root,
            bin_size_label=bin_size_label,
        )

        # 2.2) Load NWB once per session
        nwb_data= NWBUtils.read_behavior_nwb(session_name=session_id)

        # 2.3) Resolve trial ids for each trial type
        trial_ids_by_type: Dict[str, np.ndarray] = {}
        for tt in trial_types:
            trial_ids_by_type[tt] = np.asarray(find_trials(nwb_data, tt), dtype=int)

        # Union of all requested trial ids in this session
        non_empty_trial_lists = [v for v in trial_ids_by_type.values() if v.size > 0]
        if len(non_empty_trial_lists) == 0:
            # No trials of the requested types in this session
            continue
        all_trial_ids = np.unique(np.concatenate(non_empty_trial_lists))

        # 2.4) Load PSTH subset: (units x trials x time)
        psth_da, _ = load_psth_raster_subset(
            psth_path,
            trial_ids=all_trial_ids,
            unit_ids=unit_indices,
            align_to_event=align_to_event,
            time_window=time_window,
            consolidated=consolidated,
        )

        # Identify trial and unit dims/coords
        trial_dim = next(d for d in psth_da.dims if d.startswith("trial_"))
        trial_coord_name = next(
            c for c in psth_da.coords if c.startswith("trial_index_")
        )
        unit_coord_name = "unit_index"

        trial_ids_in_ds = psth_da.coords[trial_coord_name].values.astype(int)
        unit_ids_in_ds = psth_da.coords[unit_coord_name].values.astype(int)
        times = psth_da.coords["time"].values

        # 2.5) Establish or check the shared time axis
        if common_time is None:
            common_time = times.copy()
        else:
            if len(times) != len(common_time) or not np.allclose(times, common_time):
                raise ValueError(
                    "Time axis differs between sessions. "
                    "This function assumes a shared time axis across sessions."
                )

        # 2.6) Baseline mask for z-scoring
        baseline_mask = (times >= baseline_window[0]) & (times < baseline_window[1])
        if not baseline_mask.any():
            raise ValueError(
                f"Baseline window {baseline_window} does not overlap with "
                f"the PSTH time axis [{times[0]:.3f}, {times[-1]:.3f}]."
            )

        # 2.7) Loop over units and trial types
        for unit_index in unit_indices:
            # Find the index of this unit in the PSTH DataArray
            where_unit = np.where(unit_ids_in_ds == int(unit_index))[0]
            if where_unit.size == 0:
                # This unit is not present in the PSTH dataset (skip)
                continue
            u_pos = int(where_unit[0])

            # Slice to get all trials for this unit → (trial_dim, time)
            unit_psth_all = psth_da.isel(unit=u_pos)

            for tt in trial_types:
                tt_trial_ids = trial_ids_by_type[tt]
                if tt_trial_ids.size == 0:
                    # No trials of this type in this session
                    continue

                # Map requested trial ids to positions in the PSTH array
                mask = np.isin(trial_ids_in_ds, tt_trial_ids)
                if not mask.any():
                    # No overlap between PSTH trials and this trial type
                    continue

                unit_psth = unit_psth_all.isel({trial_dim: mask})
                data = unit_psth.values  # shape: (n_trials, n_time)

                if data.ndim == 1:
                    # Single trial edge case → make it 2D
                    data = data[np.newaxis, :]

                # Mean PSTH across trials
                mean_rate = np.nanmean(data, axis=0)  # (time,)

                # Baseline statistics for z-score
                baseline_vals = data[:, baseline_mask].reshape(-1)
                baseline_mean = np.nanmean(baseline_vals)
                baseline_std = np.nanstd(baseline_vals, ddof=1)

                if baseline_std <= 0 or not np.isfinite(baseline_std):
                    # Degenerate baseline (no variance or non-finite)
                    zscore = np.zeros_like(mean_rate)
                else:
                    zscore = (mean_rate - baseline_mean) / baseline_std

                key = (session_id, int(unit_index), tt)
                mean_map[key] = mean_rate
                zscore_map[key] = zscore

    # ------------------------------------------------------------------
    # 3) Assemble final matrices
    # ------------------------------------------------------------------
    if common_time is None:
        raise RuntimeError(
            "No PSTH data found for the requested units and trial types."
        )

    n_units = len(unit_keys)
    n_types = len(trial_types)
    n_time = len(common_time)

    # Initialize with NaN (or zeros if you prefer) so missing combinations are explicit
    mean_matrix = np.full((n_units, n_types, n_time), np.nan, dtype=float)
    zscore_matrix = np.full((n_units, n_types, n_time), np.nan, dtype=float)

    # Maps from (session, unit) and trial type to matrix indices
    unit_to_row: Dict[Tuple[str, int], int] = {
        uk: i for i, uk in enumerate(unit_keys)
    }
    type_to_col: Dict[str, int] = {tt: j for j, tt in enumerate(trial_types)}

    for (sid, uid), row in unit_to_row.items():
        for tt, col in type_to_col.items():
            key = (sid, uid, tt)
            if key not in mean_map:
                # No data for this (unit, trial_type) combination
                continue
            mean_matrix[row, col, :] = mean_map[key]
            zscore_matrix[row, col, :] = zscore_map[key]

    # Build a small table describing how rows map back to (session, unit)
    unit_table = pd.DataFrame(
        {
            "row_index": np.arange(n_units, dtype=int),
            "session_id": [sid for (sid, uid) in unit_keys],
            "unit_index": [uid for (sid, uid) in unit_keys],
        }
    )

    result: Dict[str, Any] = {
        "time": common_time,
        "trial_types": trial_types,
        "unit_table": unit_table,
        "mean_rate": mean_matrix,
        "zscore": zscore_matrix,
    }

    # ------------------------------------------------------------------
    # 4) Optionally save to Zarr
    # ------------------------------------------------------------------
    if save_zarr_path is not None:
        save_zarr_path = Path(save_zarr_path)

        # Create a fresh Zarr group (overwrite if exists)
        root = zarr.open_group(str(save_zarr_path), mode="w")

        # 4.1) Core arrays
        root.create_dataset("time", data=common_time)
        root.create_dataset(
            "trial_types",
            data=np.asarray(trial_types, dtype="U"),
        )
        root.create_dataset(
            "unit_session",
            data=np.asarray([sid for (sid, uid) in unit_keys], dtype="U"),
        )
        root.create_dataset(
            "unit_index",
            data=np.asarray([uid for (sid, uid) in unit_keys], dtype=int),
        )
        root.create_dataset("mean_rate", data=mean_matrix)
        root.create_dataset("zscore", data=zscore_matrix)

        # 4.2) Metadata in group attributes
        root.attrs.update(
            {
                "n_units": n_units,
                "n_trial_types": n_types,
                "n_time": n_time,
                "baseline_window": tuple(baseline_window),
                "align_to_event": align_to_event,
                "bin_size_label": bin_size_label,
            }
        )

        print(f"Saved PSTH summary to Zarr: {save_zarr_path}")

    return result







def load_psth_summary_zarr(
    zarr_path: str | Path,
) -> dict:
    """
    Load a PSTH summary saved previously with compute_average_psth_matrix(..., save_zarr_path=...).

    Parameters
    ----------
    zarr_path : str | Path
        Path to the Zarr group containing:
            time, trial_types, unit_session, unit_index,
            mean_rate, zscore, attrs metadata.

    Returns
    -------
    res : dict
        A dictionary with the identical structure as the original `res`:
            {
                "time": 1D array,
                "trial_types": list[str],
                "unit_table": DataFrame,
                "mean_rate": 3D array,
                "zscore": 3D array,
                "attrs": dict of metadata
            }
    """

    zarr_path = Path(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="r")

    # Load basic arrays
    time = root["time"][:]
    trial_types = root["trial_types"][:].astype(str).tolist()
    unit_session = root["unit_session"][:].astype(str)
    unit_index = root["unit_index"][:].astype(int)

    # Load matrices
    mean_rate = root["mean_rate"][:]    # (N, K, T)
    zscore = root["zscore"][:]          # (N, K, T)

    # Metadata from attrs
    attrs = dict(root.attrs)

    # Reconstruct unit_table
    unit_table = pd.DataFrame({
        "row_index": np.arange(len(unit_index), dtype=int),
        "session_id": unit_session,
        "unit_index": unit_index,
    })

    res = {
        "time": time,
        "trial_types": trial_types,
        "unit_table": unit_table,
        "mean_rate": mean_rate,
        "zscore": zscore,
        "attrs": attrs,
    }

    return res





def plot_psth_heatmaps_per_trial_type(
    res: Dict[str, Any],
    *,
    use_zscore: bool = True,
    trial_types: Optional[Sequence[str]] = None,
    cluster_mode: Literal["reference", "separate"] = "reference",
    cluster_trial_type: Optional[str] = None,
    cluster_method: str = "average",
    metric: str = "correlation",
    figsize: Tuple[float, float] = (10.0, 6.0),
    cmap: str = "viridis",
    color_range: Optional[Tuple[float, float]] = None,
    show_colorbar: bool = True,
    show_unit_labels: bool = True,
    show_difference: bool = False,
    difference_color_range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot hierarchically clustered PSTH heatmaps separately for each trial type.

    One subplot is created per trial type. Clustering is always along the
    unit axis (rows).

    Clustering behavior
    -------------------
    cluster_mode = "reference"
        Cluster units using one reference trial type (specified by
        cluster_trial_type). The resulting unit order is applied to all
        trial types so that rows are directly comparable across panels.

    cluster_mode = "separate"
        Cluster each trial type independently. Each panel has its own
        row order; rows are not aligned across trial types.

    Difference heatmap
    ------------------
    If `show_difference=True` and exactly two trial_types are selected,
    an additional panel is plotted showing:

        difference = PSTH(trial_types[0]) - PSTH(trial_types[1])

    This panel uses the same unit order as the *first* trial type
    (either the reference order in "reference" mode, or the clustered
    order for that type in "separate" mode).

    Parameters
    ----------
    res : dict
        Result dictionary from `compute_average_psth_matrix`. It must
        contain:
          - "time"        : 1D array, shape (T,)
          - "trial_types" : list[str], length K
          - "unit_table"  : DataFrame with columns
                             ["row_index", "session_id", "unit_index"]
          - "mean_rate"   : 3D array, shape (N, K, T)
          - "zscore"      : 3D array, shape (N, K, T)

    use_zscore : bool, default True
        If True, use z-scored PSTH (res["zscore"]).
        If False, use mean-rate PSTH (res["mean_rate"]).

    trial_types : sequence of str, optional
        Which trial types to include. If None, use all trial types in
        `res["trial_types"]`.

    cluster_mode : {"reference", "separate"}, default "reference"
        - "reference": cluster using `cluster_trial_type` once, apply
          that unit order to all panels.
        - "separate": cluster each trial type independently.

    cluster_trial_type : str, optional
        Trial type to use as the reference for clustering when
        cluster_mode == "reference". If None, the first trial type in
        the selected list is used.

    cluster_method : str, default "average"
        Linkage method for hierarchical clustering, passed to
        `scipy.cluster.hierarchy.linkage`.

    metric : str, default "correlation"
        Distance metric passed to `scipy.spatial.distance.pdist`.

    figsize : (float, float), default (10.0, 6.0)
        Overall figure size in inches.

    cmap : str, default "viridis"
        Colormap for the heatmaps.

    color_range : (float, float), optional
        (vmin, vmax) for the main heatmaps (all trial types). If None,
        these limits are computed from the data across all selected
        trial types so the scale is shared.

    show_colorbar : bool, default True
        Whether to draw a shared colorbar for the main heatmaps.

    show_unit_labels : bool, default True
        If True, label the y-axis with unit numbers 1 and N
        (the total number of units). If False, hide y-ticks entirely.

    show_difference : bool, default False
        If True and exactly two trial_types are selected, add an extra
        panel showing the difference:
            trial_types[0] - trial_types[1]
        using the same row order as the first trial type.

    difference_color_range : (float, float), optional
        (vmin, vmax) for the difference heatmap. If None, it is computed
        from the difference data only. This is separate from `color_range`.
    """
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
    except ImportError as e:
        raise ImportError(
            "SciPy is required for hierarchical clustering. "
            "Please install scipy (e.g. `pip install scipy`)."
        ) from e

    time = res["time"]
    all_trial_types = list(res["trial_types"])
    unit_table = res["unit_table"]
    mean_rate = res["mean_rate"]  # (N, K, T)
    zscore = res["zscore"]        # (N, K, T)

    data_3d = zscore if use_zscore else mean_rate
    n_units, n_types, n_time = data_3d.shape

    # --------------------------------------------------------------
    # 1) Decide which trial types to plot
    # --------------------------------------------------------------
    if trial_types is None:
        selected_trial_types = all_trial_types
    else:
        missing = [tt for tt in trial_types if tt not in all_trial_types]
        if missing:
            raise ValueError(
                f"Requested trial_types {missing} not found in res['trial_types']: "
                f"{all_trial_types}"
            )
        selected_trial_types = list(trial_types)

    type_indices = [all_trial_types.index(tt) for tt in selected_trial_types]
    if len(type_indices) == 0:
        raise ValueError("No trial types selected for plotting.")

    # --------------------------------------------------------------
    # 2) Helper: cluster rows for a given data matrix
    # --------------------------------------------------------------
    def cluster_rows_from_matrix(matrix_2d: np.ndarray) -> np.ndarray:
        """Cluster rows of a 2D matrix (units x features) and return row order."""
        valid_rows = ~np.all(~np.isfinite(matrix_2d), axis=1)
        if not np.any(valid_rows):
            raise RuntimeError(
                "All units have NaN data only for the selected trial type; "
                "cannot cluster."
            )

        clustering_data = np.nan_to_num(matrix_2d[valid_rows], nan=0.0)
        dist = pdist(clustering_data, metric=metric)
        link = linkage(dist, method=cluster_method)
        ordered_valid = leaves_list(link)

        valid_idx = np.where(valid_rows)[0]
        row_order_local = valid_idx[ordered_valid]

        if np.any(~valid_rows):
            invalid_idx = np.where(~valid_rows)[0]
            row_order_local = np.concatenate([row_order_local, invalid_idx])

        return row_order_local

    # --------------------------------------------------------------
    # 3) Compute row order(s) depending on cluster_mode
    # --------------------------------------------------------------
    if cluster_mode == "reference":
        # Reference trial type for clustering
        if cluster_trial_type is None:
            ref_tt = selected_trial_types[0]
        else:
            if cluster_trial_type not in selected_trial_types:
                raise ValueError(
                    f"cluster_trial_type {cluster_trial_type!r} is not in "
                    f"selected trial_types {selected_trial_types}."
                )
            ref_tt = cluster_trial_type

        ref_idx = all_trial_types.index(ref_tt)
        ref_data = data_3d[:, ref_idx, :]  # (N, T)
        row_order = cluster_rows_from_matrix(ref_data)
        row_orders = {tt: row_order for tt in selected_trial_types}
    elif cluster_mode == "separate":
        row_orders = {}
        for tt, ti in zip(selected_trial_types, type_indices):
            this_data = data_3d[:, ti, :]
            row_orders[tt] = cluster_rows_from_matrix(this_data)
    else:
        raise ValueError(f"Unknown cluster_mode: {cluster_mode!r}")

    # --------------------------------------------------------------
    # 4) Determine color limits for main heatmaps
    # --------------------------------------------------------------
    if color_range is not None:
        vmin, vmax = color_range
    else:
        all_vals = []
        for tt, ti in zip(selected_trial_types, type_indices):
            ro = row_orders[tt]
            all_vals.append(data_3d[ro, ti, :])
        all_vals = np.stack(all_vals, axis=1)  # (N, K_sel, T)
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))

    # --------------------------------------------------------------
    # 5) Prepare difference data (if requested)
    # --------------------------------------------------------------
    have_difference = show_difference and (len(selected_trial_types) == 2)
    diff_data_2d = None
    vmin_diff = vmax_diff = None

    if show_difference and len(selected_trial_types) != 2:
        print(
            "Warning: show_difference=True but the number of selected trial_types "
            f"is {len(selected_trial_types)} (must be exactly 2). "
            "Difference panel will not be plotted."
        )
        have_difference = False

    if have_difference:
        tt1, tt2 = selected_trial_types
        ti1 = all_trial_types.index(tt1)
        ti2 = all_trial_types.index(tt2)

        # Use the row order of the first trial type for the difference panel
        ro_diff = row_orders[tt1]
        diff_data_2d = data_3d[ro_diff, ti1, :] - data_3d[ro_diff, ti2, :]

        if difference_color_range is not None:
            vmin_diff, vmax_diff = difference_color_range
        else:
            vmin_diff = float(np.nanmin(diff_data_2d))
            vmax_diff = float(np.nanmax(diff_data_2d))

    # --------------------------------------------------------------
    # 6) Create figure and axes
    # --------------------------------------------------------------
    n_main = len(selected_trial_types)
    n_panels = n_main + (1 if have_difference else 0)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=figsize,
        sharey=True,
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    main_axes = axes[:n_main]
    diff_ax = axes[-1] if have_difference else None

    images_main = []

    # --------------------------------------------------------------
    # 7) Plot main heatmaps (one per trial type)
    # --------------------------------------------------------------
    for ax, tt, ti in zip(main_axes, selected_trial_types, type_indices):
        ro = row_orders[tt]
        data_2d = data_3d[ro, ti, :]  # (N, T)

        im = ax.imshow(
            data_2d,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[time[0], time[-1], 0, data_2d.shape[0]],
        )
        images_main.append(im)

        ax.set_title(f"{tt} ({'z-score' if use_zscore else 'mean rate'})")
        ax.set_xlabel("Time (s)")

        # Y-axis: only show unit numbers 1 and N on the first panel
        if ax is main_axes[0]:
            if show_unit_labels:
                N_units = data_2d.shape[0]
                ax.set_ylabel("Units (clustered)")
                ax.set_yticks([0.5, N_units - 0.5])
                ax.set_yticklabels(["1", str(N_units)], fontsize=9)
            else:
                ax.set_yticks([])
        else:
            ax.set_yticks([])

    # --------------------------------------------------------------
    # 8) Plot difference heatmap (if requested)
    # --------------------------------------------------------------
    if have_difference and diff_ax is not None and diff_data_2d is not None:
        im_diff = diff_ax.imshow(
            diff_data_2d,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin_diff,
            vmax=vmax_diff,
            extent=[time[0], time[-1], 0, diff_data_2d.shape[0]],
        )
        tt1, tt2 = selected_trial_types
        diff_ax.set_title(f"{tt1} − {tt2} ({'z-score' if use_zscore else 'mean rate'})")
        diff_ax.set_xlabel("Time (s)")

        if show_unit_labels:
            N_units = diff_data_2d.shape[0]
            diff_ax.set_yticks([0.5, N_units - 0.5])
            diff_ax.set_yticklabels(["1", str(N_units)], fontsize=9)
        else:
            diff_ax.set_yticks([])

        if show_colorbar:
            cbar_diff = fig.colorbar(
                im_diff,
                ax=diff_ax,
                location="right",
                fraction=0.06,
                pad=0.03,
            )
            cbar_diff.set_label(
                f"Difference ({'z-score' if use_zscore else 'spk/s'})"
            )

    # --------------------------------------------------------------
    # 9) Shared colorbar for main panels
    # --------------------------------------------------------------
    if show_colorbar and len(images_main) > 0:
        cbar = fig.colorbar(
            images_main[0],
            ax=main_axes,
            location="right",
            fraction=0.06,
            pad=0.03,
        )
        cbar.set_label("Z-score" if use_zscore else "Firing rate (spk/s)")

    plt.show()
