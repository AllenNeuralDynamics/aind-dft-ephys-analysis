# -*- coding: utf-8 -*-
# Complete plotting utilities WITHOUT _to_1d_int
# - plot_psth_raster_for_units
# - plot_raster_and_quantile_psth_by_latent
# All comments are in English.

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Tuple, Optional, Union, Literal, List, Dict

import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd

# Project-specific loader (assumed available in your environment)
from create_psth import load_psth_raster_subset


def plot_psth_raster_for_units(
    source: Union[str, Path, xr.DataArray, xr.Dataset],
    unit_ids: Optional[Sequence[int]] = None,
    trial_ids: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
    trial_types: Optional[Sequence[str]] = None,
    nwb_data: Optional[Any] = None,
    align_to_event: Optional[str] = None,
    time_window: Optional[Tuple[float, float]] = None,
    plot_type: Literal["single", "mean"] = "single",
    colors: Optional[Sequence[str]] = None,
    sem_alpha: float = 0.3,
    figsize: Tuple[float, float] = (6.0, 4.0),
    sharey: bool = False,
    legend: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    consolidated: bool = True,
    group_labels: Optional[Sequence[str]] = None,
    y_mode: Literal["auto_per_unit", "auto_global", "none"] = "auto_per_unit",
    y_pad: float = 0.05,
    group_name: Optional[Sequence[str]] = None,
    show: bool = True,  # control whether to display figures
) -> None:
    """
    Plot both PSTH and raster for each specified unit in separate figures.

    Parameters
    ----------
    source : str | Path | xr.Dataset | xr.DataArray
        Path to a Zarr folder produced by `extract_neuron_psth_to_zarr`
        or an already loaded PSTH Dataset/DataArray.
    unit_ids : sequence of int, optional
        Indices of units (neurons) to plot. If None, all units are plotted.
    trial_ids : sequence or list of sequences of int, optional
        Specific trial index values to include. Flat list → one group; nested → multiple groups.
    trial_types : sequence of str, optional
        Behavioral trial type names used to select trials via `find_trials`
        if `trial_ids` is not provided.
    nwb_data : Any, optional
        NWB handle required if `trial_types` is used to look up trial indices.
    align_to_event : str, optional
        Event name (without the `psth_` prefix) if multiple PSTHs exist.
    time_window : (float, float), optional
        Time window slice along the PSTH time axis; also sets x-limits.
    plot_type : {'single','mean'}, default 'single'
        'single' → plot each trial separately; 'mean' → mean ± SEM per group.
    colors : sequence of str, optional
        Colors for each trial group. Defaults to Matplotlib’s color cycle.
    sem_alpha : float, default 0.3
        Transparency for SEM shading when `plot_type='mean'`.
    figsize : (float, float), default (6.0, 4.0)
        Figure size in inches (width, height) per unit.
    sharey : bool, default False
        Whether to share y-axis limits between PSTH and raster axes.
    legend : bool, default True
        Show legend for mean plots when multiple groups are present.
    save_path : str | Path, optional
        Path to save each unit’s figure (PNG). If a directory is given,
        saves as ``unit_<id>.png`` inside it. If None, figures are not saved.
    dpi : int, default 300
        Resolution of saved figures.
    consolidated : bool, default True
        Passed to the Zarr loader for faster metadata reading.
    group_labels : sequence of str, optional
        Custom labels for trial groups when ``trial_ids`` or ``trial_types``
        define multiple groups. Must match the number of groups.
    y_mode : {'auto_per_unit', 'auto_global', 'none'}, default 'auto_per_unit'
        Y-axis scaling mode for PSTH:
        - 'auto_per_unit': auto-scale individually for each unit.
        - 'auto_global': compute a global maximum across all units and use it.
        - 'none': leave Matplotlib’s default y-limits.
    y_pad : float, default 0.05
        Fractional padding added above the maximum firing rate when auto-scaling.
    group_name : sequence of str, optional
        Custom group names when using ``trial_ids``.
        - If ``trial_ids`` is nested (list of lists), length of ``group_name``
          must equal the number of groups.
        - If ``trial_ids`` is flat, length of ``group_name`` must be 1.
        Ignored when ``trial_ids`` is None or when ``trial_types`` is used.
    """
    # -----------------------------
    # 0) Helpers
    # -----------------------------
    def _to_1d_int(a) -> np.ndarray:
        """Convert input to a 1-D int array (handles (n,1), lists, scalars)."""
        if a is None:
            return np.array([], dtype=int)
        arr = np.asarray(a)
        if arr.ndim > 1:
            arr = arr.reshape(-1)   # ravel but ensures 1-D even for (n,1)
        return arr.astype(int)

    if trial_ids is not None and trial_types is not None:
        raise ValueError("Provide either trial_ids or trial_types, not both.")

    trial_groups: Optional[dict[str, np.ndarray]] = None

    # -----------------------------
    # A) Build trial groups dict
    # -----------------------------
    if trial_ids is not None:

        # Case 1: Nested groups → multiple groups provided
        if len(trial_ids) > 0 and isinstance(trial_ids[0], (list, tuple, np.ndarray)):
            group_list = [_to_1d_int(g) for g in trial_ids]

            if group_name is not None:
                if len(group_name) != len(group_list):
                    raise ValueError(
                        f"Length of group_name ({len(group_name)}) must match "
                        f"number of groups ({len(group_list)})."
                    )
                labels = list(group_name)
            elif group_labels is not None:
                if len(group_labels) != len(group_list):
                    raise ValueError(
                        f"Length of group_labels ({len(group_labels)}) must match "
                        f"number of groups ({len(group_list)})."
                    )
                labels = list(group_labels)
            else:
                labels = [f"group_{i}" for i in range(len(group_list))]

            trial_groups = dict(zip(labels, group_list))

        # Case 2: Flat → one group
        else:
            arr = _to_1d_int(trial_ids)

            if group_name is not None:
                if len(group_name) != 1:
                    raise ValueError(
                        "For flat trial_ids, group_name must contain exactly one name."
                    )
                label = group_name[0]
            elif group_labels is not None:
                if len(group_labels) != 1:
                    raise ValueError(
                        "For flat trial_ids, group_labels must contain exactly one name."
                    )
                label = group_labels[0]
            else:
                label = "all_trials"

            trial_groups = {label: arr}

    elif trial_types is not None:
        if nwb_data is None:
            raise ValueError("nwb_data is required when using trial_types.")
        group_list = [_to_1d_int(find_trials(nwb_data, tt)) for tt in trial_types]
        labels = group_labels or list(trial_types)
        if group_labels is not None and len(group_labels) != len(group_list):
            raise ValueError(
                f"Length of group_labels ({len(group_labels)}) must match "
                f"number of trial types ({len(group_list)})."
            )
        trial_groups = dict(zip(labels, group_list))

    # -----------------------------
    # B) Union of all requested trials for subsetting during load
    # -----------------------------
    if trial_groups is not None and len(trial_groups) > 0:
        safe_lists = [v for v in trial_groups.values() if v is not None and v.size > 0]
        union_trials = (
            np.unique(np.concatenate([_to_1d_int(v) for v in safe_lists]))
            if len(safe_lists) > 0
            else None
        )
    else:
        union_trials = None  # Load all trials

    # -----------------------------
    # 1) Load subset (using union for selection)
    # -----------------------------
    psth_da, raster_da = load_psth_raster_subset(
        source,
        trial_ids=union_trials,
        unit_ids=unit_ids,
        align_to_event=align_to_event,
        time_window=time_window,
        consolidated=consolidated,
    )

    # Identify dim/coord names
    trial_dim = next(d for d in psth_da.dims if d.startswith("trial_"))
    trial_coord = next(c for c in psth_da.coords if c.startswith("trial_index_"))
    times = psth_da.coords["time"].values
    unit_indices = psth_da.coords["unit_index"].values

    # If unit_ids was None, plot all units
    if unit_ids is None:
        unit_ids = list(unit_indices)

    # Intersect each group's ids with available ids to avoid KeyErrors; ensure 1-D int again
    available_trial_ids = _to_1d_int(psth_da.coords[trial_coord].values)
    if trial_groups is None:
        trial_groups = {"all_trials": available_trial_ids}
    else:
        for k in list(trial_groups.keys()):
            trial_groups[k] = np.intersect1d(
                _to_1d_int(trial_groups[k]),
                available_trial_ids,
                assume_unique=False,
            )

    # Drop empty groups (no matching trials)
    trial_groups = {k: v for k, v in trial_groups.items() if v.size > 0}
    if len(trial_groups) == 0:
        raise ValueError(
            "After loading, none of the requested trial groups have matching trials."
        )

    # Colors
    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = colors or cmap
    group_colors = {g: colors[i % len(colors)] for i, g in enumerate(trial_groups)}

    # Optional global autoscale
    global_max = 0.0
    if y_mode == "auto_global":
        for unit in unit_ids:
            da = psth_da.sel(unit_index=unit)
            data = da.mean(trial_dim).values if plot_type == "mean" else da.values
            if np.size(data) > 0 and np.isfinite(data).any():
                global_max = max(global_max, float(np.nanmax(data)))

    # -----------------------------
    # 3) Plot per unit
    # -----------------------------
    for unit in unit_ids:
        # Positional index for this unit
        try:
            pos = int(np.where(psth_da.coords["unit_index"].values == unit)[0][0])
        except IndexError:
            # Skip units not present in the dataset
            continue
        pos = int(where[0])

        unit_psth = psth_da.isel(unit=pos)
        unit_raster = raster_da.isel(unit=pos)

        fig, (ax_rast, ax_psth) = plt.subplots(
            2, 1, figsize=figsize, sharex=True, sharey=sharey
        )

        # Raster (stack trials; keep color by group)
        y_raster = 0
        for label, tids in trial_groups.items():
            tids_set = set(map(int, _to_1d_int(tids)))  # ensure hashable ints
            for ti, tval in enumerate(unit_raster.coords[trial_coord].values):
                if int(tval) in tids_set:
                    y_raster += 1
                    spikes = unit_raster.isel({trial_dim: ti}).values
                    spikes = spikes[~np.isnan(spikes)]
                    ax_rast.vlines(
                        spikes, y_raster, y_raster + 1,
                        color=group_colors[label],
                        alpha=0.7,
                    )

        ax_rast.axvline(0, color="k", ls="--", lw=0.8)
        ax_rast.set_ylabel("Trial")
        ax_rast.set_title(f"Unit {unit}")

        # PSTH (single-trial or mean ± SEM)
        unit_max = 0.0
        for label, tids in trial_groups.items():
            tids_set = set(map(int, _to_1d_int(tids)))
            trial_vals = unit_psth.coords[trial_coord].values
            idxs = [i for i, tv in enumerate(trial_vals) if int(tv) in tids_set]
            if len(idxs) == 0:
                continue

            data = unit_psth.isel({trial_dim: idxs}).values  # shape: (n_trials, n_time)
            if data.ndim == 1:
                data = data[np.newaxis, :]

            if plot_type == "single":
                for trial_vec in data:
                    ax_psth.plot(
                        times,
                        trial_vec,
                        color=group_colors[label],
                        alpha=0.6,
                    )
                    if np.isfinite(trial_vec).any():
                        unit_max = max(unit_max, float(np.nanmax(trial_vec)))
            else:
                mean_f = np.nanmean(data, axis=0)
                ntr = max(data.shape[0], 1)
                sem_f = (
                    np.nanstd(data, axis=0, ddof=1) / np.sqrt(ntr)
                    if ntr > 1
                    else np.zeros_like(mean_f)
                )
                ax_psth.plot(times, mean_f, color=group_colors[label], label=label)
                ax_psth.fill_between(
                    times, mean_f - sem_f, mean_f + sem_f, alpha=sem_alpha
                )
                if np.isfinite(mean_f + sem_f).any():
                    unit_max = max(unit_max, float(np.nanmax(mean_f + sem_f)))

        ax_psth.axvline(0, color="k", ls="--", lw=0.8)
        ax_psth.set_ylabel("Firing rate (spk/s)")
        ax_psth.set_xlabel("Time (s)")

        # X-limits
        if time_window is not None:
            ax_rast.set_xlim(*time_window)
            ax_psth.set_xlim(*time_window)

        # Y-limits
        if y_mode == "auto_per_unit" and unit_max > 0:
            ax_psth.set_ylim(0, unit_max * (1 + y_pad))
        elif y_mode == "auto_global" and global_max > 0:
            ax_psth.set_ylim(0, global_max * (1 + y_pad))

        if plot_type == "mean" and legend and len(trial_groups) > 1:
            ax_psth.legend(frameon=False)

        plt.tight_layout()

        # Save / Show / Close
        if save_path:
            save_target = Path(save_path)
            if save_target.suffix == "" or save_target.is_dir():
                save_target.mkdir(parents=True, exist_ok=True)
                fp = save_target / f"unit_{unit}.png"
            else:
                base = save_target.with_suffix("")
                fp = base.parent / f"{base.name}_unit_{unit}.png"
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
            print(f"Saved Unit {unit} figure to {fp}")

        if show:
            plt.show()
        else:
            plt.close(fig)


# ---------------------------------------------------------------------
# Plot Raster (sorted by latent) + Quantile PSTH summaries
# ---------------------------------------------------------------------
def plot_raster_and_quantile_psth_by_latent(
    source: Union[str, Path, xr.DataArray, xr.Dataset],
    *,
    latent_values: Sequence[float],
    latent_trial_ids: Sequence[int],
    unit_ids: Optional[Sequence[int]] = None,
    align_to_event: Optional[str] = None,
    time_window: Optional[Tuple[float, float]] = None,
    n_bins: int = 5,
    binning: Literal["quantile", "equal"] = "quantile",
    bin_range: Optional[Tuple[float, float]] = None,
    bin_label: Literal["mean", "center"] = "mean",
    quantile_stat: Literal["mean", "median"] = "mean",
    ci: Literal["sem", "iqr", "none"] = "sem",
    figsize: Tuple[float, float] = (6.0, 5.0),
    dpi: int = 300,
    consolidated: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    title_prefix: str = "",
    cmap_name: str = "viridis",
    raster_colormap: bool = True,
    show_colormap: bool = True,
    sort_order: Literal["ascending", "descending", "not_sort"] = "ascending",
    save_prefix: Optional[str] = None,
    latent_name: Optional[str] = None,  # colorbar label
    show: bool = True,                  # control figure display
    overwrite: bool = True,             # overwrite existing figure files
    min_trial_rate: float = 0,        # NEW: include only trials with mean rate > this
) -> None:
    """
    Plot per-unit rasters and binned PSTH summaries using a trial-wise latent value.

    Parameters
    ----------
    source : str | Path | xr.DataArray | xr.Dataset
        Path to a Zarr PSTH dataset or an already loaded xarray object
        containing spike raster and PSTH data.
    latent_values : sequence of float
        Latent variable values corresponding one-to-one with trials.
    latent_trial_ids : sequence of int
        Trial IDs corresponding to `latent_values`. Must match trial indices
        used in the PSTH dataset.
    unit_ids : sequence of int, optional
        Specific unit indices to plot. If None, all available units are plotted.
    align_to_event : str, optional
        Event name used when multiple alignment types exist in the dataset.
    time_window : (float, float), optional
        Time range (in seconds) for plotting. Applies to both raster and PSTH,
        and is also used for the trial inclusion filter when `min_trial_rate` is set.
    n_bins : int, default 5
        Number of bins (quantiles or equal-width bins) used to divide latent values.
    binning : {'quantile', 'equal'}, default 'quantile'
        - 'quantile': divides data into quantiles with equal number of trials.
        - 'equal': divides into equal-width ranges.
    bin_range : (float, float), optional
        Range of latent values for equal-width binning. If None, use observed range.
    bin_label : {'mean', 'center'}, default 'mean'
        Labels for colorbar ticks:
        - 'mean': mean latent value per bin.
        - 'center': midpoint of bin edges.
    quantile_stat : {'mean', 'median'}, default 'mean'
        Aggregation function for PSTH within each bin.
    ci : {'sem', 'iqr', 'none'}, default 'sem'
        Confidence interval style: 'sem' (mean±SEM), 'iqr' (25–75%), or 'none'.
    figsize : (float, float), default (6.0, 5.0)
        Figure size for each unit.
    dpi : int, default 300
        DPI for saved figures.
    consolidated : bool, default True
        Passed to the Zarr loader for faster metadata reading.
    save_path : str | Path, optional
        Directory or file-like path used to derive output folder.
        Figures will be saved into a subfolder named `save_prefix.rstrip('_')`
        under the base directory.
    title_prefix : str, optional
        Text prefix added to each figure title (e.g., session name).
    cmap_name : str, default 'viridis'
        Name of Matplotlib colormap for bins and colorbar.
    raster_colormap : bool, default True
        If True, color-code raster trials by bin color; if False, rasters are black.
    show_colormap : bool, default True
        Whether to display a colorbar showing latent value range.
    sort_order : {'ascending','descending'}, default 'ascending'
        Sort order of trials by latent value for raster display.
    save_prefix : str, optional
        Used for:
        1) subfolder name: `save_prefix.rstrip('_')`
        2) filename prefix: `f"{save_prefix or ''}unit_{unit}.png"`
    latent_name : str, optional
        Custom label for the colorbar. Defaults to 'Latent value' if not provided.
    show : bool, default True
        If True, display each figure. If False, suppress display and close after saving.
    overwrite : bool, default True
        If True, overwrite an existing figure file. If False and the target file exists,
        skip that unit (do not generate a figure).

    min_trial_rate : float, default 0.2
        Per-unit trial inclusion threshold (in spk/s).

        For each unit, a per-trial mean firing rate is computed from that unit's PSTH
        over the plotting time window:
        - If `time_window` is provided: mean over times within `[time_window[0], time_window[1]]`
        - If `time_window` is None: mean over the full PSTH time axis

        Only trials with mean rate strictly greater than `min_trial_rate` are included.
        This filter is applied per unit and affects BOTH:
        - the raster (only retained trials are drawn)
        - the PSTH summaries by bin (only retained trials contribute to curves/CI)

        Trials with non-finite mean rates (NaN/inf) are excluded. If no trials pass the
        filter for a unit, that unit is skipped.
    """

    # -----------------------------
    # 1) Validate inputs
    # -----------------------------
    latent_values = np.asarray(latent_values, dtype=np.float32)
    latent_trial_ids = np.asarray(latent_trial_ids, dtype=np.int64)

    if latent_values.shape[0] != latent_trial_ids.shape[0]:
        raise ValueError("`latent_values` and `latent_trial_ids` must have the same length.")
    if latent_trial_ids.size != np.unique(latent_trial_ids).size:
        raise ValueError("`latent_trial_ids` contains duplicates; must be one-to-one.")

    # -----------------------------
    # 2) Load PSTH & raster subset
    # -----------------------------
    psth_da, raster_da = load_psth_raster_subset(
        source,
        trial_ids=None,
        unit_ids=unit_ids,
        align_to_event=align_to_event,
        time_window=time_window,
        consolidated=consolidated,
    )

    trial_dim = next(d for d in psth_da.dims if d.startswith("trial_"))
    trial_coord = next(c for c in psth_da.coords if c.startswith("trial_index_"))

    all_trial_ids_in_ds = psth_da.coords[trial_coord].values.astype(np.int64)
    unit_indices = psth_da.coords["unit_index"].values.astype(np.int64)
    times = np.asarray(psth_da.coords["time"].values)

    # -----------------------------
    # 3) Align latent values with trials present in dataset
    # -----------------------------
    present_mask = np.isin(latent_trial_ids, all_trial_ids_in_ds)
    if not np.any(present_mask):
        raise ValueError("None of `latent_trial_ids` are present in the dataset.")

    latent_trial_ids = latent_trial_ids[present_mask]
    latent_values = latent_values[present_mask]

    pos_idx = pd.Index(all_trial_ids_in_ds).get_indexer(latent_trial_ids)
    keep = pos_idx >= 0

    psth_da = psth_da.isel({trial_dim: pos_idx[keep]})
    raster_da = raster_da.isel({trial_dim: pos_idx[keep]})
    lat = latent_values[keep]

    trial_ids_arr = psth_da.coords[trial_coord].values.astype(np.int64)

    # -----------------------------
    # 4) Build bins (global edges/colors)
    # -----------------------------
    lat_finite = lat[np.isfinite(lat)]
    if lat_finite.size == 0:
        raise ValueError("All latent values are NaN; cannot build bins.")

    if binning == "quantile":
        q_perc = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.nanpercentile(lat_finite, q_perc))
        if edges.size < 2:
            raise ValueError("Quantile edges collapsed (insufficient latent variability).")
    else:
        lo, hi = (
            (float(np.nanmin(lat_finite)), float(np.nanmax(lat_finite)))
            if bin_range is None
            else (float(bin_range[0]), float(bin_range[1]))
        )
        if hi <= lo:
            raise ValueError("Invalid bin_range or latent range.")
        edges = np.linspace(lo, hi, n_bins + 1)
        edges[-1] = np.nextafter(edges[-1], np.inf)

    n_bins_eff = int(edges.size - 1)
    if n_bins_eff <= 0:
        raise ValueError("Invalid bin edges; cannot form bins.")
    if n_bins_eff != n_bins:
        n_bins = n_bins_eff

    bin_idx = np.digitize(lat, edges[1:-1], right=False)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    cmap = cm.get_cmap(cmap_name, n_bins)
    colors = [mcolors.to_hex(cmap(i)) for i in range(n_bins)]

    latent_min = float(np.nanmin(lat_finite))
    latent_max = float(np.nanmax(lat_finite))

    unit_list = list(unit_indices) if unit_ids is None else list(unit_ids)

    # -----------------------------
    # 4.5) Prepare output directory ONCE
    # -----------------------------
    out_dir: Optional[Path] = None
    if save_path is not None:
        save_target = Path(save_path)
        if save_target.suffix == "" or save_target.is_dir():
            base_dir = save_target
        else:
            base_dir = save_target.parent

        if save_prefix:
            subfolder_name = save_prefix.rstrip("_")
            out_dir = base_dir / subfolder_name
        else:
            out_dir = base_dir

        out_dir.mkdir(parents=True, exist_ok=True)
    # -----------------------------
    # 5) Plot per unit
    # -----------------------------
    for unit in unit_list:
        fp: Optional[Path] = None

        # File naming + overwrite logic (fast per-unit)
        if out_dir is not None:
            filename = f"{save_prefix or ''}unit_{unit}.png"
            fp = out_dir / filename
            if fp.exists() and not overwrite:
                print(f"Skipping Unit {unit} (file exists, overwrite=False): {fp}")
                continue

        where = np.where(unit_indices == unit)[0]
        if where.size == 0:
            continue
        upos = int(where[0])

        fig = None
        unit_psth_np = None
        unit_rast_vals = None
        unit_psth = None
        unit_rast = None

        try:
            # Slice xarray once per unit
            unit_psth = psth_da.isel(unit=upos)
            unit_rast = raster_da.isel(unit=upos)

            # Materialize once
            unit_psth_np = np.asarray(unit_psth.values)  # (n_trials, n_time)
            unit_rast_vals = unit_rast.values            # typically object arrays per trial

            # -----------------------------
            # Per-unit trial filter by mean firing rate within time_window
            # -----------------------------
            if time_window is None:
                tmask = np.ones(times.shape, dtype=bool)
            else:
                t0, t1 = float(time_window[0]), float(time_window[1])
                if t1 < t0:
                    raise ValueError("time_window must be (tmin, tmax) with tmax >= tmin.")
                tmask = (times >= t0) & (times <= t1)

            if not np.any(tmask):
                raise ValueError("No time points fall within the specified time_window.")

            trial_mean_rate = np.nanmean(unit_psth_np[:, tmask], axis=1)
            keep_trials = np.isfinite(trial_mean_rate) & (trial_mean_rate > float(min_trial_rate))

            if not np.any(keep_trials):
                print(f"Skipping Unit {unit}: no trials with mean rate > {min_trial_rate} in the time window.")
                continue

            # Filter per-unit arrays
            lat_u = lat[keep_trials]
            trial_ids_u = trial_ids_arr[keep_trials]
            unit_psth_np_u = unit_psth_np[keep_trials, :]
            unit_rast_vals_u = unit_rast_vals[keep_trials]

            trial_to_pos_u = {int(t): i for i, t in enumerate(trial_ids_u)}

            # -----------------------------
            # Build bins PER UNIT (after filtering)
            # -----------------------------
            lat_finite_u = lat_u[np.isfinite(lat_u)]
            if lat_finite_u.size == 0:
                print(f"Skipping Unit {unit}: all filtered latent values are NaN/inf.")
                continue

            if binning == "quantile":
                q_perc = np.linspace(0, 100, n_bins + 1)
                edges_u = np.unique(np.nanpercentile(lat_finite_u, q_perc))
                if edges_u.size < 2:
                    print(f"Skipping Unit {unit}: quantile edges collapsed after filtering.")
                    continue
            else:
                lo_u, hi_u = (
                    (float(np.nanmin(lat_finite_u)), float(np.nanmax(lat_finite_u)))
                    if bin_range is None
                    else (float(bin_range[0]), float(bin_range[1]))
                )
                if hi_u <= lo_u:
                    print(f"Skipping Unit {unit}: invalid bin_range or latent range after filtering.")
                    continue
                edges_u = np.linspace(lo_u, hi_u, n_bins + 1)
                edges_u[-1] = np.nextafter(edges_u[-1], np.inf)

            n_bins_eff_u = int(edges_u.size - 1)
            if n_bins_eff_u <= 0:
                print(f"Skipping Unit {unit}: invalid bin edges after filtering.")
                continue

            # If quantiles collapsed, n_bins_eff_u may be < n_bins; respect that per unit
            n_bins_u = n_bins_eff_u

            bin_idx_u = np.digitize(lat_u, edges_u[1:-1], right=False)
            bin_idx_u = np.clip(bin_idx_u, 0, n_bins_u - 1)

            cmap_u = cm.get_cmap(cmap_name, n_bins_u)
            colors_u = [mcolors.to_hex(cmap_u(i)) for i in range(n_bins_u)]

            if bin_label == "center":
                bin_tick_vals_u = list(0.5 * (edges_u[:-1] + edges_u[1:]))
            else:
                bin_tick_vals_u = [
                    float(np.nanmean(lat_u[bin_idx_u == i])) if np.any(bin_idx_u == i) else np.nan
                    for i in range(n_bins_u)
                ]

            latent_min_u = float(np.nanmin(lat_finite_u))
            latent_max_u = float(np.nanmax(lat_finite_u))

            # -----------------------------
            # Sorting PER UNIT (after filtering)
            # -----------------------------
            if sort_order == "ascending":
                order_u = np.argsort(lat_u, kind="mergesort")
                order_str = "ascending"

            elif sort_order == "descending":
                order_u = np.argsort(lat_u, kind="mergesort")[::-1]
                order_str = "descending"

            elif sort_order == "not_sort":
                order_u = np.arange(len(lat_u))
                order_str = "original order"

            else:
                raise ValueError("sort_order must be 'ascending', 'descending', or 'not_sort'.")

            sorted_trials_u = trial_ids_u[order_u]
            sorted_bins_u = bin_idx_u[order_u]


            fig, (ax_rast, ax_psth) = plt.subplots(
                2, 1,
                figsize=figsize,
                sharex=True,
                gridspec_kw={"height_ratios": [1, 1.3]},
            )

            # -----------------------------
            # Raster (filtered trials only)
            # -----------------------------
            y = 0
            for idx, tval in enumerate(sorted_trials_u):
                y += 1
                b = int(sorted_bins_u[idx])
                ti = trial_to_pos_u.get(int(tval))
                if ti is None:
                    continue

                spikes = unit_rast_vals_u[ti]
                if spikes is None:
                    continue

                spikes = np.asarray(spikes)
                if spikes.size == 0:
                    continue

                if spikes.dtype.kind not in ("f", "i", "u"):
                    spikes = spikes.astype(np.float32, copy=False)

                spikes = spikes[np.isfinite(spikes)]
                if spikes.size == 0:
                    continue

                color = colors_u[b] if raster_colormap else "black"
                ax_rast.vlines(spikes, y, y + 0.9, color=color, alpha=0.8, linewidth=0.6)

            ax_rast.axvline(0.0, color="k", ls="--", lw=0.8)

            ttl = f"{title_prefix} Unit {unit}" if title_prefix else f"Unit {unit}"
            ax_rast.set_title(f"{ttl} (sorted {order_str}; rate>{min_trial_rate})")
            ax_rast.set_ylabel("Trials (sorted by latent)")

            # -----------------------------
            # PSTH summaries by bin (filtered trials only; per-unit bins)
            # -----------------------------
            ymax = 0.0
            for b in range(n_bins_u):
                sel = (bin_idx_u == b)
                if not np.any(sel):
                    continue

                data = unit_psth_np_u[sel, :]

                if quantile_stat == "median":
                    center = np.nanmedian(data, axis=0)
                else:
                    center = np.nanmean(data, axis=0)

                lower = upper = None
                if ci == "sem":
                    n = np.sum(np.isfinite(data), axis=0)
                    std = np.nanstd(data, axis=0, ddof=1)
                    sem = np.where(n > 0, std / np.sqrt(np.maximum(n, 1)), np.nan)
                    lower, upper = center - sem, center + sem
                elif ci == "iqr":
                    q25 = np.nanpercentile(data, 25, axis=0)
                    q75 = np.nanpercentile(data, 75, axis=0)
                    lower, upper = q25, q75

                ax_psth.plot(times, center, color=colors_u[b], linewidth=1.5)

                if lower is not None and upper is not None:
                    ax_psth.fill_between(times, lower, upper, color=colors_u[b], alpha=0.25)

                fmax = np.nanmax(upper if upper is not None else center)
                if np.isfinite(fmax):
                    ymax = max(ymax, float(fmax))

            ax_psth.axvline(0.0, color="k", ls="--", lw=0.8)
            ax_psth.set_ylabel("Firing rate (spk/s)")
            ax_psth.set_xlabel("Time (s)")

            if time_window is not None:
                ax_rast.set_xlim(*time_window)
                ax_psth.set_xlim(*time_window)

            if ymax > 0:
                ax_psth.set_ylim(0, ymax * 1.05)

            plt.tight_layout()

            # -----------------------------
            # Colorbar (per-unit scaling/ticks)
            # -----------------------------
            if show_colormap:
                sm = plt.cm.ScalarMappable(
                    cmap=cm.get_cmap(cmap_name),
                    norm=plt.Normalize(vmin=latent_min_u, vmax=latent_max_u),
                )
                cbar = fig.colorbar(sm, ax=[ax_rast, ax_psth], orientation="vertical", pad=0.02)
                cbar.set_label(latent_name or "Latent value", rotation=270, labelpad=12)

                ticks = [v for v in bin_tick_vals_u if np.isfinite(v)]
                if len(ticks) > 0:
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels([f"{v:.2f}" for v in ticks])
                cbar.ax.tick_params(size=0)

            # -----------------------------
            # Save / Show
            # -----------------------------
            if fp is not None:
                fig.savefig(fp, dpi=dpi, bbox_inches="tight")
                print(f"Saved Unit {unit} figure to {fp}")

            if show:
                plt.show()

        finally:
            if fig is not None:
                plt.close(fig)

            if unit_psth_np is not None:
                del unit_psth_np
            if unit_rast_vals is not None:
                del unit_rast_vals
            if unit_psth is not None:
                del unit_psth
            if unit_rast is not None:
                del unit_rast







def plot_trial_mean_activity_vs_latent_per_unit(
    source: Union[str, Path, xr.DataArray, xr.Dataset],
    *,
    latent_values: Sequence[float],
    latent_trial_ids: Sequence[int],
    activity_window: Tuple[float, float],
    unit_ids: Optional[Sequence[int]] = None,
    align_to_event: Optional[str] = None,
    time_window: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (5.5, 4.5),
    dpi: int = 300,
    consolidated: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    title_prefix: str = "",
    save_prefix: Optional[str] = None,
    latent_name: Optional[str] = None,
    activity_name: str = "Trial mean firing rate (spk/s)",
    show: bool = True,
    overwrite: bool = True,
    fit_kind: Literal["linear"] = "linear",
    show_identity: bool = False,
    annotate: bool = True,                         # NEW: add annotation box
    annotation_loc: Literal[
        "upper left", "upper right", "lower left", "lower right"
    ] = "upper left",                              # NEW: annotation location
) -> None:
    """
    For each unit, compute trial-wise mean neural activity within `activity_window`,
    scatter-plot it against a trial-wise latent variable, fit a line, and show
    goodness-of-fit in the title.

    Notes
    -----
    - Uses PSTH data (trial x time) to compute trial mean activity.
    - Saves one figure per unit if `save_path` is provided.
    - Output folder is:
        base_dir / subfolder_name
      where:
        subfolder_name = f"{save_prefix.rstrip('_')}_activity_window_<w0>_<w1>"
        (or "activity_window_<w0>_<w1>" if save_prefix is None)
      and filenames are:
        f"{save_prefix or ''}unit_{unit}_latent_scatter.png"

    Parameters
    ----------
    source
        Path to a Zarr PSTH dataset or an already loaded xarray object containing PSTH/raster.
        Must be compatible with `load_psth_raster_subset`.
    latent_values
        Latent variable values, one per trial.
    latent_trial_ids
        Trial IDs corresponding to `latent_values`.
    activity_window
        Time interval (seconds) to average neural activity within each trial.
    unit_ids
        Which units to plot. If None, plot all units in the dataset.
    align_to_event
        Alignment key when multiple alignments exist.
    time_window
        Optional overall time slicing passed to loader (can reduce IO).
        This does not replace `activity_window`; it just limits loaded data.
    figsize, dpi
        Matplotlib figure size and output DPI.
    consolidated
        Passed to the Zarr loader.
    save_path
        Directory or file-like path used to derive base output folder.
    title_prefix
        Prefix used in figure title.
    save_prefix
        Used for:
        1) folder naming (with activity window appended)
        2) filename prefix: `f"{save_prefix or ''}unit_{unit}_latent_scatter.png"`
    latent_name
        Label for x-axis (defaults to 'Latent value').
    activity_name
        Label for y-axis.
    show
        If True, display figures. If False, suppress display and close after saving.
    overwrite
        If False and target file exists, skip that unit.
    fit_kind
        Currently only 'linear' is supported.
    show_identity
        If True, also plot y=x line (only meaningful if units are comparable).
    """

    # -----------------------------
    # 1) Validate inputs
    # -----------------------------
    latent_values = np.asarray(latent_values, dtype=np.float32)
    latent_trial_ids = np.asarray(latent_trial_ids, dtype=np.int64)

    if latent_values.shape[0] != latent_trial_ids.shape[0]:
        raise ValueError("`latent_values` and `latent_trial_ids` must have the same length.")
    if latent_trial_ids.size != np.unique(latent_trial_ids).size:
        raise ValueError("`latent_trial_ids` contains duplicates; must be one-to-one.")

    w0, w1 = float(activity_window[0]), float(activity_window[1])
    if w1 <= w0:
        raise ValueError("`activity_window` must satisfy activity_window[1] > activity_window[0].")

    # -----------------------------
    # 2) Load PSTH subset
    # -----------------------------
    # NOTE: This function must exist in your codebase.
    psth_da, _raster_da = load_psth_raster_subset(
        source,
        trial_ids=None,
        unit_ids=unit_ids,
        align_to_event=align_to_event,
        time_window=time_window,
        consolidated=consolidated,
    )

    trial_dim = next(d for d in psth_da.dims if d.startswith("trial_"))
    trial_coord = next(c for c in psth_da.coords if c.startswith("trial_index_"))

    all_trial_ids_in_ds = psth_da.coords[trial_coord].values.astype(np.int64)
    unit_indices = psth_da.coords["unit_index"].values.astype(np.int64)
    times = np.asarray(psth_da.coords["time"].values, dtype=np.float32)

    # -----------------------------
    # 3) Align latent values to trials present in dataset
    # -----------------------------
    present_mask = np.isin(latent_trial_ids, all_trial_ids_in_ds)
    if not np.any(present_mask):
        raise ValueError("None of `latent_trial_ids` are present in the dataset.")

    latent_trial_ids = latent_trial_ids[present_mask]
    latent_values = latent_values[present_mask]

    pos_idx = pd.Index(all_trial_ids_in_ds).get_indexer(latent_trial_ids)
    keep = pos_idx >= 0

    psth_da = psth_da.isel({trial_dim: pos_idx[keep]})
    lat = latent_values[keep]

    # -----------------------------
    # 4) Time mask for trial-mean activity
    # -----------------------------
    tmask = (times >= w0) & (times <= w1)
    if not np.any(tmask):
        raise ValueError(
            f"`activity_window`={activity_window} selects no time points. "
            f"Dataset time range is [{float(np.nanmin(times)):.3f}, {float(np.nanmax(times)):.3f}]."
        )

    # -----------------------------
    # 5) Prepare output directory ONCE
    # -----------------------------
    def _fmt_time_for_path(t: float) -> str:
        """
        Format a float time value into a filesystem-safe token.
        Examples:
          -0.25 -> m0p25
           0.0  -> 0p0
           1.5  -> 1p5
        """
        s = f"{t:.6g}"
        s = s.replace("-", "m").replace(".", "p")
        return s

    out_dir: Optional[Path] = None
    if save_path is not None:
        save_target = Path(save_path)

        if save_target.suffix == "" or save_target.is_dir():
            base_dir = save_target
        else:
            base_dir = save_target.parent

        if save_prefix:
            prefix = save_prefix.rstrip("_")
            subfolder_name = (
                f"{prefix}_activity_window_"
                f"{_fmt_time_for_path(w0)}_"
                f"{_fmt_time_for_path(w1)}"
            )
        else:
            subfolder_name = (
                f"activity_window_"
                f"{_fmt_time_for_path(w0)}_"
                f"{_fmt_time_for_path(w1)}"
            )

        out_dir = base_dir / subfolder_name
        out_dir.mkdir(parents=True, exist_ok=True)

    unit_list = list(unit_indices) if unit_ids is None else list(unit_ids)

    # -----------------------------
    # 6) Helpers: linear fit + R^2 + p-value
    # -----------------------------
    def _p_value_from_t(t_stat: float, df: int) -> float:
        """
        Two-sided p-value for t-statistic.
        Uses SciPy if available; otherwise uses a normal approximation (good for df ~ 30+).

        This keeps the function robust in minimal environments.
        """
        if not np.isfinite(t_stat) or df <= 0:
            return np.nan

        try:
            # SciPy is preferred when available.
            from scipy.stats import t as student_t  # type: ignore

            return float(2.0 * student_t.sf(abs(float(t_stat)), df=df))
        except Exception:
            # Normal approximation: p ≈ 2 * (1 - Phi(|t|))
            # Phi via erfc: 1 - Phi(z) = 0.5 * erfc(z / sqrt(2))
            z = abs(float(t_stat))
            return float(math.erfc(z / math.sqrt(2.0)))

    def _linear_fit_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Fit y = a*x + b.

        Returns
        -------
        a : float
            Slope
        b : float
            Intercept
        r2 : float
            Coefficient of determination
        p_val : float
            Two-sided p-value for slope (a)
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n = int(x.size)
        if n < 3:
            return np.nan, np.nan, np.nan, np.nan

        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))

        sxx = float(np.sum((x - x_mean) ** 2))
        if sxx <= 0:
            return np.nan, np.nan, np.nan, np.nan

        sxy = float(np.sum((x - x_mean) * (y - y_mean)))
        a = sxy / sxx
        b = y_mean - a * x_mean

        y_hat = a * x + b
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y_mean) ** 2))

        r2 = np.nan
        if ss_tot > 0:
            r2 = 1.0 - ss_res / ss_tot

        # Slope significance: t = a / SE(a), df = n - 2
        df = n - 2
        if df <= 0:
            return float(a), float(b), float(r2), np.nan

        mse = ss_res / df
        se_a = math.sqrt(mse / sxx) if (mse >= 0 and sxx > 0) else np.nan
        if not np.isfinite(se_a) or se_a <= 0:
            p_val = np.nan
        else:
            t_stat = a / se_a
            p_val = _p_value_from_t(t_stat, df=df)

        return float(a), float(b), float(r2), float(p_val)

    def _annotation_xy(loc: str) -> Tuple[float, float, str, str]:
        """
        Map a location string to (x, y, ha, va) in Axes coordinates.
        """
        if loc == "upper left":
            return 0.03, 0.97, "left", "top"
        if loc == "upper right":
            return 0.97, 0.97, "right", "top"
        if loc == "lower left":
            return 0.03, 0.03, "left", "bottom"
        if loc == "lower right":
            return 0.97, 0.03, "right", "bottom"
        return 0.03, 0.97, "left", "top"

    # -----------------------------
    # 7) Plot per unit
    # -----------------------------
    for unit in unit_list:
        fp: Optional[Path] = None
        if out_dir is not None:
            filename = f"{save_prefix or ''}unit_{unit}_latent_scatter.png"
            fp = out_dir / filename
            if fp.exists() and not overwrite:
                print(f"Skipping Unit {unit} (file exists, overwrite=False): {fp}")
                continue

        where = np.where(unit_indices == unit)[0]
        if where.size == 0:
            continue
        upos = int(where[0])

        fig = None
        try:
            # unit_psth: (trial, time)
            unit_psth = psth_da.isel(unit=upos)
            unit_psth_np = np.asarray(unit_psth.values, dtype=np.float32)  # (n_trials, n_time)

            # Trial mean activity in activity_window
            trial_mean = np.nanmean(unit_psth_np[:, tmask], axis=1)

            # Match finite pairs
            x = np.asarray(lat, dtype=np.float32)
            y = np.asarray(trial_mean, dtype=np.float32)
            ok = np.isfinite(x) & np.isfinite(y)
            n_ok = int(np.sum(ok))

            if n_ok < 3:
                a = b = r2 = p_val = np.nan
            else:
                if fit_kind != "linear":
                    raise ValueError(f"Unsupported fit_kind={fit_kind!r}. Only 'linear' is supported.")
                a, b, r2, p_val = _linear_fit_stats(x[ok], y[ok])

            fig, ax = plt.subplots(1, 1, figsize=figsize)

            ax.scatter(x[ok], y[ok], s=16, alpha=0.8)

            # Fit line
            if np.isfinite(a) and np.isfinite(b) and n_ok >= 2:
                x_min = float(np.nanmin(x[ok]))
                x_max = float(np.nanmax(x[ok]))
                xs = np.linspace(x_min, x_max, 100, dtype=np.float32)
                ys = a * xs + b
                ax.plot(xs, ys, linewidth=2.0)

            # Optional identity line
            if show_identity and n_ok > 0:
                x_min = float(np.nanmin(x[ok]))
                x_max = float(np.nanmax(x[ok]))
                xs = np.linspace(x_min, x_max, 100, dtype=np.float32)
                ax.plot(xs, xs, linewidth=1.0, linestyle="--")

            ax.set_xlabel(latent_name or "Latent value")
            ax.set_ylabel(activity_name)

            # Title 
            base_title = f"{title_prefix} Unit {unit}".strip() if title_prefix else f"Unit {unit}"
            ax.set_title(base_title)


            # Annotation box (same content as title line, but easier to read)
            if annotate and np.isfinite(a) and np.isfinite(b):
                x0, y0, ha, va = _annotation_xy(annotation_loc)
                text = (
                    f"y = {a:.3g}x + {b:.3g}\n"
                    f"R² = {r2:.3f}\n"
                    f"p = {p_val:.2e}\n"
                    f"N = {n_ok}"
                )
                ax.text(
                    x0,
                    y0,
                    text,
                    transform=ax.transAxes,
                    ha=ha,
                    va=va,
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
                )

            ax.grid(True, alpha=0.25)
            plt.tight_layout()

            if fp is not None:
                fig.savefig(fp, dpi=dpi, bbox_inches="tight")
                print(f"Saved Unit {unit} scatter to {fp}")

            if show:
                plt.show()

        finally:
            if fig is not None:
                plt.close(fig)
