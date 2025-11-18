
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple, Optional, Union
from ephys_behavior import get_units_passed_default_qc
from behavior_utils import extract_event_timestamps, find_trials
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Any, Iterable, Tuple, Optional, Union, Literal,Mapping, List, Dict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
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
) -> None:
    """
    Plot both PSTH and raster for each specified unit in separate figures.

    Parameters
    ----------
    source : str | Path | xr.Dataset | xr.DataArray
        Path to a Zarr folder produced by ``extract_neuron_psth_to_zarr``
        or an already loaded PSTH Dataset/DataArray.
    unit_ids : sequence of int, optional
        Indices of units (neurons) to plot. If None, all units are plotted.
    trial_ids : sequence or nested sequences of int, optional
        Specific trial index values to include. Can be a flat list for one group
        or a list of lists for multiple groups to be plotted with different colors.
    trial_types : sequence of str, optional
        Behavioral trial type names used to select trials via ``find_trials``
        if ``trial_ids`` is not provided.
    nwb_data : Any, optional
        NWB file handle required if ``trial_types`` is used to look up trial indices.
    align_to_event : str, optional
        Event name (without the ``psth_`` prefix) when multiple PSTHs exist.
        Ignored if only one event is present.
    time_window : (float, float), optional
        Time window slice along the PSTH time axis, also sets x-axis limits.
    plot_type : {'single', 'mean'}, default 'single'
        'single' → plot each trial separately;
        'mean' → plot mean firing rate per group with SEM shading.
    colors : sequence of str, optional
        Colors for each trial group. Defaults to Matplotlib’s color cycle.
    sem_alpha : float, default 0.3
        Transparency for SEM shading when ``plot_type='mean'``.
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

    # -----------------------------
    # 2) Global autoscale (optional)
    # -----------------------------
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

        # X-limits from requested window (if any)
        if time_window is not None:
            ax_rast.set_xlim(*time_window)
            ax_psth.set_xlim(*time_window)

        # Y-limits strategy
        if y_mode == "auto_per_unit" and unit_max > 0:
            ax_psth.set_ylim(0, unit_max * (1 + y_pad))
        elif y_mode == "auto_global" and global_max > 0:
            ax_psth.set_ylim(0, global_max * (1 + y_pad))

        if plot_type == "mean" and legend and len(trial_groups) > 1:
            ax_psth.legend(frameon=False)

        plt.tight_layout()

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

        plt.show()

from typing import Sequence, Optional, Tuple, Union, Literal
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
from create_psth import load_psth_raster_subset


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
    sort_order: Literal["ascending", "descending"] = "ascending",
    save_prefix: Optional[str] = None,
    latent_name: Optional[str] = None,  # NEW
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
        Time range (in seconds) for plotting. Applies to both raster and PSTH.
    n_bins : int, default 5
        Number of bins (quantiles or equal-width bins) used to divide latent values.
    binning : {'quantile', 'equal'}, default 'quantile'
        How to compute bin edges:
        - 'quantile': divides data into quantiles with equal number of trials.
        - 'equal': divides into equal-width ranges.
    bin_range : (float, float), optional
        Range of latent values for equal-width binning.
        If None, the full observed range is used.
    bin_label : {'mean', 'center'}, default 'mean'
        How to label bins in the colorbar:
        - 'mean': use mean latent value per bin.
        - 'center': use midpoint of bin edges.
    quantile_stat : {'mean', 'median'}, default 'mean'
        Aggregation function for PSTH within each bin.
    ci : {'sem', 'iqr', 'none'}, default 'sem'
        Confidence interval style:
        - 'sem': plot mean ± SEM.
        - 'iqr': plot interquartile range (25–75%).
        - 'none': no shading.
    figsize : (float, float), default (6.0, 5.0)
        Figure size for each unit.
    dpi : int, default 300
        DPI for saved figures.
    consolidated : bool, default True
        Passed to the Zarr loader for faster metadata reading.
    save_path : str | Path, optional
        Directory or full path to save each unit’s plot.
        If a directory is provided, files are named `unit_<id>.png`.
    title_prefix : str, optional
        Text prefix added to each figure title (e.g. session name).
    cmap_name : str, default 'viridis'
        Name of Matplotlib colormap for bins and colorbar.
    raster_colormap : bool, default True
        If True, color-code raster trials by bin color.
        If False, plot all rasters in black.
    show_colormap : bool, default True
        Whether to display a colorbar showing latent value range.
    sort_order : {'ascending', 'descending'}, default 'ascending'
        Sort order of trials by latent value for raster display.
    save_prefix : str, optional
        Optional filename prefix added to each saved figure.
    latent_name : str, optional
        Custom label for the colorbar. Defaults to 'Latent value' if not provided.
    """

    # -----------------------------
    # 1) Validate inputs
    # -----------------------------
    latent_values = np.asarray(latent_values, dtype=float)
    latent_trial_ids = np.asarray(latent_trial_ids, dtype=int)
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
    all_trial_ids_in_ds = psth_da.coords[trial_coord].values.astype(int)
    unit_indices = psth_da.coords["unit_index"].values.astype(int)
    times = psth_da.coords["time"].values

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
    trial_ids_arr = psth_da.coords[trial_coord].values.astype(int)

    # -----------------------------
    # 4) Build bins
    # -----------------------------
    lat_finite = lat[np.isfinite(lat)]
    if lat_finite.size == 0:
        raise ValueError("All latent values are NaN; cannot build bins.")

    if binning == "quantile":
        q_perc = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.nanpercentile(lat_finite, q_perc))
    else:
        lo, hi = (float(np.nanmin(lat_finite)), float(np.nanmax(lat_finite))) if bin_range is None else bin_range
        if hi <= lo:
            raise ValueError("Invalid bin_range or latent range.")
        edges = np.linspace(lo, hi, n_bins + 1)
        edges[-1] = np.nextafter(edges[-1], np.inf)

    n_bins = edges.size - 1
    bin_idx = np.digitize(lat, edges[1:-1], right=False)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    cmap = cm.get_cmap(cmap_name, n_bins)
    colors = [mcolors.to_hex(cmap(i)) for i in range(n_bins)]
    if bin_label == "center":
        bin_tick_vals = list(0.5 * (edges[:-1] + edges[1:]))
    else:
        bin_tick_vals = [float(np.nanmean(lat[bin_idx == i])) if np.any(bin_idx == i) else np.nan
                         for i in range(n_bins)]

    latent_min, latent_max = float(np.nanmin(lat_finite)), float(np.nanmax(lat_finite))
    unit_list = list(unit_indices) if unit_ids is None else list(unit_ids)
    trial_to_pos = {int(t): i for i, t in enumerate(trial_ids_arr)}

    # -----------------------------
    # 5) Plot per unit
    # -----------------------------
    for unit in unit_list:
        where = np.where(unit_indices == unit)[0]
        if where.size == 0:
            continue
        upos = int(where[0])

        unit_psth = psth_da.isel(unit=upos)
        unit_rast = raster_da.isel(unit=upos)

        sort_order_bool = sort_order == "ascending"
        sort_order_arr = np.argsort(lat, kind="mergesort")
        if not sort_order_bool:
            sort_order_arr = sort_order_arr[::-1]
        sorted_trials = trial_ids_arr[sort_order_arr]
        sorted_bins = bin_idx[sort_order_arr]

        fig, (ax_rast, ax_psth) = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [1, 1.3]}
        )

        # --- Raster ---
        y = 0
        for idx, tval in enumerate(sorted_trials):
            y += 1
            b = int(sorted_bins[idx])
            ti = trial_to_pos.get(int(tval))
            if ti is None:
                continue
            spikes = unit_rast.isel({trial_dim: ti}).values
            spikes = spikes[np.isfinite(spikes)]
            color = colors[b] if raster_colormap else "black"
            if spikes.size > 0:
                ax_rast.vlines(spikes, y, y + 0.9, color=color, alpha=0.8, linewidth=0.6)
        ax_rast.axvline(0.0, color="k", ls="--", lw=0.8)
        ttl = f"{title_prefix} Unit {unit}" if title_prefix else f"Unit {unit}"
        order_str = "ascending" if sort_order_bool else "descending"
        ax_rast.set_title(f"{ttl} (sorted {order_str})")
        ax_rast.set_ylabel("Trials (sorted by latent)")

        # --- PSTH summaries ---
        ymax = 0.0
        for b in range(n_bins):
            sel = (bin_idx == b)
            if not np.any(sel):
                continue
            data = unit_psth.isel({trial_dim: sel}).values
            center = np.nanmedian(data, axis=0) if quantile_stat == "median" else np.nanmean(data, axis=0)
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
            ax_psth.plot(times, center, color=colors[b], linewidth=1.5)
            if lower is not None and upper is not None:
                ax_psth.fill_between(times, lower, upper, color=colors[b], alpha=0.25)
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

        # --- Colorbar ---
        if show_colormap:
            sm = plt.cm.ScalarMappable(
                cmap=cm.get_cmap(cmap_name),
                norm=plt.Normalize(vmin=latent_min, vmax=latent_max),
            )
            cbar = fig.colorbar(sm, ax=[ax_rast, ax_psth], orientation="vertical", pad=0.02)
            cbar.set_label(latent_name or "Latent value", rotation=270, labelpad=12)
            ticks = [v for v in bin_tick_vals if np.isfinite(v)]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{v:.2f}" for v in ticks])
            cbar.ax.tick_params(size=0)

        # --- Save ---
        if save_path:
            save_target = Path(save_path)
            if save_target.suffix == "" or save_target.is_dir():
                save_target.mkdir(parents=True, exist_ok=True)
                filename = f"{save_prefix or ''}unit_{unit}.png"
                fp = save_target / filename
            else:
                base = save_target.with_suffix("")
                filename = f"{save_prefix or ''}{base.name}_unit_{unit}.png"
                fp = base.parent / filename
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
            print(f"Saved Unit {unit} figure to {fp}")

        plt.show()
