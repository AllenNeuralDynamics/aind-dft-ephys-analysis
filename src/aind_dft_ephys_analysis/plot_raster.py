
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



def plot_raster_and_quantile_psth_by_latent(
    source: Union[str, Path, xr.DataArray, xr.Dataset],
    *,
    latent_values: Union[Sequence[float], Mapping[int, float]],
    unit_ids: Optional[Sequence[int]] = None,
    align_to_event: Optional[str] = None,
    time_window: Optional[Tuple[float, float]] = None,
    n_quantiles: int = 5,
    quantile_stat: Literal["mean", "median"] = "mean",
    ci: Literal["sem", "iqr", "none"] = "sem",
    colors: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (6.0, 5.0),
    legend: bool = True,
    dpi: int = 300,
    consolidated: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    title_prefix: str = "",
) -> None:
    """
    Plot, for each unit, a trial-sorted raster and PSTH quantile summaries binned by a per-trial latent value.

    Parameters
    ----------
    source : str | Path | xr.Dataset | xr.DataArray
        Zarr path or already-loaded PSTH Dataset/DataArray created by `extract_neuron_psth_to_zarr`.
    latent_values : Sequence[float] | Mapping[int, float]
        Per-trial latent values used to (1) sort the raster and (2) form quantile bins.
        - If a sequence/array: must be aligned with the PSTH trial_index order.
        - If a mapping: keys are trial_index values (ints), values are floats.
    unit_ids : sequence of int, optional
        Units to plot. If None, all available units in `source` are used.
    align_to_event : str, optional
        Event name (without the "psth_" prefix) when multiple events exist.
    time_window : (float, float), optional
        Time window to slice the PSTH time axis; also sets x-limits for plots.
    n_quantiles : int, default 5
        Number of quantile bins (e.g., 5 → quintiles).
    quantile_stat : {"mean","median"}, default "mean"
        Summary statistic across trials within each quantile for the PSTH line.
    ci : {"sem","iqr","none"}, default "sem"
        Uncertainty band shown around the quantile summary:
        - "sem": +/- standard error of the mean
        - "iqr": 25th–75th percentile band around the median/mean
        - "none": no band
    colors : sequence of str, optional
        Colors for each quantile, from low to high. Defaults to Matplotlib's cycle.
    figsize : (float, float), default (6.0, 5.0)
        Figure size per unit (width, height).
    legend : bool, default True
        Whether to show the legend mapping colors to quantile ranges.
    dpi : int, default 300
        Resolution when saving figures.
    consolidated : bool, default True
        Passed to the Zarr loader for faster metadata reading.
    save_path : str | Path, optional
        If provided, save PNGs here. If a directory, files are named `unit_<id>.png`.
    title_prefix : str, default ""
        Optional prefix for figure titles.

    Notes
    -----
    * Trials with NaN latent values are excluded from both raster and PSTH summaries.
    * Quantile edges are computed using `np.nanpercentile` and are made unique; if
      too many duplicate edges occur (e.g., many identical latent values), the
      number of effective bins may be smaller than `n_quantiles`.
    """
    # 1) Load PSTH and raster (all trials; unit subset will be applied here)
    psth_da, raster_da = load_psth_raster_subset(
        source,
        trial_ids=None,
        unit_ids=unit_ids,
        align_to_event=align_to_event,
        time_window=time_window,  # window applies to PSTH and masks raster values
        consolidated=consolidated,
    )

    # Identify names and coordinates
    trial_dim = next(d for d in psth_da.dims if d.startswith("trial_"))
    trial_coord = next(c for c in psth_da.coords if c.startswith("trial_index_"))
    trial_index_vals = psth_da.coords[trial_coord].values.astype(int)
    times = psth_da.coords["time"].values
    unit_indices = psth_da.coords["unit_index"].values.astype(int)

    # 2) Normalize latent values into an array aligned to `trial_index_vals`
    if isinstance(latent_values, Mapping):
        lat = np.full(trial_index_vals.shape, np.nan, dtype=float)
        for i, tid in enumerate(trial_index_vals):
            if tid in latent_values:
                lat[i] = float(latent_values[tid])
    else:
        lat_arr = np.asarray(latent_values, dtype=float)
        if lat_arr.shape[0] != trial_index_vals.shape[0]:
            raise ValueError(
                "When `latent_values` is a sequence, its length must match the number "
                "of trials in the PSTH. If your latent is keyed by trial IDs, pass a dict."
            )
        lat = lat_arr

    # Mask trials with NaN latent
    valid_mask = np.isfinite(lat)
    if not np.any(valid_mask):
        raise ValueError("All latent values are NaN; nothing to plot.")

    # 3) Compute quantile bin edges (unique edges to avoid empty bins)
    #    Example: for 5 quantiles, percentiles at [0, 20, 40, 60, 80, 100]
    q_perc = np.linspace(0, 100, n_quantiles + 1)
    edges = np.nanpercentile(lat[valid_mask], q_perc)
    # Make edges non-decreasing unique to avoid zero-width bins
    edges = np.unique(edges)
    if edges.size < 2:
        raise ValueError("Latent values lack variation; cannot form quantile bins.")
    # When fewer unique edges than requested, we will get fewer bins.
    # Build bins as half-open [edge[i], edge[i+1]) except last which is closed on the right.
    n_bins = edges.size - 1

    # Assign each valid trial to a bin index in 0..n_bins-1
    # We use right=False so that the last edge is exclusive; we then manually include max at the end.
    bin_idx = np.digitize(lat, edges[1:-1], right=False)
    # Ensure max value goes into the last bin
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_idx[~valid_mask] = -1  # mark invalid trials

    # Define labels and colors
    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if colors is None:
        # Pick first n_bins colors from the cycle
        colors = [cmap[i % len(cmap)] for i in range(n_bins)]
    elif len(colors) < n_bins:
        raise ValueError(f"Provided {len(colors)} colors but need at least {n_bins} for the quantile bins.")

    # Build legend labels like "Q1 [min, q20)" ... "Q5 [q80, max]"
    def _fmt_edge(x: float) -> str:
        # Compact formatting for edges
        if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 1e-2):
            return f"{x:.2e}"
        return f"{x:.3g}"

    q_labels: List[str] = []
    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]
        if i < n_bins - 1:
            lab = f"Q{i+1} [{_fmt_edge(left)}, {_fmt_edge(right)})"
        else:
            lab = f"Q{i+1} [{_fmt_edge(left)}, {_fmt_edge(right)}]"
        q_labels.append(lab)

    # 4) Decide which units to iterate over
    if unit_ids is None:
        unit_list = list(unit_indices)
    else:
        unit_list = list(unit_ids)

    # 5) Per-unit plotting
    for unit in unit_list:
        # Map unit id to positional index
        try:
            upos = int(np.where(unit_indices == unit)[0][0])
        except IndexError:
            # Skip if the requested unit is not present
            continue

        unit_psth = psth_da.isel(unit=upos)     # dims: (trial_dim, time)
        unit_rast = raster_da.isel(unit=upos)   # dims: (trial_dim, spike)
        trial_ids_arr = unit_psth.coords[trial_coord].values.astype(int)

        # Keep only trials with finite latent and that exist for this unit
        keep = (bin_idx >= 0)
        if not np.any(keep):
            continue

        # Sort trials by latent ascending for a clean raster
        sort_order = np.argsort(lat, kind="mergesort")  # stable sort to keep groups coherent
        sort_order = sort_order[keep[sort_order]]       # drop NaN latent trials
        sorted_trials = trial_ids_arr[sort_order]
        sorted_bins = bin_idx[sort_order]

        # Prepare figure: top raster, bottom PSTH quantile summaries
        fig, (ax_rast, ax_psth) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # 5a) Raster: plot each trial's spikes, colored by its quantile bin
        y = 0
        for idx_in_sorted, tval in enumerate(sorted_trials):
            y += 1
            b = int(sorted_bins[idx_in_sorted])
            spikes = unit_rast.sel({trial_coord: tval}).values  # shape: (spike,)
            spikes = spikes[np.isfinite(spikes)]
            if spikes.size > 0:
                ax_rast.vlines(spikes, y, y + 0.9, color=colors[b], alpha=0.8, linewidth=0.6)

        ax_rast.axvline(0.0, color="k", ls="--", lw=0.8)
        ax_rast.set_ylabel("Trials (sorted by latent)")
        ttl = f"Unit {unit}"
        if title_prefix:
            ttl = f"{title_prefix} {ttl}".strip()
        ax_rast.set_title(ttl)

        # 5b) PSTH quantile summaries
        ymax = 0.0
        for b in range(n_bins):
            # Trials belonging to this bin
            trial_mask_b = (bin_idx == b)
            if not np.any(trial_mask_b):
                continue
            # Extract indices of trials for this unit that are in bin b
            trials_b = trial_ids_arr[trial_mask_b]
            if trials_b.size == 0:
                continue

            da_b = unit_psth.sel({trial_coord: trials_b})  # dims: (trial_dim, time)
            data = da_b.values  # shape: (n_trials_b, n_time)

            # Summary line
            if quantile_stat == "median":
                center = np.nanmedian(data, axis=0)
            else:
                center = np.nanmean(data, axis=0)

            # Uncertainty band
            lower, upper = None, None
            if ci == "sem":
                n = np.sum(np.isfinite(data), axis=0)
                std = np.nanstd(data, axis=0, ddof=1)
                sem = np.where(n > 0, std / np.sqrt(np.maximum(n, 1)), np.nan)
                lower, upper = center - sem, center + sem
            elif ci == "iqr":
                q25 = np.nanpercentile(data, 25, axis=0)
                q75 = np.nanpercentile(data, 75, axis=0)
                lower, upper = q25, q75

            # Plot
            ax_psth.plot(times, center, color=colors[b], label=q_labels[b], linewidth=1.5)
            if lower is not None and upper is not None:
                ax_psth.fill_between(times, lower, upper, color=colors[b], alpha=0.25)

            # Track ymax for nice padding
            finite_max = np.nanmax(upper if upper is not None else center)
            if np.isfinite(finite_max):
                ymax = max(ymax, float(finite_max))

        ax_psth.axvline(0.0, color="k", ls="--", lw=0.8)
        ax_psth.set_ylabel("Firing rate (spk/s)")
        ax_psth.set_xlabel("Time (s)")

        if time_window is not None:
            ax_rast.set_xlim(*time_window)
            ax_psth.set_xlim(*time_window)

        if ymax > 0:
            ax_psth.set_ylim(0, ymax * 1.05)

        if legend:
            ax_psth.legend(frameon=False, title="Latent quantiles", fontsize=9)

        plt.tight_layout()

        # Save if requested
        if save_path:
            save_target = Path(save_path)
            save_target.mkdir(parents=True, exist_ok=True) if (save_target.suffix == "" or save_target.is_dir()) else None
            if save_target.suffix == "" or save_target.is_dir():
                fp = save_target / f"unit_{unit}.png"
            else:
                base = save_target.with_suffix("")
                fp = base.parent / f"{base.name}_unit_{unit}.png"
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
            print(f"Saved Unit {unit} figure to {fp}")

        plt.show()
