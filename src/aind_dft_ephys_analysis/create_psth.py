import numpy as np
import xarray as xr
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple, Optional, Union
from ephys_behavior import get_units_passed_default_qc
from behavior_utils import extract_event_timestamps, find_trials
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Any, Iterable, Tuple, Optional, Union, Literal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def extract_neuron_psth_to_zarr(
    nwb_data: Any,
    units: Optional[Iterable[int]] = None,
    align_to_event: Union[str, Sequence[str]] = "go_cue",
    time_window: Tuple[float, float] = (-3.0, 5.0),
    bin_size: float = 0.05,
    trial_type: Optional[str] = None,
    save_folder: Union[str, Path] = "/root/capsule/results",
    save_name: Optional[str] = None,
    overwrite: bool = True,
) -> xr.Dataset:
    """
    Parameters
    ----------
    nwb_data : Any
        NWB file handle. Must have:
        - `nwb_data.units["spike_times"]`: list-of-arrays of spike timestamps per unit
        - `nwb_data.session_id` attribute for naming output.
    units : Optional[Iterable[int]], default None
        Specific unit indices to include. If None, uses
        `get_units_passed_default_qc(nwb_data)` to select all good units.
    align_to_event : str or sequence of str, default "go_cue"
        Name(s) of behavioral event(s) to align spike trains to. For each
        event you get one `psth_<event>` and one `raster_<event>` variable.
    time_window : tuple of (float, float), default (-3.0, 5.0)
        Time window (in seconds) around each event timestamp. First value
        is start (can be negative), second is end.
    bin_size : float, default 0.05
        Bin width in seconds for PSTH computation. Firing rate is
        `counts / bin_size` (spikes per second).
    trial_type : Optional[str], default None
        If provided, filters trials via `find_trials(nwb_data, trial_type)`.
        Otherwise all trials for each event are kept.
    save_folder : str or Path, default "/root/capsule/results"
        Directory where the Zarr folder will be created (will be made if needed).
    save_name : Optional[str], default None
        Name of the output Zarr folder. If None, uses
        `<session_id>.zarr`; “.zarr” is appended automatically otherwise.
    overwrite : bool, default True
        If True and the destination folder exists, it is deleted before writing.

    Returns
    -------
    xr.Dataset
        Contains, for each event in `align_to_event`:
        - `psth_<event>` : DataArray of shape (unit × trial × time) with firing rates.
        - `raster_<event>` : DataArray of shape (unit × trial × spike) with
          raw spike-time offsets (s), padded with NaN.

    Raises
    ------
    ValueError
        If `align_to_event` is empty or no trials match a given event.
    """

    # 1. Determine units and events
    if units is None:
        units = get_units_passed_default_qc(nwb_data)
    units = list(units)

    if isinstance(align_to_event, (str, bytes)):
        events = [str(align_to_event)]
    else:
        events = [str(e) for e in align_to_event]
    if not events:
        raise ValueError("`align_to_event` cannot be empty.")

    start, end = time_window
    edges = np.arange(start, end + bin_size, bin_size)
    centers = edges[:-1] + bin_size / 2.0

    data_vars = {}
    n_trials_per_event = {}

    for evt in events:
        # 2. Extract event times and trial indices
        all_times = np.asarray(extract_event_timestamps(nwb_data, evt), dtype=float)
        if trial_type is not None:
            trial_idx = np.asarray(find_trials(nwb_data, trial_type), dtype=int)
            event_times = all_times[trial_idx]
        else:
            trial_idx = np.arange(len(all_times), dtype=int)
            event_times = all_times

        if len(event_times) == 0:
            raise ValueError(f"No trials matched for event '{evt}'.")

        n_u, n_t = len(units), len(event_times)

        # 3. Compute PSTH
        psth = np.zeros((n_u, n_t, len(centers)), dtype=np.float32)

        # 4. Collect raw spike-time offsets per (unit, trial)
        raster_lists = [[[] for _ in range(n_t)] for _ in range(n_u)]
        for ui, u in enumerate(units):
            spikes = np.asarray(nwb_data.units["spike_times"][u], dtype=float)
            for ti, t0 in enumerate(event_times):
                rel = spikes - t0
                # PSTH bin counts
                counts, _ = np.histogram(rel, bins=edges)
                psth[ui, ti, :] = counts / bin_size
                # Raster: keep only offsets within [start, end]
                mask = (rel >= start) & (rel <= end)
                raster_lists[ui][ti] = rel[mask]

        # 5. Pad raster lists into fixed-size array
        max_spikes = max(len(raster_lists[ui][ti]) for ui in range(n_u) for ti in range(n_t))
        raster = np.full((n_u, n_t, max_spikes), np.nan, dtype=np.float32)
        for ui in range(n_u):
            for ti in range(n_t):
                sl = raster_lists[ui][ti]
                raster[ui, ti, : len(sl)] = sl

        trial_dim = f"trial_{evt}"
        coord_trial = f"trial_index_{evt}"

        # 6. Build DataArrays
        da_psth = xr.DataArray(
            psth,
            dims=("unit", trial_dim, "time"),
            coords={
                "unit_index": ("unit", units),
                coord_trial: (trial_dim, trial_idx),
                "time": ("time", centers),
            },
            name=f"psth_{evt}",
            attrs={
                "align_to_event": evt,
                "time_window": time_window,
                "bin_size": bin_size,
                "trial_type": trial_type or "all",
            },
        )
        da_raster = xr.DataArray(
            raster,
            dims=("unit", trial_dim, "spike"),
            coords={
                "unit_index": ("unit", units),
                coord_trial: (trial_dim, trial_idx),
                "spike": np.arange(max_spikes),
            },
            name=f"raster_{evt}",
            attrs={
                "align_to_event": evt,
                "time_window": time_window,
                "description": "relative spike times (s), padded with NaN",
            },
        )

        data_vars[f"psth_{evt}"] = da_psth
        data_vars[f"raster_{evt}"] = da_raster
        n_trials_per_event[evt] = n_t

    # 7. Assemble Dataset and save to Zarr
    ds = xr.Dataset(data_vars)
    ds.attrs.update({
        "session_id": getattr(nwb_data, "session_id", "unknown"),
        "align_to_events": events,
        "bin_size": bin_size,
        "n_units": len(units),
        "n_trials_per_event": n_trials_per_event,
        "created_with": "extract_neuron_psth_to_zarr",
    })

    session_id = getattr(nwb_data, "session_id", "session")
    if save_name is None:
        save_name = f"{session_id}.zarr"
    elif not save_name.endswith(".zarr"):
        save_name += ".zarr"

    dest = Path(save_folder).expanduser() / save_name
    if dest.exists() and overwrite:
        import shutil
        shutil.rmtree(dest)
    ds.to_zarr(dest, mode="w", consolidated=True)
    print(f"PSTH and raster data saved to {dest} [events={events}]")

    return ds


def load_psth_raster_subset(
    source: Union[str, Path, xr.DataArray, xr.Dataset],
    trial_ids: Optional[Sequence[int]] = None,
    unit_ids: Optional[Sequence[int]] = None,
    align_to_event: Optional[str] = None,
    time_window: Optional[Tuple[float, float]] = None,
    consolidated: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Load and subset both PSTH and raster data from a Zarr store or in‑memory object.

    Parameters
    ----------
    source : str or Path or xr.DataArray or xr.Dataset
        - If a string or Path: path to a Zarr folder created by `extract_neuron_psth_to_zarr`.
        - If an xarray.Dataset: must contain variables named "psth_<event>" and "raster_<event>".
        - If an xarray.DataArray: assumed to be a single psth_<event> array; wrapped into a Dataset.
    trial_ids : sequence of int, optional
        Values of the trial_index coordinate to keep. If None, all trials are included.
    unit_ids : sequence of int, optional
        Values of the unit_index coordinate to keep. If None, all units are included.
    align_to_event : str, optional
        Event name (without the "psth_" prefix) selecting which psth_ and raster_
        variables to use when multiple exist. If None and only one event is present,
        that one is selected automatically.
    time_window : tuple of two floats, optional
        (start, end) in seconds to slice the PSTH time axis. If None, the full range
        is kept. Raster data is not time-sliced.
    consolidated : bool, default True
        Whether to use consolidated metadata when opening a Zarr store (faster).

    Returns
    -------
    psth_da : xr.DataArray
        The subsetted PSTH DataArray of shape (unit × trial × time).
    raster_da : xr.DataArray
        The subsetted raster DataArray of shape (unit × trial × spike).

    Raises
    ------
    FileNotFoundError
        If a provided Zarr path does not exist.
    KeyError
        If the necessary psth_ or raster_ variables are not found.
    ValueError
        If multiple psth_ variables exist but no `align_to_event` is given.
    TypeError
        If `source` is not a supported type.
    """
    # 1) Obtain an xarray.Dataset containing both psth_ and raster_ variables
    if isinstance(source, xr.Dataset):
        ds = source
    elif isinstance(source, xr.DataArray):
        ds = source.to_dataset(name=source.name)
    else:
        # Assume source is a path/str
        ds = load_psth_raster_from_zarr(
            source,
            load_type="both",
            align_to_event=align_to_event,
            consolidated=consolidated,
        )

    # 2) Determine variable names
    psth_vars   = [v for v in ds.data_vars if v.startswith("psth_")]
    raster_vars = [v for v in ds.data_vars if v.startswith("raster_")]
    if not psth_vars or not raster_vars:
        raise KeyError("Dataset must contain both psth_ and raster_ variables.")
    if align_to_event:
        psth_name   = f"psth_{align_to_event}"
        raster_name = f"raster_{align_to_event}"
    else:
        if len(psth_vars) > 1:
            raise ValueError(f"Multiple psth_ vars {psth_vars}, please specify `align_to_event`.")
        psth_name   = psth_vars[0]
        raster_name = psth_name.replace("psth_", "raster_")
    psth_da   = ds[psth_name]
    raster_da = ds[raster_name]

    # 3) Identify trial dimension and coordinate
    trial_dim   = next((d for d in psth_da.dims if d.startswith("trial_")), "trial")
    trial_coord = next((c for c in psth_da.coords if c.startswith("trial_index_")), "trial_index")

    # 4) Subset by trial_ids
    if trial_ids is not None:
        if trial_coord in psth_da.coords and psth_da.indexes.get(trial_coord) is not None:
            psth_da   = psth_da.sel({trial_coord: list(trial_ids)})
            raster_da = raster_da.sel({trial_coord: list(trial_ids)})
        else:
            mask = np.isin(psth_da.coords[trial_coord].values, trial_ids)
            psth_da   = psth_da.isel({trial_dim: mask})
            raster_da = raster_da.isel({trial_dim: mask})

    # 5) Subset by unit_ids
    if unit_ids is not None:
        if "unit_index" in psth_da.coords and psth_da.indexes.get("unit_index") is not None:
            psth_da   = psth_da.sel(unit_index=list(unit_ids))
            raster_da = raster_da.sel(unit_index=list(unit_ids))
        else:
            mask = np.isin(psth_da.coords["unit_index"].values, unit_ids)
            psth_da   = psth_da.isel(unit=mask)
            raster_da = raster_da.isel(unit=mask)

    # 6) Apply time window slice to PSTH only
    if time_window is not None:
        t0, t1 = time_window
        psth_da = psth_da.sel(time=slice(t0, t1))
        raster_da = raster_da.where((raster_da >= t0) & (raster_da <= t1), other=np.nan)
        
    return psth_da, raster_da




def load_zarr(
    zarr_path: Union[str, Path],
    *,
    consolidated: bool = True,
) -> xr.Dataset:
    """
    Directly open a Zarr folder (no PSTH variable selection).

    Parameters
    ----------
    zarr_path : str or Path
        Path to the ``*.zarr`` folder produced by
        :func:`extract_neuron_psth_to_zarr`.
    consolidated : bool, default True
        Whether to use consolidated metadata (faster when the folder
        was written with ``consolidated=True``).

    Returns
    -------
    xarray.Dataset
        The raw Dataset stored in the Zarr folder.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    zarr_path = Path(zarr_path).expanduser()
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr folder not found: {zarr_path}")
    return xr.open_zarr(zarr_path, consolidated=consolidated)


def load_psth_raster_from_zarr(
    zarr_path: Union[str, Path],
    load_type: Literal["psth", "raster", "both"] = "psth",
    align_to_event: Optional[str] = None,
    consolidated: bool = True,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Load PSTH, raster, or both from a Zarr store created by extract_neuron_psth_to_zarr.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the “*.zarr” folder produced by `extract_neuron_psth_to_zarr`.
    load_type : {"psth", "raster", "both"}, default "psth"
        • "psth"   → return a single PSTH DataArray (psth_<event>).
        • "raster" → return a single raster DataArray (raster_<event>).
        • "both"   → return a Dataset containing all psth_* and raster_* variables.
    align_to_event : str or None, optional
        When loading a single DataArray ("psth" or "raster") and multiple events exist,
        this selects which event (without prefix) to load. If omitted and only one
        matching variable exists, that one is returned automatically.
    consolidated : bool, default True
        Whether to use consolidated metadata (faster when written with consolidated=True).

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Depending on `load_type`:
        - DataArray for a single psth_ or raster_ variable.
        - Dataset containing all psth_* and raster_* variables if `load_type=="both"`.
    """
    zarr_path = Path(zarr_path).expanduser()
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr folder not found: {zarr_path}")

    ds = xr.open_zarr(zarr_path, consolidated=consolidated)

    psth_vars   = [v for v in ds.data_vars if v.startswith("psth_")]
    raster_vars = [v for v in ds.data_vars if v.startswith("raster_")]

    # BOTH → return a Dataset with both sets of variables
    if load_type == "both":
        if not (psth_vars or raster_vars):
            raise KeyError("No psth_ or raster_ variables found in dataset.")
        return ds[psth_vars + raster_vars]

    # determine which set to pick
    if load_type == "psth":
        var_list, prefix = psth_vars, "psth_"
    elif load_type == "raster":
        var_list, prefix = raster_vars, "raster_"
    else:
        raise ValueError("`load_type` must be 'psth', 'raster', or 'both'.")

    if not var_list:
        raise KeyError(f"No variables starting with '{prefix}' found in dataset.")

    # select single variable
    if align_to_event is None:
        if len(var_list) == 1:
            chosen = var_list[0]
        else:
            raise ValueError(f"Multiple variables found {var_list}; please specify `align_to_event=`.")
    else:
        chosen = f"{prefix}{align_to_event}"
        if chosen not in ds:
            raise KeyError(f"Variable '{chosen}' not found. Available: {var_list}")

    return ds[chosen]


def mean_firing_rate_matrix(
    source: Union[str, Path, xr.Dataset, xr.DataArray],
    trial_ids: Optional[Sequence[int]] = None,
    unit_ids: Optional[Sequence[int]] = None,
    time_window: Optional[Tuple[float, float]] = None,
    align_to_event: Optional[str] = None,
    standardize_names: bool = False,
    consolidated: bool = True,
) -> xr.DataArray:
    """
    Compute a 2‑D matrix (unit × trial) of mean firing rates (spikes/s)
    across a requested time window, for a chosen alignment event.

    Parameters
    ----------
    source : str | Path | xr.Dataset | xr.DataArray
        Zarr path or already-loaded PSTH Dataset/DataArray.
    trial_ids : Sequence[int] | None
        Values of the *trial_index_<event>* coordinate to keep. None → all.
    unit_ids : Sequence[int] | None
        Values of the *unit_index* coordinate to keep. None → all.
    time_window : (float, float) | None
        (start, end) in seconds along the PSTH time axis. None → full range.
    align_to_event : str | None
        Event name (without "psth_") when multiple PSTHs exist in *source*.
        Ignored if *source* is already a single PSTH DataArray.
    standardize_names : bool, default False
        If True, rename the trial dimension to "trial" and its index coord to
        "trial_index" for downstream convenience.
    consolidated : bool, default True
        Passed through when opening a Zarr store.

    Returns
    -------
    xr.DataArray
        Dimensions: ("unit", "<trial_dim>") or ("unit", "trial" if standardized)
        Coordinates: *unit_index*, *<trial_index_coord>* (or *trial_index*)
        Name: "mean_firing_rate"

    Raises
    ------
    ValueError, KeyError, FileNotFoundError
        See `load_psth_raster_subset`.
    """
    # 1) Reuse the slicer (now event-aware)
    da_sub = load_psth_raster_subset(
        source,
        trial_ids=trial_ids,
        unit_ids=unit_ids,
        time_window=time_window,
        align_to_event=align_to_event,
        consolidated=consolidated,
    )

    if da_sub.size == 0:
        raise ValueError("Selected subset is empty – nothing to average.")

    # Identify trial dim / coord (event-specific)
    trial_dim_candidates = [d for d in da_sub.dims if d.startswith("trial_")] or [d for d in da_sub.dims if d == "trial"]
    if len(trial_dim_candidates) != 1:
        raise ValueError("Could not uniquely determine the trial dimension.")
    trial_dim = trial_dim_candidates[0]

    trial_coord_candidates = [c for c in da_sub.coords if c.startswith("trial_index_")] or [c for c in da_sub.coords if c == "trial_index"]
    if len(trial_coord_candidates) != 1:
        raise ValueError("Could not uniquely determine the trial_index coordinate.")
    trial_coord = trial_coord_candidates[0]

    # 2) Mean across time → unit × trial matrix
    mean_da = da_sub.mean("time")  # dims now ('unit', trial_dim)
    mean_da = mean_da.rename("mean_firing_rate")
    mean_da.attrs = {**da_sub.attrs, "computed_with": "mean_firing_rate_matrix"}

    # 3) Optional normalization of names
    if standardize_names:
        mean_da = mean_da.rename({trial_dim: "trial"})
        mean_da = mean_da.rename({trial_coord: "trial_index"})

    return mean_da

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
) -> None:
    """
    Plot both PSTH and raster for each specified unit in separate figures.

    Parameters
    ----------
    source : str | Path | xr.Dataset | xr.DataArray
        Path to Zarr folder or loaded PSTH Dataset/DataArray.
    unit_ids : sequence of int, optional
        Unit indices to plot. If None, plots all available units.
    trial_ids : sequence or nested sequences of int, optional
        Specific trial_index values to include, or list of lists for groups.
    trial_types : sequence of str, optional
        Trial type names used to select trials via find_trials if trial_ids not provided.
    nwb_data : Any, optional
        NWB file handle required if trial_types is used.
    align_to_event : str, optional
        Event name (without prefix) when multiple PSTHs exist.
    time_window : (float, float), optional
        Time window slice along the PSTH time axis; also sets x-axis limits.
    plot_type : {'single', 'mean'}, default 'single'
        'single' -> plot each trial separately; 'mean' -> plot mean and SEM.
    colors : sequence of str, optional
        Colors for each trial group. Defaults to matplotlib cycle.
    sem_alpha : float, default 0.3
        Transparency for SEM shading when plot_type='mean'.
    figsize : (float, float), default (6.0, 4.0)
        Figure size in inches (width, height) per unit.
    sharey : bool, default False
        Whether to share y-axis limits between PSTH and raster.
    legend : bool, default True
        Show legend for mean plots when multiple groups.
    save_path : str | Path, optional
        Path to save each unit's figure (PNG). If None, figures are not saved.
    dpi : int, default 300
        Resolution of saved figure.
    consolidated : bool, default True
        Passed to Zarr loader for metadata.
    group_labels : sequence of str, optional
        Custom labels for trial groups when nested trial_ids given.
    y_mode : {'auto_per_unit', 'auto_global', 'none'}, default 'auto_per_unit'
        Y-axis scaling mode for PSTH.
    y_pad : float, default 0.05
        Fractional padding added above max firing rate for autoscaling.
    """
    # Load PSTH and raster
    psth_da, raster_da = load_psth_raster_subset(
        source,
        trial_ids=trial_ids,
        unit_ids=unit_ids,
        align_to_event=align_to_event,
        time_window=time_window,
        consolidated=consolidated,
    )

    # Determine dims
    trial_dim = next(d for d in psth_da.dims if d.startswith("trial_"))
    trial_coord = next(c for c in psth_da.coords if c.startswith("trial_index_"))
    times = psth_da.coords["time"].values
    unit_indices = psth_da.coords["unit_index"].values

    # Prepare trial groups
    def _group_ids_by_type():
        if trial_ids is not None:
            lists = trial_ids if isinstance(trial_ids[0], (list, tuple, np.ndarray)) else [trial_ids]
            lists = [np.asarray(lst, dtype=int) for lst in lists]
            labels = group_labels or [f"group_{i}" for i in range(len(lists))]
            return dict(zip(labels, lists))
        if trial_types is not None:
            if nwb_data is None:
                raise ValueError("nwb_data is required for trial_types.")
            return {tt: np.asarray(find_trials(nwb_data, tt), dtype=int) for tt in trial_types}
        return {"all_trials": psth_da.coords[trial_coord].values}

    trial_groups = _group_ids_by_type()
    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = colors or cmap
    group_colors = {g: colors[i % len(colors)] for i, g in enumerate(trial_groups)}

    # For autoscaling global PSTH
    global_max = 0.0
    if y_mode == "auto_global":
        for unit in unit_indices:
            da = psth_da.sel(unit_index=unit)
            data = da.mean(trial_dim).values if plot_type == 'mean' else da.values
            global_max = max(global_max, np.nanmax(data))

    # Plot each unit in its own figure
    for unit in unit_indices:
        pos = int(np.where(psth_da.coords["unit_index"].values == unit)[0][0])
        unit_psth = psth_da.isel(unit=pos)
        unit_raster = raster_da.isel(unit=pos)

        fig, (ax_rast, ax_psth) = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=sharey)

        # Raster plot
        y_raster=0
        for label, tids in trial_groups.items():
            for ti, tval in enumerate(unit_raster.coords[trial_coord].values):
                if tval in tids:
                    y_raster = y_raster+1
                    spikes = unit_raster.isel({trial_dim: ti}).values
                    spikes = spikes[~np.isnan(spikes)]
                    ax_rast.vlines(spikes, y_raster, y_raster + 1, color=group_colors[label], alpha=0.7)
        ax_rast.axvline(0, color='k', ls='--', lw=0.8)  # event line
        ax_rast.set_ylabel('Trial')
        ax_rast.set_title(f'Unit {unit}')

        # PSTH plot
        unit_max = 0.0
        for label, tids in trial_groups.items():
            idxs = [i for i, tv in enumerate(unit_psth.coords[trial_coord].values) if tv in tids]
            data = unit_psth.isel({trial_dim: idxs}).values
            if plot_type == 'single':
                for trial_vec in data:
                    ax_psth.plot(times, trial_vec, color=group_colors[label], alpha=0.6)
                    unit_max = max(unit_max, np.nanmax(trial_vec))
            else:
                mean_f = np.nanmean(data, axis=0)
                sem_f = np.nanstd(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
                ax_psth.plot(times, mean_f, color=group_colors[label], label=label)
                ax_psth.fill_between(times, mean_f - sem_f, mean_f + sem_f, alpha=sem_alpha)
                unit_max = max(unit_max, np.nanmax(mean_f + sem_f))

        ax_psth.axvline(0, color='k', ls='--', lw=0.8)
        ax_psth.set_ylabel('Firing rate (spk/s)')
        ax_psth.set_xlabel('Time (s)')

        # Set x-limits to requested window
        if time_window is not None:
            ax_rast.set_xlim(*time_window)
            ax_psth.set_xlim(*time_window)

        # Set y-limits
        if y_mode == 'auto_per_unit' and unit_max > 0:
            ax_psth.set_ylim(0, unit_max * (1 + y_pad))
        elif y_mode == 'auto_global' and global_max > 0:
            ax_psth.set_ylim(0, global_max * (1 + y_pad))

        if plot_type == 'mean' and legend and len(trial_groups) > 1:
            ax_psth.legend(frameon=False)

        plt.tight_layout()

        if save_path:
            save_target = Path(save_path)
            # If save_path is a directory, create and save under unit-specific filename
            if save_target.suffix == '' or save_target.is_dir():
                save_target.mkdir(parents=True, exist_ok=True)
                fp = save_target / f"unit_{unit}.png"
            else:
                # Treat save_path as base filename, append unit ID
                base = save_target.with_suffix('')
                fp = base.parent / f"{base.name}_unit_{unit}.png"
            fig.savefig(fp, dpi=dpi, bbox_inches='tight')
            print(f"Saved Unit {unit} figure to {fp}")

        plt.show()









