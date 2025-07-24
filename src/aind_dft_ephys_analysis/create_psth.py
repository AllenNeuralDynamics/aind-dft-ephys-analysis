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



def load_psth_subset(
    source: Union[str, Path, xr.DataArray, xr.Dataset],
    trial_ids: Optional[Sequence[int]] = None,
    unit_ids: Optional[Sequence[int]] = None,
    time_window: Optional[Tuple[float, float]] = None,
    align_to_event: Optional[str] = None,
    consolidated: bool = True,
) -> xr.DataArray:
    """
    Return a view of the PSTH cube restricted to the requested trials,
    units and/or time range, with explicit control of which alignment event
    to use when multiple PSTHs are stored in the same Dataset.

    Parameters
    ----------
    source : str | Path | xr.DataArray | xr.Dataset
        • Path / string  → Zarr folder produced by `extract_neuron_psth_to_zarr`.  
        • xr.Dataset     → already-opened dataset (must contain variables named "psth_<event>").  
        • xr.DataArray   → the PSTH array itself.
    trial_ids : Sequence[int] | None
        Trial indices (values of the *trial_index_<event>* coordinate) to keep.
        None → keep all trials.
    unit_ids : Sequence[int] | None
        Unit indices (values of the *unit_index* coordinate) to keep.
        None → keep all units.
    time_window : (float, float) | None
        (start, end) in seconds. None → full time range.
    align_to_event : str | None
        Event name (without the "psth_" prefix) selecting which PSTH to use
        when `source` is a Dataset or Zarr path that contains multiple events.
        Ignored if `source` is already a DataArray.
    consolidated : bool, default True
        Use consolidated metadata when opening Zarr.

    Returns
    -------
    xr.DataArray
        Lazily-sliced PSTH array with attrs & coords preserved.

    Raises
    ------
    FileNotFoundError
        If a provided path does not exist.
    ValueError
        If multiple PSTHs are present but `align_to_event` is not given.
    KeyError
        If the requested PSTH variable is missing.
    """
    # ------------------------------------------------------------------ #
    # 1) Resolve a single PSTH DataArray                                 #
    # ------------------------------------------------------------------ #
    if isinstance(source, xr.DataArray):
        da = source
    else:
        # Open dataset if needed
        if isinstance(source, (str, Path)):
            zarr_path = Path(source).expanduser()
            if not zarr_path.exists():
                raise FileNotFoundError(f"Zarr folder not found: {zarr_path}")
            ds = xr.open_zarr(zarr_path, consolidated=consolidated)
        elif isinstance(source, xr.Dataset):
            ds = source
        else:
            raise TypeError("Unsupported type for `source`.")

        # figure out which variable to use
        psth_vars = [v for v in ds.data_vars if v.startswith("psth_")]
        if not psth_vars:
            raise KeyError("No variables beginning with 'psth_' found in the dataset.")

        if align_to_event is None:
            if len(psth_vars) == 1:
                var_name = psth_vars[0]
            else:
                raise ValueError(
                    f"Multiple PSTHs present {psth_vars}; please set `align_to_event=`."
                )
        else:
            var_name = f"psth_{align_to_event}"
            if var_name not in ds:
                raise KeyError(f"PSTH variable '{var_name}' not found. Available: {psth_vars}")

        da = ds[var_name]

    # ------------------------------------------------------------------ #
    # 2) Identify trial dim / coord names (event-specific)               #
    # ------------------------------------------------------------------ #
    # Expect dims: ("unit", "trial_<evt>", "time")
    trial_dim_candidates = [d for d in da.dims if d.startswith("trial_")]
    if len(trial_dim_candidates) != 1:
        # fallback to a generic "trial" dim if present
        trial_dim_candidates = [d for d in da.dims if d == "trial"]
        if len(trial_dim_candidates) != 1:
            raise ValueError("Could not determine the trial dimension name.")
    trial_dim = trial_dim_candidates[0]

    trial_coord_candidates = [c for c in da.coords if c.startswith("trial_index_")]
    if len(trial_coord_candidates) != 1:
        # fallback to generic
        trial_coord_candidates = [c for c in da.coords if c == "trial_index"]
        if len(trial_coord_candidates) != 1:
            raise ValueError("Could not determine the trial_index coordinate name.")
    trial_coord = trial_coord_candidates[0]

    # ------------------------------------------------------------------ #
    # 3) Apply trial / unit / time subsetting                            #
    # ------------------------------------------------------------------ #
    # Trial subset
    if trial_ids is not None:
        if trial_coord in da.coords and da.indexes.get(trial_coord) is not None:
            da = da.sel({trial_coord: list(trial_ids)})
        else:
            mask = np.isin(da.coords[trial_coord].values, trial_ids)
            da = da.isel({trial_dim: mask})

    # Unit subset
    if unit_ids is not None:
        if "unit_index" in da.coords and da.indexes.get("unit_index") is not None:
            da = da.sel(unit_index=list(unit_ids))
        else:
            mask = np.isin(da.coords["unit_index"].values, unit_ids)
            da = da.isel(unit=mask)

    # Time slice
    if time_window is not None:
        t0, t1 = time_window
        da = da.sel(time=slice(t0, t1))

    return da


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


def load_psth_from_zarr(
    zarr_path: Union[str, Path],
    as_object: Literal["dataarray", "dataset"] = "dataarray",
    align_to_event: Optional[str] = None,
    consolidated: bool = True,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Load a Zarr folder created by `extract_neuron_psth_to_zarr`.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the “*.zarr” folder.
    as_object : {"dataarray", "dataset"}, default "dataarray"
        * "dataset"   → return the whole Dataset (all events).
        * "dataarray" → return a single PSTH DataArray.
            - Multi‑event case: you must specify *event*.
            - Single‑event case: the sole variable is returned automatically.
    align_to_event : str or None, optional
        Event name (without the ``psth_`` prefix) selecting which PSTH to
        return when ``as_object == "dataarray"``.  Ignored if
        ``as_object == "dataset"``.  If omitted and exactly one PSTH
        variable exists, that one is used.
    consolidated : bool, default True
        Use consolidated metadata (faster when written with
        ``consolidated=True``).

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Dataset with all PSTHs, or the selected single PSTH DataArray.

    Raises
    ------
    FileNotFoundError
        If the Zarr path does not exist.
    KeyError
        If the requested event variable is missing.
    ValueError
        If multiple events exist but *event* is not provided.
    """
    zarr_path = Path(zarr_path).expanduser()
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr folder not found: {zarr_path}")

    ds = xr.open_zarr(zarr_path, consolidated=consolidated)

    if as_object == "dataset":
        return ds
    elif as_object == "dataarray":
        # collect variables that follow the multi‑event naming scheme
        psth_vars = [v for v in ds.data_vars if v.startswith("psth_")]

        if not psth_vars:
            raise KeyError(
                f"No variables starting with 'psth_' found in dataset at {zarr_path}."
            )

        if align_to_event is None:
            if len(psth_vars) == 1:
                var_name = psth_vars[0]
            else:
                raise ValueError(
                    "Multiple PSTH variables found "
                    f"{psth_vars} – please specify `align_to_event=`."
                )
        else:
            var_name = f"psth_{align_to_event}"
            if var_name not in ds:
                raise KeyError(
                    f"PSTH variable '{var_name}' not found. "
                    f"Available: {psth_vars}"
                )

        return ds[var_name]
    else:
        raise ValueError("`as_object` must be 'dataarray' or 'dataset'")



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
        See `load_psth_subset`.
    """
    # 1) Reuse the slicer (now event-aware)
    da_sub = load_psth_subset(
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


def plot_psth_for_units(
    source: Union[str, Path, xr.DataArray, xr.Dataset],
    *,
    unit_ids: Optional[Sequence[int]] = None,
    trial_ids: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
    trial_types: Optional[Sequence[str]] = None,
    nwb_data: Optional[Any] = None,
    align_to_event: Optional[str] = None,
    time_window: Optional[Tuple[float, float]] = None,
    plot_type: Literal["single", "mean"] = "single",
    colors: Optional[Sequence[str]] = None,
    sem_alpha: float = 0.3,
    figsize: Tuple[float, float] = (6.0, 2.5),
    sharey: bool = False,
    legend: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    consolidated: bool = True,
    group_labels: Optional[Sequence[str]] = None,
    y_mode: Literal["auto_per_unit", "auto_global", "none"] = "auto_per_unit",
    y_pad: float = 0.05,
) -> Figure:
    """
    Plot peri‑stimulus time histograms (PSTHs) for one or many units, with support for:
    - Multiple alignment events (`align_to_event`)
    - Explicit trial selection via `trial_ids` (flat or nested list)
    - Trial grouping by behavioural labels (`trial_types`, using `find_trials`)
    - Automatic y‑axis scaling (`y_mode`, `y_pad`)

    Parameters
    ----------
    source : str | Path | xr.DataArray | xr.Dataset
        Zarr path, Dataset, or DataArray containing PSTHs produced by `extract_neuron_psth_to_zarr`.
    unit_ids : Sequence[int] | None
        Values of the *unit_index* coordinate to plot. ``None`` → all units.
    trial_ids : list[int] | list[list[int]] | None
        Explicit trial indices. If nested (list of lists), each sublist is treated as a separate group.
        Empty sublists are allowed and skipped. If provided, `trial_types` is ignored.
    trial_types : list[str] | None
        Behavioural labels. If `trial_ids` is None, each label defines a trial group via `find_trials(nwb_data, tt)`.
    nwb_data : Any | None
        NWB handle needed when `trial_types` is used (to call `find_trials`).
    align_to_event : str | None
        Event name (without the "psth_" prefix) to select the PSTH variable when multiple exist.
    time_window : (float, float) | None
        Slice of the time axis in seconds, e.g. (-1.0, 3.0). ``None`` → full stored range.
    plot_type : {"single", "mean"}
        "single": overlay every trial curve; "mean": plot mean ± SEM per group.
    colors : list[str] | None
        Matplotlib colours. One colour per group (in "mean" mode) or reused per trial (in "single" mode).
        Defaults to Matplotlib's current colour cycle.
    sem_alpha : float
        Transparency for the SEM band in "mean" mode.
    figsize : (float, float)
        Size per subplot (width, height). Total height scales with the number of units.
    sharey : bool
        Share the y-axis among subplots (Matplotlib's `sharey` in `plt.subplots`).
    legend : bool
        Show a legend (only meaningful in "mean" mode with multiple groups).
    save_path : str | Path | None
        If provided, the figure is saved to this path (extension determines format).
    dpi : int
        DPI used when saving the figure.
    consolidated : bool
        Passed through to `xr.open_zarr` when `source` is a path.
    group_labels : list[str] | None
        Optional custom labels for groups when `trial_ids` is nested. Length must match number of groups.
    y_mode : {"auto_per_unit", "auto_global", "none"}
        How to auto-set y-limits:
          • "auto_per_unit"  → per-unit max (ignored if `sharey=True`; set `sharey=False` to differ).
          • "auto_global"    → single global max for all subplots.
          • "none"           → leave Matplotlib defaults.
    y_pad : float
        Fractional padding added to the max value when y autoscaling.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.

    Notes
    -----
    • Uses `load_psth_subset` (event-aware) to obtain PSTH slices lazily.
    • Automatically detects the event-specific trial dimension/coordinate names.
    • Empty trial groups are silently skipped.
    """
    # ------------------------------------------------------------------ #
    # Helpers to normalize trial grouping                                #
    # ------------------------------------------------------------------ #
    def _group_from_trial_ids(tids: Union[Sequence[int], Sequence[Sequence[int]], None]
                              ) -> dict[str, np.ndarray]:
        if tids is None:
            return {}
        # nested?
        if len(tids) > 0 and isinstance(tids[0], (list, tuple, np.ndarray)):
            groups = [np.asarray(g, dtype=int) for g in tids]
        else:
            groups = [np.asarray(tids, dtype=int)]
        groups = [g for g in groups if g.size > 0]  # drop empties
        names = group_labels if group_labels is not None else [f"group_{i}" for i in range(len(groups))]
        if len(names) != len(groups):
            raise ValueError("`group_labels` length must match number of non-empty groups.")
        return {n: g for n, g in zip(names, groups)}

    def _group_from_trial_types(ttypes: Optional[Sequence[str]]) -> dict[str, np.ndarray]:
        if ttypes is None:
            return {}
        if nwb_data is None:
            raise ValueError("`nwb_data` is required when using `trial_types`.")
        d = {str(tt): np.asarray(find_trials(nwb_data, tt), dtype=int) for tt in ttypes}
        d = {k: v for k, v in d.items() if v.size > 0}
        if not d:
            raise ValueError("No trials found for the requested trial_types.")
        return d

    if trial_ids is not None:
        trial_groups = _group_from_trial_ids(trial_ids)
    elif trial_types is not None:
        trial_groups = _group_from_trial_types(trial_types)
    else:
        trial_groups = {}

    # ------------------------------------------------------------------ #
    # 1) Load a base PSTH slice (no trial filter if we will split later) #
    # ------------------------------------------------------------------ #
    base_da = load_psth_subset(
        source,
        trial_ids=None,                 # groups handled later
        unit_ids=unit_ids,
        time_window=time_window,
        align_to_event=align_to_event,
        consolidated=consolidated,
    )
    if base_da.size == 0:
        raise ValueError("Selected subset is empty – nothing to plot.")

    # Identify trial dim/coord (event-specific)
    trial_dim_candidates = [d for d in base_da.dims if d.startswith("trial_")] or [d for d in base_da.dims if d == "trial"]
    if len(trial_dim_candidates) != 1:
        raise ValueError("Could not determine the trial dimension name.")
    trial_dim = trial_dim_candidates[0]

    trial_coord_candidates = [c for c in base_da.coords if c.startswith("trial_index_")] or [c for c in base_da.coords if c == "trial_index"]
    if len(trial_coord_candidates) != 1:
        raise ValueError("Could not determine the trial_index coordinate name.")
    trial_coord = trial_coord_candidates[0]

    n_units = base_da.sizes["unit"]
    times   = base_da.coords["time"].values

    # ------------------------------------------------------------------ #
    # 2) Colours                                                         #
    # ------------------------------------------------------------------ #
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = colors or default_cycle

    if trial_groups:
        group_names  = list(trial_groups.keys())
        group_colors = {g: colors[i % len(colors)] for i, g in enumerate(group_names)}
    else:
        group_names  = ["all_trials"]
        group_colors = {"all_trials": colors[0]}

    # ------------------------------------------------------------------ #
    # 3) Figure & axes                                                   #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(
        n_units,
        1,
        sharex=True,
        sharey=sharey,
        figsize=(figsize[0], figsize[1] * n_units),
        squeeze=False,
    )
    axes = axes.ravel()

    global_max = 0.0  # for global autoscaling

    # ------------------------------------------------------------------ #
    # 4) Plot per unit                                                   #
    # ------------------------------------------------------------------ #
    for idx, (ax, unit_da_full) in enumerate(zip(axes, base_da)):
        unit_idx_val = int(unit_da_full.coords["unit_index"])
        unit_max = 0.0

        active_groups = group_names if trial_groups else ["all_trials"]

        for gname in active_groups:
            if trial_groups:
                da_grp = load_psth_subset(
                    base_da,
                    trial_ids=trial_groups[gname],
                    unit_ids=[unit_idx_val],
                    time_window=None,
                    align_to_event=None,
                ).squeeze("unit")
            else:
                da_grp = unit_da_full

            n_trials_grp = da_grp.sizes[trial_dim]

            if plot_type == "single":
                for t_i in range(n_trials_grp):
                    fr = da_grp.isel({trial_dim: t_i}).values
                    unit_max = max(unit_max, np.nanmax(fr))
                    ax.plot(times, fr, color=group_colors[gname], lw=1.0, alpha=0.6)
            elif plot_type == "mean":
                fr_mat  = da_grp.values          # trials × time
                mean_fr = fr_mat.mean(axis=0)
                sem_fr  = fr_mat.std(axis=0, ddof=1) / np.sqrt(n_trials_grp)
                unit_max = max(unit_max, np.nanmax(mean_fr + sem_fr))

                ax.plot(times, mean_fr, color=group_colors[gname], lw=1.6, label=gname)
                ax.fill_between(times,
                                mean_fr - sem_fr,
                                mean_fr + sem_fr,
                                color=group_colors[gname],
                                alpha=sem_alpha)
            else:
                raise ValueError("plot_type must be 'single' or 'mean'")

        global_max = max(global_max, unit_max)

        # Cosmetics
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_ylabel(f"Unit {unit_idx_val}\nspk/s")
        ax.margins(x=0)

        # Per-unit ylim (effective only if sharey=False)
        if y_mode == "auto_per_unit" and not sharey and unit_max > 0:
            ax.set_ylim(0, unit_max * (1.0 + y_pad))

        if idx == 0:
            title = "Per‑trial PSTH" if plot_type == "single" else "Trial‑average PSTH (mean ± SEM)"
            if align_to_event is not None:
                title += f" • event={align_to_event}"
            ax.set_title(title)

        if idx == n_units - 1:
            ax.set_xlabel("Time (s)")

    # Global ylim
    if y_mode == "auto_global" and global_max > 0:
        for ax in axes:
            ax.set_ylim(0, global_max * (1.0 + y_pad))

    # Legend
    if legend and plot_type == "mean" and len(group_names) > 1:
        axes[0].legend(frameon=False, loc="upper right", fontsize="small")

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path).expanduser()
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".png")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path.resolve()}")

    return fig





