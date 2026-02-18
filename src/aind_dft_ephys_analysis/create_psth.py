from __future__ import annotations

# ==============================
# Standard library
# ==============================
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Sequence,
    Tuple,
    Optional,
    Union,
    Literal,
    Mapping,
    List,
    Dict,
)

# ==============================
# Third-party libraries
# ==============================
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ==============================
# Project-specific imports
# ==============================
from ephys_behavior import get_units_passed_default_qc
from behavior_utils import extract_event_timestamps, find_trials


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



def extract_neuron_psth_full_session_to_zarr(
    nwb_data: Any,
    units: Optional[Iterable[int]] = None,
    *,
    bin_size: float = 0.1,
    time_range: Optional[Tuple[float, float]] = None,
    t0_mode: Literal["none", "min_spike"] = "min_spike",
    save_folder: Union[str, Path] = "/root/capsule/results",
    save_name: Optional[str] = None,
    overwrite: bool = True,
) -> xr.Dataset:
    """
    Compute a full-session binned spike count matrix and firing-rate PSTH for a set of units,
    then save the result as a Zarr-backed xarray.Dataset.

    This function treats the entire recording as one continuous time axis (not trial-aligned).
    It reads spike timestamps from:
        nwb_data.units["spike_times"][unit_index]
    where each entry is expected to be an array-like of spike times (seconds).

    Parameters
    ----------
    nwb_data
        An NWB-like object that exposes `units["spike_times"]` and optionally `session_id`.

    units
        Iterable of unit indices to include. If None, uses `get_units_passed_default_qc(nwb_data)`
        to select units that pass a default QC filter.

    bin_size
        Bin width in seconds used to histogram spikes. PSTH firing rate is computed as:
            rate (Hz) = counts / bin_size

    time_range
        (t_start, t_end) for the histogram time axis *after* applying the `t0_mode` shift.
        - If None: inferred from global min/max spike time across ALL selected units.
        - If provided: used as-is (assumed to match the spike time base after t0 shift).

    t0_mode
        Controls whether spike times are shifted before histogramming:
        - "min_spike": subtract global minimum spike time (across selected units), so time starts near 0.
        - "none": do not shift spike times (t0 = 0).

    save_folder
        Folder where the Zarr dataset will be written.

    save_name
        Name of the Zarr store. If None, defaults to "{session_id}_full_psth.zarr".
        If provided without ".zarr" suffix, the suffix is appended.

    overwrite
        If True and the destination exists, deletes the existing Zarr store before writing.

    Returns
    -------
    xr.Dataset
        Dataset with:
          - counts_full: int32 array, shape (n_units, n_bins)
          - psth_full: float32 array (Hz), shape (n_units, n_bins)
        Coordinates:
          - unit_index: unit ids/indices used (len = n_units)
          - time: bin centers in seconds (len = n_bins)
        Attributes include session_id, bin_size, time_range, raw spike time range, and t0 info.
    """

    # ---------------------------------------------------------------------
    # 1) Determine which units to include
    # ---------------------------------------------------------------------
    # If the caller didn't pass units explicitly, select units using a default QC rule.
    if units is None:
        units = get_units_passed_default_qc(nwb_data)

    # Convert to a concrete list so we can:
    #   - compute length
    #   - iterate multiple times
    #   - preserve ordering
    units = list(units)

    # Defensive check: a PSTH over "no units" is meaningless.
    if len(units) == 0:
        raise ValueError("No units selected.")

    # ---------------------------------------------------------------------
    # 2) Load spike trains and compute GLOBAL min/max across ALL selected units
    # ---------------------------------------------------------------------
    # We store each unit's spike train to avoid reading from NWB twice.
    spike_trains: List[np.ndarray] = []

    # Track the minimum and maximum spike time observed across the selected units.
    # These are used to infer session-wide time_range when time_range=None.
    global_min = np.inf
    global_max = -np.inf

    for u in units:
        # Convert spike times to a float numpy array.
        # Assumes nwb_data.units["spike_times"][u] is array-like (possibly ragged lists).
        st = np.asarray(nwb_data.units["spike_times"][u], dtype=float)
        spike_trains.append(st)

        # Update global min/max only if this unit actually has spikes.
        if st.size > 0:
            smin = float(st.min())
            smax = float(st.max())
            if smin < global_min:
                global_min = smin
            if smax > global_max:
                global_max = smax

    # If all spike trains were empty (or invalid), global_min/max remain +/- inf.
    # Also reject degenerate ranges where max <= min.
    if not np.isfinite(global_min) or not np.isfinite(global_max) or global_max <= global_min:
        raise ValueError("Could not infer global min/max spike time across units (all empty?).")

    # ---------------------------------------------------------------------
    # 3) Choose the reference t0 (time shift) based on t0_mode
    # ---------------------------------------------------------------------
    # t0 is subtracted from all spike times before histogramming.
    # "min_spike" shifts the earliest spike in the selected set to ~0.
    if t0_mode == "min_spike":
        t0 = global_min
    elif t0_mode == "none":
        t0 = 0.0
    else:
        raise ValueError("t0_mode must be one of {'none','min_spike'}")

    # ---------------------------------------------------------------------
    # 4) Determine histogram time_range in the shifted time base
    # ---------------------------------------------------------------------
    # If user didn't provide time_range, infer it from the global min/max (raw),
    # then convert into the post-shift time base by subtracting t0.
    if time_range is None:
        t_start = global_min - t0
        t_end = global_max - t0
    else:
        # If time_range is provided, we assume it is already in the same time base
        # as spike times after shifting by t0 (i.e., it matches st - t0).
        t_start, t_end = float(time_range[0]), float(time_range[1])

    # Validate final range (must be finite and have positive width).
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        raise ValueError(f"Invalid time_range: ({t_start}, {t_end})")

    # ---------------------------------------------------------------------
    # 5) Build histogram bin edges and bin centers
    # ---------------------------------------------------------------------
    # Number of bins: ceil ensures we cover the full span even if not divisible by bin_size.
    n_bins = int(np.ceil((t_end - t_start) / bin_size))

    # Bin edges: length = n_bins + 1
    # Edges are [t_start, t_start+bin_size, ..., t_start+n_bins*bin_size]
    edges = t_start + np.arange(n_bins + 1, dtype=float) * bin_size

    # Bin centers: used as the time coordinate in the output.
    centers = edges[:-1] + bin_size / 2.0

    # ---------------------------------------------------------------------
    # 6) Histogram spikes for each unit
    # ---------------------------------------------------------------------
    # counts: (n_units, n_bins)
    counts = np.zeros((len(units), n_bins), dtype=np.int32)

    for ui, st in enumerate(spike_trains):
        # Skip units with no spikes.
        if st.size == 0:
            continue

        # Shift spike timestamps by t0 (either 0 or global_min).
        st = st - t0

        # Restrict spikes to the desired [t_start, t_end] range before histogramming.
        # Note: np.histogram uses half-open bins [edge_i, edge_{i+1}) except the last bin,
        # which includes the right edge. Here we filter using <= t_end which is consistent
        # with allowing spikes exactly at t_end to potentially land in the last bin.
        st = st[(st >= t_start) & (st <= t_end)]
        if st.size == 0:
            continue

        # Histogram: returns counts per bin (length n_bins).
        c, _ = np.histogram(st, bins=edges)

        # Store as int32 without unnecessary copies.
        counts[ui, :] = c.astype(np.int32, copy=False)

    # Convert to firing rate (Hz) as float32 for compactness.
    psth = counts.astype(np.float32) / float(bin_size)

    # ---------------------------------------------------------------------
    # 7) Package results into an xarray.Dataset
    # ---------------------------------------------------------------------
    ds = xr.Dataset(
        data_vars={
            # Raw binned spike counts
            "counts_full": (("unit", "time"), counts),
            # Firing rate (Hz)
            "psth_full": (("unit", "time"), psth),
        },
        coords={
            # The original unit indices (from NWB indexing) for each row
            "unit_index": ("unit", units),
            # Bin centers for the time coordinate
            "time": ("time", centers),
        },
        attrs={
            # Helpful metadata for provenance / reproducibility
            "session_id": getattr(nwb_data, "session_id", "unknown"),
            "bin_size": float(bin_size),
            # Time range in the post-shift time base used for edges/centers
            "time_range": (float(t_start), float(t_end)),
            # Raw time range in the original spike time base (before subtracting t0)
            "global_spike_time_range_raw": (float(global_min), float(global_max)),
            "t0_mode": t0_mode,
            "t0_subtracted": float(t0),
            "n_units": int(len(units)),
            "created_with": "extract_neuron_psth_full_session_to_zarr",
        },
    )

    # ---------------------------------------------------------------------
    # 8) Save the dataset to a Zarr store on disk
    # ---------------------------------------------------------------------
    session_id = getattr(nwb_data, "session_id", "session")

    # If the caller didn't provide a save_name, build a default.
    if save_name is None:
        save_name = f"{session_id}_full_psth.zarr"
    elif not str(save_name).endswith(".zarr"):
        # Ensure the name ends in .zarr for consistency.
        save_name = f"{save_name}.zarr"

    # Destination path: expand ~ and join folder/name.
    dest = Path(save_folder).expanduser() / str(save_name)

    # Overwrite behavior: remove existing directory-based Zarr store.
    if dest.exists() and overwrite:
        import shutil
        shutil.rmtree(dest)

    # Write the dataset. consolidated=True writes consolidated metadata
    # (faster opening in many workflows).
    ds.to_zarr(dest, mode="w", consolidated=True)

    print(
        f"Saved full-session PSTH to {dest} "
        f"[t0_mode={t0_mode}, time_range={ds.attrs['time_range']}]"
    )

    return ds


