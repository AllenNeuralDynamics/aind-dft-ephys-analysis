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
    Compute per‑trial PSTHs for one or many alignment events and persist them as Zarr.

    Each alignment event produces an independent PSTH cube stored as a separate
    variable inside the returned Dataset. Because different events can have
    different numbers of trials, each PSTH uses a **unique trial dimension**
    name: ``trial_<event>`` and a matching coordinate ``trial_index_<event>``.

    Parameters
    ----------
    nwb_data : Any
        NWB handle that provides:
        * ``units['spike_times']`` – list‑of‑arrays (one array per unit)
        * ``session_id`` – recording identifier (attribute)
        * behavioural event data accessible through
          :func:`extract_event_timestamps` and :func:`find_trials`.
    units : Iterable[int] or None, optional
        Explicit unit indices to include. ``None`` → automatically select all
        units that pass QC via :func:`get_units_passed_default_qc`.
    align_to_event : str or Sequence[str], default "go_cue"
        Single event name or a sequence of event names. For each event a
        variable named ``psth_<event>`` is created.
    time_window : (float, float), default (-3.0, 5.0)
        Time window relative to each event timestamp (seconds).
        The first value is the start (negative allowed), the second the end.
    bin_size : float, default 0.05
        Histogram bin width in seconds. Firing rate is reported as
        ``counts / bin_size`` (spikes per second).
    trial_type : str or None, optional
        Behavioural trial type label used to filter trials. ``None`` keeps all
        trials for each event.
    save_folder : str or Path, default "/root/capsule/results"
        Directory where the Zarr folder will be written (created if missing).
    save_name : str or None, optional
        Output folder name. When ``None`` uses ``<session_id>.zarr``.
        ``.zarr`` suffix is appended if absent.
    overwrite : bool, default True
        If ``True`` and the destination folder already exists, it is deleted
        before writing.

    Returns
    -------
    xarray.Dataset
        Dataset containing one DataArray per event (variables
        ``psth_<event>``). Each DataArray has dimensions
        ``("unit", "trial_<event>", "time")`` and coordinates:
        * ``unit_index`` – unit indices
        * ``trial_index_<event>`` – integer trial indices for that event
        * ``time`` – bin centres (seconds)

        Dataset‑level attributes include ``align_to_events``,
        ``n_trials_per_event``, and other metadata.

    Raises
    ------
    ValueError
        If ``align_to_event`` is empty or if no trials match the criteria for
        a given event.
    """
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
    centres = edges[:-1] + bin_size / 2.0

    data_vars = {}
    n_trials_per_event = {}

    for evt in events:
        all_event_times = np.asarray(extract_event_timestamps(nwb_data, evt), dtype=float)
        if trial_type is not None:
            trial_index = np.asarray(find_trials(nwb_data, trial_type), dtype=int)
            event_times = all_event_times[trial_index]
        else:
            event_times = all_event_times
            trial_index = np.arange(len(event_times), dtype=int)

        if len(event_times) == 0:
            raise ValueError(f"No trials matched criteria for event '{evt}'.")

        psth = np.zeros((len(units), len(event_times), len(centres)), dtype=np.float32)
        for ui, u in enumerate(units):
            spikes = np.asarray(nwb_data.units["spike_times"][u], dtype=float)
            for ti, t0 in enumerate(event_times):
                rel = spikes - t0
                counts, _ = np.histogram(rel, bins=edges)
                psth[ui, ti, :] = counts / bin_size

        trial_dim = f"trial_{evt}"
        da = xr.DataArray(
            psth,
            dims=("unit", trial_dim, "time"),
            coords={
                "unit_index": ("unit", units),
                f"trial_index_{evt}": (trial_dim, trial_index),
                "time": ("time", centres),
            },
            name=f"psth_{evt}",
            attrs={
                "session_id": getattr(nwb_data, "session_id", "unknown"),
                "align_to_event": evt,
                "time_window": time_window,
                "bin_size": bin_size,
                "trial_type": trial_type if trial_type else "all_trials",
                "n_units": len(units),
                "n_trials": len(event_times),
                "created_with": "extract_neuron_psth_to_zarr",
            },
        )
        data_vars[f"psth_{evt}"] = da
        n_trials_per_event[evt] = len(event_times)

    ds = xr.Dataset(data_vars)
    ds.attrs.update(
        {
            "session_id": getattr(nwb_data, "session_id", "unknown"),
            "align_to_events": events,
            "time_window": time_window,
            "bin_size": bin_size,
            "trial_type": trial_type if trial_type else "all_trials",
            "n_units": len(units),
            "n_trials_per_event": n_trials_per_event,
            "created_with": "extract_neuron_psth_to_zarr",
        }
    )

    session_id = getattr(nwb_data, "session_id", "unknown_session")
    if save_name is None:
        save_name = f"{session_id}.zarr"
    elif not save_name.endswith(".zarr"):
        save_name += ".zarr"

    save_folder = Path(save_folder).expanduser()
    save_folder.mkdir(parents=True, exist_ok=True)
    dest = save_folder / save_name
    if dest.exists() and overwrite:
        import shutil
        shutil.rmtree(dest)
    ds.to_zarr(dest, mode="w", consolidated=True)
    print(f"PSTHs saved to {dest}  [events={events}]")
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
                    f"{psth_vars} – please specify `event=`."
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


# ─────────────────────────────────────────────────────────────────────────────
# Helper ── plot PSTHs for a set of units with either per‑trial curves
#           or trial‑average (mean ± SEM) curves.
# ─────────────────────────────────────────────────────────────────────────────
def plot_psth_for_units(
    source: Union[str, Path, xr.DataArray, xr.Dataset],
    *,
    unit_ids: Optional[Sequence[int]] = None,
    trial_ids: Optional[Sequence[int]] = None,
    time_window: Optional[Tuple[float, float]] = None,
    plot_type: Literal["single", "mean"] = "single",
    colors: Optional[Sequence[str]] = None,
    sem_alpha: float = 0.3,
    figsize: Tuple[float, float] = (6.0, 2.5),
    sharey: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
):
    """
    Plot peri‑stimulus time histograms (PSTHs) for one or many units.

    Parameters
    ----------
    source : str | Path | xr.DataArray | xr.Dataset
        Location of the PSTHs.
        * **str/Path** — path to a “*.zarr” folder written by
          `extract_neuron_psth_to_zarr`.
        * **xr.Dataset** — an already‑opened dataset; must contain the
          variable ``"psth"``.
        * **xr.DataArray** — the “psth” array itself.
    unit_ids : Sequence[int] | None, default None
        • Explicit list of *unit_index* values to plot.  
        • **None** → plot **all** units present after filtering.
    trial_ids : Sequence[int] | None, optional
        Trial identifiers (values of the *trial_index* coordinate) to keep.
        ``None`` keeps all trials.
    time_window : (float, float) | None, optional
        Slice of the time axis in seconds, e.g. ``(-1.0, 3.0)``.
        ``None`` → use the full window stored in the PSTH.
    plot_type : {"single", "mean"}, default "single"
        • ``"single"`` — overlay every selected trial.  
        • ``"mean"``   — plot the trial‑average curve and shade ± SEM.
    colors : Sequence[str] | None, optional
        Matplotlib colours.  If ``None`` the default colour cycle is used.
        In ``"single"`` mode colours cycle over trials; in ``"mean"`` mode
        the first colour is used for mean ± SEM.
    sem_alpha : float, default 0.3
        Alpha (transparency) for the SEM band when ``plot_type == "mean"``.
    figsize : (float, float), default (6, 2.5)
        Size *per subplot* in inches – ``(width, height)``.  The total
        figure height scales with the number of units plotted.
    sharey : bool, default True
        Share the y‑axis among subplots.
    save_path : str | Path | None, optional
        If provided, save the figure to this path.  Extension determines
        format (``.png``, ``.pdf``, ``.svg`` …).  When omitted, the plot is
        not saved automatically.
    dpi : int, default 300
        Resolution of the saved figure (only used if *save_path* is given).

    Returns
    -------
    matplotlib.figure.Figure
        Handle to the created figure (handy for further tweaking).

    Notes
    -----
    * Relies on ``load_psth_subset`` to slice the PSTH cube.
    * Draws a vertical dashed line at *t = 0* (alignment marker) in every
      subplot.
    """
    # ------------------------------------------------------------------ #
    # 1) Obtain the desired PSTH slice (lazy)                            #
    # ------------------------------------------------------------------ #
    psth_da = load_psth_subset(
        source,
        trial_ids=trial_ids,
        unit_ids=unit_ids,      # may be None → keep all units
        time_window=time_window,
    )
    if psth_da.size == 0:
        raise ValueError("Selected subset is empty – nothing to plot.")

    n_units  = psth_da.sizes["unit"]
    n_trials = psth_da.sizes["trial"]
    times    = psth_da.coords["time"].values

    # ------------------------------------------------------------------ #
    # 2) Figure scaffolding                                              #
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

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = colors or default_colors

    # ------------------------------------------------------------------ #
    # 3) Plot per‑unit panels                                            #
    # ------------------------------------------------------------------ #
    for idx, (ax, unit_da) in enumerate(zip(axes, psth_da)):
        unit_idx = int(unit_da.coords["unit_index"])

        if plot_type == "single":
            # overlay every trial
            for t_i in range(n_trials):
                fr = unit_da.isel(trial=t_i).values
                ax.plot(times, fr,
                        color=colors[t_i % len(colors)],
                        lw=1.0, alpha=0.7)
        elif plot_type == "mean":
            # mean ± SEM
            fr_mat  = unit_da.values                     # trials × time
            mean_fr = fr_mat.mean(axis=0)
            sem_fr  = fr_mat.std(axis=0, ddof=1) / np.sqrt(n_trials)

            ax.plot(times, mean_fr, color=colors[0], lw=1.5)
            ax.fill_between(times, mean_fr - sem_fr, mean_fr + sem_fr,
                            color=colors[0], alpha=sem_alpha)
        else:
            raise ValueError("plot_type must be 'single' or 'mean'")

        # cosmetic tweaks
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_ylabel(f"Unit {unit_idx}\nspk/s")
        ax.margins(x=0)

        if idx == 0:
            title = "Per‑trial PSTH" if plot_type == "single" \
                    else "Trial‑average PSTH (mean ± SEM)"
            ax.set_title(title)
        if idx == n_units - 1:
            ax.set_xlabel("Time (s)")

    plt.tight_layout()

    # ------------------------------------------------------------------ #
    # 4) Optional save                                                   #
    # ------------------------------------------------------------------ #
    if save_path is not None:
        save_path = Path(save_path).expanduser()
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".png")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path.resolve()}")

    return fig



