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

def extract_neuron_psth_to_zarr(
    nwb_data: Any,
    units: Optional[Iterable[int]] = None,
    *,
    align_to_event: str = "go_cue",
    time_window: Tuple[float, float] = (-3.0, 5.0),
    bin_size: float = 0.05,
    trial_type: Optional[str] = None,
    save_folder: Union[str, Path] = "/root/capsule/results",
    save_name: Optional[str] = None, 
    overwrite: bool = True,
) -> xr.DataArray:
    """
    Compute per‑trial PSTHs for the requested units and persist them as Zarr.

    Parameters
    ----------
    nwb_data : Any
        NWB handle that provides
        • ``units['spike_times']`` – list‑of‑arrays, one per unit
        • ``session_id``           – recording identifier
        • behavioural event data accessible by ``extract_event_timestamps``.
    units : iterable[int] or None
        Unit indices to include; None → all units that pass QC.
    align_to_event, time_window, bin_size, trial_type
        Standard PSTH parameters (see previous version).
    save_folder : str or Path, default "/root/capsule/results"
        Directory in which the Zarr folder will be created.
    save_name : str or None
        • None → use `<session_id>.zarr`
        • Otherwise use the given string (".zarr" appended if missing)..
    overwrite : bool, default True
        Overwrite an existing destination folder if it exists.

    Returns
    -------
    xr.DataArray
        The PSTH array with dimensions ('unit', 'trial', 'time').
    """
    # ──────────────── 1  resolve units ────────────────
    if units is None:
        units = get_units_passed_default_qc(nwb_data)
    units = list(units)

    # ──────────────── 2  alignment times ─────────────
    all_event_times = np.asarray(
        extract_event_timestamps(nwb_data, align_to_event), dtype=float
    )
    if trial_type is not None:
        trial_mask   = find_trials(nwb_data, trial_type)
        event_times  = all_event_times[trial_mask]
        trial_index  = np.nonzero(trial_mask)[0]
    else:
        event_times  = all_event_times
        trial_index  = np.arange(len(event_times), dtype=int)

    if len(event_times) == 0:
        raise ValueError("No trials matched your criteria.")

    # ──────────────── 3  binning grid ───────────────
    start, end = time_window
    edges   = np.arange(start, end + bin_size, bin_size)
    centres = edges[:-1] + bin_size / 2.0

    # ──────────────── 4  build PSTH cube ────────────
    psth = np.zeros((len(units), len(event_times), len(centres)), dtype=np.float32)
    for ui, u in enumerate(units):
        spikes = np.asarray(nwb_data.units["spike_times"][u], dtype=float)
        for ti, t0 in enumerate(event_times):
            rel = spikes - t0
            counts, _ = np.histogram(rel, bins=edges)
            psth[ui, ti, :] = counts / bin_size  # spikes / s

    # ──────────────── 5  wrap in xarray ─────────────
    da = xr.DataArray(
        psth,
        dims   = ("unit", "trial", "time"),
        coords = {
            "unit_index": ("unit", units),
            "trial_index": ("trial", trial_index),
            "time": ("time", centres),
        },
        name   = "psth",
        attrs  = {
            "session_id": getattr(nwb_data, "session_id", "unknown"),
            "align_to_event": align_to_event,
            "time_window":   time_window,
            "bin_size":      bin_size,
            "trial_type":    trial_type if trial_type else "all_trials",
            "n_units":       len(units),
            "n_trials":      len(event_times),
            "created_with":  "extract_neuron_psth_to_zarr",
        },
    )

    # ──────────────── 6  save to Zarr ───────────────
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

    da.to_dataset().to_zarr(dest, mode="w", consolidated=True)
    print(f"PSTH saved to {dest}  [shape={psth.shape}]")

    return da

def load_psth_subset(
    source: Union[str, Path, xr.DataArray, xr.Dataset],
    trial_ids: Optional[Sequence[int]] = None,
    *,
    unit_ids: Optional[Sequence[int]] = None,
    time_window: Optional[Tuple[float, float]] = None,
) -> xr.DataArray:
    """
    Return a view of the PSTH cube restricted to the requested trials,
    units and/or time range.

    Parameters
    ----------
    source : str | Path | xr.DataArray | xr.Dataset
        • Path / string  → Zarr folder created by `extract_neuron_psth_to_zarr`.  
        • `xr.Dataset`   → loaded dataset (must contain variable *psth*).  
        • `xr.DataArray` → loaded *psth* array.
    trial_ids : Sequence[int] | None
        Trial indices (i.e. values of the *trial* coordinate) to keep.
        `None` keeps all trials.
    unit_ids : Sequence[int] | None, keyword‑only
        Unit indices (values of the *unit* coordinate) to keep.
        `None` keeps all units.
    time_window : (float, float) | None, keyword‑only
        `(start, end)` in seconds relative to the alignment event.
        Must fall within the original PSTH window.  `None` keeps all bins.

    Returns
    -------
    xr.DataArray
        A **view** (lazy slice) of the PSTH array; attrs & coords preserved.
    """
    # ------------------------------------------------------------------ #
    # 1  resolve DataArray                                               #
    # ------------------------------------------------------------------ #
    if isinstance(source, xr.DataArray):
        da = source
    elif isinstance(source, xr.Dataset):
        if "psth" not in source:
            raise KeyError("'psth' variable not found in the supplied Dataset")
        da = source["psth"]
    else:  # assume path‑like
        zarr_path = Path(source).expanduser()
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr folder not found: {zarr_path}")
        da = xr.open_zarr(zarr_path)["psth"]

    # ------------------------------------------------------------------ #
    # 2  trial subset                                                    #
    # ------------------------------------------------------------------ #
    if trial_ids is not None:
        da = da.sel(trial=trial_ids)

    # ------------------------------------------------------------------ #
    # 3  unit subset                                                     #
    # ------------------------------------------------------------------ #
    if unit_ids is not None:
        da = da.sel(unit=unit_ids)

    # ------------------------------------------------------------------ #
    # 4  time‑window slice                                               #
    # ------------------------------------------------------------------ #
    if time_window is not None:
        t0, t1 = time_window
        da = da.sel(time=slice(t0, t1))

    return da



def load_psth_from_zarr(
    zarr_path: Union[str, Path],
    *,
    as_object: Literal["dataarray", "dataset"] = "dataarray",
    consolidated: bool = True,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Load a Zarr folder created by `extract_neuron_psth_to_zarr`.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the “*.zarr” folder.
    as_object : {"dataarray", "dataset"}, default "dataarray"
        • "dataarray" → return the single variable **psth** (dims: unit×trial×time)  
        • "dataset"   → return the whole Dataset (identical to `.to_dataset()` output)
    consolidated : bool, default True
        Use the consolidated metadata path (faster when the folder was written
        with `consolidated=True`, which is the default of the extractor).

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The entire PSTH cube with coordinates and attributes intact.
    """
    zarr_path = Path(zarr_path).expanduser()
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr folder not found: {zarr_path}")

    ds = xr.open_zarr(zarr_path, consolidated=consolidated)

    if as_object == "dataset":
        return ds
    elif as_object == "dataarray":
        if "psth" not in ds:
            raise KeyError(
                f"'psth' variable not found in dataset at {zarr_path}. "
                "Did you pass the correct Zarr folder?"
            )
        return ds["psth"]
    else:
        raise ValueError("`as_object` must be 'dataarray' or 'dataset'")
