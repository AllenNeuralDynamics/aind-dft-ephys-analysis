import numpy as np
import xarray as xr
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple, Optional, Union
from ephys_behavior import get_units_passed_default_qc
from behavior_utils import extract_event_timestamps, find_trials
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Any, Iterable, Tuple, Optional, Union

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
