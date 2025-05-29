import os
import glob
import json

from typing import Union, Sequence, Optional, Any, Dict, Iterable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def save_ccf_locations_to_json(
    units_table: Any,
    save_path: str,
    selected_units: Iterable[int] = None
):
    """
    Save CCF-location info for a subset of units to a JSON file.

    Parameters
    ----------
    units_table : pd.DataFrame-like
        The NWB units table (e.g. `self.nwb_ephys_data.units[:]`). Must have a
        'ccf_location' column of dicts.
    save_path : str
        Full path (including filename) where the JSON will be written.
    selected_units : iterable of int, optional
        Absolute unit IDs (i.e. the DataFrame index) to include. If None, saves all units.
    """
    # 1) ensure directory exists
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # 2) report what we're about to do
    print("Current working directory:", os.getcwd())
    print(f"Writing JSON to: {save_path}")

    out = {}
    for unit_id, row in units_table.iterrows():
        if selected_units is not None and unit_id not in selected_units:
            continue
        loc = row.get('ccf_location', {})
        if not isinstance(loc, dict):
            continue
        entry = {k: loc[k] for k in (
            'x','y','z','axial','lateral','brain_region_id','brain_region'
        ) if k in loc}
        out[f"unit_{unit_id}"] = entry

    try:
        with open(save_path, 'w') as fp:
            json.dump(out, fp, indent=2)
    except Exception as e:
        print("Error saving JSON:", e)
        return

    print(f"✔ Wrote {len(out)} units"
          f"{' (filtered)' if selected_units is not None else ''} to {save_path}")



def load_ccf_channel_locations(
    session_name: str,
    probe_name: str,
    shank_id: int,
    alignment_root: str = '/root/capsule/data/IBL_alignment'
) -> Dict[str, Any]:
    """
    Load the ccf_channel_locations_shank{shank_id}.json for a given session.
    Returns the parsed JSON as a dict, or an empty dict if not found.

    Parameters
    ----------
    session_name : str
        e.g. '753126_2024-10-15_12-20-35'
    probe_name : str
        e.g. 'ProbeB'
    shank_id : int
        e.g. 0, 1, 2, or 3
    alignment_root : str
        Base path to the IBL_alignment folder.

    Returns
    -------
    Dict[str, Any]
        Loaded JSON data, or {} if the file was not found.
    """
    # Build the glob pattern to find the JSON file
    pattern = os.path.join(
        alignment_root,
        'result-*',
        f'ecephys_{session_name}',
        probe_name,
        f'ccf_channel_locations_shank{shank_id}.json'
    )
    matches = glob.glob(pattern)
    if not matches:
        # No file found
        return {}

    # Load and return the first matching JSON
    file_path = matches[0]
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_channel_info(
    ccf_data: Dict[str, Any],
    channel_id: Union[int, str]
) -> Dict[str, Any]:
    """
    Extract the info for a specific channel from the loaded ccf JSON data.

    Parameters
    ----------
    ccf_data : Dict[str, Any]
        The JSON data loaded from ccf_channel_locations_shankX.json, e.g.:
        {
          "channel_0": { ... },
          "channel_1": { ... },
          ...
        }
    channel_id : int or str
        - If int, the channel index (e.g. 0, 1, 2, ...).
        - If str, the full key (e.g. "channel_2").

    Returns
    -------
    Dict[str, Any]
        The sub-dictionary for that channel, or {} if not found.
    """
    # Determine the lookup key
    key = f"channel_{channel_id}" if isinstance(channel_id, int) else channel_id

    # Return the channel info or empty dict if missing
    return ccf_data.get(key, {})


def find_best_electrode(
    waveform_mean: np.ndarray,
    unit_index: Optional[Union[int, Sequence[int]]] = None
) -> Union[int, np.ndarray]:
    """
    For given unit(s), compute trough-to-peak distance on each electrode
    and return the electrode index(es) with the maximal distance.

    Parameters
    ----------
    waveform_mean : np.ndarray
        Array of shape (n_units, n_timepoints, n_electrodes)
    unit_index : int or sequence of int or None, default=None
        - If int, returns a single electrode index.
        - If sequence of ints, returns an array of electrode indices, one per unit.
        - If None or empty sequence, processes all units and returns a 1D array
          of length n_units.

    Returns
    -------
    int or np.ndarray
        Electrode index (if single unit) or array of electrode indices.
    """
    # determine which units to process
    n_units = waveform_mean.shape[0]
    if unit_index is None:
        indices = np.arange(n_units)
    elif isinstance(unit_index, (list, tuple, np.ndarray)):
        if len(unit_index) == 0:
            indices = np.arange(n_units)
        else:
            indices = np.array(unit_index, dtype=int)
    else:
        indices = np.array([int(unit_index)], dtype=int)

    # pre-allocate result
    best_ch = np.empty(indices.shape, dtype=int)

    # compute for each requested unit
    for i, u in enumerate(indices):
        unit_wave = waveform_mean[u]          # shape (timepoints, electrodes)
        troughs   = unit_wave.min(axis=0)
        peaks     = unit_wave.max(axis=0)
        tp_dist   = peaks - troughs
        best_ch[i] = int(np.argmax(tp_dist))

    # if they asked for a single unit_index int, return an int
    if best_ch.size == 1 and not isinstance(unit_index, (list, tuple, np.ndarray)):
        return int(best_ch[0])
    return best_ch

def plot_unit_waveforms(
    waveform_mean: np.ndarray,
    unit_index: int,
    highlight_electrode: int = None
):
    """
    Plot all electrode waveforms for a given unit, highlighting one channel.
    
    Parameters
    ----------
    waveform_mean : np.ndarray
        Array of shape (n_units, n_timepoints, n_electrodes)
    unit_index : int
        Index of the unit to plot.
    highlight_electrode : int, optional
        Electrode index to highlight. If None, no channel is highlighted.
    """
    unit_wave = waveform_mean[unit_index]        # shape (timepoints, electrodes)
    n_time, n_ch = unit_wave.shape
    time_axis = np.arange(n_time)
    
    plt.figure(figsize=(10, 5))
    # plot every channel in light grey
    for ch in range(n_ch):
        plt.plot(time_axis, unit_wave[:, ch],
                 color='lightgrey', linewidth=0.8)
    
    if highlight_electrode is not None:
        # overplot the highlighted channel in bold
        plt.plot(time_axis,
                 unit_wave[:, highlight_electrode],
                 color='C0', linewidth=2,
                 label=f'Electrode {highlight_electrode}')
        plt.legend()
    
    plt.title(f'Unit {unit_index}: Waveforms across {n_ch} electrodes')
    plt.xlabel('Timepoint')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def cluster_estimated_x(
    estimated_x: np.ndarray,
    n_clusters: int = 4,
    threshold: float = 0.5,
    plot: bool = False,
    random_state: int = 0
) -> np.ndarray:
    """
    Cluster 1D electrode positions into n_clusters, re-label so clusters
    are ordered by their center, and mark ambiguous units as label 5.

    Parameters
    ----------
    estimated_x : np.ndarray
        1D array of shape (n_units,) of x-positions.
    n_clusters : int, default=4
        Number of KMeans clusters to form.
    threshold : float, default=0.5
        Relative gap threshold: if (second_best - best)/best < threshold,
        unit is ambiguous.
    plot : bool, default=False
        If True, scatter-plot the units colored by final label,
        marking ambiguous (5) with 'x'.
    random_state : int, default=0
        Random seed for KMeans.

    Returns
    -------
    final_labels : np.ndarray of int, shape (n_units,)
        Cluster labels 0…n_clusters−1, with ambiguous units set to 5.
    """
    # ensure 1D array
    x = np.asarray(estimated_x).reshape(-1)
    N = x.size

    # run KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    raw_labels = kmeans.fit_predict(x.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()

    # sort cluster IDs by ascending center
    order = np.argsort(centers)
    label_map = {old: new for new, old in enumerate(order)}
    sorted_labels = np.vectorize(label_map.get)(raw_labels)
    sorted_centers = centers[order]

    # compute distances to each sorted center
    dists = np.abs(x.reshape(N, 1) - sorted_centers.reshape(1, -1))
    closest, second = np.partition(dists, 1, axis=1)[:, 0], np.partition(dists, 1, axis=1)[:, 1]

    # ambiguous if gap is small relative to best
    ambiguous = (second - closest) / closest < threshold

    # assign final labels (ambiguous → 5)
    final_labels = sorted_labels.copy()
    final_labels[ambiguous] = 5

    # optional plotting
    if plot:
        plt.figure(figsize=(10, 4))
        for cid in list(range(n_clusters)) + [5]:
            mask = final_labels == cid
            if cid == 5:
                plt.scatter(np.where(mask)[0], x[mask], c='k', marker='x', s=30, label='Ambiguous')
            else:
                plt.scatter(np.where(mask)[0], x[mask], s=15, label=f'Cluster {cid}')
        plt.xlabel('Unit Index')
        plt.ylabel('Estimated X')
        plt.title('Clustered Units (ambiguous=5)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return final_labels

#    
#wf_x = EphysBehaviorD.nwb_ephys_data.units['estimated_x'].data[:]
#labels = cluster_estimated_x(wf_x, n_clusters=4, threshold=0.5, plot=True)
#print("Final labels:", labels)

def append_units_locations(nwb_data: Any):
    """Add a *single* ``ccf_location`` column to ``nwb_data.units``.

    This is a functional, side‑effect–friendly refactor of the former
    ``EphysBehavior.append_units_locations`` *instance* method.  It no longer
    depends on *self* and instead operates directly on the supplied NWBFile.

    Parameters
    ----------
    nwb_data : pynwb.NWBFile
        The electrophysiology NWB object whose ``units`` DynamicTable should be
        augmented.  The function mutates this object **in‑place** and returns
        it so that it can be chained if desired.

    Returns
    -------
    pynwb.NWBFile
        The same ``nwb_data`` object, with the extra column appended (or left
        untouched if prerequisites are missing).
    """

    # ─── Sanity checks ────────────────────────────────────────────────────────
    if nwb_data is None or not hasattr(nwb_data, "units"):
        return nwb_data

    units = nwb_data.units

    # Avoid double‑insertion
    if "ccf_location" in units.colnames:
        return nwb_data

    # Extract pre‑computed per‑unit features
    wm = units["waveform_mean"].data[:]
    ex = units["estimated_x"].data[:]

    if wm.size == 0 or ex.size == 0:
        return nwb_data

    # ─── 1. Best electrode per unit (trough‑to‑peak) ─────────────────────────
    best_ch = find_best_electrode(wm, unit_index=None)  # (n_units,)

    # ─── 2. Assign shank IDs by clustering ML‑estimated X coordinates ────────
    shank_ids = cluster_estimated_x(ex, n_clusters=4, threshold=0.5, plot=False)

    device_names = units["device_name"][:]  # e.g. ['ProbeA', 'ProbeA', …]
    n_units = best_ch.shape[0]

    # ─── 3. Cache CCF look‑ups per *probe, shank* pair ───────────────────────
    session_name = getattr(nwb_data, "session_id", getattr(nwb_data, "session_description", ""))
    unique_keys = set(zip(device_names, shank_ids))
    ccf_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for probe, shank in unique_keys:
        ccf_cache[(probe, int(shank))] = load_ccf_channel_locations(
            session_name, probe, int(shank) + 1
        )

    # ─── 4. Gather arrays for vectorised fill‑in ─────────────────────────────
    x_arr = np.full(n_units, np.nan)
    y_arr = np.full(n_units, np.nan)
    z_arr = np.full(n_units, np.nan)
    axial_arr = np.full(n_units, np.nan)
    lateral_arr = np.full(n_units, np.nan)
    region_id = np.full(n_units, np.nan)
    region_name = np.array([""] * n_units, dtype=object)

    for i in range(n_units):
        probe = device_names[i]
        shank = int(shank_ids[i])
        ccf = ccf_cache.get((probe, shank), {})
        chan = extract_channel_info(ccf, int(best_ch[i]))
        if chan:
            x_arr[i] = chan.get("x", np.nan)
            y_arr[i] = chan.get("y", np.nan)
            z_arr[i] = chan.get("z", np.nan)
            axial_arr[i] = chan.get("axial", np.nan)
            lateral_arr[i] = chan.get("lateral", np.nan)
            region_id[i] = chan.get("brain_region_id", np.nan)
            region_name[i] = chan.get("brain_region", "")

    # ─── 5. Pack everything into a dict per unit and append as one column ────
    ccf_location_col = []
    for i in range(n_units):
        entry = {
            "best_electrode": int(best_ch[i]),
            "shank": int(shank_ids[i]),
            "probe": str(device_names[i]),
        }
        # Add CCF‑derived fields only if valid
        if np.isfinite(x_arr[i]):
            entry.update(
                {
                    "x": float(x_arr[i]),
                    "y": float(y_arr[i]),
                    "z": float(z_arr[i]),
                    "axial": float(axial_arr[i]),
                    "lateral": float(lateral_arr[i]),
                    "brain_region_id": int(region_id[i]),
                    "brain_region": region_name[i],
                }
            )
        ccf_location_col.append(entry)

    units.add_column(
        name="ccf_location",
        description=(
            "Per‑unit dict with keys: best_electrode, shank, x, y, z, axial, "
            "lateral, brain_region_id, brain_region"
        ),
        data=ccf_location_col,
    )

    return nwb_data