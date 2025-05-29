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
