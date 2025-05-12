import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Optional, Union, Tuple

from behavior_utils import extract_event_timestamps  # adjust import based on module path


def plot_raster_graph(
    nwb_data: Any,
    unit_index: int,
    align_to_event: str = 'go_cue',
    time_window: List[float] = [-2, 3],
    bin_size: float = 0.05,
    fitted_data: Optional[np.ndarray] = None,
    latent_name: Optional[str] = None,
    exclude_trials: Optional[List[int]] = None,
    save_figure: bool = False,
    save_format: str = 'eps',
    save_folder: str = '/root/capsule/results',
    figure_name_prefix: Optional[str] = None,
) -> None:
    """
    Plot spike raster and PSTH for a single unit from a combined NWB file.

    This function takes a combined NWB object (containing both behavior trials and ephys unit data),
    extracts spike times for the specified unit, aligns them to a behavioral event for each trial,
    constructs a raster plot and corresponding peri-stimulus time histogram (PSTH),
    and optionally saves the figure.

    Parameters
    ----------
    nwb_data : NWB file handle
        Combined NWB object containing:
        - `trials` table or times as used by `extract_event_timestamps`.
        - `units` DynamicTable with a `spike_times` column.
    unit_index : int
        Index of the unit to plot (row in `nwb_data.units`).
    align_to_event : str, optional
        The event name to align spikes to (default 'go_cue'). Passed to `extract_event_timestamps`.
    time_window : list of two floats, optional
        [start, end] times (seconds) relative to event for alignment (default [-2, 3]).
    bin_size : float, optional
        Bin width (seconds) for PSTH histogram (default 0.05).
    fitted_data : 1D numpy.ndarray, optional
        Trial-specific values (length equal to number of trials) to sort the raster by;
        must match number of included trials if provided (default None).
    latent_name : str, optional
        Name of the latent variable in `fitted_data`; used for title annotation if sorting.
    exclude_trials : list of int, optional
        List of trial indices to exclude before sorting and plotting (default None).
    save_figure : bool, optional
        If True, saves the figure to disk (default False).
    save_format : str, optional
        File format for saving figure (e.g., 'eps', 'pdf'; default 'eps').
    save_folder : str, optional
        Directory path to save figures (default '/root/capsule/results').
    figure_name_prefix : str, optional
        Filename prefix for saved figure (omit extension). If None, filename built from unit
        index and sorting information.

    Returns
    -------
    None
        Displays the raster and PSTH plot. If `save_figure` is True, prints the saved file path.
    """
    # Create save directory if needed
    if save_figure:
        os.makedirs(save_folder, exist_ok=True)

    # 1. Retrieve event timestamps per trial
    all_times = np.array(extract_event_timestamps(nwb_data, align_to_event))
    n_trials = len(all_times)

    # 2. Exclude specified trials if any
    trials_idx = np.arange(n_trials)
    if exclude_trials:
        mask_exc = np.ones(n_trials, dtype=bool)
        mask_exc[exclude_trials] = False
        all_times = all_times[mask_exc]
        trials_idx = trials_idx[mask_exc]

    # 3. Validate fitted_data length
    if fitted_data is not None and len(fitted_data) != len(all_times):
        raise ValueError(
            f"After exclusion, fitted_data length ({len(fitted_data)}) does not match number of trials ({len(all_times)})"
        )

    # 4. Sort trials by fitted_data if provided
    if fitted_data is not None:
        sort_idx = np.argsort(fitted_data)
        all_times = all_times[sort_idx]
        trials_idx = trials_idx[sort_idx]

    # 5. Load spike times for the unit
    try:
        unit_spike_times = np.array(nwb_data.units['spike_times'][unit_index])
    except Exception:
        raise ValueError(f"Spike times not found for unit {unit_index}.")

    # 6. Align spikes and collect for PSTH
    spikes_aligned = []  # list of (trial_idx, aligned_spikes)
    all_spikes = []
    for row_idx, t0 in enumerate(all_times):
        start, end = t0 + time_window[0], t0 + time_window[1]
        mask_spk = (unit_spike_times >= start) & (unit_spike_times <= end)
        aligned = unit_spike_times[mask_spk] - t0
        spikes_aligned.append((row_idx, aligned))
        all_spikes.extend(aligned.tolist())

    num_plotted = len(all_times)

    # 7. Plotting
    fig, (ax_raster, ax_psth) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # Raster plot
    for row_idx, spks in spikes_aligned:
        ax_raster.vlines(spks, row_idx + 0.5, row_idx + 1.5, color='black')
    ax_raster.axvline(0, color='red', linestyle='--')
    ax_raster.set_ylabel('Trials')
    title = f'Raster: Unit {unit_index} aligned to {align_to_event}'
    if fitted_data is not None and latent_name:
        title += f' (sorted by {latent_name})'
    ax_raster.set_title(title)
    ax_raster.set_ylim(0.5, num_plotted + 0.5)

    # If sorted by fitted_data, annotate low/high positions
    if fitted_data is not None:
        ax_raster.text(
            time_window[1], 1, 'Low', va='center', ha='right', color='blue'
        )
        ax_raster.text(
            time_window[1], num_plotted, 'High', va='center', ha='right', color='blue'
        )

    # PSTH plot
    bins = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
    counts, _ = np.histogram(all_spikes, bins=bins)
    rates = counts / (num_plotted * bin_size)
    ax_psth.bar(bins[:-1], rates, width=bin_size, align='edge', color='black')
    ax_psth.axvline(0, color='red', linestyle='--')
    ax_psth.set_xlabel('Time from event (s)')
    ax_psth.set_ylabel('Firing rate (Hz)')
    ax_psth.set_title('PSTH')

    plt.tight_layout()

    # Save if requested
    if save_figure:
        # Build base filename
        parts = []
        if figure_name_prefix:
            parts.append(figure_name_prefix)
        parts.append(f'unit_{unit_index}')
        if fitted_data is not None and latent_name:
            parts.append(f'sorted_by_{latent_name}')
        filename = "_".join(parts) + f'.{save_format}'
        filepath = os.path.join(save_folder, filename)
        fig.savefig(filepath, format=save_format)
        print(f'Figure saved as {filepath}')

    plt.show()


def get_units_passed_default_qc(nwb_data: Any) -> np.ndarray:
    """
    Retrieves the indices of units in a combined NWB dataset that have passed
    the automated default QC checks and are not labeled as 'noise'.

    The `default_qc` flag is precomputed based on the
    following metrics:

    - **presence_ratio** (float): Fraction of the recording session during which the unit was active.
      Units must have `presence_ratio >= 0.8`.
    - **isi_violations_ratio** (float): Proportion of inter-spike intervals violating the refractory period.
      Units must have `isi_violations_ratio <= 0.5`.
    - **amplitude_cutoff** (float): Estimated fraction of missed spikes due to detection threshold.
      Units must have `amplitude_cutoff <= 0.1`.

    After computing these, the `default_qc` field should be True. Additionally, units
    labeled as 'noise' by `decoder_label` are excluded.

    Parameters
    ----------
    nwb_data : NWB file handle
        Combined NWB object containing a `units` table with columns:
        `default_qc` and `decoder_label`.

    Returns
    -------
    indices : numpy.ndarray
        1D array of integer indices of units where `default_qc` is True
        and `decoder_label` != 'noise'.
    """
    tbl = nwb_data.units
    default_qc = np.array(tbl['default_qc'].data)
    labels = np.array(tbl['decoder_label'].data)

    # Mask for QC-passed, non-noise units
    mask = ((default_qc == True) | (default_qc == 'True')) & (labels != 'noise')

    # Return indices
    indices = np.nonzero(mask)[0]
    print(f"Number of units passing QC: {len(indices)}")
    return indices


def get_the_mean_firing_rate(
    nwb_data: Any,
    unit_index: Union[int, List[int], None] = None,
    align_to_event: str = 'go_cue',
    time_windows: List[List[float]] = [[-1, 0]],
    z_score: bool = False
) -> pd.DataFrame:
    """
    Calculate mean firing rates per trial for specified units over one or more time windows
    aligned to a behavioral event. If `unit_index` is None, only units passing default QC
    (via `default_qc`) are processed.

    Parameters
    ----------
    nwb_data : NWB file handle
        Combined NWB object containing behavior trials and `units` spike_times.
    unit_index : int, list of int, or None
        Unit index or list of indices to process. If None, uses all units passing default QC.
    align_to_event : str
        Event name to align spikes to (default 'go_cue').
    time_windows : list of [start, end]
        Windows (seconds relative to event) for computing firing rates; e.g., [[-2,1],[1,2]].
    z_score : bool
        If True, normalize each unit's firing rates (per window) across trials to z-scores.

    Returns
    -------
    firing_rate_df : pd.DataFrame
        MultiIndex DataFrame with:
          - Rows: Trial number (Trial_1, Trial_2, ...).
          - Columns: MultiIndex (Unit, Window) indicating unit index and window label
            ("window_<start>_<end>").
        Each cell contains the mean firing rate (spikes/sec) for that trial, unit, and window.
    """
    # Retrieve event times per trial
    all_times = np.array(extract_event_timestamps(nwb_data, align_to_event))
    num_trials = len(all_times)

    # Determine units to process
    if unit_index is None:
        units_to_process = list(get_units_passed_default_qc(nwb_data))
    elif isinstance(unit_index, list):
        units_to_process = unit_index
    else:
        units_to_process = [unit_index]

    # Prepare labels
    window_labels = [f"window_{int(w[0])}_{int(w[1])}" for w in time_windows]
    # Data dict: unit -> matrix trials x windows
    data = {}

    for u in units_to_process:
        unit_spike_times = np.array(nwb_data.units['spike_times'][u])
        rates_matrix = np.zeros((num_trials, len(time_windows)))
        for wi, (start_offset, end_offset) in enumerate(time_windows):
            duration = end_offset - start_offset
            for ti, t0 in enumerate(all_times):
                start = t0 + start_offset
                end = t0 + end_offset
                count = np.sum((unit_spike_times >= start) & (unit_spike_times <= end))
                rates_matrix[ti, wi] = count / duration if duration > 0 else np.nan
        if z_score:
            means = rates_matrix.mean(axis=0)
            stds = rates_matrix.std(axis=0)
            rates_matrix = (rates_matrix - means) / stds
            rates_matrix = np.nan_to_num(rates_matrix)
        data[u] = rates_matrix

    # Build DataFrame: MultiIndex columns (unit, window, trial)
    tuples = []
    values = []
    for u, matrix in data.items():
        for wi, label in enumerate(window_labels):
            tuples.append((u, label))
            values.append(matrix[:, wi])
    index = pd.Index([f'Trial_{i+1}' for i in range(num_trials)], name='Trial')
    columns = pd.MultiIndex.from_tuples(tuples, names=['Unit', 'Window'])
    df = pd.DataFrame(np.stack(values, axis=1), index=index, columns=columns)

    return df