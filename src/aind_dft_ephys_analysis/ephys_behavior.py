import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Optional,Tuple

from behavior_utils import extract_event_timestamps  # adjust import based on module path

def plot_raster_graph(
    nwb_behavior_data: Any,
    nwb_ephys_data: Any,
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
    Plot spike raster and PSTH for a single unit, aligned to a specified event,
    with optional trial sorting based on fitted model data and exclusion of specific trials.
    Optionally save the figure in the given format and folder. If `figure_name_prefix` is provided,
    it will be prepended to the saved filename; otherwise, filenames start with `unit_<unit_index>`.

    Parameters
    ----------
    nwb_behavior_data : NWB behavior handle
        NWB object containing trial timestamps and behavior metadata.
    nwb_ephys_data : NWB ephys handle
        NWB object containing unit spike times under `.units['spike_times']`.
    unit_index : int
        Index of the unit to plot.
    align_to_event : str, optional
        The event to align spikes to (default 'go_cue').
    time_window : list of float, optional
        [start, end] times (in seconds) relative to event for alignment (default [-2, 3]).
    bin_size : float, optional
        Bin size for PSTH in seconds (default 0.05).
    fitted_data : ndarray, optional
        1D array of trial-specific values for sorting; must correspond to each trial.
    latent_name : str, optional
        Name of the latent variable used for sorting; displayed in title/annotations.
    exclude_trials : list of int, optional
        Trial indices to drop before sorting and plotting.
    save_figure : bool, optional
        If True, save the figure to disk (default False).
    save_format : str, optional
        File format for saving (e.g., 'eps', 'pdf'; default 'eps').
    save_folder : str, optional
        Directory to which the figure will be saved (default '/root/capsule/results').
    figure_name_prefix : str, optional
        Identifier for the recording session, used in saved filenames. If None, omitted.
    Returns
    -------
    None
        Displays a raster (top) and PSTH (bottom) plot, and saves if requested.
    """
    # Create save directory if needed
    if save_figure:
        os.makedirs(save_folder, exist_ok=True)

    # 1. Retrieve all event timestamps (one per trial)
    all_times = np.array(
        extract_event_timestamps(nwb_behavior_data, align_to_event)
    )
    n_trials = len(all_times)
    trials_idx = np.arange(n_trials)

    # 2. Exclude specified trials
    if exclude_trials:
        mask_exclude = np.ones(n_trials, dtype=bool)
        mask_exclude[exclude_trials] = False
        all_times = all_times[mask_exclude]
        trials_idx = trials_idx[mask_exclude]

    # 3. Validate fitted_data length
    if fitted_data is not None and len(fitted_data) != len(all_times):
        raise ValueError(
            f"After exclusion, fitted_data length ({len(fitted_data)}) does not match "
            f"number of trials ({len(all_times)})."
        )

    # 4. Sort trials by fitted_data if provided
    if fitted_data is not None:
        sort_idx = np.argsort(fitted_data)
        all_times = all_times[sort_idx]
        trials_idx = trials_idx[sort_idx]

    # 5. Load spike times
    try:
        unit_spike_times = nwb_ephys_data.units['spike_times'][unit_index]
    except Exception:
        raise ValueError(f"Spike times not found for unit {unit_index}.")

    # 6. Align spikes and collect for PSTH
    spikes_aligned = []  # (row_idx, aligned_times)
    all_spikes = []
    for row_idx, t0 in enumerate(all_times):
        start, end = t0 + time_window[0], t0 + time_window[1]
        mask = (unit_spike_times >= start) & (unit_spike_times <= end)
        aligned = unit_spike_times[mask] - t0
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


def get_units_passed_default_qc(nwb_ephys_data: Any) -> np.ndarray:
    """
    Retrieves the indices of units in an NWB ephys dataset that have passed the
    automated default QC (`default_qc` flag) and are not labeled as 'noise'.

    The `default_qc` flag is precomputed based on internal thresholds applied to the
    following metrics:

    - **presence_ratio** (float): Fraction of the recording session during which the unit was active.
      Units must have `presence_ratio >= 0.8`.
    - **isi_violations_ratio** (float): Proportion of inter-spike intervals violating the refractory period.
      Units must have `isi_violations_ratio <= 0.5`.
    - **amplitude_cutoff** (float): Estimated fraction of missed spikes due to detection threshold.
      Units must have `amplitude_cutoff <= 0.1`.

    After computing these, the `default_qc` field should be True. Additionally, units
    labeled as 'noise' by `decoder_label` are excluded.

    Input
    -----
    nwb_ephys_data : pynwb.NWBFile or similar
        NWB file handle containing a `.units` DynamicTable with at least:
        - `default_qc` (bool or str): Automated QC pass flag.
        - `decoder_label` (str): Unit classification; 'noise' to exclude.

    Output
    ------
    indices : numpy.ndarray of int
        1D array of integer indices of units where `default_qc` is True 
        (or 'True') and `decoder_label` != 'noise'.
    """
    tbl = nwb_ephys_data.units
    default_qc = np.array(tbl['default_qc'].data)
    labels = np.array(tbl['decoder_label'].data)

    # Mask for QC-passed, non-noise units
    mask = ((default_qc == True) | (default_qc == 'True')) & (labels != 'noise')

    # Return indices
    indices = np.nonzero(mask)[0]
    print(f"Number of units passing QC: {len(indices)}")
    return indices

