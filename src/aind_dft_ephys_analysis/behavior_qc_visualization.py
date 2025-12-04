import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional, Tuple

from behavior_qc import compute_behavior_qc_from_nwb


# =========================================================
# Generic 1D histogram panel
# =========================================================

def _plot_hist_panel(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    bins: int = 30,
    xlabel: str = "",
    title: str = "",
) -> None:
    """
    Generic helper to plot a 1D histogram on a given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    values : np.ndarray
        1D array of values to histogram. NaNs are removed inside.
    bins : int, optional
        Number of histogram bins. Default is 30.
    xlabel : str, optional
        Label for the x-axis.
    title : str, optional
        Title of the panel.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if values.size > 0:
        ax.hist(values, bins=bins)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_first_lick_latency_hist(
    ax: plt.Axes,
    first_lick_latencies: np.ndarray,
    *,
    bins: int = 30,
) -> None:
    """
    Plot the distribution of first lick latency on a given Axes.
    """
    _plot_hist_panel(
        ax,
        first_lick_latencies,
        bins=bins,
        xlabel="First lick latency (s)",
        title="First lick latency",
    )


def plot_iti_hist(
    ax: plt.Axes,
    iti_time: np.ndarray,
    *,
    bins: int = 30,
) -> None:
    """
    Plot the distribution of ITI duration on a given Axes.
    """
    _plot_hist_panel(
        ax,
        iti_time,
        bins=bins,
        xlabel="ITI time (s)",
        title="ITI duration",
    )


def plot_delay_hist(
    ax: plt.Axes,
    delay_time: np.ndarray,
    *,
    bins: int = 30,
) -> None:
    """
    Plot the distribution of delay duration on a given Axes.
    """
    _plot_hist_panel(
        ax,
        delay_time,
        bins=bins,
        xlabel="Delay time (s)",
        title="Delay duration",
    )


# =========================================================
# Shared PSTH helper (after ITI start)
# =========================================================

def _compute_lick_psth_after_iti(
    left_licks: np.ndarray,
    right_licks: np.ndarray,
    go_times: np.ndarray,
    start_times: np.ndarray,
    *,
    trial_mask: Optional[np.ndarray] = None,
    psth_window: Tuple[float, float] = (-8.0, 5.0),
    psth_bin_width: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute lick PSTH (left/right/total) aligned to go cue, counting only
    licks that occur after the ITI start of each trial.

    For each trial i:
        - ITI start is start_times[i].
        - Licks earlier than ITI start are excluded for that trial.
        - Licks are aligned to go_times[i] (time 0).

    For each time bin, the denominator is the number of trials whose ITI
    start occurs at or before the bin center (relative to go), i.e. trials
    for which this time point is within the current trial rather than the
    previous trial.

    Parameters
    ----------
    left_licks : np.ndarray
        1D array of absolute timestamps of left licks (seconds).
    right_licks : np.ndarray
        1D array of absolute timestamps of right licks (seconds).
    go_times : np.ndarray
        1D array of go cue times for each trial (seconds).
    start_times : np.ndarray
        1D array of trial start times (ITI start; seconds).
    trial_mask : np.ndarray or None, optional
        Boolean mask over trials indicating which trials to include.
        If None, all trials are considered.
    psth_window : tuple of float, optional
        Time window around go cue for lick PSTH, in seconds,
        as (t_min, t_max). Default is (-8.0, 5.0).
    psth_bin_width : float, optional
        Bin width for lick PSTH (seconds). Default is 0.1.

    Returns
    -------
    bin_centers : np.ndarray
        1D array of bin centers (relative time from go cue).
    left_rate : np.ndarray
        1D array of lick rate for left licks (licks/s/trial).
    right_rate : np.ndarray
        1D array of lick rate for right licks (licks/s/trial).
    total_rate : np.ndarray
        1D array of lick rate for total licks (licks/s/trial).
    """
    t_min, t_max = psth_window
    if t_min >= t_max:
        raise ValueError("psth_window must satisfy t_min < t_max")

    go_times = np.asarray(go_times, dtype=float)
    start_times = np.asarray(start_times, dtype=float)

    # Valid go times; optionally also apply external trial_mask
    valid_go = ~np.isnan(go_times)
    if trial_mask is not None:
        trial_mask = np.asarray(trial_mask, dtype=bool)
        mask = valid_go & trial_mask
    else:
        mask = valid_go

    go_valid = go_times[mask]
    start_valid = start_times[mask]
    n_valid_trials = go_valid.size

    # Bin edges and centers (defined regardless, so caller can safely plot)
    bin_edges = np.arange(t_min, t_max + psth_bin_width, psth_bin_width)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if n_valid_trials == 0:
        # No valid trials → return NaN arrays
        nan_arr = np.full(bin_centers.shape, np.nan)
        return bin_centers, nan_arr, nan_arr, nan_arr

    # Per-trial relative minimum time allowed (ITI start relative to go cue)
    iti_rel_min = start_valid - go_valid  # shape (n_valid_trials,)

    # Collect lick times relative to go cue, enforcing "after ITI start"
    left_rel_all = []
    right_rel_all = []

    for t0, iti_start_abs in zip(go_valid, start_valid):
        # Absolute time window for this trial
        win_start_abs = t0 + t_min
        win_end_abs = t0 + t_max

        # Effective lower bound: not earlier than ITI start
        eff_start_abs = max(win_start_abs, iti_start_abs)

        # Left licks in window for this trial
        li_start = np.searchsorted(left_licks, eff_start_abs, side="left")
        li_end = np.searchsorted(left_licks, win_end_abs, side="right")
        if li_end > li_start:
            left_rel_all.append(left_licks[li_start:li_end] - t0)

        # Right licks in window for this trial
        ri_start = np.searchsorted(right_licks, eff_start_abs, side="left")
        ri_end = np.searchsorted(right_licks, win_end_abs, side="right")
        if ri_end > ri_start:
            right_rel_all.append(right_licks[ri_start:ri_end] - t0)

    if len(left_rel_all) > 0:
        left_rel_all_arr = np.concatenate(left_rel_all)
    else:
        left_rel_all_arr = np.array([], dtype=float)

    if len(right_rel_all) > 0:
        right_rel_all_arr = np.concatenate(right_rel_all)
    else:
        right_rel_all_arr = np.array([], dtype=float)

    # Histogram (counts per bin)
    left_counts, _ = np.histogram(left_rel_all_arr, bins=bin_edges)
    right_counts, _ = np.histogram(right_rel_all_arr, bins=bin_edges)
    total_counts = left_counts + right_counts

    # For each bin center, compute how many trials "cover" that time point:
    # bin_center[j] >= iti_rel_min[i]
    n_trials_per_bin = np.array(
        [np.sum(bin_c >= iti_rel_min) for bin_c in bin_centers],
        dtype=float,
    )

    # Convert counts to lick rate (licks/s/trial)
    with np.errstate(invalid="ignore", divide="ignore"):
        left_rate = left_counts / (n_trials_per_bin * psth_bin_width)
        right_rate = right_counts / (n_trials_per_bin * psth_bin_width)
        total_rate = total_counts / (n_trials_per_bin * psth_bin_width)

    # Bins with no trials contributing → set to NaN
    mask_zero = n_trials_per_bin == 0
    left_rate[mask_zero] = np.nan
    right_rate[mask_zero] = np.nan
    total_rate[mask_zero] = np.nan

    return bin_centers, left_rate, right_rate, total_rate


# =========================================================
# Panel D: Overall lick rate PSTH after ITI start
# =========================================================

def plot_lick_rate_psth_after_iti(
    ax: plt.Axes,
    left_licks: np.ndarray,
    right_licks: np.ndarray,
    go_times: np.ndarray,
    start_times: np.ndarray,
    *,
    psth_window: Tuple[float, float] = (-8.0, 5.0),
    psth_bin_width: float = 0.1,
) -> None:
    """
    Plot lick rate (left/right/total) aligned to go cue, counting only
    licks that occur after the ITI start of each trial.
    """
    bin_centers, left_rate, right_rate, total_rate = _compute_lick_psth_after_iti(
        left_licks=left_licks,
        right_licks=right_licks,
        go_times=go_times,
        start_times=start_times,
        trial_mask=None,
        psth_window=psth_window,
        psth_bin_width=psth_bin_width,
    )

    if np.any(np.isfinite(left_rate)):
        ax.plot(bin_centers, left_rate, label="Left licks")
    if np.any(np.isfinite(right_rate)):
        ax.plot(bin_centers, right_rate, label="Right licks")
    if np.any(np.isfinite(total_rate)):
        ax.plot(bin_centers, total_rate, linestyle="--", label="Total")

    ax.axvline(0.0, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Time from go cue (s)")
    ax.set_ylabel("Lick rate (licks/s/trial)")
    ax.set_title("Lick rate aligned to go cue\n(only licks after ITI start)")
    ax.grid(True, alpha=0.3)
    ax.legend()


# =========================================================
# Extra panel: lick rate PSTH for reward vs no-reward trials
# =========================================================

def plot_lick_rate_psth_by_reward_after_iti(
    ax: plt.Axes,
    left_licks: np.ndarray,
    right_licks: np.ndarray,
    go_times: np.ndarray,
    start_times: np.ndarray,
    reward: np.ndarray,
    *,
    psth_window: Tuple[float, float] = (-8.0, 5.0),
    psth_bin_width: float = 0.1,
) -> None:
    """
    Plot lick rate aligned to go cue (only licks after ITI start),
    separated for reward vs no-reward trials.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    left_licks, right_licks : np.ndarray
        Absolute lick timestamps.
    go_times : np.ndarray
        Go cue timestamps for each trial.
    start_times : np.ndarray
        ITI start timestamps for each trial.
    reward : np.ndarray
        Boolean array: reward[i] = True if trial i was rewarded.
    psth_window : tuple, optional
        Time window relative to go cue, as (t_min, t_max).
    psth_bin_width : float, optional
        Bin width in seconds.
    """
    reward = np.asarray(reward, dtype=bool)

    # Rewarded trials
    centers_r, _, _, total_r = _compute_lick_psth_after_iti(
        left_licks=left_licks,
        right_licks=right_licks,
        go_times=go_times,
        start_times=start_times,
        trial_mask=reward,
        psth_window=psth_window,
        psth_bin_width=psth_bin_width,
    )

    # Non-rewarded trials
    centers_n, _, _, total_n = _compute_lick_psth_after_iti(
        left_licks=left_licks,
        right_licks=right_licks,
        go_times=go_times,
        start_times=start_times,
        trial_mask=~reward,
        psth_window=psth_window,
        psth_bin_width=psth_bin_width,
    )

    if np.any(np.isfinite(total_r)):
        ax.plot(centers_r, total_r, label="Rewarded trials", linestyle="-")
    if np.any(np.isfinite(total_n)):
        ax.plot(centers_n, total_n, label="No-reward trials", linestyle="--")

    ax.axvline(0.0, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Time from go cue (s)")
    ax.set_ylabel("Lick rate (licks/s/trial)")
    ax.set_title("Lick rate: reward vs no-reward\n(only licks after ITI start)")
    ax.grid(True, alpha=0.3)
    ax.legend()


# =========================================================
# Wrapper: Build QC figure (now 2×3 with extra PSTH panel)
# =========================================================

def plot_behavior_qc_summary(
    nwb_data: Any,
    response_latency_window: Optional[float] = 2.0,
    bins: int = 30,
    figsize: Tuple[float, float] = (15.0, 8.0),
    save_path: Optional[str] = None,
    psth_window: Tuple[float, float] = (-8.0, 5.0),
    psth_bin_width: float = 0.1,
):
    """
    High-level QC visualization for one behavior session.

    This function:
    1) calls `compute_behavior_qc_from_nwb` to get metrics and trial_df
    2) extracts arrays for:
        - first_lick_latency_all
        - iti_time
        - delay_time
        - goCue_start_time
        - start_time
        - reward
    3) reads lick timestamps from `nwb_data.acquisition`
    4) creates a 2×3 figure and plots:
        [0, 0] : first lick latency histogram
        [0, 1] : ITI duration histogram
        [0, 2] : delay duration histogram
        [1, 0] : lick rate PSTH (overall, after ITI start)
        [1, 1] : lick rate PSTH (reward vs no-reward, after ITI start)
        [1, 2] : currently unused (axis turned off)

    Parameters
    ----------
    nwb_data : Any
        NWB behavior object passed to `compute_behavior_qc_from_nwb`.
    response_latency_window : float or None, optional
        Passed through to `compute_behavior_qc_from_nwb`.
    bins : int, optional
        Number of histogram bins for timing distributions. Default is 30.
    figsize : tuple of float, optional
        Figure size (width, height). Default is (15.0, 8.0).
    save_path : str or None, optional
        If not None, path where the figure will be saved (e.g. 'qc_summary.png').
    psth_window : tuple of float, optional
        Time window around go cue for lick PSTH, in seconds,
        as (t_min, t_max). Default is (-8.0, 5.0).
    psth_bin_width : float, optional
        Bin width for lick PSTH (seconds). Default is 0.1.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : np.ndarray
        2×3 array of Axes objects.
    metrics : dict
        Metrics returned by `compute_behavior_qc_from_nwb`.
    trial_df : pandas.DataFrame
        Per-trial DataFrame returned by `compute_behavior_qc_from_nwb`.
    """
    # 1) Compute QC metrics and per-trial table
    metrics, trial_df = compute_behavior_qc_from_nwb(
        nwb_data,
        response_latency_window=response_latency_window,
    )

    # 2) Extract arrays for histograms
    first_lick_latencies = np.asarray(metrics["first_lick_latency_all"], dtype=float)
    iti_time = np.asarray(trial_df["iti_time"], dtype=float)
    delay_time = np.asarray(trial_df["delay_time"], dtype=float)
    go_times = np.asarray(trial_df["goCue_start_time"], dtype=float)
    start_times = np.asarray(trial_df["start_time"], dtype=float)
    reward = np.asarray(trial_df["reward"], dtype=bool)

    # 3) Read lick timestamps from NWB
    left_licks = np.asarray(
        nwb_data.acquisition["left_lick_time"].timestamps[:], dtype=float
    )
    right_licks = np.asarray(
        nwb_data.acquisition["right_lick_time"].timestamps[:], dtype=float
    )

    # 4) Create figure and axes (2×3 grid)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    ax_lat = axes[0, 0]
    ax_iti = axes[0, 1]
    ax_delay = axes[0, 2]
    ax_psth_all = axes[1, 0]
    ax_psth_reward = axes[1, 1]
    ax_empty = axes[1, 2]

    # Panel A: first lick latency
    plot_first_lick_latency_hist(ax_lat, first_lick_latencies, bins=bins)

    # Panel B: ITI distribution
    plot_iti_hist(ax_iti, iti_time, bins=bins)

    # Panel C: delay distribution
    plot_delay_hist(ax_delay, delay_time, bins=bins)

    # Panel D: lick rate PSTH (overall)
    plot_lick_rate_psth_after_iti(
        ax_psth_all,
        left_licks=left_licks,
        right_licks=right_licks,
        go_times=go_times,
        start_times=start_times,
        psth_window=psth_window,
        psth_bin_width=psth_bin_width,
    )

    # Panel E: lick rate PSTH (reward vs no-reward)
    plot_lick_rate_psth_by_reward_after_iti(
        ax_psth_reward,
        left_licks=left_licks,
        right_licks=right_licks,
        go_times=go_times,
        start_times=start_times,
        reward=reward,
        psth_window=psth_window,
        psth_bin_width=psth_bin_width,
    )

    # Panel F: unused axis (turn off for now)
    ax_empty.axis("off")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes, metrics, trial_df
