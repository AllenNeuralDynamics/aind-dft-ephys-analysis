import io
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional, Tuple

from behavior_qc import compute_behavior_qc_from_nwb
from general_visualization import plot_behavior_session

from model_fitting import (
    fit_choice_logistic_regression_from_nwb,
    visualize_choice_logistic_regression,
)

# =========================================================
# Generic 1D histogram panels
# =========================================================

def _plot_hist_panel(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    bins: int = 30,
    xlabel: str = "",
    title: str = "",
) -> None:
    """Generic helper to plot a 1D histogram on a given Axes."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if values.size > 0:
        ax.hist(values, bins=bins)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_iti_hist(
    ax: plt.Axes,
    iti_time: np.ndarray,
    *,
    bins: int = 30,
) -> None:
    """Plot the distribution of ITI duration on a given Axes."""
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
    """Plot the distribution of delay duration on a given Axes."""
    _plot_hist_panel(
        ax,
        delay_time,
        bins=bins,
        xlabel="Delay time (s)",
        title="Delay duration",
    )


# =========================================================
# PSTH helper (aligned to go cue, trial start → post-go)
# =========================================================

def _compute_lick_psth_after_iti(
    left_licks: np.ndarray,
    right_licks: np.ndarray,
    go_times: np.ndarray,
    start_times: np.ndarray,
    *,
    trial_mask: Optional[np.ndarray] = None,
    psth_window: Tuple[float, float] = (-5.0, 2.0),
    psth_bin_width: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute lick PSTH (left/right/total) aligned to go cue.

    Time is aligned to go cue: t = lick_time - go_time.
    Licks are kept only if they occur at or after trial start_time.
    Trials must fully cover the pre-go part of the window
    (go_time - start_time >= -t_min).
    """
    t_min, t_max = psth_window
    if t_min >= t_max:
        raise ValueError("psth_window must satisfy t_min < t_max")

    go_times = np.asarray(go_times, dtype=float)
    start_times = np.asarray(start_times, dtype=float)

    valid = (~np.isnan(go_times)) & (~np.isnan(start_times))
    iti_duration = go_times - start_times
    valid &= iti_duration >= (-t_min)

    if trial_mask is not None:
        trial_mask = np.asarray(trial_mask, dtype=bool)
        valid &= trial_mask

    go_valid = go_times[valid]
    start_valid = start_times[valid]
    n_valid_trials = go_valid.size

    bin_edges = np.arange(t_min, t_max + psth_bin_width, psth_bin_width)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if n_valid_trials == 0:
        nan_arr = np.full(bin_centers.shape, np.nan)
        return bin_centers, nan_arr, nan_arr, nan_arr

    left_rel_all = []
    right_rel_all = []

    for t0, s in zip(go_valid, start_valid):
        win_start_abs = max(t0 + t_min, s)
        win_end_abs = t0 + t_max
        if win_end_abs <= win_start_abs:
            continue

        li_start = np.searchsorted(left_licks, win_start_abs, side="left")
        li_end = np.searchsorted(left_licks, win_end_abs, side="right")
        if li_end > li_start:
            left_rel_all.append(left_licks[li_start:li_end] - t0)

        ri_start = np.searchsorted(right_licks, win_start_abs, side="left")
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

    left_counts, _ = np.histogram(left_rel_all_arr, bins=bin_edges)
    right_counts, _ = np.histogram(right_rel_all_arr, bins=bin_edges)
    total_counts = left_counts + right_counts

    n_trials_per_bin = float(n_valid_trials)

    with np.errstate(invalid="ignore", divide="ignore"):
        left_rate = left_counts / (n_trials_per_bin * psth_bin_width)
        right_rate = right_counts / (n_trials_per_bin * psth_bin_width)
        total_rate = total_counts / (n_trials_per_bin * psth_bin_width)

    return bin_centers, left_rate, right_rate, total_rate


def plot_lick_rate_psth_after_iti(
    ax: plt.Axes,
    left_licks: np.ndarray,
    right_licks: np.ndarray,
    go_times: np.ndarray,
    start_times: np.ndarray,
    *,
    psth_window: Tuple[float, float] = (-5.0, 2.0),
    psth_bin_width: float = 0.1,
) -> None:
    """Plot lick rate (left/right/total) aligned to go cue."""
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
    ax.set_title("Lick rate aligned to go cue\n(trial start → post-go)")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_lick_rate_psth_by_reward_after_iti(
    ax: plt.Axes,
    left_licks: np.ndarray,
    right_licks: np.ndarray,
    go_times: np.ndarray,
    start_times: np.ndarray,
    reward: np.ndarray,
    *,
    psth_window: Tuple[float, float] = (-5.0, 2.0),
    psth_bin_width: float = 0.1,
) -> None:
    """Plot lick rate aligned to go cue, split by reward vs no-reward."""
    reward = np.asarray(reward, dtype=bool)

    centers_r, _, _, total_r = _compute_lick_psth_after_iti(
        left_licks=left_licks,
        right_licks=right_licks,
        go_times=go_times,
        start_times=start_times,
        trial_mask=reward,
        psth_window=psth_window,
        psth_bin_width=psth_bin_width,
    )

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
    ax.set_title("Lick rate aligned to go cue\nReward vs no-reward")
    ax.grid(True, alpha=0.3)
    ax.legend()


# =========================================================
# PSTH helper aligned to start_time (only licks before go)
# =========================================================

def _compute_lick_psth_from_start_before_go(
    left_licks: np.ndarray,
    right_licks: np.ndarray,
    start_times: np.ndarray,
    go_times: np.ndarray,
    *,
    trial_mask: Optional[np.ndarray] = None,
    psth_window: Tuple[float, float] = (0.0, 5.0),
    psth_bin_width: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute lick PSTH aligned to trial start_time, including only
    licks between trial start and go cue.
    """
    t_min, t_max = psth_window
    if t_min >= t_max:
        raise ValueError("psth_window must satisfy t_min < t_max")
    if t_min < 0:
        raise ValueError("For start-aligned pre-go PSTH, psth_window[0] must be >= 0.")

    start_times = np.asarray(start_times, dtype=float)
    go_times = np.asarray(go_times, dtype=float)

    valid = (~np.isnan(start_times)) & (~np.isnan(go_times))
    rel_duration = go_times - start_times
    valid &= rel_duration >= t_max

    if trial_mask is not None:
        trial_mask = np.asarray(trial_mask, dtype=bool)
        valid &= trial_mask

    start_valid = start_times[valid]
    go_valid = go_times[valid]
    n_valid_trials = start_valid.size

    bin_edges = np.arange(t_min, t_max + psth_bin_width, psth_bin_width)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if n_valid_trials == 0:
        nan_arr = np.full(bin_centers.shape, np.nan)
        return bin_centers, nan_arr, nan_arr, nan_arr

    left_rel_all = []
    right_rel_all = []

    for s, g in zip(start_valid, go_valid):
        win_start_abs = s + t_min
        win_end_abs = min(s + t_max, g)
        if win_end_abs <= win_start_abs:
            continue

        li_start = np.searchsorted(left_licks, win_start_abs, side="left")
        li_end = np.searchsorted(left_licks, win_end_abs, side="right")
        if li_end > li_start:
            left_rel_all.append(left_licks[li_start:li_end] - s)

        ri_start = np.searchsorted(right_licks, win_start_abs, side="left")
        ri_end = np.searchsorted(right_licks, win_end_abs, side="right")
        if ri_end > ri_start:
            right_rel_all.append(right_licks[ri_start:ri_end] - s)

    if len(left_rel_all) > 0:
        left_rel_all_arr = np.concatenate(left_rel_all)
    else:
        left_rel_all_arr = np.array([], dtype=float)

    if len(right_rel_all) > 0:
        right_rel_all_arr = np.concatenate(right_rel_all)
    else:
        right_rel_all_arr = np.array([], dtype=float)

    left_counts, _ = np.histogram(left_rel_all_arr, bins=bin_edges)
    right_counts, _ = np.histogram(right_rel_all_arr, bins=bin_edges)
    total_counts = left_counts + right_counts

    n_trials_per_bin = float(n_valid_trials)

    with np.errstate(invalid="ignore", divide="ignore"):
        left_rate = left_counts / (n_trials_per_bin * psth_bin_width)
        right_rate = right_counts / (n_trials_per_bin * psth_bin_width)
        total_rate = total_counts / (n_trials_per_bin * psth_bin_width)

    return bin_centers, left_rate, right_rate, total_rate


def plot_lick_rate_psth_from_start_before_go(
    ax: plt.Axes,
    left_licks: np.ndarray,
    right_licks: np.ndarray,
    start_times: np.ndarray,
    go_times: np.ndarray,
    *,
    psth_window: Tuple[float, float] = (0.0, 5.0),
    psth_bin_width: float = 0.1,
) -> None:
    """Plot lick rate aligned to trial start_time (start → go)."""
    bin_centers, left_rate, right_rate, total_rate = _compute_lick_psth_from_start_before_go(
        left_licks=left_licks,
        right_licks=right_licks,
        start_times=start_times,
        go_times=go_times,
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
    ax.set_xlabel("Time from trial start (s)")
    ax.set_ylabel("Lick rate (licks/s/trial)")
    ax.set_title("Lick rate (start → go)\nAligned to trial start")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_lick_rate_psth_from_start_by_reward_before_go(
    ax: plt.Axes,
    left_licks: np.ndarray,
    right_licks: np.ndarray,
    start_times: np.ndarray,
    go_times: np.ndarray,
    reward: np.ndarray,
    *,
    psth_window: Tuple[float, float] = (0.0, 5.0),
    psth_bin_width: float = 0.1,
) -> None:
    """Plot start-aligned lick rate (start → go), split by reward vs no-reward."""
    reward = np.asarray(reward, dtype=bool)

    centers_r, _, _, total_r = _compute_lick_psth_from_start_before_go(
        left_licks=left_licks,
        right_licks=right_licks,
        start_times=start_times,
        go_times=go_times,
        trial_mask=reward,
        psth_window=psth_window,
        psth_bin_width=psth_bin_width,
    )

    centers_n, _, _, total_n = _compute_lick_psth_from_start_before_go(
        left_licks=left_licks,
        right_licks=right_licks,
        start_times=start_times,
        go_times=go_times,
        trial_mask=~reward,
        psth_window=psth_window,
        psth_bin_width=psth_bin_width,
    )

    if np.any(np.isfinite(total_r)):
        ax.plot(centers_r, total_r, label="Rewarded trials", linestyle="-")
    if np.any(np.isfinite(total_n)):
        ax.plot(centers_n, total_n, label="No-reward trials", linestyle="--")

    ax.axvline(0.0, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Time from trial start (s)")
    ax.set_ylabel("Lick rate (licks/s/trial)")
    ax.set_title("Start-aligned lick rate (start → go)\nReward vs no-reward")
    ax.grid(True, alpha=0.3)
    ax.legend()


# =========================================================
# Raster helper (aligned to go cue, with side)
# =========================================================

def _compute_lick_raster_with_side(
    left_licks: np.ndarray,
    right_licks: np.ndarray,
    go_times: np.ndarray,
    *,
    raster_window: Tuple[float, float] = (-8.0, 5.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute lick raster aligned to the go cue, with side information.

    Returns:
        t_rel_all      : relative lick times (sec from go cue)
        trial_idx_all  : trial indices (0-based)
        side_all       : 0 for left licks, 1 for right licks
    """
    t_min, t_max = raster_window
    go_times = np.asarray(go_times, dtype=float)

    t_rel_list = []
    trial_idx_list = []
    side_list = []

    for i, t0 in enumerate(go_times):
        if np.isnan(t0):
            continue

        win_start = t0 + t_min
        win_end = t0 + t_max

        li_start = np.searchsorted(left_licks, win_start, side="left")
        li_end = np.searchsorted(left_licks, win_end, side="right")
        if li_end > li_start:
            t_rel_list.append(left_licks[li_start:li_end] - t0)
            trial_idx_list.append(np.full(li_end - li_start, i, dtype=int))
            side_list.append(np.zeros(li_end - li_start, dtype=int))

        ri_start = np.searchsorted(right_licks, win_start, side="left")
        ri_end = np.searchsorted(right_licks, win_end, side="right")
        if ri_end > ri_start:
            t_rel_list.append(right_licks[ri_start:ri_end] - t0)
            trial_idx_list.append(np.full(ri_end - ri_start, i, dtype=int))
            side_list.append(np.ones(ri_end - ri_start, dtype=int))

    if len(t_rel_list) == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )

    t_rel_all = np.concatenate(t_rel_list)
    trial_idx_all = np.concatenate(trial_idx_list)
    side_all = np.concatenate(side_list)
    return t_rel_all, trial_idx_all, side_all


# =========================================================
# Helper: embed an existing Figure as an image into an Axes
# =========================================================

def _embed_figure_as_image(src_fig: plt.Figure, ax: plt.Axes) -> None:
    """
    Render a Matplotlib Figure to an RGB image (cropped tightly around
    the content) and display it in the target Axes.
    """

    if src_fig is None or src_fig.canvas is None:
        ax.text(
            0.5,
            0.5,
            "Behavior figure unavailable",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.axis("off")
        return

    # Render figure to an in-memory PNG with tight bbox so axes (incl. x-axis)
    # are fully included and margins are trimmed.
    buf_io = io.BytesIO()
    try:
        src_fig.savefig(
            buf_io,
            format="png",
            bbox_inches="tight",
            pad_inches=0.0,
            dpi=src_fig.dpi,
        )
        buf_io.seek(0)
        img = plt.imread(buf_io)
    except Exception as e:
        ax.text(
            0.5,
            0.5,
            f"Cannot render behavior figure\n({e})",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.axis("off")
        return

    ax.imshow(img, aspect="auto")
    ax.axis("off")



# =========================================================
# Wrapper: full QC figure (6×3, behavior rows span full width)
# =========================================================

def plot_behavior_qc_summary(
    nwb_data: Any,
    response_latency_window: Optional[float] = 1.0,
    bins: int = 30,
    figsize: Tuple[float, float] = (15.0, 26.0),
    save_path: Optional[str] = None,
    psth_window_go: Tuple[float, float] = (-5.0, 2.0),
    psth_bin_width: float = 0.1,
    psth_window_start: Tuple[float, float] = (0.0, 5.0),
    logreg_lag: int = 8,
):
    """
    High-level QC visualization for one behavior session (6×4 layout).

    Rows:
      0: behavior session with model      (spans all 4 columns)
      1: behavior session without model   (spans all 4 columns)

      2: logistic regression coefficients | logistic predicted vs actual
         (two panels, each occupying 1 column)

      3: first-lick latency (all)         | first-lick latency (L vs R)
         go-aligned PSTH (all)            | go-aligned PSTH (reward vs no-reward)

      4: start-aligned PSTH (all)         | start-aligned PSTH (by reward)
         lick raster (all)                | lick raster (grouped by reward)

      5: ITI histogram                    | delay histogram
         metrics summary (left)           | metrics summary (right)
    """
    # ---------------------------------------------------------
    # 1) Compute QC metrics and per-trial table
    # ---------------------------------------------------------
    metrics, trial_df = compute_behavior_qc_from_nwb(
        nwb_data,
        response_latency_window=response_latency_window,
    )

    # First-lick latency arrays
    lat_all = np.asarray(metrics["first_lick_latency_all"], dtype=float)
    lat_left = np.asarray(metrics["first_lick_latency_all_left"], dtype=float)
    lat_right = np.asarray(metrics["first_lick_latency_all_right"], dtype=float)

    # ITI, delay
    iti_time = np.asarray(trial_df["iti_time"], dtype=float)
    delay_time = np.asarray(trial_df["delay_time"], dtype=float)

    mean_iti = float(metrics.get("mean_iti_time", np.nan))
    mean_delay = float(metrics.get("mean_delay_time", np.nan))

    # Trial info
    go_times = np.asarray(trial_df["goCue_start_time"], dtype=float)
    start_times = np.asarray(trial_df["start_time"], dtype=float)
    reward = np.asarray(trial_df["reward"], dtype=bool)

    # Lick timestamps (absolute)
    left_licks = np.asarray(
        nwb_data.acquisition["left_lick_time"].timestamps[:], dtype=float
    )
    right_licks = np.asarray(
        nwb_data.acquisition["right_lick_time"].timestamps[:], dtype=float
    )

    # Lick raster (go-aligned)
    t_rel_all, trial_idx_all, side_all = _compute_lick_raster_with_side(
        left_licks=left_licks,
        right_licks=right_licks,
        go_times=go_times,
        raster_window=(-8.0, 5.0),
    )

    # ---------------------------------------------------------
    # 1b) Logistic regression fit + figures
    # ---------------------------------------------------------
    logreg_fig_coef = None
    logreg_fig_pred = None

    try:
        fit_output = fit_choice_logistic_regression_from_nwb(
            nwb_data,
            lag=logreg_lag,
        )
        viz_out = visualize_choice_logistic_regression(
            fit_output,
            plot_coefficients=True,
            plot_predictions=True,
            title_font_size=16,
            label_font_size=14,
        )
        logreg_fig_coef, _ = viz_out["coefficients"]
        logreg_fig_pred, _ = viz_out["predictions"]
    except Exception as e:
        print(f"Warning: logistic regression in plot_behavior_qc_summary failed: {e}")
        logreg_fig_coef = None
        logreg_fig_pred = None

    # ---------------------------------------------------------
    # 2) Create figure and axes via GridSpec (6×4)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        6,
        4,
        height_ratios=[1.1, 1.1, 1.0, 1.0, 1.0, 1.0],
    )

    axes = np.empty((6, 4), dtype=object)

    # Row 0: behavior with model spans all columns
    ax_behavior_model = fig.add_subplot(gs[0, :])
    axes[0, 0] = ax_behavior_model
    axes[0, 1] = ax_behavior_model
    axes[0, 2] = ax_behavior_model
    axes[0, 3] = ax_behavior_model

    # Row 1: behavior without model spans all columns
    ax_behavior_nomodel = fig.add_subplot(gs[1, :])
    axes[1, 0] = ax_behavior_nomodel
    axes[1, 1] = ax_behavior_nomodel
    axes[1, 2] = ax_behavior_nomodel
    axes[1, 3] = ax_behavior_nomodel

    # ------------------------
    # Row 2: logistic regression
    # ------------------------

    # Create a nested GridSpec inside row 2 with custom width ratios
    gs_log = gs[2, :].subgridspec(1, 2, width_ratios=[1, 1.0], wspace=0.05)

    ax_log_coef = fig.add_subplot(gs_log[0, 0])   # wider panel
    ax_log_pred = fig.add_subplot(gs_log[0, 1])   # narrower panel

    # Register them in axes[] array
    axes[2, 0] = ax_log_coef
    axes[2, 1] = ax_log_coef
    axes[2, 2] = ax_log_pred
    axes[2, 3] = ax_log_pred



    # Row 3: (old row2) latency + go-aligned PSTH
    axes[3, 0] = fig.add_subplot(gs[3, 0])
    axes[3, 1] = fig.add_subplot(gs[3, 1])
    axes[3, 2] = fig.add_subplot(gs[3, 2])
    axes[3, 3] = fig.add_subplot(gs[3, 3])

    ax_lat_all = axes[3, 0]
    ax_lat_lr = axes[3, 1]
    ax_psth_all_go = axes[3, 2]
    ax_psth_reward_go = axes[3, 3]

    # Row 4: (old row3) start-aligned PSTHs + rasters
    axes[4, 0] = fig.add_subplot(gs[4, 0])
    axes[4, 1] = fig.add_subplot(gs[4, 1])
    axes[4, 2] = fig.add_subplot(gs[4, 2])
    axes[4, 3] = fig.add_subplot(gs[4, 3])

    ax_psth_start_all = axes[4, 0]
    ax_psth_start_reward = axes[4, 1]
    ax_raster_all = axes[4, 2]
    ax_raster_reward = axes[4, 3]

    # Row 5: (old row4) ITI / delay / metrics summary
    axes[5, 0] = fig.add_subplot(gs[5, 0])
    axes[5, 1] = fig.add_subplot(gs[5, 1])
    axes[5, 2] = fig.add_subplot(gs[5, 2])
    axes[5, 3] = fig.add_subplot(gs[5, 3])

    ax_iti = axes[5, 0]
    ax_delay = axes[5, 1]
    ax_unused_1 = axes[5, 2]   # metrics summary left
    ax_unused_2 = axes[5, 3]   # metrics summary right

    # ---------------------------------------------------------
    # Row 2: logistic regression (coeffs + predictions)
    # ---------------------------------------------------------
    # coeffs
    if logreg_fig_coef is not None:
        _embed_figure_as_image(logreg_fig_coef, ax_log_coef)
        plt.close(logreg_fig_coef)
    else:
        ax_log_coef.axis("off")
        ax_log_coef.text(
            0.5,
            0.5,
            "Logistic regression unavailable",
            ha="center",
            va="center",
            transform=ax_log_coef.transAxes,
            fontsize=10,
        )

    # predicted vs actual
    if logreg_fig_pred is not None:
        _embed_figure_as_image(logreg_fig_pred, ax_log_pred)
        plt.close(logreg_fig_pred)
    else:
        ax_log_pred.axis("off")
        ax_log_pred.text(
            0.5,
            0.5,
            "Logistic regression unavailable",
            ha="center",
            va="center",
            transform=ax_log_pred.transAxes,
            fontsize=10,
        )


    # ---------------------------------------------------------
    # Row 3: first-lick latency + go-aligned PSTHs (old row2)
    # ---------------------------------------------------------
    _plot_hist_panel(
        ax_lat_all,
        lat_all,
        bins=bins,
        xlabel="First lick latency (s)",
        title="First lick latency (all trials)",
    )

    vals_left = lat_left[~np.isnan(lat_left)]
    vals_right = lat_right[~np.isnan(lat_right)]

    if vals_left.size + vals_right.size > 0:
        combined = np.concatenate([v for v in (vals_left, vals_right) if v.size > 0])
        bin_edges = np.histogram_bin_edges(combined, bins=bins)

        if vals_left.size > 0:
            ax_lat_lr.hist(vals_left, bins=bin_edges, alpha=0.5, label="Left choice")
        if vals_right.size > 0:
            ax_lat_lr.hist(vals_right, bins=bin_edges, alpha=0.5, label="Right choice")

    ax_lat_lr.set_xlabel("First lick latency (s)")
    ax_lat_lr.set_ylabel("Count")
    ax_lat_lr.set_title("First lick latency (left vs right)")
    ax_lat_lr.grid(True, alpha=0.3)
    ax_lat_lr.legend()

    plot_lick_rate_psth_after_iti(
        ax_psth_all_go,
        left_licks=left_licks,
        right_licks=right_licks,
        go_times=go_times,
        start_times=start_times,
        psth_window=psth_window_go,
        psth_bin_width=psth_bin_width,
    )

    plot_lick_rate_psth_by_reward_after_iti(
        ax_psth_reward_go,
        left_licks=left_licks,
        right_licks=right_licks,
        go_times=go_times,
        start_times=start_times,
        reward=reward,
        psth_window=psth_window_go,
        psth_bin_width=psth_bin_width,
    )

    # ---------------------------------------------------------
    # Row 4: start-aligned PSTHs + rasters (old row3)
    # ---------------------------------------------------------
    plot_lick_rate_psth_from_start_before_go(
        ax_psth_start_all,
        left_licks=left_licks,
        right_licks=right_licks,
        start_times=start_times,
        go_times=go_times,
        psth_window=psth_window_start,
        psth_bin_width=psth_bin_width,
    )

    plot_lick_rate_psth_from_start_by_reward_before_go(
        ax_psth_start_reward,
        left_licks=left_licks,
        right_licks=right_licks,
        start_times=start_times,
        go_times=go_times,
        reward=reward,
        psth_window=psth_window_start,
        psth_bin_width=psth_bin_width,
    )

    # Lick rasters (go-aligned)
    if t_rel_all.size > 0:
        mask_left = side_all == 0
        mask_right = side_all == 1

        if np.any(mask_left):
            ax_raster_all.scatter(
                t_rel_all[mask_left],
                trial_idx_all[mask_left],
                s=2,
                label="Left licks",
            )
        if np.any(mask_right):
            ax_raster_all.scatter(
                t_rel_all[mask_right],
                trial_idx_all[mask_right],
                s=2,
                label="Right licks",
            )

        ax_raster_all.axvline(0.0, linestyle=":", linewidth=1.0)
        ax_raster_all.set_xlabel("Time from go cue (s)")
        ax_raster_all.set_ylabel("Trial index")
        ax_raster_all.set_title("Lick raster (all trials)")
        ax_raster_all.grid(True, alpha=0.3)
        ax_raster_all.legend()
    else:
        ax_raster_all.set_title("Lick raster (no licks found)")
        ax_raster_all.set_xlabel("Time from go cue (s)")
        ax_raster_all.set_ylabel("Trial index")
        ax_raster_all.grid(True, alpha=0.3)

    if t_rel_all.size > 0 and reward.size > 0:
        idx_rewarded = np.where(reward)[0]
        idx_unrewarded = np.where(~reward)[0]
        order = np.concatenate([idx_rewarded, idx_unrewarded])

        if order.size > 0:
            rank = np.empty_like(order)
            rank[order] = np.arange(order.size)
            sorted_trial_idx_all = rank[trial_idx_all]

            lick_reward = reward[trial_idx_all]
            mask_rewarded = lick_reward
            mask_unrewarded = ~lick_reward

            if np.any(mask_rewarded):
                ax_raster_reward.scatter(
                    t_rel_all[mask_rewarded],
                    sorted_trial_idx_all[mask_rewarded],
                    s=2,
                    label="Rewarded trials",
                )
            if np.any(mask_unrewarded):
                ax_raster_reward.scatter(
                    t_rel_all[mask_unrewarded],
                    sorted_trial_idx_all[mask_unrewarded],
                    s=2,
                    label="Unrewarded trials",
                )

            ax_raster_reward.axvline(0.0, linestyle=":", linewidth=1.0)
            ax_raster_reward.set_xlabel("Time from go cue (s)")
            ax_raster_reward.set_ylabel("Sorted trial index")
            ax_raster_reward.set_title("Lick raster (grouped by reward)")
            ax_raster_reward.grid(True, alpha=0.3)
            ax_raster_reward.legend()
        else:
            ax_raster_reward.set_title("Lick raster (no valid trials)")
            ax_raster_reward.set_xlabel("Time from go cue (s)")
            ax_raster_reward.set_ylabel("Sorted trial index")
            ax_raster_reward.grid(True, alpha=0.3)
    else:
        ax_raster_reward.set_title("Lick raster (no licks found)")
        ax_raster_reward.set_xlabel("Time from go cue (s)")
        ax_raster_reward.set_ylabel("Sorted trial index")
        ax_raster_reward.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Row 5: ITI / delay / metrics summary (old row4)
    # ---------------------------------------------------------
    plot_iti_hist(ax_iti, iti_time, bins=bins)
    if np.isfinite(mean_iti):
        ax_iti.set_title(f"ITI duration (mean = {mean_iti:.2f} s)")

    plot_delay_hist(ax_delay, delay_time, bins=bins)
    if np.isfinite(mean_delay):
        ax_delay.set_title(f"Delay duration (mean = {mean_delay:.2f} s)")

    # ---------- grouped metrics text panels ----------
    ax_unused_1.clear()
    ax_unused_2.clear()
    ax_unused_1.axis("off")
    ax_unused_2.axis("off")

    skip_keys = {
        "first_lick_latency_all",
        "first_lick_latency_all_left",
        "first_lick_latency_all_right",
    }

    clean_metrics = {}
    for key, val in metrics.items():
        if key in skip_keys:
            continue

        if isinstance(val, (int, float, np.integer, np.floating)):
            if np.isnan(val):
                clean_metrics[key] = "NA"
            else:
                clean_metrics[key] = f"{float(val):.3f}"
        else:
            clean_metrics[key] = str(val)

    groups = {
        "Session": [
            "n_trials",
            "is_opto_session",
            "n_opto_trials",
        ],
        "Response / Choices": [
            "response_rate",
            "choice_fraction_left",
            "choice_fraction_right",
        ],
        "Reward": [
            "reward_fraction",
            "reward_fraction_left",
            "reward_fraction_right",
            "reward_volume_total",
            "reward_volume_left",
            "reward_volume_right",
        ],
        "Timing": [
            "mean_iti_time",
            "mean_delay_time",
            "mean_first_lick_latency",
            "mean_first_lick_latency_left",
            "mean_first_lick_latency_right",
            "early_lick_fraction",
        ],
        "Lick Fractions": [
            "lick_fraction_iti",
            "lick_fraction_iti_left",
            "lick_fraction_iti_right",
            "lick_fraction_delay",
            "lick_fraction_delay_left",
            "lick_fraction_delay_right",
        ],
        "Win–Stay / Lose–Switch": [
            "win_stay_rate",
            "win_stay_rate_left",
            "win_stay_rate_right",
            "lose_switch_rate",
            "lose_switch_rate_left",
            "lose_switch_rate_right",
        ],
    }

    grouped_lines = []
    used_keys = set()

    for group_name, keys in groups.items():
        available = [k for k in keys if k in clean_metrics]
        if not available:
            continue

        grouped_lines.append(f"{group_name}:")
        for k in available:
            grouped_lines.append(f"  {k}: {clean_metrics[k]}")
            used_keys.add(k)
        grouped_lines.append("")

    leftover = [k for k in clean_metrics.keys() if k not in used_keys]
    if leftover:
        grouped_lines.append("Other:")
        for k in leftover:
            grouped_lines.append(f"  {k}: {clean_metrics[k]}")
        grouped_lines.append("")

    if len(grouped_lines) == 0:
        grouped_lines = ["(no scalar metrics to display)"]

    mid = (len(grouped_lines) + 1) // 2
    left_text = "\n".join(grouped_lines[:mid])
    right_text = "\n".join(grouped_lines[mid:])

    ax_unused_1.text(
        0.0,
        1.0,
        left_text,
        transform=ax_unused_1.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontfamily="monospace",
    )

    ax_unused_2.text(
        0.0,
        1.0,
        right_text,
        transform=ax_unused_2.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontfamily="monospace",
    )

    # ---------------------------------------------------------
    # Row 0 & 1: behavior session panels (with / without model)
    # ---------------------------------------------------------
    fig_model = plot_behavior_session(
        nwb_data=nwb_data,
        model_alias="ForagingCompareThreshold",
        latent_name="right_choice_probability",
    )
    if fig_model is None:
        fig_model = plt.gcf()
    _embed_figure_as_image(fig_model, ax_behavior_model)
    ax_behavior_model.set_title("Behavior session (fit with ForagingCompareThreshold)")
    plt.close(fig_model)

    fig_nomodel = plot_behavior_session(
        nwb_data=nwb_data,
        model_alias=None,
        latent_name=None,
    )
    if fig_nomodel is None:
        fig_nomodel = plt.gcf()
    _embed_figure_as_image(fig_nomodel, ax_behavior_nomodel)
    ax_behavior_nomodel.set_title("Behavior session (no model)")
    plt.close(fig_nomodel)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes, metrics, trial_df



