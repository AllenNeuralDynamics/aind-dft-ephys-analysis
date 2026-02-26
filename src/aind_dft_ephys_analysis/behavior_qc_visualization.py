import io
import os
import ast
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
from typing import Any, Optional, Dict, Tuple, Sequence, Iterable, Union, List

from behavior_qc import compute_behavior_qc_from_nwb
from general_visualization import plot_behavior_session
from model_fitting import (
    fit_choice_logistic_regression_from_nwb,
    visualize_choice_logistic_regression,
)
from behavior_utils import generate_behavior_summary, get_fitted_latent, extract_fitted_data
from nwb_utils import NWBUtils

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

def _embed_figure_as_image(
    src_fig: plt.Figure,
    ax: plt.Axes,
    pad_inches: float = 0.08,   # small margin around the embedded figure
) -> None:
    """
    Render a Matplotlib Figure to an RGB image and display it in the target Axes.
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

    buf_io = io.BytesIO()
    try:
        src_fig.savefig(
            buf_io,
            format="png",
            bbox_inches="tight",
            pad_inches=pad_inches,   # ← was 0.0
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
    figsize: Tuple[float, float] = (18.0, 22.0),
    save_path: Optional[str] = None,
    psth_window_go: Tuple[float, float] = (-8.0, 4.0),
    psth_bin_width: float = 0.1,
    psth_window_start: Tuple[float, float] = (0.0, 8.0),
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
         lick raster (all)                | lick raster (reward / unreward / no-response)

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

    # Trial info (all aligned to trial_df length)
    n_trials = len(trial_df)
    go_times = np.asarray(trial_df["goCue_start_time"], dtype=float)
    start_times = np.asarray(trial_df["start_time"], dtype=float)
    reward = np.asarray(trial_df["reward"], dtype=bool)

    # Response / no response: 0 left, 1 right, 2 no response
    # Clip to n_trials in case NWB has extra trials
    animal_response = np.asarray(
        nwb_data.trials["animal_response"][:],
        dtype=float,
    )
    animal_response = animal_response[:n_trials]

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
    gs_log = gs[2, :].subgridspec(1, 2, width_ratios=[1, 1.0], wspace=0.05)
    ax_log_coef = fig.add_subplot(gs_log[0, 0])
    ax_log_pred = fig.add_subplot(gs_log[0, 1])

    axes[2, 0] = ax_log_coef
    axes[2, 1] = ax_log_coef
    axes[2, 2] = ax_log_pred
    axes[2, 3] = ax_log_pred

    # Row 3: latency + go-aligned PSTH
    axes[3, 0] = fig.add_subplot(gs[3, 0])
    axes[3, 1] = fig.add_subplot(gs[3, 1])
    axes[3, 2] = fig.add_subplot(gs[3, 2])
    axes[3, 3] = fig.add_subplot(gs[3, 3])

    ax_lat_all = axes[3, 0]
    ax_lat_lr = axes[3, 1]
    ax_psth_all_go = axes[3, 2]
    ax_psth_reward_go = axes[3, 3]

    # Row 4: start-aligned PSTHs + rasters
    axes[4, 0] = fig.add_subplot(gs[4, 0])
    axes[4, 1] = fig.add_subplot(gs[4, 1])
    axes[4, 2] = fig.add_subplot(gs[4, 2])
    axes[4, 3] = fig.add_subplot(gs[4, 3])

    ax_psth_start_all = axes[4, 0]
    ax_psth_start_reward = axes[4, 1]
    ax_raster_all = axes[4, 2]
    ax_raster_grouped = axes[4, 3]

    # Row 5: ITI / delay / metrics summary
    axes[5, 0] = fig.add_subplot(gs[5, 0])
    axes[5, 1] = fig.add_subplot(gs[5, 1])
    axes[5, 2] = fig.add_subplot(gs[5, 2])
    axes[5, 3] = fig.add_subplot(gs[5, 3])

    ax_iti = axes[5, 0]
    ax_delay = axes[5, 1]
    ax_metrics_left = axes[5, 2]
    ax_metrics_right = axes[5, 3]

    # ---------------------------------------------------------
    # Row 2: logistic regression (coeffs + predictions)
    # ---------------------------------------------------------
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
    # Row 3: first-lick latency + go-aligned PSTHs
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
    # Row 4: start-aligned PSTHs
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

    # ---------------------------------------------------------
    # Row 4: lick rasters (go-aligned)
    # ---------------------------------------------------------
    # Panel 1: all trials, colored by side
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
        ax_raster_all.legend(fontsize=8, markerscale=2, handlelength=1.5,framealpha=0.3)
    else:
        ax_raster_all.set_title("Lick raster (no licks found)")
        ax_raster_all.set_xlabel("Time from go cue (s)")
        ax_raster_all.set_ylabel("Trial index")
        ax_raster_all.grid(True, alpha=0.3)

    # Panel 2: grouped by reward / unreward / no response
    ax_raster_grouped.clear()
    if t_rel_all.size > 0 and reward.size > 0 and animal_response.size == reward.size:
        # Define per-trial groups:
        #   0: rewarded trials
        #   1: unrewarded but responded (animal_response 0 or 1)
        #   2: no response (animal_response == 2 or NaN)
        resp_codes = animal_response.copy()

        is_noresp = np.isnan(resp_codes) | (resp_codes == 2)
        is_responded = ~is_noresp
        is_rewarded = reward & is_responded
        is_unrewarded_resp = (~reward) & is_responded

        # Trial indices for each group
        idx_rewarded = np.where(is_rewarded)[0]
        idx_unrewarded = np.where(is_unrewarded_resp)[0]
        idx_noresp = np.where(is_noresp)[0]

        # Order trials: rewarded → unrewarded → no response
        order = np.concatenate([idx_rewarded, idx_unrewarded, idx_noresp])
        if order.size > 0:
            # Map original trial index → row index in raster
            rank = np.empty(n_trials, dtype=int)
            rank[order] = np.arange(order.size)

            # Per-trial group codes
            trial_group = np.full(n_trials, 2, dtype=int)  # default = no response
            trial_group[is_unrewarded_resp] = 1
            trial_group[is_rewarded] = 0

            # For each lick, get its trial group and new row index
            lick_groups = trial_group[trial_idx_all]
            sorted_trial_idx_all = rank[trial_idx_all]

            mask_g_reward = lick_groups == 0
            mask_g_unreward = lick_groups == 1
            mask_g_noresp = lick_groups == 2

            # Plot licks for each group (if any)
            if np.any(mask_g_reward):
                ax_raster_grouped.scatter(
                    t_rel_all[mask_g_reward],
                    sorted_trial_idx_all[mask_g_reward],
                    s=2,
                    label=f"Rewarded (n={idx_rewarded.size})",
                )
            else:
                # Dummy point to keep legend entry
                ax_raster_grouped.scatter([], [], s=2,
                    label=f"Rewarded (n={idx_rewarded.size})"
                )

            if np.any(mask_g_unreward):
                ax_raster_grouped.scatter(
                    t_rel_all[mask_g_unreward],
                    sorted_trial_idx_all[mask_g_unreward],
                    s=2,
                    label=f"Unrewarded resp (n={idx_unrewarded.size})",
                )
            else:
                ax_raster_grouped.scatter([], [], s=2,
                    label=f"Unrewarded resp (n={idx_unrewarded.size})"
                )

            if np.any(mask_g_noresp):
                ax_raster_grouped.scatter(
                    t_rel_all[mask_g_noresp],
                    sorted_trial_idx_all[mask_g_noresp],
                    s=2,
                    label=f"No response (n={idx_noresp.size})",
                )
            else:
                ax_raster_grouped.scatter([], [], s=2,
                    label=f"No response (n={idx_noresp.size})"
                )

            ax_raster_grouped.axvline(0.0, linestyle=":", linewidth=1.0)
            ax_raster_grouped.set_xlabel("Time from go cue (s)")
            ax_raster_grouped.set_ylabel("Sorted trial index")
            ax_raster_grouped.set_title("Lick raster\n(reward / unreward / no response)")
            ax_raster_grouped.grid(True, alpha=0.3)
            # Create legend and capture the Legend object
            leg = ax_raster_grouped.legend(
                fontsize=8,
                markerscale=2,
                handlelength=1.5,
                framealpha=0.3
            )

            # Adjust legend *text* transparency
            for text in leg.get_texts():
                text.set_alpha(0.5)
        else:
            ax_raster_grouped.set_title("Lick raster (no valid trials)")
            ax_raster_grouped.set_xlabel("Time from go cue (s)")
            ax_raster_grouped.set_ylabel("Sorted trial index")
            ax_raster_grouped.grid(True, alpha=0.3)
    else:
        ax_raster_grouped.set_title("Lick raster (no licks found)")
        ax_raster_grouped.set_xlabel("Time from go cue (s)")
        ax_raster_grouped.set_ylabel("Sorted trial index")
        ax_raster_grouped.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Row 5: ITI / delay / metrics summary
    # ---------------------------------------------------------
    plot_iti_hist(ax_iti, iti_time, bins=bins)
    if np.isfinite(mean_iti):
        ax_iti.set_title(f"ITI duration (mean = {mean_iti:.2f} s)")

    plot_delay_hist(ax_delay, delay_time, bins=bins)
    if np.isfinite(mean_delay):
        ax_delay.set_title(f"Delay duration (mean = {mean_delay:.2f} s)")

    # ---------- grouped metrics text panels ----------
    ax_metrics_left.clear()
    ax_metrics_right.clear()
    ax_metrics_left.axis("off")
    ax_metrics_right.axis("off")

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

    ax_metrics_left.text(
        0.0,
        1.0,
        left_text,
        transform=ax_metrics_left.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontfamily="monospace",
    )

    ax_metrics_right.text(
        0.0,
        1.0,
        right_text,
        transform=ax_metrics_right.transAxes,
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

# ---------------------------------------------------------------------
# Helper: pull one per-trial array out of the summary
# ---------------------------------------------------------------------
def _get_q_array_from_summary(summary: pd.DataFrame, col_name: str) -> np.ndarray:
    """
    Extract a 1D float array from a summary column like
    summary['QLearning_L1F1_CK1_softmax-QR'][0].

    Assumes each cell stores a list/ndarray of per-trial values.
    If the column does not exist, or there is no usable numeric data,
    returns an empty array.
    """
    if col_name not in summary.columns:
        return np.array([], dtype=float)

    col = summary[col_name]
    if len(col) == 0:
        return np.array([], dtype=float)

    cell = col.iloc[0]

    if cell is None:
        return np.array([], dtype=float)

    if isinstance(cell, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(cell, dtype=float).ravel()
        except (TypeError, ValueError):
            return np.array([], dtype=float)
    else:
        try:
            arr = np.array([float(cell)], dtype=float)
        except (TypeError, ValueError):
            return np.array([], dtype=float)

    if arr.size == 0:
        return np.array([], dtype=float)

    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([], dtype=float)

    return arr


def _get_model_field_array(
    summary: pd.DataFrame,
    model_name: str,
    field_key: str,
) -> np.ndarray:
    """
    Wrapper around _get_q_array_from_summary that is tolerant to suffixes.

    It first looks for an exact column name:
        f"{model_name}-{field_key}"

    If that does not exist, it searches for any column that:
        - starts with f"{model_name}-"
        - and whose suffix contains `field_key` as a substring.

    This allows, e.g., requesting field_key="value" and matching
    "ForagingCompareThreshold-value-1" automatically.
    """
    exact_name = f"{model_name}-{field_key}"
    if exact_name in summary.columns:
        return _get_q_array_from_summary(summary, exact_name)

    prefix = f"{model_name}-"
    candidates = [
        c for c in summary.columns
        if c.startswith(prefix) and field_key in c[len(prefix):]
    ]
    if not candidates:
        return np.array([], dtype=float)

    # Use the first candidate in sorted order for determinism
    chosen = sorted(candidates)[0]
    return _get_q_array_from_summary(summary, chosen)


# ---------------------------------------------------------------------
# Main function: hist row + scatter row per model
# ---------------------------------------------------------------------
def plot_qlearning_hist_and_scatter_from_nwb(
    nwb_data: Any,
    *,
    bins: int = 30,
    panel_width: float = 3,
    panel_height: float = 2.3,
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """
    For each model, plot:

      Row A (per model): histograms of model-specific latent variables
      Row B (per model): scatter plots of model-specific latent pairs

    Models included:
      - QLearning_L1F1_CK1_softmax
      - QLearning_L2F1_softmax
      - QLearning_L2F1_CK1_softmax
      - QLearning_L2F1_CKfull_softmax
      - ForagingCompareThreshold (value / RPE only)

    Any model-field combination that is missing or empty in `summary`
    is skipped and the corresponding axis is turned off.
    """
    # Fonts
    title_fs = 11          # per-panel title
    label_fs = 9           # axis labels
    tick_fs = 8            # tick labels
    main_title_fs = 18     # figure title

    # -----------------------------------------------------------------
    # 1) Get summary
    # -----------------------------------------------------------------
    summary = generate_behavior_summary(nwb_data)

    # -----------------------------------------------------------------
    # 2) Per-model configuration
    # -----------------------------------------------------------------
    model_configs = [
        {
            "name": "QLearning_L1F1_CK1_softmax",
            "label": "L1F1_CK1",
            "hist_fields": ["QL", "QR", "chosenQ", "unchosenQ", "deltaQ", "sumQ", "RPE"],
            "scatter_pairs": [
                ("chosenQ", "deltaQ"),
                ("chosenQ", "sumQ"),
                ("deltaQ",  "sumQ"),
                ("deltaQ",  "RPE"),
                ("sumQ",    "RPE"),
            ],
        },
        {
            "name": "QLearning_L2F1_softmax",
            "label": "L2F1",
            "hist_fields": ["QL", "QR", "chosenQ", "unchosenQ", "deltaQ", "sumQ", "RPE"],
            "scatter_pairs": [
                ("chosenQ", "deltaQ"),
                ("chosenQ", "sumQ"),
                ("deltaQ",  "sumQ"),
                ("deltaQ",  "RPE"),
                ("sumQ",    "RPE"),
            ],
        },
        {
            "name": "QLearning_L2F1_CK1_softmax",
            "label": "L2F1_CK1",
            "hist_fields": ["QL", "QR", "chosenQ", "unchosenQ", "deltaQ", "sumQ", "RPE"],
            "scatter_pairs": [
                ("chosenQ", "deltaQ"),
                ("chosenQ", "sumQ"),
                ("deltaQ",  "sumQ"),
                ("deltaQ",  "RPE"),
                ("sumQ",    "RPE"),
            ],
        },
        {
            "name": "QLearning_L2F1_CKfull_softmax",
            "label": "L2F1_CKFull",
            "hist_fields": ["QL", "QR", "chosenQ", "unchosenQ", "deltaQ", "sumQ", "RPE"],
            "scatter_pairs": [
                ("chosenQ", "deltaQ"),
                ("chosenQ", "sumQ"),
                ("deltaQ",  "sumQ"),
                ("deltaQ",  "RPE"),
                ("sumQ",    "RPE"),
            ],
        },
        # New non–Q-learning model: ForagingCompareThreshold
        {
            "name": "ForagingCompareThreshold",
            "label": "Foraging",
            # We conceptually want value and RPE; suffixes like "value-1" are handled
            "hist_fields": ["value", "RPE"],
            "scatter_pairs": [
                ("value", "RPE"),
            ],
        },
    ]

    n_models = len(model_configs)
    n_hist_cols = max(len(cfg["hist_fields"]) for cfg in model_configs)
    n_scatter_cols = max(len(cfg["scatter_pairs"]) for cfg in model_configs)
    n_cols = max(n_hist_cols, n_scatter_cols)

    # 2 rows per model
    n_rows = 2 * n_models

    # Figure size based on desired panel size
    fig_width = panel_width * n_cols
    fig_height = panel_height * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    # -----------------------------------------------------------------
    # 3) Fill panels
    # -----------------------------------------------------------------
    for m_idx, cfg in enumerate(model_configs):
        model_name = cfg["name"]
        model_label = cfg["label"]
        hist_fields = cfg["hist_fields"]
        scatter_pairs = cfg["scatter_pairs"]

        row_hist = 2 * m_idx
        row_scatter = 2 * m_idx + 1

        # ---------- Histogram row ----------
        for col, field in enumerate(hist_fields):
            ax = axes[row_hist, col]

            values = _get_model_field_array(summary, model_name, field)
            if values.size == 0:
                ax.axis("off")
                continue

            ax.hist(values, bins=bins)
            ax.set_title(f"{model_label} – {field}", fontsize=title_fs)
            ax.set_xlabel(field, fontsize=label_fs)
            ax.set_ylabel("Count", fontsize=label_fs)
            ax.tick_params(axis="both", labelsize=tick_fs)
            ax.grid(True, alpha=0.3)

        # Any extra histogram columns beyond hist_fields are turned off
        for col in range(len(hist_fields), n_cols):
            axes[row_hist, col].axis("off")

        # ---------- Scatter row ----------
        for col, (x_field, y_field) in enumerate(scatter_pairs):
            ax = axes[row_scatter, col]

            x_vals = _get_model_field_array(summary, model_name, x_field)
            y_vals = _get_model_field_array(summary, model_name, y_field)

            # Need both variables with data
            if (x_vals.size == 0) or (y_vals.size == 0):
                ax.axis("off")
                continue

            n = min(x_vals.size, y_vals.size)
            x_vals = x_vals[:n]
            y_vals = y_vals[:n]

            mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            x_vals = x_vals[mask]
            y_vals = y_vals[mask]

            if x_vals.size == 0:
                ax.axis("off")
                continue

            ax.scatter(x_vals, y_vals, s=8, alpha=0.5)
            ax.set_xlabel(x_field, fontsize=label_fs)
            ax.set_ylabel(y_field, fontsize=label_fs)
            ax.set_title(
                f"{model_label}: {x_field} vs {y_field}",
                fontsize=title_fs,
            )
            ax.tick_params(axis="both", labelsize=tick_fs)
            ax.grid(True, alpha=0.3)

        # Turn off any scatter columns beyond the defined pairs
        for col in range(len(scatter_pairs), n_cols):
            axes[row_scatter, col].axis("off")

    # -----------------------------------------------------------------
    # 4) Global formatting
    # -----------------------------------------------------------------
    fig.suptitle(
        "Q-learning and Foraging model latents: histograms and pairwise relationships",
        fontsize=main_title_fs,
        y=0.95,
    )

    fig.subplots_adjust(
        top=0.90,
        hspace=0.55,
        wspace=0.45,
    )

    return fig, axes, summary



def save_combined_behavior_and_qlearning_summary(
    nwb_data: Any,
    *,
    qc_kwargs: Optional[Dict] = None,
    q_kwargs: Optional[Dict] = None,
    rpe_kwargs: Optional[Dict] = None,          # passed to RPE regression only
    save_dir: Optional[str] = None,
    save_basepath: Optional[str] = "behavior_qlearning_summary",
    formats: Sequence[str] = ("pdf", "png", "eps"),
    dpi: int = 300,
    bins: Optional[int] = None,                 # ← NEW: shared bins for QC and Q-learning plots
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Generate:
      1) Behavior QC figure
      2) Q-learning / Foraging latent figure
      3) RPE history regression figure

    Combine them vertically and save in multiple formats.

    Parameters
    ----------
    qc_kwargs : dict or None
        Passed to plot_behavior_qc_summary.
    q_kwargs : dict or None
        Passed to plot_qlearning_hist_and_scatter_from_nwb.
    rpe_kwargs : dict or None
        Passed to plot_rpe_history_regression_from_nwb (in addition to nwb_data
        and the shared `summary`).
    save_dir : str or None
        Directory to save into (created automatically if missing).
        If None → saves in current working directory.
    save_basepath : str
        Base filename *without extension*.
    formats : list[str]
        Formats to save, e.g. ("pdf","png","eps").
    dpi : int
        Dots per inch for saved figures.
    bins : int or None
        If not None, used as the 'bins' argument for both
        plot_behavior_qc_summary and plot_qlearning_hist_and_scatter_from_nwb,
        unless those kwargs already specify their own 'bins'.
    """

    if qc_kwargs is None:
        qc_kwargs = {}
    if q_kwargs is None:
        q_kwargs = {}
    if rpe_kwargs is None:
        rpe_kwargs = {}

    # If a global 'bins' is provided, inject it into qc_kwargs and q_kwargs
    # without overwriting explicit per-call choices.
    if bins is not None:
        qc_kwargs.setdefault("bins", bins)
        q_kwargs.setdefault("bins", bins)

    # ----------------------------------------------------------
    # 1. Generate the three figures
    # ----------------------------------------------------------
    fig_qc, _, _, _ = plot_behavior_qc_summary(
        nwb_data,
        **qc_kwargs,
    )

    fig_q, _, summary = plot_qlearning_hist_and_scatter_from_nwb(
        nwb_data,
        **q_kwargs,
    )

    # Re-use the same summary for the RPE regression
    fig_rpe, _, _ = plot_rpe_history_regression_from_nwb(
        nwb_data,
        summary=summary,
        **rpe_kwargs,
    )

    # ----------------------------------------------------------
    # 2. Compute combined figure size
    # ----------------------------------------------------------
    w_qc, h_qc = fig_qc.get_size_inches()
    w_q, h_q = fig_q.get_size_inches()
    w_rpe, h_rpe = fig_rpe.get_size_inches()

    fig_width = max(w_qc, w_q, w_rpe)
    fig_height = h_qc + h_q + h_rpe

    combined_fig = plt.figure(figsize=(fig_width, fig_height))
    gs = combined_fig.add_gridspec(
        3, 1,
        height_ratios=[h_qc, h_q, h_rpe],
    )

    ax_top = combined_fig.add_subplot(gs[0, 0])
    ax_mid = combined_fig.add_subplot(gs[1, 0])
    ax_bottom = combined_fig.add_subplot(gs[2, 0])

    # ----------------------------------------------------------
    # 3. Embed the figures as images
    # ----------------------------------------------------------
    _embed_figure_as_image(fig_qc, ax_top)
    _embed_figure_as_image(fig_q, ax_mid)
    _embed_figure_as_image(fig_rpe, ax_bottom)

    # Close individual figs to free memory
    plt.close(fig_qc)
    plt.close(fig_q)
    plt.close(fig_rpe)

    # Put the title closer to the panels and give subplots more vertical space
    combined_fig.suptitle(
        "Behavior QC and model latent summaries\n",
        fontsize=20,
        y=0.985,  # closer to the top edge
    )

    # rect = [left, bottom, right, top]; increase "top" so axes go higher
    combined_fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.98])


    # ----------------------------------------------------------
    # 4. Save
    # ----------------------------------------------------------
    if save_basepath is not None:

        if save_dir is None:
            save_dir = "."
        os.makedirs(save_dir, exist_ok=True)

        full_base = os.path.join(save_dir, save_basepath)

        for ext in formats:
            ext = ext.replace(".", "").lower()
            if not ext:
                continue

            outfile = f"{full_base}.{ext}"

            if ext in ("pdf", "eps", "svg"):
                # Vector formats → do NOT use tight bbox
                combined_fig.savefig(outfile, dpi=dpi)
            else:
                # Raster formats → use bbox tight
                combined_fig.savefig(
                    outfile,
                    dpi=dpi,
                    bbox_inches="tight",
                    pad_inches=0.05,
                )

            print(f"Saved: {outfile}")

    return combined_fig, summary



# ---------------------------------------------------------------------
# Helper: RPE history regression
# ---------------------------------------------------------------------
def _fit_rpe_history_regression(
    rpe: np.ndarray,
    reward: np.ndarray,
    choice: np.ndarray,
    max_lag: int,
    target_mask: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fit a linear regression:

        RPE(t) ~ bias
                 + reward(t) + reward(t-1) + ... + reward(t-max_lag)
                 + choice(t) + choice(t-1) + ... + choice(t-max_lag)

    Parameters
    ----------
    rpe : array-like
        Per-trial RPE values (length T).
    reward : array-like
        Per-trial reward values (length T).
    choice : array-like
        Per-trial choice values (length T).
    max_lag : int
        Maximum history lag (in trials) to include.
    target_mask : array-like of bool, optional
        Length T. If provided, only trials t for which target_mask[t] is True
        can be used as regression targets (RPE(t)).
        History terms always come from the full sequences.

    Returns
    -------
    dict or None
        {
          "bias": float,
          "reward_coefs": np.ndarray of shape (max_lag + 1,),
          "choice_coefs": np.ndarray of shape (max_lag + 1,),
          "max_lag": int,
          "n_samples": int,
        }
        or None if there is not enough data to fit.

    Notes
    -----
    rpe, reward and choice must have the same length. If not, a ValueError is raised.
    """
    # ------------------------------------------------------------------
    # 0) Basic checks and masking
    # ------------------------------------------------------------------
    rpe = np.asarray(rpe, dtype=float).ravel()
    reward = np.asarray(reward, dtype=float).ravel()
    choice = np.asarray(choice, dtype=float).ravel()

    # Enforce identical length
    if not (rpe.size == reward.size == choice.size):
        raise ValueError(
            f"rpe, reward and choice must have the same length, got "
            f"len(rpe)={rpe.size}, len(reward)={reward.size}, len(choice)={choice.size}"
        )

    T = rpe.size
    # Need at least more than max_lag trials to have any target with full history
    if T <= max_lag + 1:
        return None

    base_valid = np.isfinite(rpe) & np.isfinite(reward) & np.isfinite(choice)

    if target_mask is not None:
        target_mask = np.asarray(target_mask, dtype=bool).ravel()
        if target_mask.size > T:
            target_mask = target_mask[:T]
        elif target_mask.size < T:
            target_mask = np.pad(
                target_mask,
                (0, T - target_mask.size),
                constant_values=False,
            )
        target_valid = base_valid & target_mask
    else:
        target_valid = base_valid

    # ------------------------------------------------------------------
    # 1) Candidate indices: must have full history t, t-1, ..., t-max_lag
    # ------------------------------------------------------------------
    candidate_indices = np.arange(T)
    candidate_indices = candidate_indices[candidate_indices >= max_lag]
    candidate_indices = candidate_indices[target_valid[candidate_indices]]

    if candidate_indices.size == 0:
        return None

    # We now include lag 0..max_lag → (max_lag+1) history terms per regressor
    n_lags = max_lag + 1
    n_samples = candidate_indices.size
    X = np.zeros((n_samples, 1 + 2 * n_lags), dtype=float)
    y = np.zeros(n_samples, dtype=float)

    # ------------------------------------------------------------------
    # 2) Build design matrix
    # ------------------------------------------------------------------
    for row_idx, t in enumerate(candidate_indices):
        # indices t, t-1, ..., t-max_lag
        lag_indices = np.arange(0, max_lag + 1)
        r_hist = reward[t - lag_indices]
        c_hist = choice[t - lag_indices]

        # If any history element is non-finite, mark this row invalid
        if not (np.all(np.isfinite(r_hist)) and np.all(np.isfinite(c_hist))):
            y[row_idx] = np.nan
            continue

        X[row_idx, 0] = 1.0  # bias
        X[row_idx, 1 : 1 + n_lags] = r_hist
        X[row_idx, 1 + n_lags : 1 + 2 * n_lags] = c_hist
        y[row_idx] = rpe[t]

    # ------------------------------------------------------------------
    # 3) Remove invalid rows and solve least-squares
    # ------------------------------------------------------------------
    valid_rows = np.isfinite(y)
    if np.count_nonzero(valid_rows) < (max_lag + 1):
        return None

    X = X[valid_rows]
    y = y[valid_rows]
    n_samples = y.size

    if n_samples <= (max_lag + 1):
        return None

    try:
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    # Layout:
    #   beta[0]                → bias
    #   beta[1 : 1+n_lags]     → reward(t .. t-max_lag)
    #   beta[1+n_lags : ...]   → choice(t .. t-max_lag)
    bias = float(beta[0])
    reward_coefs = beta[1 : 1 + n_lags]
    choice_coefs = beta[1 + n_lags : 1 + 2 * n_lags]

    return {
        "bias": bias,
        "reward_coefs": reward_coefs,
        "choice_coefs": choice_coefs,
        "max_lag": max_lag,
        "n_samples": int(n_samples),
    }



def _get_reward_and_choice_from_nwb(
    nwb_data: Any,
    max_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-trial reward and choice arrays from NWB.

    NO-RESPONSE TRIALS ARE REMOVED:
        animal_response == 2 or NaN → trial dropped.
    This keeps reward and choice aligned with RPE arrays,
    which already exclude no-response trials.

    Returns reward_raw and choice arrays AFTER filtering.
    """

    # Defensive checks
    if nwb_data is None:
        raise ValueError("nwb_data is None. Behavior NWB could not be loaded for this session.")
    if not hasattr(nwb_data, "trials") or nwb_data.trials is None:
        raise ValueError("nwb_data.trials is missing. This does not look like a valid behavior NWB.")



    # Rewarded left/right
    rewarded_L = np.asarray(
        nwb_data.trials["rewarded_historyL"][:],
        dtype=bool,
    )
    rewarded_R = np.asarray(
        nwb_data.trials["rewarded_historyR"][:],
        dtype=bool,
    )

    # Raw choice codes
    choice_raw = np.asarray(
        nwb_data.trials["animal_response"][:],
        dtype=float,
    )

    # ----------------------------------------------------
    # Remove no-response trials BEFORE constructing arrays
    # ----------------------------------------------------
    valid_mask = np.isfinite(choice_raw) & (choice_raw != 2)

    rewarded_L = rewarded_L[valid_mask]
    rewarded_R = rewarded_R[valid_mask]
    choice_raw = choice_raw[valid_mask]

    # ----------------------------------------------------
    # Apply max_len *after* filtering
    # ----------------------------------------------------
    if max_len is not None:
        rewarded_L = rewarded_L[:max_len]
        rewarded_R = rewarded_R[:max_len]
        choice_raw = choice_raw[:max_len]

    # ----------------------------------------------------
    # Compute reward
    # ----------------------------------------------------
    reward_raw = (rewarded_L | rewarded_R).astype(float)

    # ----------------------------------------------------
    # Map choice codes
    # 0 → -1  (left)
    # 1 → +1  (right)
    # ----------------------------------------------------
    choice = np.zeros_like(choice_raw, dtype=float)
    choice[choice_raw == 0] = -1.0
    choice[choice_raw == 1] =  1.0

    return reward_raw, choice


def plot_rpe_history_regression_from_nwb(
    nwb_data: Any,
    *,
    summary: Optional[pd.DataFrame] = None,
    session_name: str=None,
    max_lag: int = 8,
    panel_width: float = 3.2,
    panel_height: float = 2.3,
    make_figure: bool = True,
    show_figure: bool = True,
):
    """
    Plot regression:

        RPE(t) ~ bias
                 + reward(t .. t-max_lag)
                 + choice(t .. t-max_lag)

    for each model and for several trial subsets.

    Parameters
    ----------
    nwb_data : Any
        NWB-like object with trials and acquisition data.
    max_lag : int
        Maximum number of history lags (in trials) to include.
    panel_width, panel_height : float
        Size of each panel in inches.
    make_figure : bool, default True
        Whether to construct the matplotlib figure and axes.
        If False, plotting is skipped and only coefficients are returned.
    show_figure : bool, default True
        Whether to display the figure via plt.show(). Ignored if
        make_figure is False.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    axes : np.ndarray of Axes or None
    coeffs : dict
        Nested dict: coeffs[model_name][subset_key] -> regression dict.
    """

    model_configs = [
        {"name": "QLearning_L1F0_CKfull_softmax", "label": "L1F0_CKFull"},
        {"name": "QLearning_L1F1_CK1_softmax",    "label": "L1F1_CK1"},
        {"name": "QLearning_L1F1_CKfull_softmax", "label": "L1F1_CKFull"},
        {"name": "QLearning_L2F1_softmax",        "label": "L2F1"},
        {"name": "QLearning_L2F1_CK1_softmax",    "label": "L2F1_CK1"},
        {"name": "QLearning_L2F1_CKfull_softmax", "label": "L2F1_CKFull"},
        {"name": "ForagingCompareThreshold",      "label": "Foraging"},
    ]


    # ---------------------------------------------------
    # 1) Obtain summary
    # ---------------------------------------------------
    if session_name is None:
        session_name = getattr(nwb_data, "session_id", None)
        if session_name is None:
            raise ValueError("session_name is None and nwb_data.session_id is missing.")

    subsets = [
        ("all",         "All trials"),
        ("rpe_pos",     "RPE > 0"),
        ("rpe_neg",     "RPE < 0"),
        ("choice_left", "Left choices"),
        ("choice_right","Right choices"),
    ]

    n_models = len(model_configs)
    n_conditions = len(subsets)

    # ---------------------------------------------------
    # 2) Create figure only if requested
    # ---------------------------------------------------
    if make_figure:
        fig_width = panel_width * n_conditions
        fig_height = panel_height * n_models

        fig, axes = plt.subplots(
            n_models,
            n_conditions,
            figsize=(fig_width, fig_height),
            squeeze=False,
            sharex=False,
            constrained_layout=False,
        )
    else:
        fig = None
        axes = None

    reward_all, choice_all = _get_reward_and_choice_from_nwb(nwb_data)
    coeffs: Dict[str, Dict[str, Any]] = {}

    # =====================================================
    # 3) LOOP: models
    # =====================================================
    for m_idx, cfg in enumerate(model_configs):
        model_name = cfg["name"]
        model_label = cfg["label"]

        res = get_fitted_latent(session_name=session_name,model_alias=model_name)
        if res is None:
            continue
        lat = res.get("fitted_latent_variables", {})
        if "rpe" not in lat:
            # No RPE saved for this model/session → skip
            rpe=extract_fitted_data(nwb_behavior_data=nwb_data,fitted_latent=res,model_alias=model_name,latent_name='RPE')
        else:
            rpe = np.asarray(lat["rpe"], dtype=float).ravel()

        coeffs[model_name] = {}

        if rpe.size == 0:
            continue

        reward = reward_all
        choice = choice_all
        T = rpe.size
        row_values: list[float] = []

        for c_idx, (subset_key, subset_label) in enumerate(subsets):
            if subset_key == "all":
                target_mask = np.ones(T, dtype=bool)
            elif subset_key == "rpe_pos":
                target_mask = rpe > 0
            elif subset_key == "rpe_neg":
                target_mask = rpe < 0
            elif subset_key == "choice_left":
                target_mask = choice[:T] < 0
            elif subset_key == "choice_right":
                target_mask = choice[:T] > 0
            else:
                target_mask = np.ones(T, dtype=bool)

            try:
                reg = _fit_rpe_history_regression(
                    rpe=rpe,
                    reward=reward,
                    choice=choice,
                    max_lag=max_lag,
                    target_mask=target_mask,
                )
            except ValueError as e:
                print(
                    f"Warning in plot_rpe_history_regression_from_nwb "
                    f"for model {model_name}, subset {subset_key}: {e}"
                )
                reg = None

            coeffs[model_name][subset_key] = reg or {}

            if (not make_figure) or reg is None:
                continue

            ax = axes[m_idx, c_idx]

            reward_coefs = np.asarray(reg["reward_coefs"], dtype=float)
            choice_coefs = np.asarray(reg["choice_coefs"], dtype=float)

            n_lags_eff = min(reward_coefs.size, choice_coefs.size)
            if n_lags_eff == 0:
                ax.axis("off")
                continue

            reward_coefs = reward_coefs[:n_lags_eff]
            choice_coefs = choice_coefs[:n_lags_eff]
            lags = np.arange(n_lags_eff)

            row_values.extend(reward_coefs)
            row_values.extend(choice_coefs)

            ax.axhline(0, ls="--", lw=0.7, color="k", alpha=0.7)
            ax.plot(lags, reward_coefs, marker="o", label="Reward history")
            ax.plot(lags, choice_coefs, marker="s", label="Choice history")

            ax.set_xticks(lags)
            ax.tick_params(axis="both", labelsize=8)

            if c_idx == 0:
                ax.set_ylabel(f"{model_label}\nCoefficient", fontsize=10)

            ax.set_title(
                f"{subset_label}\n"
                f"bias = {reg['bias']:.3f}, n = {reg['n_samples']}",
                fontsize=10,
            )

            if c_idx == 0:
                ax.legend(fontsize=7, framealpha=0.4, loc="upper right")

            ax.grid(True, alpha=0.3)

        if make_figure and row_values:
            ymin = np.min(row_values) * 1.1
            ymax = np.max(row_values) * 1.1
            for c_idx in range(n_conditions):
                if axes[m_idx, c_idx].has_data():
                    axes[m_idx, c_idx].set_ylim(ymin, ymax)

    # -----------------------------------------------------
    # 4) Final layout & display
    # -----------------------------------------------------
    if make_figure:
        fig.tight_layout(rect=[0.03, 0.08, 0.97, 0.90])
        fig.suptitle(
            "RPE(t) ~ Reward and choice history\nPer model and subset",
            fontsize=16,
            y=0.96,
        )
        fig.supxlabel("Lag index (0 = most recent)", fontsize=14, y=0.04)

        if show_figure:
            plt.show()

    return fig, axes, coeffs






def plot_x_vs_y_by_stage(
    summary,
    models: Optional[Sequence[str]] = None,
    *,
    suffix_x: str = "LPT",
    suffix_y: str = "reward_coefs_sum6",
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    stage: Optional[Union[str, Sequence[Optional[str]]]] = None,
    min_points: int = 3,
    alpha: float = 0.6,
    figsize_per_panel: tuple = (4, 4),
    xlim_percentiles: Optional[Tuple[float, float]] = None,
    ylim_percentiles: Optional[Tuple[float, float]] = None,
    show_grid: bool = True,
    show: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Plot x vs y in a multi-panel figure.
    Rows correspond to training stages, columns correspond to models.

    This function visualizes the relationship between two summary metrics
    (x and y) across different behavioral models and training stages.
    For each stage–model combination, paired observations are filtered,
    optionally trimmed by percentile thresholds, and then used to fit
    a simple linear regression that is overlaid on a scatter plot.

    Parameters
    ----------
    summary : pd.DataFrame
        Summary table containing model-specific metrics and the column
        'auto_train_stage'.
    models : sequence of str, optional
        List of model names. Each model corresponds to one column in the
        multi-panel figure. If None, a default model list is used.
    suffix_x : str, default "LPT"
        Suffix used to construct the x-axis column name as
        f"{model}_{suffix_x}" when `x_col` is not provided.
    suffix_y : str, default "reward_coefs_sum6"
        Suffix used to construct the y-axis column name as
        f"{model}_{suffix_y}" when `y_col` is not provided.
    x_col : str, optional
        Explicit column name to use for the x-axis. If provided, this column
        is used for all models and overrides `suffix_x`. If None, the x-axis
        column is constructed separately for each model using `suffix_x`.
    y_col : str, optional
        Explicit column name to use for the y-axis. If provided, this column
        is used for all models and overrides `suffix_y`. If None, the y-axis
        column is constructed separately for each model using `suffix_y`.
    stage : str or sequence of str or None, optional
        Training stage or stages to include, matched against the
        'auto_train_stage' column. If None, all observed stages are plotted.
        Missing stages (NaN) are labeled as "None".
    min_points : int, default 3
        Minimum number of paired (x, y) observations required to perform
        linear regression and generate a plot for a given model–stage panel.
    alpha : float, default 0.6
        Transparency level for scatter points.
    figsize_per_panel : tuple of (float, float), default (4, 4)
        Size (width, height) of each subplot panel in inches.
        The total figure size scales with the number of models and stages.
    xlim_percentiles : tuple of (float, float), optional
        Fractional percentile range (low, high) in the interval [0, 1] used
        to filter x-axis values before plotting and linear fitting.
        For example, (0.05, 0.95) keeps the central 90% of x values.
        Values must satisfy 0 <= low < high <= 1.
        If None, no percentile-based filtering is applied to x.
    ylim_percentiles : tuple of (float, float), optional
        Fractional percentile range (low, high) in the interval [0, 1] used
        to filter y-axis values before plotting and linear fitting.
        For example, (0.05, 0.95) keeps the central 90% of y values.
        Values must satisfy 0 <= low < high <= 1.
        If None, no percentile-based filtering is applied to y.
    show_grid : bool, default True
        Whether to display a grid on each subplot.
    show : bool, default True
        Whether to call `plt.show()` to display the figure.

    Returns
    -------
    results : dict
        Nested dictionary with structure:
        results[stage_label][model] = {
            "slope": float,
            "intercept": float,
            "r2": float,
            "n": int
        }

        where:
        - stage_label is the training stage (or "None" for NaN),
        - model is the model name,
        - slope and intercept define the fitted linear regression,
        - r2 is the coefficient of determination,
        - n is the number of data points used in the fit.

    Notes
    -----
    - NaN values are removed before any filtering or fitting.
    - Percentile-based filtering (if specified) is applied before checking
      `min_points` and affects both visualization and regression.
    - x and y percentile filters are applied independently; if both are
      provided, a data point must satisfy both constraints to be retained.
    """
    if models is None:
        models = [
            "QLearning_L1F1_CK1_softmax",
            "QLearning_L2F1_softmax",
            "QLearning_L2F1_CK1_softmax",
            "QLearning_L2F1_CKfull_softmax",
            "ForagingCompareThreshold",
            "QLearning_L1F0_CKfull_softmax",
            "QLearning_L1F1_CKfull_softmax",
        ]

    if "auto_train_stage" not in summary.columns:
        raise ValueError("summary must contain column 'auto_train_stage'.")

    # Validate percentile inputs (fractions in [0, 1])
    if xlim_percentiles is not None:
        lo, hi = xlim_percentiles
        if not (0 <= lo < hi <= 1):
            raise ValueError("xlim_percentiles must satisfy 0 <= low < high <= 1.")

    if ylim_percentiles is not None:
        lo, hi = ylim_percentiles
        if not (0 <= lo < hi <= 1):
            raise ValueError("ylim_percentiles must satisfy 0 <= low < high <= 1.")

    # Determine stages to plot
    if stage is None:
        stage_series = summary["auto_train_stage"]
        stage_list = (
            stage_series.astype(object)
            .where(stage_series.notna(), other="None")
            .unique()
            .tolist()
        )
        stage_list = sorted(stage_list, key=str)
    else:
        stage_list = list(stage) if isinstance(stage, (list, tuple, set)) else [stage]

    n_rows = len(stage_list)
    n_cols = len(models)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for i, st in enumerate(stage_list):
        stage_label = "None" if st is None else str(st)

        if stage_label == "None":
            df_stage = summary[summary["auto_train_stage"].isna()]
        else:
            df_stage = summary[summary["auto_train_stage"] == st]

        results[stage_label] = {}

        for j, model in enumerate(models):
            ax = axes[i, j]

            x_col_use = x_col if x_col is not None else f"{model}_{suffix_x}"
            y_col_use = y_col if y_col is not None else f"{model}_{suffix_y}"

            if x_col_use not in df_stage.columns or y_col_use not in df_stage.columns:
                ax.set_title(f"{model}\n(missing columns)")
                ax.axis("off")
                continue

            df_xy = df_stage[[x_col_use, y_col_use]].dropna()

            # Fractional percentile-based filtering (affects both fitting and visualization)
            if xlim_percentiles is not None and not df_xy.empty:
                x_lo, x_hi = np.percentile(
                    df_xy[x_col_use].to_numpy(), [xlim_percentiles[0] * 100, xlim_percentiles[1] * 100]
                )
                df_xy = df_xy[
                    (df_xy[x_col_use] >= x_lo) & (df_xy[x_col_use] <= x_hi)
                ]

            if ylim_percentiles is not None and not df_xy.empty:
                y_lo, y_hi = np.percentile(
                    df_xy[y_col_use].to_numpy(), [ylim_percentiles[0] * 100, ylim_percentiles[1] * 100]
                )
                df_xy = df_xy[
                    (df_xy[y_col_use] >= y_lo) & (df_xy[y_col_use] <= y_hi)
                ]

            x = df_xy[x_col_use].to_numpy()
            y = df_xy[y_col_use].to_numpy()
            n = len(x)

            if n < min_points:
                ax.set_title(f"{model}\n(n={n}, insufficient)")
                ax.axis("off")
                continue

            # Linear fit
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            # Plot
            ax.scatter(x, y, alpha=alpha)
            x_sorted = np.sort(x)
            ax.plot(x_sorted, slope * x_sorted + intercept, linewidth=2)

            ax.set_title(f"{model}\n$R^2$={r2:.3f}, n={n}")
            ax.set_xlabel(x_col_use)
            ax.set_ylabel(y_col_use)

            if show_grid:
                ax.grid(True)

            results[stage_label][model] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": float(r2) if np.isfinite(r2) else np.nan,
                "n": int(n),
            }

    fig.suptitle(
        f"{x_col if x_col is not None else suffix_x} vs "
        f"{y_col if y_col is not None else suffix_y}",
        y=1.02,
    )

    plt.tight_layout()

    if show:
        plt.show()

    return results


def load_behavior_model_summary_csv(
    csv_path: str,
    *,
    parse_arrays: bool = True,
    array_cols: Optional[Sequence[str]] = None,
    keep_strings_if_parse_fails: bool = True,
) -> pd.DataFrame:
    """
    Load the wide-format behavior model summary saved by:
        summary.to_csv(output_path, index=False)

    CSV stores numpy arrays as strings. This loader converts those strings back
    into np.ndarray (object-dtype columns), handling common formats including:
      1) Python literal lists: "[1, 2, 3]"
      2) numpy-style strings without commas (your case): "[ 1.0  2.0\\n 3.0 ]"
      3) "array([ ... ])"

    Parameters
    ----------
    csv_path : str
        Path to the saved CSV.
    parse_arrays : bool, default True
        If True, attempt to parse array-like columns that were written as strings.
    array_cols : sequence of str or None, default None
        Columns to parse as arrays. If None, auto-detect columns:
          - endswith("_reward_coefs") or endswith("_choice_coefs") or contains("_coefs")
    keep_strings_if_parse_fails : bool, default True
        If parsing fails for a cell, keep the original string (True) or set NaN (False).

    Returns
    -------
    df : pd.DataFrame
        Loaded DataFrame with optional array columns parsed into np.ndarray objects.
    """
    df = pd.read_csv(csv_path)

    if not parse_arrays or df.empty:
        return df

    if array_cols is None:
        array_cols = [
            c for c in df.columns
            if (c.endswith("_reward_coefs") or c.endswith("_choice_coefs") or ("_coefs" in c))
        ]

    def _parse_cell(x):
        # Preserve already-parsed arrays/lists
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float).ravel()
            return arr if arr.size > 0 else np.nan

        # Handle NaN/None
        if x is None:
            return np.nan
        if isinstance(x, float) and np.isnan(x):
            return np.nan

        # Non-string scalar: try float
        if not isinstance(x, str):
            try:
                return float(x)
            except Exception:
                return x if keep_strings_if_parse_fails else np.nan

        s = x.strip()
        if s == "" or s.lower() == "nan":
            return np.nan

        # Remove wrapping quotes sometimes introduced by CSV
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            s = s[1:-1].strip()

        # Case A: "array([...])"
        s_for_eval = s
        if s_for_eval.startswith("array(") and s_for_eval.endswith(")"):
            s_for_eval = s_for_eval[len("array(") : -1].strip()

        # Case B: Python literal list/tuple (commas present) -> ast.literal_eval
        # This will NOT work for numpy-style "[ 1  2  3 ]" without commas.
        if ("," in s_for_eval) or s_for_eval.startswith("(") or s_for_eval.startswith("["):
            try:
                obj = ast.literal_eval(s_for_eval)
                if isinstance(obj, (list, tuple, np.ndarray)):
                    arr = np.asarray(obj, dtype=float).ravel()
                    return arr if arr.size > 0 else np.nan
                # Scalar literal
                return float(obj)
            except Exception:
                pass

        # Case C: numpy-style space/newline separated numbers, e.g.
        # "[ 1.0  2.0\\n 3.0 ]"
        s_num = s_for_eval
        if s_num.startswith("[") and s_num.endswith("]"):
            s_num = s_num[1:-1].strip()

        # First try whitespace-separated parsing
        arr = np.fromstring(s_num, sep=" ", dtype=float)

        # Fallback: sometimes values are separated by commas but ast failed
        if arr.size == 0 and ("," in s_num):
            arr = np.fromstring(s_num.replace(",", " "), sep=" ", dtype=float)

        if arr.size > 0:
            return arr

        return s if keep_strings_if_parse_fails else np.nan

    for c in array_cols:
        if c in df.columns:
            df[c] = df[c].apply(_parse_cell)

    return df


def plot_summary_column_distributions(
    summary: pd.DataFrame,
    columns: Sequence[str],
    *,
    bins: int = 30,
    figsize_per_panel: Tuple[float, float] = (4.2, 3.2),
    ncols: int = 4,
    titles: Optional[Dict[str, str]] = None,
    dropna: bool = True,
    # ---- Per-panel x-range controls ----
    xlim_percentiles: Optional[Tuple[float, float]] = (1.0, 99.0),
    xlim_pad_frac: float = 0.05,
    xlim_override: Optional[Dict[str, Tuple[float, float]]] = None,
    # ---- NEW: annotate how much data remains after range filtering ----
    annotate_range_filter: bool = True,
    annotate_loc: Tuple[float, float] = (0.98, 0.78),
    show_grid: bool = True,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray, Dict[str, Dict[str, Union[int, float]]]]:
    """
    Plot distributions (histograms) for selected columns in a summary DataFrame.

    This function is intended for scalar numeric columns (one value per session/row).
    If a column cannot be coerced to numeric (or has no finite values), its panel is disabled.

    Parameters
    ----------
    summary : pd.DataFrame
        Wide-format summary table.
    columns : sequence of str
        Column names to plot.
    bins : int
        Number of histogram bins.
    figsize_per_panel : (float, float)
        Size (width, height) per subplot panel in inches.
    ncols : int
        Number of subplot columns in the grid.
    titles : dict[str, str] or None
        Optional mapping from column name -> display title.
    dropna : bool
        If True, drop NaN/inf values before plotting and stats.
    xlim_percentiles : (low, high) or None
        If provided, per-panel x-limits are computed from these percentiles of the data.
        Example: (1, 99). If None, no percentile-based x-limits are applied unless overridden.
    xlim_pad_frac : float
        Fractional padding to apply around percentile-derived limits.
        Example: 0.05 expands the range by 5% on each side.
    xlim_override : dict[str, (xmin, xmax)] or None
        Per-column explicit x-limits. If provided for a column, overrides percentile behavior.
    annotate_range_filter : bool
        If True, annotate each panel with how many values are retained after x-range filtering.
    annotate_loc : (float, float)
        Annotation location in axes coordinates (x, y). Default places it below the stats box.
    show_grid : bool
        If True, show a light grid.
    show : bool
        If True, calls plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of Axes
    stats : dict
        Per-column summary stats (computed on the plotted values):
        {"n", "mean", "std", "min", "max"}.
    """
    if summary is None or not isinstance(summary, pd.DataFrame):
        raise ValueError("summary must be a pandas DataFrame.")
    if columns is None or len(columns) == 0:
        raise ValueError("columns must be a non-empty sequence of column names.")
    if xlim_percentiles is not None:
        p_lo, p_hi = xlim_percentiles
        if not (0.0 <= p_lo < p_hi <= 100.0):
            raise ValueError("xlim_percentiles must satisfy 0 <= low < high <= 100.")
    if xlim_pad_frac < 0:
        raise ValueError("xlim_pad_frac must be >= 0.")

    n_plots = len(columns)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    stats: Dict[str, Dict[str, Union[int, float]]] = {}
    xlim_override = xlim_override or {}

    for idx, col in enumerate(columns):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        if col not in summary.columns:
            ax.axis("off")
            ax.set_title(f"{col}\n(missing)", fontsize=10)
            continue

        # Coerce to numeric; non-numeric becomes NaN
        values_all = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=float)

        if dropna:
            values_all = values_all[np.isfinite(values_all)]

        if values_all.size == 0:
            ax.axis("off")
            ax.set_title(f"{col}\n(no numeric data)", fontsize=10)
            continue

        # -------------------------------------------------
        # Decide x-range for THIS panel
        # -------------------------------------------------
        xmin = xmax = None

        if col in xlim_override:
            xmin, xmax = xlim_override[col]
            if (xmin is None) or (xmax is None) or (not np.isfinite(xmin)) or (not np.isfinite(xmax)) or (xmax <= xmin):
                raise ValueError(f"xlim_override for '{col}' must be finite and satisfy xmin < xmax.")
        elif xlim_percentiles is not None:
            p_lo, p_hi = xlim_percentiles
            lo = float(np.percentile(values_all, p_lo))
            hi = float(np.percentile(values_all, p_hi))

            # Fallback if degenerate
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(np.nanmin(values_all))
                hi = float(np.nanmax(values_all))

            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                pad = (hi - lo) * float(xlim_pad_frac)
                xmin, xmax = lo - pad, hi + pad

        # -------------------------------------------------
        # Drop out-of-range values BEFORE binning and edges
        # -------------------------------------------------
        if xmin is not None and xmax is not None and xmax > xmin:
            values_plot = values_all[(values_all >= xmin) & (values_all <= xmax)]
        else:
            values_plot = values_all

        # -------------------------------------------------
        # Plot histogram using ONLY in-range values
        # -------------------------------------------------
        if values_plot.size >= 2:
            if xmin is not None and xmax is not None and xmax > xmin:
                bin_edges = np.linspace(xmin, xmax, bins + 1)
                ax.hist(values_plot, bins=bin_edges)
                ax.set_xlim(xmin, xmax)
            else:
                ax.hist(values_plot, bins=bins)
        else:
            ax.axis("off")
            ax.set_title(f"{col}\n(insufficient in-range data)", fontsize=10)
            continue

        # Titles/labels
        title = titles.get(col, col) if titles is not None else col
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        if show_grid:
            ax.grid(True, alpha=0.3)

        # Stats are computed on the plotted values (post filtering)
        col_stats = {
            "n": int(values_plot.size),
            "mean": float(np.nanmean(values_plot)),
            "std": float(np.nanstd(values_plot)),
            "min": float(np.nanmin(values_plot)),
            "max": float(np.nanmax(values_plot)),
        }
        stats[col] = col_stats

        # Stats annotation
        ax.text(
            0.98,
            0.98,
            f"n={col_stats['n']}\nmean={col_stats['mean']:.3g}\nstd={col_stats['std']:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            family="monospace",
            alpha=0.9,
        )

        # -------------------------------------------------
        # NEW: annotate how much data survived the range filter
        # -------------------------------------------------
        if annotate_range_filter and (values_all.size > 0):
            kept = int(values_plot.size)
            total = int(values_all.size)
            frac = kept / total if total > 0 else np.nan

            if xmin is not None and xmax is not None and np.isfinite(xmin) and np.isfinite(xmax):
                range_str = f"[{xmin:.3g}, {xmax:.3g}]"
            else:
                range_str = "full range"

            ax.text(
                annotate_loc[0],
                annotate_loc[1],
                f"kept={kept}/{total} ({frac:.1%})\nrange={range_str}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                family="monospace",
                alpha=0.7,
            )

    # Turn off unused axes
    for idx in range(n_plots, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes, stats
