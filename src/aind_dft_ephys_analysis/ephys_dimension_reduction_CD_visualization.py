import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Literal
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


def plot_cd_window_distribution(
    time: np.ndarray,
    trace_A: np.ndarray,   # shape: (nA, T)
    trace_B: np.ndarray,   # shape: (nB, T)
    *,
    window: Tuple[float, float],
    kind: Literal["hist", "violin", "box"] = "hist",
    bins: int = 30,
    hist_overlay: bool = True,     # for kind="hist": overlay vs. side-by-side
    density: bool = True,          # histogram density
    figsize: Tuple[int, int] = (6, 4),
    labels: Tuple[str, str] = ("Type A", "Type B"),
    title: Optional[str] = None,
    xlabel: str = "Mean projection in window",
    ylabel: Optional[str] = None,  # defaults per-kind
    return_values: bool = False,   # if True, also return the per-trial means
):
    """
    For each trial, average its CD trace over `window=(t0, t1)` and plot A vs B distributions.

    Parameters
    ----------
    time : (T,)
        Time vector (seconds).
    trace_A, trace_B : (n_trials, T)
        Time-resolved projections per trial for the two classes.
    window : (t0, t1)
        Time window in *seconds* over which to average each trial's trace.
    kind : {'hist','violin','box'}
        Which distribution plot to draw.
    bins : int
        Histogram bins (for kind='hist').
    hist_overlay : bool
        Overlay the two histograms (True) or draw them side-by-side (False).
    density : bool
        Normalize histogram to a probability density.
    figsize : (w, h)
        Figure size.
    labels : (label_A, label_B)
        Legend/axis labels for the two classes.
    title : str or None
        Optional figure title.
    xlabel, ylabel : str or None
        Axis labels. If ylabel is None, a sensible default is chosen per plot type.
    return_values : bool
        If True, returns (means_A, means_B).

    Notes
    -----
    - Trials with all-NaN values in the selected window are dropped automatically.
    - If the window extends beyond the time range, it is clipped to valid samples.
    """
    t0, t1 = window
    if t0 > t1:
        t0, t1 = t1, t0

    # Build mask for the requested window; clip to valid time range
    mask = (time >= t0) & (time < t1)
    if not np.any(mask):
        # If nothing selected, try inclusive end if t1 equals last sample
        mask = (time >= t0) & (time <= t1)
    if not np.any(mask):
        raise ValueError(f"Window {window} selects no samples within time range [{time.min()}, {time.max()}].")

    # Compute per-trial means over the window (ignore NaNs)
    def _per_trial_mean(tr):
        # tr: (n_trials, T)
        sub = tr[:, mask]
        means = np.nanmean(sub, axis=1)
        # Drop all-NaN trials (mean becomes NaN)
        return means[np.isfinite(means)]

    means_A = _per_trial_mean(trace_A)
    means_B = _per_trial_mean(trace_B)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    if kind == "hist":
        if hist_overlay:
            ax.hist(means_A, bins=bins, density=density, alpha=0.5, label=labels[0])
            ax.hist(means_B, bins=bins, density=density, alpha=0.5, label=labels[1])
        else:
            # side by side using two axes sharing y
            ax.hist(means_A, bins=bins, density=density, alpha=0.7, label=labels[0])
            ax.hist(means_B, bins=bins, density=density, alpha=0.7, label=labels[1])
        ax.set_ylabel("Density" if density else "Count" if ylabel is None else ylabel)

    elif kind == "violin":
        parts = ax.violinplot([means_A, means_B], showmeans=True, showextrema=True, widths=0.8)
        ax.set_xticks([1, 2], labels)
        ax.set_ylabel("Mean projection" if ylabel is None else ylabel)

        # Optional: faint scatter of individual points (jitter)
        xA = np.random.uniform(0.85, 1.15, size=means_A.size)
        xB = np.random.uniform(1.85, 2.15, size=means_B.size)
        ax.scatter(xA, means_A, alpha=0.4, s=10)
        ax.scatter(xB, means_B, alpha=0.4, s=10)

    elif kind == "box":
        ax.boxplot([means_A, means_B], widths=0.6, showmeans=True)
        ax.set_xticks([1, 2], labels)
        ax.set_ylabel("Mean projection" if ylabel is None else ylabel)

        # Optional: faint scatter of individual points (jitter)
        xA = np.random.uniform(0.85, 1.15, size=means_A.size)
        xB = np.random.uniform(1.85, 2.15, size=means_B.size)
        ax.scatter(xA, means_A, alpha=0.4, s=10)
        ax.scatter(xB, means_B, alpha=0.4, s=10)

    else:
        raise ValueError("kind must be one of {'hist','violin','box'}")

    ax.set_xlabel(xlabel)
    if kind == "hist":
        ax.legend(frameon=False)
    if title:
        ax.set_title(title + f"  (window: {t0:.3f}–{t1:.3f}s)")
    else:
        ax.set_title(f"Distribution of trial means (window: {t0:.3f}–{t1:.3f}s)")
    plt.tight_layout()
    plt.show()

    if return_values:
        return means_A, means_B


def plot_cd_projection(
    time: np.ndarray,
    trace_A: np.ndarray,
    trace_B: np.ndarray,
    *,
    average: bool = True,
    error: Optional[Literal["sem", "std", "ci"]] = "ci",
    ci_level: float = 0.95,
    smooth: Optional[float] = None,
    smooth_mode: Literal["gaussian", "moving"] = "gaussian",
    dt: Optional[float] = None,
    edge_handling: Literal["reflect", "nearest", "mirror", "wrap", "none"] = "reflect",
    labels: Tuple[str, str] = ("Type A", "Type B"),
    colors: Tuple[str, str] = ("#1f77b4", "#d62728"),
    figsize: Tuple[int, int] = (6, 4),
    alpha_single: float = 0.15,
    linewidth_mean: float = 2.0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    xlabel: str = "Time (s)",
    ylabel: str = "Projection along CD",
) -> None:
    """
    Plot coding-direction projection traces for two trial types (A vs B),
    with optional temporal smoothing and error bands.

    Parameters
    ----------
    time : (T,)
        Time vector (seconds).
    trace_A, trace_B : (n_trials, T)
        Time-resolved projections per trial for the two classes.
    average : bool
        If True, plot only the mean (with optional error band).
        If False, plot all trials (faint) plus the mean.
    error : {'sem', 'std', 'ci', None}, optional
        Type of error band to plot around the mean:
        - 'sem': mean ± standard error of the mean
        - 'std': mean ± standard deviation
        - 'ci' : mean ± z * SEM, where z is based on `ci_level`
        - None : no error band
    ci_level : float
        Confidence level for 'ci' (e.g. 0.95 for ~95% CI).
    smooth : float or None
        If >0, smoothing window. Interpreted in *seconds* if `dt` is provided,
        otherwise in *samples*.
    smooth_mode : {'gaussian','moving'}
        Gaussian kernel (softer) or moving average (boxcar).
    dt : float or None
        Sample spacing (sec). Needed only if `smooth` is in seconds.
    edge_handling : {'reflect','nearest','mirror','wrap','none'}
        How to treat edges when smoothing. 'reflect' (default) avoids boundary artefacts.
        If 'none', no padding is used; instead the plot is trimmed to the valid region.
    labels : (label_A, label_B)
        Legend labels for the two classes.
    colors : (color_A, color_B)
        Line colors for the two classes.
    figsize : (w, h)
        Figure size in inches.
    alpha_single : float
        Alpha for single-trial lines when `average=False`.
    linewidth_mean : float
        Line width for the mean trace.
    xlim, ylim : (min, max) or None
        Optional axis limits.
    title : str or None
        Optional plot title.
    xlabel, ylabel : str
        Axis labels.
    """
    # --- Helper: determine kernel width in samples ---
    def _kernel_pts() -> int:
        if smooth is None or smooth <= 0:
            return 0
        return max(1, int(round((smooth if dt is None else smooth / dt))))

    # --- Helper: smoothing ---
    def _smooth_traces(tr: np.ndarray) -> Tuple[np.ndarray, Optional[slice]]:
        k = _kernel_pts()
        if k <= 1:
            return tr, None

        if smooth_mode == "gaussian":
            tr_sm = gaussian_filter1d(
                tr,
                sigma=k,
                axis=-1,
                mode=("nearest" if edge_handling == "none" else edge_handling),
                truncate=3.0,
            )
            if edge_handling == "none":
                trim = int(3.0 * k)
                sl = slice(trim, tr.shape[-1] - trim)
                tr_sm = tr_sm[..., sl]
                return tr_sm, sl
            return tr_sm, None

        elif smooth_mode == "moving":
            if edge_handling == "none":
                # Valid-only convolution: use cumulative-sum trick and trim ends
                w = k

                def _boxcar_valid(x):
                    c = np.cumsum(np.pad(x, (1, 0), mode="constant"))
                    y = (c[w:] - c[:-w]) / w
                    return y

                sm = np.apply_along_axis(_boxcar_valid, -1, tr)
                sl = slice(k // 2, tr.shape[-1] - (k - 1 - k // 2))
                return sm, sl
            else:
                tr_sm = uniform_filter1d(tr, size=k, axis=-1, mode=edge_handling)
                return tr_sm, None
        else:
            raise ValueError("smooth_mode must be 'gaussian' or 'moving'.")

    # --- Smooth A/B (and get any trimming slices if edge_handling=='none') ---
    trace_A_sm, slA = _smooth_traces(trace_A)
    trace_B_sm, slB = _smooth_traces(trace_B)

    # Align time to valid region if trimming occurred
    time_plot = time
    if slA is not None or slB is not None:
        def _to_slice(s, n):
            return s if s is not None else slice(0, n)

        n = time.shape[0]
        sA = _to_slice(slA, n)
        sB = _to_slice(slB, n)
        start = max(sA.start, sB.start)
        stop = min(sA.stop, sB.stop)
        sl = slice(start, stop)

        len_common = stop - start
        offA = sA.start - start
        offB = sB.start - start
        trace_A_sm = trace_A_sm[..., offA:offA + len_common]
        trace_B_sm = trace_B_sm[..., offB:offB + len_common]
        time_plot = time[sl]

    fig, ax = plt.subplots(figsize=figsize)

    # --- Helper: error computation ---
    def _compute_error(traces: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if error is None:
            return None

        mean = np.nanmean(traces, axis=0)
        std = np.nanstd(traces, axis=0)

        # Number of non-NaN samples per time bin
        n = np.sum(np.isfinite(traces), axis=0).astype(float)
        # Avoid division by zero
        n[n == 0] = 1.0

        if error == "sem":
            err = std / np.sqrt(n)
        elif error == "std":
            err = std
        elif error == "ci":
            from scipy.stats import norm
            z = norm.ppf(0.5 + ci_level / 2.0)
            err = z * std / np.sqrt(n)
        else:
            raise ValueError("error must be one of {'sem','std','ci',None}.")

        lower = mean - err
        upper = mean + err
        return lower, upper

    # --- Helper: plotting for one group ---
    def _plot_group(traces: np.ndarray, color: str, label: str) -> None:
        mean = np.nanmean(traces, axis=0)

        if average:
            # Error band if requested
            if error is not None:
                band = _compute_error(traces)
                if band is not None:
                    lower, upper = band
                    ax.fill_between(
                        time_plot,
                        lower,
                        upper,
                        color=color,
                        alpha=0.25,
                        lw=0,
                    )
            ax.plot(time_plot, mean, color=color, lw=linewidth_mean, label=label)
        else:
            # Faint single trials + thick mean on top
            for tr in traces:
                ax.plot(time_plot, tr, color=color, alpha=alpha_single, lw=1.0)
            if error is not None:
                band = _compute_error(traces)
                if band is not None:
                    lower, upper = band
                    ax.fill_between(
                        time_plot,
                        lower,
                        upper,
                        color=color,
                        alpha=0.25,
                        lw=0,
                    )
            ax.plot(time_plot, mean, color=color, lw=linewidth_mean, label=label)

    _plot_group(trace_A_sm, colors[0], labels[0])
    _plot_group(trace_B_sm, colors[1], labels[1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
