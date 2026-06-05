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
    show_mean: bool = True,
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
            # Faint single trials + optional thick mean on top
            for tr in traces:
                ax.plot(time_plot, tr, color=color, alpha=alpha_single, lw=1.0)
            if show_mean:
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
                # Add a thin legend handle so the label still shows up.
                ax.plot([], [], color=color, lw=linewidth_mean, label=label)

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


def plot_cd_heatmap(
    time: np.ndarray,
    trace_A: np.ndarray,
    trace_B: Optional[np.ndarray] = None,
    *,
    smooth: Optional[float] = None,
    smooth_mode: Literal["gaussian", "moving"] = "gaussian",
    dt: Optional[float] = None,
    edge_handling: Literal["reflect", "nearest", "mirror", "wrap"] = "reflect",
    sort_by: Optional[Literal["mean", "peak_time", "peak_value", "none"]] = "mean",
    sort_window: Optional[Tuple[float, float]] = None,
    sort_ascending: bool = False,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vrange_quantile: float = 0.98,
    symmetric_colorbar: bool = True,
    labels: Tuple[str, str] = ("Type A", "Type B"),
    figsize: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    xlabel: str = "Time (s)",
    ylabel: str = "Trial (sorted)",
    show_colorbar: bool = True,
    show_event_line: bool = True,
) -> None:
    """
    Single-trial CD-projection heatmap for one or two classes.

    Parameters
    ----------
    time : (T,)
        Time vector (seconds).
    trace_A, trace_B : (n_trials, T)
        Per-trial projection traces. ``trace_B`` may be None or empty to plot
        a single panel.
    smooth, smooth_mode, dt, edge_handling
        Temporal smoothing applied to each trial before display. ``smooth`` is
        interpreted in *seconds* if ``dt`` is given, otherwise in *samples*.
    sort_by : {'mean','peak_time','peak_value','none'} or None
        Sort each class's rows. ``'mean'`` (default) sorts by within-window
        average; ``'peak_time'`` by argmax time; ``'peak_value'`` by maximum
        value; ``'none'``/``None`` keeps input order.
    sort_window : (t0, t1), optional
        Time window used by the sort statistic. Defaults to the full ``xlim``
        if set, otherwise the full time axis.
    sort_ascending : bool, default False
        Sort direction (default: largest at top).
    cmap : str
        Matplotlib colormap.
    vmin, vmax : float, optional
        Color limits. If both None, computed from the central ``vrange_quantile``
        of the (smoothed) data, optionally made symmetric around 0.
    vrange_quantile : float, default 0.98
        When auto-computing limits, use this two-sided quantile (i.e.
        clips at the (1-q)/q tails — e.g. ``0.98`` -> 2nd / 98th
        percentiles for higher contrast in the presence of outliers).
    symmetric_colorbar : bool, default True
        Force ``vmin = -vmax`` when auto-computing limits (useful for diverging
        colormaps like ``RdBu_r``).
    labels : (label_A, label_B)
        Subplot titles (suffixed with trial counts).
    figsize : (w, h), optional
        Figure size. Default scales with number of panels.
    xlim, title, xlabel, ylabel
        Standard axis controls.
    show_colorbar : bool
        Append a colorbar to the right of the heatmaps.
    show_event_line : bool
        Draw a vertical line at t=0 if it lies within ``xlim``.
    """
    # ---- normalize inputs ----
    have_A = isinstance(trace_A, np.ndarray) and trace_A.ndim == 2 and trace_A.size > 0
    have_B = isinstance(trace_B, np.ndarray) and trace_B.ndim == 2 and trace_B.size > 0
    if not have_A and not have_B:
        print("[heatmap] No data to plot.")
        return

    # ---- smoothing helpers (mirror plot_cd_projection) ----
    def _kernel_pts() -> int:
        if smooth is None or smooth <= 0:
            return 0
        return max(1, int(round(smooth if dt is None else smooth / dt)))

    def _smooth(tr: np.ndarray) -> np.ndarray:
        k = _kernel_pts()
        if k <= 1 or tr.size == 0:
            return tr
        if smooth_mode == "gaussian":
            return gaussian_filter1d(tr, sigma=k, axis=-1, mode=edge_handling, truncate=3.0)
        elif smooth_mode == "moving":
            return uniform_filter1d(tr, size=k, axis=-1, mode=edge_handling)
        else:
            raise ValueError("smooth_mode must be 'gaussian' or 'moving'.")

    A_sm = _smooth(trace_A) if have_A else None
    B_sm = _smooth(trace_B) if have_B else None

    # ---- sort window mask ----
    def _sort_mask() -> np.ndarray:
        if sort_window is not None:
            t0, t1 = sort_window
        elif xlim is not None:
            t0, t1 = xlim
        else:
            return np.ones_like(time, dtype=bool)
        return (time >= t0) & (time <= t1)

    def _sort_order(mat: np.ndarray) -> np.ndarray:
        if sort_by in (None, "none"):
            return np.arange(mat.shape[0])
        m = _sort_mask()
        seg = mat[:, m] if np.any(m) else mat
        if sort_by == "mean":
            stat = np.nanmean(seg, axis=1)
        elif sort_by == "peak_value":
            stat = np.nanmax(seg, axis=1)
        elif sort_by == "peak_time":
            # argmax along sorted segment; map back to global time axis
            idx = np.nanargmax(np.where(np.isnan(seg), -np.inf, seg), axis=1)
            local_time = time[m] if np.any(m) else time
            stat = local_time[idx]
        else:
            raise ValueError(f"Unsupported sort_by={sort_by!r}")
        # NaN-safe argsort: push NaNs to bottom
        nan_mask = np.isnan(stat)
        order = np.argsort(np.where(nan_mask, -np.inf if sort_ascending else np.inf, stat))
        if not sort_ascending:
            order = order[::-1]
        return order

    A_plot = A_sm[_sort_order(A_sm)] if have_A else None
    B_plot = B_sm[_sort_order(B_sm)] if have_B else None

    # ---- color limits ----
    # Compute auto vmin/vmax only from the *visible* time window (``xlim``)
    # when given, so outliers outside the displayed range don't shrink the
    # effective contrast inside it. Data values are NOT clipped — only the
    # colorbar mapping is.
    if vmin is None and vmax is None:
        if xlim is not None:
            cmask = (time >= float(xlim[0])) & (time <= float(xlim[1]))
            if not np.any(cmask):
                cmask = np.ones_like(time, dtype=bool)
        else:
            cmask = np.ones_like(time, dtype=bool)
        pool = []
        if A_plot is not None:
            pool.append(A_plot[:, cmask][np.isfinite(A_plot[:, cmask])])
        if B_plot is not None:
            pool.append(B_plot[:, cmask][np.isfinite(B_plot[:, cmask])])
        if pool:
            flat = np.concatenate(pool)
            if flat.size:
                lo = float(np.quantile(flat, 1.0 - vrange_quantile))
                hi = float(np.quantile(flat, vrange_quantile))
                if symmetric_colorbar:
                    m = max(abs(lo), abs(hi))
                    vmin, vmax = -m, m
                else:
                    vmin, vmax = lo, hi
    if vmin is None:
        vmin = -1.0
    if vmax is None:
        vmax = 1.0

    # ---- figure / axes ----
    n_panels = int(have_A) + int(have_B)
    if figsize is None:
        figsize = (7.0, 3.0 + 1.5 * n_panels)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize, sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    extent_lo = float(time[0])
    extent_hi = float(time[-1])

    panels = []
    if have_A:
        panels.append((A_plot, f"{labels[0]} (n={A_plot.shape[0]})"))
    if have_B:
        panels.append((B_plot, f"{labels[1]} (n={B_plot.shape[0]})"))

    im = None
    for ax, (mat, panel_title) in zip(axes, panels):
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            extent=[extent_lo, extent_hi, 0, mat.shape[0]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        if show_event_line:
            lo_x = xlim[0] if xlim else extent_lo
            hi_x = xlim[1] if xlim else extent_hi
            if lo_x <= 0.0 <= hi_x:
                ax.axvline(0.0, color="k", lw=0.8, ls="--", alpha=0.7)

    axes[-1].set_xlabel(xlabel)
    if xlim is not None:
        axes[-1].set_xlim(xlim)
    if title:
        fig.suptitle(title)

    if show_colorbar and im is not None:
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cax, label="CD projection")
    else:
        plt.tight_layout()
    plt.show()
