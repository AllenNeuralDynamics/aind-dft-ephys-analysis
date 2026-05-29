"""
Per-trial *bump* detection on CD projections.

A "bump" here is a localized transient excursion (positive or negative) of the
single-trial CD projection trace. The two main entry points are:

- :func:`detect_bumps`    — core peak detection on a ``(n_trials, n_time)``
  array. Returns a long-form DataFrame with one row per detected bump and
  columns ``trial_id, trial_index, polarity, t_peak, amplitude,
  abs_amplitude, prominence, width_sec, onset, offset, auc``.
- :func:`plot_bumps`      — 5-panel summary figure (heatmap with peak markers,
  peak-time / amplitude / width histograms, mean peak-aligned bump shape).
- :func:`summarize_bumps` — short DataFrame of medians + IQRs per polarity.

The pipeline-level wrapper :func:`plot_cd_session_bumps` (in
``ephys_dimension_reduction_CD_pipeline``) handles trial selection,
per-trial re-alignment, masking, and optional subsampling before calling
these.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d


def detect_bumps(
    traces: np.ndarray,
    time: np.ndarray,
    *,
    trial_ids: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
    search_window: Optional[Tuple[float, float]] = None,
    polarity: Literal["pos", "neg", "both"] = "both",
    smooth_sigma_sec: Optional[float] = None,
    min_amplitude: Optional[float] = None,
    min_prominence: Optional[float] = None,
    min_width_sec: Optional[float] = None,
    max_per_trial: int = 1,
) -> pd.DataFrame:
    """Detect transient bumps in single-trial projection traces.

    Parameters
    ----------
    traces : (n_trials, n_time) array
        Per-trial signal. NaN samples (e.g. from per-trial windowing) are
        treated as missing; trials with too few valid samples are skipped.
    time : (n_time,) array
        Time axis in seconds (same reference as ``search_window``).
    trial_ids : (n_trials,), optional
        Trial IDs to record in the output. Defaults to row index.
    dt : float, optional
        Sample period. Inferred from ``time`` if omitted.
    search_window : (t0, t1), optional
        Only look for peaks inside this time interval. None → full ``time``.
    polarity : {'pos', 'neg', 'both'}
        Detect positive peaks, negative peaks (run on ``-trace``), or both.
    smooth_sigma_sec : float, optional
        Per-trial Gaussian smoothing (σ in seconds) applied inside the
        search window before peak detection. None disables smoothing.
    min_amplitude, min_prominence : float, optional
        Forwarded to :func:`scipy.signal.find_peaks` as ``height`` and
        ``prominence`` thresholds (applied to the polarity-flipped signal,
        so always positive numbers).
    min_width_sec : float, optional
        Minimum peak width (at half prominence) in seconds.
    max_per_trial : int, default 1
        Keep at most this many peaks per trial × polarity, ranked by
        prominence (descending).

    Returns
    -------
    pandas.DataFrame
        One row per detected bump. Empty if no peaks meet the thresholds.
    """
    if traces.ndim != 2 or traces.size == 0:
        return pd.DataFrame()
    n_trials, n_time = traces.shape
    if dt is None or not np.isfinite(dt):
        dt = float(np.mean(np.diff(time))) if len(time) > 1 else 1.0
    if trial_ids is None:
        trial_ids = np.arange(n_trials, dtype=int)
    trial_ids = np.asarray(trial_ids)

    if search_window is None:
        idx_window = np.arange(n_time)
    else:
        t0, t1 = search_window
        idx_window = np.where((time >= t0) & (time <= t1))[0]
    if idx_window.size < 3:
        return pd.DataFrame()
    t_sub = time[idx_window]

    sigma_pts = max(1, int(round(smooth_sigma_sec / dt))) if smooth_sigma_sec else 0

    polarities = (
        ["pos"] if polarity == "pos"
        else ["neg"] if polarity == "neg"
        else ["pos", "neg"]
    )

    rows = []
    for i in range(n_trials):
        row = traces[i, idx_window].astype(float, copy=True)
        valid = np.isfinite(row)
        if valid.sum() < 3:
            continue
        if not valid.all():
            # Replace missing samples with 0 so find_peaks can run; the peak
            # itself must sit on a finite sample to be reported (we check
            # below). This avoids find_peaks NaN-propagation issues.
            row = np.where(valid, row, 0.0)
        if sigma_pts > 0:
            row = gaussian_filter1d(row, sigma=sigma_pts, mode="nearest")

        for pol in polarities:
            sig = row if pol == "pos" else -row
            kwargs = {}
            if min_amplitude is not None:
                kwargs["height"] = float(min_amplitude)
            if min_prominence is not None:
                kwargs["prominence"] = float(min_prominence)
            if min_width_sec is not None:
                kwargs["width"] = max(1.0, min_width_sec / dt)
            peaks, props = find_peaks(sig, **kwargs)
            if peaks.size == 0:
                continue
            # Reject peaks that landed on originally-NaN samples.
            keep = valid[peaks]
            peaks = peaks[keep]
            for k in list(props.keys()):
                if isinstance(props[k], np.ndarray) and props[k].shape[0] == keep.size:
                    props[k] = props[k][keep]
            if peaks.size == 0:
                continue
            score = props.get("prominences", sig[peaks])
            order = np.argsort(score)[::-1][:max_per_trial]
            peaks_sel = peaks[order]
            prom_sel = (
                props["prominences"][order] if "prominences" in props
                else np.full(peaks_sel.shape, np.nan)
            )
            try:
                widths_pts, _wh, left_ips, right_ips = peak_widths(
                    sig, peaks_sel, rel_height=0.5
                )
            except Exception:  # noqa: BLE001
                widths_pts = np.full(peaks_sel.shape, np.nan)
                left_ips = np.full(peaks_sel.shape, np.nan)
                right_ips = np.full(peaks_sel.shape, np.nan)

            x_axis = np.arange(len(t_sub))
            for k, p in enumerate(peaks_sel):
                amp = float(row[p])
                t_peak = float(t_sub[p])
                width_sec = (
                    float(widths_pts[k] * dt) if np.isfinite(widths_pts[k]) else np.nan
                )
                onset = (
                    float(np.interp(left_ips[k], x_axis, t_sub))
                    if np.isfinite(left_ips[k]) else np.nan
                )
                offset = (
                    float(np.interp(right_ips[k], x_axis, t_sub))
                    if np.isfinite(right_ips[k]) else np.nan
                )
                if np.isfinite(left_ips[k]) and np.isfinite(right_ips[k]):
                    li = max(0, int(np.floor(left_ips[k])))
                    ri = min(len(row) - 1, int(np.ceil(right_ips[k])))
                    seg = row[li:ri + 1]
                    auc = float(np.trapezoid(seg, dx=dt))
                    if pol == "neg":
                        auc = -auc
                else:
                    auc = np.nan
                rows.append({
                    "trial_id": int(trial_ids[i]),
                    "trial_index": int(i),
                    "polarity": pol,
                    "t_peak": t_peak,
                    "amplitude": amp,
                    "abs_amplitude": abs(amp),
                    "prominence": float(prom_sel[k]) if np.isfinite(prom_sel[k]) else np.nan,
                    "width_sec": width_sec,
                    "onset": onset,
                    "offset": offset,
                    "auc": auc,
                })
    return pd.DataFrame(rows)


def summarize_bumps(df: pd.DataFrame) -> pd.DataFrame:
    """Median + IQR summary of bump stats per polarity."""
    if df is None or df.empty:
        return pd.DataFrame()

    def _iqr(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < 2:
            return float("nan")
        q75, q25 = np.percentile(x, [75, 25])
        return float(q75 - q25)

    g = df.groupby("polarity")
    out = g.agg(
        n=("t_peak", "size"),
        t_peak_median=("t_peak", "median"),
        t_peak_iqr=("t_peak", _iqr),
        abs_amp_median=("abs_amplitude", "median"),
        abs_amp_iqr=("abs_amplitude", _iqr),
        width_median=("width_sec", "median"),
        width_iqr=("width_sec", _iqr),
        prom_median=("prominence", "median"),
        auc_median=("auc", "median"),
    )
    return out.reset_index()


def plot_bumps(
    traces: np.ndarray,
    time: np.ndarray,
    bumps_df: pd.DataFrame,
    *,
    title: str = "",
    cmap: str = "RdBu_r",
    vrange_quantile: float = 0.99,
    xlim: Optional[Tuple[float, float]] = None,
    shape_window: Tuple[float, float] = (-1.0, 2.0),
    dt: Optional[float] = None,
) -> None:
    """Five-panel bump-detection summary for one trace matrix.

    Panels
    ------
    1. Heatmap of traces (rows sorted by detected ``t_peak``) with peak
       markers overlaid (black = positive peak, cyan = negative peak).
    2. Histogram of peak times (one color per polarity).
    3. Histogram of signed peak amplitudes.
    4. Histogram of widths at half prominence (seconds).
    5. Mean ± SEM bump shape, peak-aligned (negative bumps are sign-flipped
       so both polarities show as positive deflections).
    """
    if traces.ndim != 2 or traces.size == 0:
        print("[bumps] empty traces; skip.")
        return
    if dt is None or not np.isfinite(dt):
        dt = float(np.mean(np.diff(time))) if len(time) > 1 else 1.0
    n_trials = traces.shape[0]

    # Pick a single representative peak per trial (largest |amp|) for sort key.
    if bumps_df is not None and not bumps_df.empty:
        first = (
            bumps_df.sort_values("abs_amplitude", ascending=False)
            .drop_duplicates("trial_index")
            .set_index("trial_index")["t_peak"]
        )
        order_key = np.array([first.get(i, np.inf) for i in range(n_trials)])
    else:
        order_key = np.arange(n_trials, dtype=float)
    order = np.argsort(order_key)
    traces_sorted = traces[order]
    rank_map = {int(t_idx): r for r, t_idx in enumerate(order)}

    finite = traces[np.isfinite(traces)]
    lim = float(np.nanquantile(np.abs(finite), vrange_quantile)) if finite.size else 1.0
    vmin, vmax = -lim, lim

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # 1) Heatmap + peak markers
    ax = fig.add_subplot(gs[0, 0])
    extent = [time[0], time[-1], 0, n_trials]
    ax.imshow(
        traces_sorted, aspect="auto", origin="lower", extent=extent,
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    if bumps_df is not None and not bumps_df.empty:
        for pol, color in (("pos", "black"), ("neg", "cyan")):
            sub = bumps_df[bumps_df["polarity"] == pol]
            if sub.empty:
                continue
            xs = sub["t_peak"].values
            ys = np.array([rank_map.get(int(t), -1) + 0.5 for t in sub["trial_index"].values])
            keep = ys >= 0
            ax.scatter(
                xs[keep], ys[keep], s=10, c=color, marker="o",
                edgecolors="white", linewidths=0.4, label=f"{pol} peak",
            )
        ax.legend(loc="upper right", fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trial (sorted by t_peak)")
    ax.set_title("Heatmap + detected peaks")

    # 2) Peak-time hist
    ax = fig.add_subplot(gs[0, 1])
    if bumps_df is not None and not bumps_df.empty:
        for pol, color in (("pos", "C3"), ("neg", "C0")):
            sub = bumps_df[bumps_df["polarity"] == pol]
            if not sub.empty:
                ax.hist(sub["t_peak"], bins=30, alpha=0.55, color=color, label=pol)
        ax.legend(fontsize=8)
    ax.set_xlabel("Peak time (s)")
    ax.set_ylabel("# bumps")
    ax.set_title("Peak time distribution")

    # 3) Amplitude hist
    ax = fig.add_subplot(gs[0, 2])
    if bumps_df is not None and not bumps_df.empty:
        ax.hist(bumps_df["amplitude"], bins=30, color="C2", alpha=0.75)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Peak amplitude (signed)")
    ax.set_ylabel("# bumps")
    ax.set_title("Amplitude distribution")

    # 4) Width hist
    ax = fig.add_subplot(gs[1, 0])
    if bumps_df is not None and not bumps_df.empty:
        widths = bumps_df["width_sec"].dropna()
        if widths.size:
            ax.hist(widths, bins=30, color="C4", alpha=0.75)
    ax.set_xlabel("Width at half-prominence (s)")
    ax.set_ylabel("# bumps")
    ax.set_title("Width distribution")

    # 5) Peak-aligned mean shape
    ax = fig.add_subplot(gs[1, 1:])
    win_pre, win_post = shape_window
    pre_pts = int(round(win_pre / dt))   # negative
    post_pts = int(round(win_post / dt))
    rel_t = np.arange(pre_pts, post_pts + 1) * dt
    plotted_any = False
    if bumps_df is not None and not bumps_df.empty:
        for pol, color in (("pos", "C3"), ("neg", "C0")):
            sub = bumps_df[bumps_df["polarity"] == pol]
            if sub.empty:
                continue
            chunks = []
            for _, r in sub.iterrows():
                i = int(r["trial_index"])
                p = int(np.argmin(np.abs(time - r["t_peak"])))
                lo, hi = p + pre_pts, p + post_pts + 1
                if lo < 0 or hi > traces.shape[1]:
                    continue
                seg = traces[i, lo:hi].astype(float)
                if pol == "neg":
                    seg = -seg
                chunks.append(seg)
            if not chunks:
                continue
            M = np.vstack(chunks)
            mean = np.nanmean(M, axis=0)
            denom = np.sqrt(np.maximum(1, np.sum(np.isfinite(M), axis=0)))
            sem = np.nanstd(M, axis=0) / denom
            ax.plot(rel_t, mean, color=color, label=f"{pol} (n={M.shape[0]})")
            ax.fill_between(rel_t, mean - sem, mean + sem, color=color, alpha=0.25)
            plotted_any = True
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Time relative to peak (s)")
    ax.set_ylabel("Projection (neg flipped)")
    ax.set_title("Mean bump shape (peak-aligned, mean ± SEM)")
    if plotted_any:
        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1))
    plt.show()
