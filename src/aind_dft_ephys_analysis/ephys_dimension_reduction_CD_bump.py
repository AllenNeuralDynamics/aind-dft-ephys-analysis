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
    max_per_trial: int = 5,
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
    baseline_window: Optional[Tuple[float, float]] = (-1.0, -0.3),
    baseline_stat: Literal["median", "mean"] = "median",
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
       so both polarities show as positive deflections). If
       ``baseline_window`` is given (relative to peak, in shape-window
       coordinates), each snippet is baseline-subtracted using
       ``baseline_stat`` over that window before averaging, so both
       polarities deflect from 0.
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

    # Indices within the snippet used for baseline subtraction.
    bl_idx = None
    if baseline_window is not None:
        bl0, bl1 = baseline_window
        bl_idx = np.where((rel_t >= bl0) & (rel_t <= bl1))[0]
        if bl_idx.size == 0:
            bl_idx = None
    _bl_fn = np.nanmedian if baseline_stat == "median" else np.nanmean

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
                if bl_idx is not None:
                    bl_vals = seg[bl_idx]
                    if np.any(np.isfinite(bl_vals)):
                        seg = seg - _bl_fn(bl_vals)
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
    if baseline_window is not None:
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.axvspan(baseline_window[0], baseline_window[1], color="gray", alpha=0.1)
    ax.set_xlabel("Time relative to peak (s)")
    ylabel = "Projection (neg flipped"
    if baseline_window is not None:
        ylabel += f", baseline-subtracted [{baseline_window[0]:.2f},{baseline_window[1]:.2f}]s"
    ylabel += ")"
    ax.set_ylabel(ylabel)
    ax.set_title("Mean bump shape (peak-aligned, mean ± SEM)")
    if plotted_any:
        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1))
    plt.show()


def plot_inter_peak_intervals(
    bumps_df: pd.DataFrame,
    *,
    title: str = "",
    bins: int = 30,
    max_ipi_sec: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Per-polarity inter-peak-interval (IPI) distribution.

    For each trial, sort that trial's peaks of one polarity by ``t_peak`` and
    take consecutive differences. Trials with fewer than 2 peaks of that
    polarity contribute nothing.

    Returns a long-form DataFrame with columns
    ``trial_id, trial_index, polarity, ipi_sec`` (or ``None`` if no IPIs).
    Also shows a 2-panel histogram (pos / neg) with median markers.
    """
    if bumps_df is None or bumps_df.empty:
        print("[ipi] no peaks; skip.")
        return None

    rows = []
    for (tid, pol), sub in bumps_df.groupby(["trial_index", "polarity"]):
        if len(sub) < 2:
            continue
        ts = np.sort(sub["t_peak"].values)
        ipis = np.diff(ts)
        trial_id = int(sub["trial_id"].iloc[0])
        for v in ipis:
            rows.append({
                "trial_id": trial_id,
                "trial_index": int(tid),
                "polarity": pol,
                "ipi_sec": float(v),
            })
    ipi_df = pd.DataFrame(rows)
    if ipi_df.empty:
        print("[ipi] no trials with >= 2 peaks of either polarity; skip.")
        return ipi_df

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    for ax, pol, color in zip(axes, ("pos", "neg"), ("C3", "C0")):
        sub = ipi_df[ipi_df["polarity"] == pol]["ipi_sec"]
        ax.set_title(f"{pol} IPI (n={len(sub)})")
        ax.set_xlabel("Inter-peak interval (s)")
        if sub.empty:
            ax.text(0.5, 0.5, "no IPIs", ha="center", va="center", transform=ax.transAxes)
            continue
        clipped = sub if max_ipi_sec is None else sub[sub <= max_ipi_sec]
        ax.hist(clipped, bins=bins, color=color, alpha=0.75)
        med = float(np.median(sub))
        ax.axvline(med, color="k", lw=1, ls="--", label=f"median={med:.2f}s")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("# IPIs")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if title else 1))
    plt.show()
    return ipi_df


def plot_bump_density(
    traces: np.ndarray,
    time: np.ndarray,
    bumps_df: pd.DataFrame,
    *,
    bin_width_sec: float = 0.25,
    title: str = "",
    xlim: Optional[Tuple[float, float]] = None,
) -> Optional[pd.DataFrame]:
    """Bump density (bumps / trial / second) over time, per polarity.

    For each time bin of width ``bin_width_sec``:
      - **numerator**   = number of detected bumps (from ``bumps_df``) whose
        ``t_peak`` falls in that bin, per polarity.
      - **denominator** = number of trials with valid (finite) data anywhere
        in that bin, after restrict_events masking (NaN samples don't count).

    Density is then ``numerator / (denominator * bin_width_sec)`` so units
    are bumps per trial per second.

    Returns a DataFrame with columns
    ``bin_center, bin_left, bin_right, n_valid_trials, count_pos, count_neg,
    density_pos, density_neg``.
    """
    if traces is None or traces.ndim != 2 or traces.size == 0:
        print("[density] empty traces; skip.")
        return None
    if time.size < 2:
        print("[density] time axis too short; skip.")
        return None

    t0 = float(time[0])
    t1 = float(time[-1])
    if xlim is not None:
        t0 = max(t0, float(xlim[0]))
        t1 = min(t1, float(xlim[1]))
    if t1 <= t0:
        print("[density] empty time range; skip.")
        return None

    edges = np.arange(t0, t1 + bin_width_sec, bin_width_sec)
    if edges.size < 2:
        print("[density] bin_width too large; skip.")
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = centers.size

    # Per-bin denominator: count trials with ANY finite sample in the bin.
    valid_mask = np.isfinite(traces)
    n_valid_per_bin = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        col_mask = (time >= edges[b]) & (time < edges[b + 1])
        if not np.any(col_mask):
            continue
        # trial is "valid in bin" if any sample within the bin is finite
        n_valid_per_bin[b] = int(np.sum(np.any(valid_mask[:, col_mask], axis=1)))

    # Per-bin numerator: # bumps per polarity whose t_peak falls in bin.
    count_pos = np.zeros(n_bins, dtype=int)
    count_neg = np.zeros(n_bins, dtype=int)
    if bumps_df is not None and not bumps_df.empty:
        for pol, arr in (("pos", count_pos), ("neg", count_neg)):
            sub = bumps_df[bumps_df["polarity"] == pol]
            if sub.empty:
                continue
            t_peaks = sub["t_peak"].values
            in_range = (t_peaks >= edges[0]) & (t_peaks < edges[-1])
            if not np.any(in_range):
                continue
            idx = np.digitize(t_peaks[in_range], edges, right=False) - 1
            idx = np.clip(idx, 0, n_bins - 1)
            np.add.at(arr, idx, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        density_pos = np.where(n_valid_per_bin > 0,
                               count_pos / (n_valid_per_bin * bin_width_sec), np.nan)
        density_neg = np.where(n_valid_per_bin > 0,
                               count_neg / (n_valid_per_bin * bin_width_sec), np.nan)

    df = pd.DataFrame({
        "bin_center": centers,
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "n_valid_trials": n_valid_per_bin,
        "count_pos": count_pos,
        "count_neg": count_neg,
        "density_pos": density_pos,
        "density_neg": density_neg,
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    plotted = False
    if np.any(count_pos > 0):
        ax1.step(centers, density_pos, where="mid", color="C3", label="pos")
        plotted = True
    if np.any(count_neg > 0):
        ax1.step(centers, density_neg, where="mid", color="C0", label="neg")
        plotted = True
    ax1.axvline(0, color="k", lw=0.6)
    ax1.set_ylabel("Bump density (bumps / trial / s)")
    ax1.set_title("Bump density vs time")
    if plotted:
        ax1.legend(fontsize=8)

    ax2.fill_between(centers, 0, n_valid_per_bin, step="mid",
                     color="gray", alpha=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("# trials\nwith data")
    if xlim is not None:
        ax2.set_xlim(*xlim)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if title else 1))
    plt.show()
    return df


