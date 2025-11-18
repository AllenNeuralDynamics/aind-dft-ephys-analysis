
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d

# your project utilities
from behavior_utils import extract_event_timestamps, extract_fitted_data
from general_utils import load_temporary_data, extract_session_name_core

# -------------------------------
#  QC filter (as specified by you)
# -------------------------------
def get_units_passed_default_qc(nwb_data: Any) -> np.ndarray:
    """
    Return indices of units that pass default QC and are not labeled 'noise'.
    """
    tbl = nwb_data.units
    default_qc = np.array(tbl['default_qc'].data)
    labels     = np.array(tbl['decoder_label'].data)
    mask = ((default_qc == True) | (default_qc == 'True')) & (labels != 'noise')
    idx = np.nonzero(mask)[0]
    print(f"Number of units passing QC: {len(idx)}")
    return idx


# ------------------------------------------------------------
#  Helpers to mine the *new* wide-format significance table
# ------------------------------------------------------------
def _pick_sig_and_coef_columns(
    df: pd.DataFrame,
    *,
    model: str,
    variable: str,
    shift_tag: str = "s0",          # e.g. "s0", "s+1", "s-1"
    direction: int = 0,             # 0 = behavior→FR; 1 = FR→behavior (your pipeline)
    group: Optional[int] = None     # None → take minimal p across groups; else fixed g#
) -> Tuple[List[str], List[str]]:
    """
    Return lists of matching p-value and coef columns for a given (model, variable)
    and (optionally) group. Matches the columns created by correlation_results_summary.
    """
    # Example columns:
    #   simple_LR-<var>-g0-s0-d0-pval
    #   simple_LR-<var>-g0-s0-d0-coef
    patt_p = f"{model}-{variable}-g{{g}}-{shift_tag}-d{direction}-pval"
    patt_b = f"{model}-{variable}-g{{g}}-{shift_tag}-d{direction}-coef"

    if group is not None:
        p_cols = [patt_p.format(g=group)]
        b_cols = [patt_b.format(g=group)]
        missing = [c for c in (p_cols + b_cols) if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in summary for requested group: {missing}")
        return p_cols, b_cols

    # Otherwise, include all groups present in the table for this (model, variable, shift, direction)
    p_cols: List[str] = []
    b_cols: List[str] = []
    g_idx = 0
    while True:
        p = patt_p.format(g=g_idx)
        b = patt_b.format(g=g_idx)
        if (p in df.columns) and (b in df.columns):
            p_cols.append(p)
            b_cols.append(b)
            g_idx += 1
        else:
            break

    if not p_cols:
        # fallback: scan columns to see what groups exist even if not contiguous
        candidates_p = [c for c in df.columns if c.startswith(f"{model}-{variable}-g") and c.endswith(f"{shift_tag}-d{direction}-pval")]
        candidates_b = [c for c in df.columns if c.startswith(f"{model}-{variable}-g") and c.endswith(f"{shift_tag}-d{direction}-coef")]
        # keep only groups that appear in both p and coef
        gs = sorted(
            set(int(c.split("-g")[1].split("-")[0]) for c in candidates_p)
            & set(int(c.split("-g")[1].split("-")[0]) for c in candidates_b)
        )
        for g in gs:
            p_cols.append(patt_p.format(g=g))
            b_cols.append(patt_b.format(g=g))

    if not p_cols:
        raise KeyError(
            f"No matching significance/coef columns for model='{model}', variable='{variable}', "
            f"shift='{shift_tag}', direction={direction}. "
            f"Check your summary table column names."
        )
    return p_cols, b_cols


def _select_significant_units_from_summary(
    summary_df: pd.DataFrame,
    *,
    model: str,
    variable: str,
    time_window_label: str = "-1_0",
    z_score_flag: bool = True,
    p_alpha: float = 0.05,
    shift_tag: str = "s0",
    direction: int = 0,
    group: Optional[int] = None,
    slope_sign: Literal["positive","negative","both"] = "both",
    session_id: Optional[str] = None,   # ← NEW: filter to this session_id if provided
) -> Tuple[List[int], pd.Series]:
    """
    Select units that pass the p-value threshold from the wide-format summary.

    Parameters
    ----------
    summary_df : pd.DataFrame
        The wide correlation-summary table produced by your pipeline.
        Must include columns: 'session_id', 'time_window', 'z_score',
        '<model>-<variable>-g<i>-<shift_tag>-d<direction>-pval/coef', and 'unit_index' (or 'unit').
    model : str
        Correlation model family (e.g., 'simple_LR', 'ARDL_model').
    variable : str
        Behaviour/latent variable suffix used in summary column names.
    time_window_label : str, default '-1_0'
        Facet to select in the 'time_window' column.
    z_score_flag : bool, default True
        Facet to select in the 'z_score' column.
    p_alpha : float, default 0.05
        Significance cutoff; units with p <= p_alpha are kept.
    shift_tag : str, default 's0'
        Trial-shift tag in column names (e.g., 's0', 's+1', 's-1').
    direction : int, default 0
        Regression direction index used in column names.
    group : int or None, default None
        Variable-group index (g#). If None, min p across all groups is used and
        the corresponding coef is selected.
    slope_sign : {'positive','negative','both'}, default 'both'
        Slope sign filter applied to the selected coefficient.
    session_id : str or None, default None
        If provided, restrict rows to this exact 'session_id' value
        (e.g., '776293_2025-02-18_12-51-36').

    Returns
    -------
    tuple[list[int], pd.Series]
        (unit_indices, coef_series_for_selected_group)
    """
    # ---- facet filter (time_window & z_score) ----
    facet = summary_df[
        (summary_df["time_window"] == time_window_label) &
        (summary_df["z_score"] == z_score_flag)
    ]

    # ---- NEW: session filter ----
    if session_id is not None and "session_id" in facet.columns:
        facet = facet[facet["session_id"] == session_id]

    if facet.empty:
        return [], pd.Series(dtype=float)

    # ---- find matching column sets ----
    p_cols, b_cols = _pick_sig_and_coef_columns(
        facet,
        model=model,
        variable=variable,
        shift_tag=shift_tag,
        direction=direction,
        group=group
    )

    P = facet[p_cols].astype(float).copy()
    B = facet[b_cols].astype(float).copy()

    if len(p_cols) == 1:
        p_best = P[p_cols[0]]
        b_best = B[b_cols[0]]
    else:
        argmin = P.values.argmin(axis=1)
        p_best = pd.Series(P.values.min(axis=1), index=facet.index)
        b_best = pd.Series([B.iloc[i, argmin[i]] for i in range(len(facet))], index=facet.index)

    keep_mask = p_best <= p_alpha

    if slope_sign != "both":
        keep_mask &= (b_best >= 0) if slope_sign == "positive" else (b_best < 0)

    kept = facet[keep_mask]
    if kept.empty:
        return [], pd.Series(dtype=float)

    unit_col = "unit_index" if "unit_index" in kept.columns else "unit"
    units = kept[unit_col].astype(int).tolist()
    return units, b_best.loc[kept.index]


# -------------------------------------------------------------------
#  Main plotting function (adapted to your current repo & summary)
# -------------------------------------------------------------------
def plot_quantiles_adapted(
    nwb_data: Any,
    *,
    # significance source
    zarr_path: str = "/root/capsule/scratch/correlation_results/sig_dir_all_sessions.zarr",
    model: str = "simple_LR",
    variable: str = "chosenQ",
    time_window_label: str = "-1_0",
    z_score_flag: bool = True,
    p_value_threshold: float = 0.05,
    shift_tag: str = "s0",
    direction: int = 0,
    group: Optional[int] = 0,
    slope_selection: Sequence[Literal["positive","negative"]] = ("positive","negative"),

    # trial/latent & PSTH
    behavior_model: str = "QLearning_L2F1_softmax",
    latent_name: str = "chosenQ",
    align_event: str = "go_cue",
    time_window: Tuple[float, float] = (-1.0, 0.0),
    bin_size: float = 0.05,
    z_score_psth: bool = True,
    smooth_sigma: Optional[float] = None,

    # baseline normalization
    normalize_to_baseline: bool = False,
    baseline_period: Optional[Tuple[float, float]] = (-1.0, 0.0),

    # unit/probe filters
    unit_indices: Optional[Sequence[int]] = None,
    probes: Optional[Sequence[Union[str,int]]] = None,

    # quantiles & labeling
    quantiles_num: int = 5,
    title_font_size: int = 16,
    label_font_size: int = 12,

    # save
    save_folder: str = "/root/capsule/results",
    save_basename: str = "PSTH_quantiles",
    save_format: Literal["png","pdf","eps"] = "png",
) -> None:
    """
    Plot mean±SEM PSTHs grouped by quantiles of a fitted latent variable, using
    significant units determined from a wide-format correlation summary (.zarr).

    Parameters
    ----------
    nwb_data : Any
        Combined NWB session object; must expose:
        - `units['spike_times']` (absolute spike times in seconds)
        - `intervals['trials']` with at least `goCue_start_time` and `animal_response`.
    zarr_path : str, default "/root/capsule/results/sig_dir_all_sessions.zarr"
        Path to the wide-format correlation summary produced by your pipeline
        (loadable via `general_utils.load_temporary_data`).
    model : str, default "simple_LR"
        Correlation model family to select in the summary (e.g., "simple_LR", "ARDL_model").
    variable : str, default "q_value_difference"
        Behavioural/latent variable suffix used in summary column names; must match
        the `<model>-<variable>-g*-s*-d*-{pval,coef}` pattern in the table.
    time_window_label : str, default "-1_0"
        Facet of the summary to use (must equal entries in the summary's `time_window` column).
    z_score_flag : bool, default True
        Facet flag for the summary; must equal entries in the summary's `z_score` column.
    p_value_threshold : float, default 0.05
        Significance cutoff; units with min p-value (over groups if `group=None`) ≤ this threshold are kept.
    shift_tag : str, default "s0"
        Trial-shift tag in the summary columns (e.g., "s0", "s+1", "s-1").
    direction : int, default 0
        Regression direction index used in summary columns:
        0 = behaviour → firing rate; 1 = firing rate → behaviour.
    group : int or None, default None
        Variable-group index (g#) to use. If None, the function takes the **minimum p-value**
        across all available groups and uses its corresponding coefficient.
    slope_selection : sequence of {"positive","negative"}, default ("positive","negative")
        Which slope-signed unit sets to include. If both are included, negative-slope units
        flip quantile ordering (to align “high latent” with higher PSTH).

    behavior_model : str, default "QLearning_L2F1_softmax"
        Behaviour model to pass to `extract_fitted_data` for latent extraction.
    latent_name : str, default "q_value_difference"
        Latent variable name to request from `extract_fitted_data`.
    align_event : str, default "go_cue"
        Behaviour event name used for per-trial alignment (x=0 in PSTH).
    time_window : (float, float), default (-1.0, 0.0)
        Time range (sec, relative to `align_event`) for PSTH binning.
    bin_size : float, default 0.05
        PSTH histogram bin width in seconds.
    z_score_psth : bool, default True
        If True, z-score each unit’s quantile PSTHs across all bins/quantiles before averaging.
    smooth_sigma : float or None, default None
        Optional Gaussian σ (in bins) to smooth the **mean** PSTH curve per quantile.

    normalize_to_baseline : bool, default False
        If True, subtract the mean firing rate over `baseline_period` from each quantile’s mean PSTH.
    baseline_period : (float, float) or None, default (-1.0, 0.0)
        Baseline window (sec, relative to `align_event`) used when `normalize_to_baseline=True`.
        If None, no baseline subtraction is applied.

    unit_indices : sequence[int] or None, default None
        Optional explicit subset of unit indices to intersect with the significant+QC set.
    probes : sequence[str|int] or None, default None
        Optional filter by `units['device_name']`; only units whose device name string
        matches any entry are kept.

    quantiles_num : int, default 5
        Number of quantile bins for the latent variable (must be ≥ 2).
    title_font_size : int, default 16
        Matplotlib title font size.
    label_font_size : int, default 12
        Matplotlib axis label font size.

    save_folder : str, default "/root/capsule/results"
        Directory to write the output figure.
    save_basename : str, default "PSTH_quantiles"
        Base filename (without extension) for the saved figure.
    save_format : {"png","pdf","eps"}, default "png"
        Output image format.

    Returns
    -------
    None
        The function displays the plot and writes it to disk if requested.
        
    Adapted quantile-PSTH plot using modern significance table and current repo utilities.
    """
    # ---------------------------
    # Load wide-format summary
    # ---------------------------
    sig_df = load_temporary_data(zarr_path)
    if not isinstance(sig_df, pd.DataFrame):
        # load_temporary_data may return xarray Dataset; convert safely
        try:
            sig_df = sig_df.to_dataframe()
        except Exception:
            raise TypeError("Loaded significance is not a DataFrame or convertible xarray Dataset.")

    if "unit_index" not in sig_df.columns and "unit" in sig_df.columns:
        sig_df = sig_df.rename(columns={"unit": "unit_index"})

    # ---------------------------
    # Select significant units
    # ---------------------------
    sess_core = extract_session_name_core(nwb_data.session_id)

    units_sig, coef_series = _select_significant_units_from_summary(
        sig_df,
        model=model,
        variable=variable,
        time_window_label=time_window_label,
        z_score_flag=z_score_flag,
        p_alpha=p_value_threshold,
        shift_tag=shift_tag,
        direction=direction,
        group=group,
        slope_sign="both",
        session_id=sess_core,  # ← NEW: restrict to this session
    )

    if len(units_sig) == 0:
        print(f"[INFO] No significant units at p ≤ {p_value_threshold}.")
        return

    # split by slope sign using the same coef_series
    units_pos = [u for u in units_sig if coef_series.loc[coef_series.index[sig_df.loc[coef_series.index, 'unit_index'].astype(int)==u]].iloc[0] >= 0] \
        if len(units_sig) > 0 else []
    units_neg = [u for u in units_sig if u not in units_pos]

    use_pos = "positive" in slope_selection
    use_neg = "negative" in slope_selection

    units_to_use: List[int] = []
    if use_pos:
        units_to_use.extend(units_pos)
    if use_neg:
        units_to_use.extend(units_neg)

    # ---------------------------
    # Intersect with QC and probe filters
    # ---------------------------
    qc_units = set(map(int, get_units_passed_default_qc(nwb_data)))
    if unit_indices is not None:
        qc_units &= set(map(int, unit_indices))

    # probe filter via units['device_name']
    if probes:
        probe_set = {str(p) for p in probes}
        kept = []
        for i in qc_units:
            try:
                dn = nwb_data.units["device_name"][i]
                if dn is None:
                    continue
                if str(dn) in probe_set:
                    kept.append(i)
            except Exception:
                continue
        qc_units = set(kept)

    units_to_use = [u for u in units_to_use if u in qc_units]
    if len(units_to_use) == 0:
        print("[INFO] No units left after QC/probe/unit-index filters.")
        return

    # ---------------------------
    # Trials & latent per trial
    # ---------------------------
    trials = nwb_data.intervals["trials"][:]
    go_cue_times = np.asarray(trials["goCue_start_time"].to_numpy(), dtype=float)

    latent_data = np.asarray(
        extract_fitted_data(nwb_behavior_data=nwb_data, session_name=nwb_data.session_id,model_alias=behavior_model, latent_name=latent_name),
        dtype=float
    )

    # Align lengths by excluding "no response" (animal_response == 2), if needed
    if latent_data.shape[0] != go_cue_times.shape[0]:
        choice_hist = trials["animal_response"].to_numpy()
        valid_mask = (choice_hist != 2)
        go_cue_times = go_cue_times[valid_mask]
        # If your extract_fitted_data is already filtered, we assume it's aligned to valid trials.
        if latent_data.shape[0] != valid_mask.sum():
            # Try to trim to min length to be safe
            min_len = min(len(latent_data), valid_mask.sum())
            go_cue_times = go_cue_times[:min_len]
            latent_data = latent_data[:min_len]

    # ---------------------------
    # Quantile edges & labels
    # ---------------------------
    if quantiles_num < 2:
        raise ValueError("quantiles_num must be >= 2.")
    q_edges = np.linspace(np.min(latent_data), np.max(latent_data), quantiles_num + 1)
    q_labels = [f"Q{i+1}" for i in range(quantiles_num)]

    # which slope groups to compute
    slope_groups: List[Tuple[str, List[int], bool]] = []
    if use_pos:
        slope_groups.append(("positive", [u for u in units_to_use if u in units_pos], False))
    if use_neg:
        # flip_quintile means we invert bin labels for negative slope group (optional behavior from your old code)
        slope_groups.append(("negative", [u for u in units_to_use if u in units_neg], True if use_pos else False))

    # ---------------------------
    # PSTH bins & containers
    # ---------------------------
    bins = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
    bin_centers = bins[:-1] + np.diff(bins)[0] / 2

    # Per-quantile, stack unit PSTHs → mean ± SEM
    def _unit_psth_for_quantiles(unit: int, flip_quintiles: bool) -> Dict[str, np.ndarray]:
        # collect trial-wise histograms per quantile
        per_q: Dict[str, List[np.ndarray]] = {ql: [] for ql in q_labels}
        spikes_abs = np.asarray(nwb_data.units["spike_times"][unit], dtype=float)

        for go_t, val in zip(go_cue_times, latent_data):
            # find which quantile
            q_idx = None
            for i in range(quantiles_num):
                if (q_edges[i] <= val <= q_edges[i+1]):
                    q_idx = i
                    break
            if q_idx is None:
                continue
            if flip_quintiles:
                q_idx = quantiles_num - q_idx - 1
            q_name = q_labels[q_idx]

            rel = spikes_abs[(spikes_abs >= go_t + time_window[0]) & (spikes_abs <= go_t + time_window[1])] - go_t
            counts, _ = np.histogram(rel, bins=bins)
            per_q[q_name].append(counts)

        # per-unit mean rate per quantile
        out = {}
        for qn in q_labels:
            if len(per_q[qn]) == 0:
                out[qn] = np.zeros(len(bins)-1, dtype=float)
            else:
                mean_counts = np.mean(np.vstack(per_q[qn]), axis=0)
                out[qn] = mean_counts / bin_size
        return out

    # accumulate across units
    all_quants: Dict[str, List[np.ndarray]] = {ql: [] for ql in q_labels}
    total_units_count = 0
    for group_name, units_group, flip in slope_groups:
        for u in units_group:
            total_units_count += 1
            psth_map = _unit_psth_for_quantiles(u, flip_quintiles=flip)
            vecs = np.vstack([psth_map[q] for q in q_labels])  # shape (Q, T)

            if z_score_psth:
                # z-score across all bins and quantiles for this unit
                mu = np.nanmean(vecs)
                sd = np.nanstd(vecs)
                if sd > 0:
                    vecs = (vecs - mu) / sd
                else:
                    vecs = vecs * 0.0

            for i, qn in enumerate(q_labels):
                all_quants[qn].append(vecs[i])

    if total_units_count == 0:
        print("[INFO] No units contributed PSTHs after filters.")
        return

    # mean & SEM per quantile; optional baseline normalization & smoothing
    q_stats: Dict[str, Dict[str, np.ndarray]] = {}
    for qn in q_labels:
        arr = np.vstack(all_quants[qn])  # shape (N_units, T)
        mean_psth = np.nanmean(arr, axis=0)
        sem_psth = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])

        if normalize_to_baseline and baseline_period is not None:
            m = (bin_centers >= baseline_period[0]) & (bin_centers <= baseline_period[1])
            if m.any():
                baseline = np.nanmean(mean_psth[m])
                mean_psth = mean_psth - baseline

        if smooth_sigma is not None and smooth_sigma > 0:
            mean_psth = gaussian_filter1d(mean_psth, sigma=smooth_sigma)

        q_stats[qn] = {"mean": mean_psth, "sem": sem_psth}

    # trial counts per quantile (for legend label)
    trial_counts: Dict[str, int] = {
        q_labels[i]: int(np.sum((latent_data >= q_edges[i]) & (latent_data <= q_edges[i+1])))
        for i in range(quantiles_num)
    }

    # ---------------------------
    # Plot
    # ---------------------------
    fig, ax = plt.subplots(figsize=(12, 8))

    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=float(np.min(latent_data)), vmax=float(np.max(latent_data)))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(latent_data)

    # color per quantile by latent midpoint
    for i, qn in enumerate(q_labels):
        mid = 0.5 * (q_edges[i] + q_edges[i+1])
        col = cmap(norm(mid))
        m = q_stats[qn]["mean"]
        s = q_stats[qn]["sem"]
        ax.plot(bin_centers, m, color=col, label=f"{qn} (n={trial_counts[qn]})")
        ax.fill_between(bin_centers, m - s, m + s, color=col, alpha=0.30)

    # event marker at 0 (go-cue)
    ax.axvline(0.0, color="red", linestyle="--", label="Go Cue")

    ax.set_xlabel("Time from Go Cue (s)", fontsize=label_font_size)
    ax.set_ylabel("Z-scored Firing Rate" if z_score_psth else "Firing Rate (Hz)", fontsize=label_font_size)

    ax.set_title(
        f"PSTH by {latent_name} quantiles  |  units={total_units_count}  |  "
        f"facet time_window={time_window_label}, z_score={z_score_flag}",
        fontsize=title_font_size, weight="bold"
    )

    # Colorbar labeling
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f"Latent: {latent_name}")
    cbar.set_ticks(q_edges)
    cbar.set_ticklabels([f"{v:.2f}" for v in q_edges])
    # optional midpoint ticks
    midpoints = [(q_edges[i] + q_edges[i+1]) / 2 for i in range(len(q_edges)-1)]
    cbar.ax.yaxis.set_ticks(midpoints, minor=False)
    cbar.ax.set_yticklabels([f"{v:.2f}" for v in midpoints])

    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()

    # save
    os.makedirs(save_folder, exist_ok=True)
    out_path = os.path.join(save_folder, f"{save_basename}.{save_format}")
    fig.savefig(out_path, format=save_format, dpi=150, bbox_inches="tight")
    print(f"[OK] Figure saved → {out_path}")

    plt.show()
