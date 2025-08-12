import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional, Union


def plot_stay_switch_over_window(
    combined_dataframe: pd.DataFrame,
    vis_types: List[str] = ("stay", "switch", "win_stay", "lose_switch", "response"),
    window: Union[int, Tuple[int, int]] = (-2, 2),
    criteria: Optional[Dict[str, Any]] = None,
    session_col: str = "session",
    subject_col: str = "subject_id",
    laser_col: str = "laser_on_trial",
    time_col: str = "trial_num",
    line_by: str = "session",  # {"session", "subject", "all"}
    # (optional) narrow the dataset
    subject: Optional[str] = None,
    session: Optional[str] = None,
    share_y: bool = True,
    return_table: bool = False,
    figsize: Tuple[float, float] = (6.0, 4.5),
):
    """
    Plot one figure per metric (stay / switch / win_stay / lose_switch / response) as line charts
    over integer trial offsets relative to **opto anchor trials**.

    Offset definition:
      - offset = 0   → the selected **opto** trials themselves (anchors)
      - offset = ±k  → **non-opto** trials at k trials before/after each anchor (within the same session)

    Each line represents either a **session** or a **subject**, controlled by `line_by`.
      - line_by="session"  → one line per session (default behavior)
      - line_by="subject"  → sessions are pooled within subject using a weighted mean by `n_trials`

    Parameters
    ----------
    combined_dataframe : pandas.DataFrame
        Per-trial DataFrame (e.g., from `create_opto_data_frame_combined` or `load_opto_data_frame`).
        Must include:
          - boolean-like metrics: 'stay', 'switch', 'win_stay', 'lose_switch' (and optionally 'response')
          - session id column (default: 'session')
          - subject id column (default: 'subject_id')
          - opto flag column (default: 'laser_on_trial'): accepts 1/0, '1'/'0', True/False, etc.
          - within-session order column (default: 'trial_num'); if missing, falls back to 'start_time' or row order.

    vis_types : list[str], default ("stay", "switch", "win_stay", "lose_switch", "response")
        Which metrics to plot.

    window : int | tuple[int, int], default (-2, 2)
        Inclusive offset range around anchors. If int n → offsets = [-n..n]. If (lo, hi) → [lo..hi].

    criteria : dict | None, default None
        Extra filters to **select opto anchors** (applied after opto=True).
        Values: None → isna; list/tuple/set → isin; else → equality.

    session_col : str, default "session"
        Column name for session identifier.

    subject_col : str, default "subject_id"
        Column name for subject (mouse) identifier.

    laser_col : str, default "laser_on_trial"
        Column indicating opto vs non-opto. Internally normalized to boolean.

    time_col : str, default "trial_num"
        Column defining within-session order (used for sorting only).
        A fresh per-session index '_trial_idx' is created for offset math.

    line_by:
      - "session": one line per session (rates computed within-session)
      - "subject": pool trials across sessions for each subject, then compute rate from pooled trials
      - "all": pool trials across all subjects/sessions into a single line

    subject : str | None, default None
        Optional: restrict to a single subject before plotting.

    session : str | None, default None
        Optional: restrict to a single session before plotting.

    share_y : bool, default True
        If True, fix y-limits to [0, 1] for all figures.

    return_table : bool, default False
        If True, also return the tidy summary table used to make the plots.

    figsize : tuple[float, float], default (6.0, 4.5)
        Figure size for each metric plot.

    Returns
    -------
    figs : dict[str, matplotlib.figure.Figure]
        Mapping from metric name to its matplotlib Figure.

    summary_df : pandas.DataFrame, optional
        Returned when `return_table=True`. Columns:
        ['session', 'subject_id', 'metric', 'offset', 'rate', 'n_trials'] (pre-aggregation).

    Raises
    ------
    ValueError
        If required columns are missing or if no data remains after filtering.
    """

    df = combined_dataframe.copy()

    # ---------- validation ----------
    base_needed = {session_col, subject_col, laser_col}
    missing = sorted(base_needed - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    valid_metrics = {"stay", "switch", "win_stay", "lose_switch", "response"}
    vis_types = list(vis_types)
    bad = sorted(set(vis_types) - valid_metrics)
    if bad:
        raise ValueError(f"Unknown vis_types: {bad}. Valid: {sorted(valid_metrics)}")
    miss_metrics = [m for m in vis_types if m not in df.columns]
    if miss_metrics:
        raise ValueError(f"Missing metric columns: {miss_metrics}")

    if line_by not in {"session", "subject", "all"}:
        raise ValueError("line_by must be 'session', 'subject', or 'all'.")

    # ---------- optional narrowing ----------
    if subject is not None:
        df = df[df[subject_col].astype(str) == str(subject)]
    if session is not None:
        df = df[df[session_col].astype(str) == str(session)]
    if df.empty:
        raise ValueError("No data left after applying subject/session filters.")

    # ---------- helpers ----------
    def _as_opto_bool(s: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False)
        truthy = {"1", "true", "yes", "on", "y", "t",1}
        falsy  = {"0", "false", "no", "off", "n", "f", "",0}
        def parse(x):
            if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
                return False
            if isinstance(x, (bool, np.bool_)):   return bool(x)
            if isinstance(x, (int, np.integer)):  return x != 0
            if isinstance(x, (float, np.floating)): return int(x) != 0
            if isinstance(x, str):
                xs = x.strip().lower()
                if xs in truthy: return True
                if xs in falsy:  return False
            return False
        return s.map(parse)

    def _as_bool_series(s: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(s):
            return s.astype("boolean")
        truthy = {"1", "true", "yes", "on", "y", "t"}
        falsy  = {"0", "false", "no", "off", "n", "f", ""}
        def parse(x):
            if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
                return pd.NA
            if isinstance(x, (bool, np.bool_)):   return bool(x)
            if isinstance(x, (int, np.integer)):  return x != 0
            if isinstance(x, (float, np.floating)): return int(x) != 0
            if isinstance(x, str):
                xs = x.strip().lower()
                if xs in truthy: return True
                if xs in falsy:  return False
                return pd.NA
            return pd.NA
        return s.map(parse).astype("boolean")

    def _apply_criteria(x: pd.DataFrame, crit: Optional[Dict[str, Any]]) -> pd.DataFrame:
        if not crit:
            return x
        mask = pd.Series(True, index=x.index)
        for k, v in crit.items():
            if k not in x.columns:
                continue
            if v is None:                        mask &= x[k].isna()
            elif isinstance(v, (list, tuple, set)): mask &= x[k].isin(list(v))
            else:                                 mask &= (x[k] == v)
        return x[mask]

    # ---------- normalize flags & ordering ----------
    df["_is_opto"] = _as_opto_bool(df[laser_col])

    order_cols = [session_col]
    if time_col in df.columns:
        order_cols.append(time_col)
    elif "start_time" in df.columns:
        order_cols.append("start_time")
    df = df.sort_values(order_cols).reset_index(drop=True)
    df["_trial_idx"] = df.groupby(session_col).cumcount()

    for m in vis_types:
        df[m] = _as_bool_series(df[m])

    # ---------- offsets ----------
    if isinstance(window, int):
        lo, hi = -abs(window), abs(window)
    else:
        lo, hi = window
        if lo > hi:
            lo, hi = hi, lo
    offsets = list(range(lo, hi + 1))

    # ---------- compute per-session counts (successes & trials) ----------
    # We keep counts so higher-level pooling can be exact (no averaging of session rates).
    rows = []
    for sess_id, g in df.groupby(session_col, sort=False):
        if g.empty:
            continue

        subj_id = str(g[subject_col].iloc[0])
        n_trials_sess = len(g)

        anchors = _apply_criteria(g[g["_is_opto"]], criteria)
        if anchors.empty:
            continue

        anchor_idx = anchors["_trial_idx"].astype(int).values

        for off in offsets:
            if off == 0:
                cand_df = anchors
            else:
                tgt = anchor_idx + off
                tgt = tgt[(tgt >= 0) & (tgt < n_trials_sess)]
                if tgt.size == 0:
                    cand_df = g.iloc[[]]
                else:
                    hit = g[g["_trial_idx"].isin(tgt)]
                    cand_df = hit[~hit["_is_opto"]].drop_duplicates(subset=["_trial_idx"])

            for metric in vis_types:
                vals = cand_df[metric]
                n_used = int(vals.notna().sum())
                n_success = int(vals.fillna(False).sum())
                rate = (n_success / n_used) if n_used > 0 else np.nan
                rows.append({
                    "session": str(sess_id),
                    "subject_id": subj_id,
                    "metric": metric,
                    "offset": off,
                    "n_success": n_success,
                    "n_trials": n_used,
                    "rate": rate,  # per-session rate (for line_by='session')
                })

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise ValueError("No data to plot. Check filters/window/criteria and that opto trials exist.")

    # ---------- optionally pool by subject or all (trial-level exact) ----------
    if line_by == "subject":
        agg = (
            summary_df
            .groupby(["subject_id", "metric", "offset"], as_index=False)[["n_success", "n_trials"]]
            .sum()
        )
        agg["rate"] = np.where(agg["n_trials"] > 0, agg["n_success"] / agg["n_trials"], np.nan)
        agg["session"] = agg["subject_id"]  # fabricate plotting label
        plot_table = agg[["session", "subject_id", "metric", "offset", "rate", "n_trials"]].copy()
        line_labels = sorted(plot_table["session"].unique().tolist())  # subjects
    elif line_by == "all":
        agg = (
            summary_df
            .groupby(["metric", "offset"], as_index=False)[["n_success", "n_trials"]]
            .sum()
        )
        agg["rate"] = np.where(agg["n_trials"] > 0, agg["n_success"] / agg["n_trials"], np.nan)
        agg["session"] = "ALL"
        agg["subject_id"] = "ALL"
        plot_table = agg[["session", "subject_id", "metric", "offset", "rate", "n_trials"]].copy()
        line_labels = ["ALL"]
    else:
        # per-session lines (as before)
        plot_table = summary_df[["session", "subject_id", "metric", "offset", "rate", "n_trials"]].copy()
        line_labels = sorted(plot_table["session"].unique().tolist())

    # ---------- plot ----------
    subjects_present = sorted(plot_table["subject_id"].dropna().astype(str).unique().tolist())
    cmap = plt.get_cmap("tab20", max(1, len(subjects_present)))
    subject_to_color = {s: cmap(i % cmap.N) for i, s in enumerate(subjects_present)}

    xticks = sorted(plot_table["offset"].dropna().unique().tolist())
    figs = {}

    for metric in vis_types:
        sub = plot_table[plot_table["metric"] == metric]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=figsize)

        for label in line_labels:
            s_df = sub[sub["session"] == label].sort_values("offset")
            if s_df.empty:
                continue
            subj = s_df["subject_id"].iloc[0]
            color = subject_to_color.get(subj, None)
            ax.plot(
                s_df["offset"].values,
                s_df["rate"].values,
                marker="o",
                linewidth=1.4,
                alpha=0.95,
                color=color,
            )

        title_map = {"session": "Per-session", "subject": "Subject-pooled", "all": "All-data pooled"}
        ax.set_title(f"{title_map[line_by]}: {metric.replace('_',' ').title()} rate vs. offset")
        ax.set_xlabel("Offset (trials) relative to opto anchors (0 = opto)")
        ax.set_ylabel("Rate")
        if share_y:
            ax.set_ylim(0.0, 1.0)
        ax.set_xticks(xticks)
        ax.grid(True, axis="y", alpha=0.25)

        # Legend by subject (for 'all' this will just show ALL)
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=subject_to_color[s], lw=2, label=s)
            for s in subjects_present
        ]
        if handles:
            ax.legend(handles=handles, title="Subject", bbox_to_anchor=(1.02, 1), loc="upper left")

        figs[metric] = fig

    return (figs, summary_df) if return_table else figs

def plot_rates_vs_latent(
    combined_dataframe: pd.DataFrame,
    latent_col: str,
    vis_types: List[str] = ("stay", "switch", "win_stay", "lose_switch", "response"),
    window: Union[int, Tuple[int, int]] = (-2, 2),
    bins: int = 12,
    binning: str = "quantile",  # {"quantile","uniform"}
    criteria: Optional[Dict[str, Any]] = None,
    session_col: str = "session",
    subject_col: str = "subject_id",
    laser_col: str = "laser_on_trial",
    time_col: str = "trial_num",
    line_by: str = "subject",   # {"session","subject","all"}
    subject: Optional[str] = None,
    session: Optional[str] = None,
    response_latent_fill: str = "ffill",  # {"ffill","bfill","nearest","none"}
    share_y: bool = True,
    return_table: bool = False,
    figsize: Tuple[float, float] = (6.0, 4.5),
):
    """
    Plot binned rate vs. latent variable under different trial offsets relative to opto anchors.

    Parameters
    ----------
    combined_dataframe : pandas.DataFrame
        Trial-level DataFrame. Must include:
        - `session_col` (default "session")
        - `subject_col` (default "subject_id")
        - `laser_col` (default "laser_on_trial") — opto flag
        - `time_col` (default "trial_num") or `start_time` (for within-session sort)
        - Boolean-like metric columns for all entries in `vis_types`
        - The latent column `latent_col` (will be coerced to numeric)
    latent_col : str
        Name of the latent variable column. At least 2 non-NA values are required to form bins.
    vis_types : list[str], optional
        Metrics to plot. Valid: {"stay","switch","win_stay","lose_switch","response"}.
    window : int or (int, int), optional
        Offset range around anchors.
        - int n  -> offsets = [-n, ..., 0, ..., +n]
        - (lo,hi) -> offsets = [lo, ..., hi] (order normalized if lo>hi)
        Selection rule:
        - 0: use **opto anchor** trials that match `criteria`
        - ±k: use **non-opto** trials k before/after each anchor (same session), de-duplicated
    bins : int, optional
        Number of bins for the latent variable (global, shared across all lines).
    binning : {"quantile","uniform"}, optional
        - "quantile": equal-count bins via `np.quantile`; falls back if edges collapse
        - "uniform": equal-width bins from min to max
        Bins are right-closed; bin centers are (left+right)/2.
    criteria : dict or None, optional
        Extra filters applied **only** to opto anchors, after `laser_col` is True.
        For each key/value:
        - value is None      -> keep rows where `col.isna()`
        - value is list/seq  -> keep rows where `col.isin(value)`
        - otherwise          -> keep rows where `col == value`
        Missing columns are ignored.
    session_col : str, optional
        Session identifier column. Default "session".
    subject_col : str, optional
        Subject/animal identifier column. Default "subject_id".
    laser_col : str, optional
        Opto flag column. Accepts 1/0, True/False, and truthy/falsy strings; coerced to bool.
        Unknown/missing treated as False (non-opto).
    time_col : str, optional
        Column used to sort within a session before building the per-session index "_trial_idx".
        If absent, falls back to "start_time", else row order.
    line_by : {"session","subject","all"}, optional
        Controls line grouping and aggregation:
        - "session": one line per session (rates computed within each session)
        - "subject": **pool trials across sessions per subject** (sum successes/trials per bin, then rate)
        - "all": **pool trials across all subjects/sessions** (sum successes/trials per bin, then rate)
        Note: pooling is exact (no averaging of session means).
    subject : str or None, optional
        If provided, pre-filter to this subject (string-compared).
    session : str or None, optional
        If provided, pre-filter to this session (string-compared).
    response_latent_fill : {"ffill","bfill","nearest","none"}, optional
        Fill strategy for latent values used **only** when binning the "response" metric.
        Many pipelines attach latent only to responded trials; filling ensures non-response
        trials land in bins so `response` is not trivially 1. "nearest" does ffill then bfill.
        "none" uses the raw latent (may exclude non-response trials from bins).
    share_y : bool, optional
        If True, y-limits are fixed to [0, 1] for all figures.
    return_table : bool, optional
        If True, also return the tidy table used to construct the plots.
    figsize : (float, float), optional
        Matplotlib figure size for each metric.

    Returns
    -------
    figs : dict[str, matplotlib.figure.Figure]
        Mapping from metric -> Figure. X: latent bin centers; Y: rate within bin.
        Line color encodes offset; subjects/sessions sharing the same offset share color.
    summary_df : pandas.DataFrame, optional
        Only if `return_table=True`. Contains (superset of):
        ["session","subject_id","metric","offset",
         "bin_left","bin_right","bin_center",
         "n_success","n_trials","rate"].

    Raises
    ------
    ValueError
        If required columns are missing, filters yield no data, bins cannot be formed,
        or a requested metric column is absent.

    Notes
    -----
    - Anchor selection is opto==True AND matches `criteria`. ±offset sets include only non-opto trials.
    - For "response", binning uses the filled latent per `response_latent_fill` so non-response trials
      contribute to denominators; other metrics use the original latent.
    - When `line_by` is "subject" or "all", rates are computed from **summed counts** (exact pooling).
    """

    df = combined_dataframe.copy()

    # ---------- validation ----------
    base_needed = {session_col, subject_col, laser_col, latent_col}
    missing = sorted(base_needed - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    valid_metrics = {"stay", "switch", "win_stay", "lose_switch", "response"}
    vis_types = list(vis_types)
    bad = sorted(set(vis_types) - valid_metrics)
    if bad:
        raise ValueError(f"Unknown vis_types: {bad}. Valid: {sorted(valid_metrics)}")
    miss_metrics = [m for m in vis_types if m not in df.columns]
    if miss_metrics:
        raise ValueError(f"Missing metric columns: {miss_metrics}")

    if line_by not in {"session", "subject", "all"}:
        raise ValueError("line_by must be one of {'session','subject','all'}.")

    # ---------- optional narrowing ----------
    if subject is not None:
        df = df[df[subject_col].astype(str) == str(subject)]
    if session is not None:
        df = df[df[session_col].astype(str) == str(session)]
    if df.empty:
        raise ValueError("No data left after applying subject/session filters.")

    # ---------- helpers ----------
    def _as_opto_bool(s: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False)
        truthy = {"1", "true", "yes", "on", "y", "t"}
        falsy  = {"0", "false", "no", "off", "n", "f", ""}
        def parse(x):
            if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
                return False
            if isinstance(x, (bool, np.bool_)):   return bool(x)
            if isinstance(x, (int, np.integer)):  return x != 0
            if isinstance(x, (float, np.floating)): return int(x) != 0
            if isinstance(x, str):
                xs = x.strip().lower()
                if xs in truthy: return True
                if xs in falsy:  return False
            return False
        return s.map(parse)

    def _as_bool_series(s: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(s):
            return s.astype("boolean")
        truthy = {"1", "true", "yes", "on", "y", "t"}
        falsy  = {"0", "false", "no", "off", "n", "f", ""}
        def parse(x):
            if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
                return pd.NA
            if isinstance(x, (bool, np.bool_)):   return bool(x)
            if isinstance(x, (int, np.integer)):  return x != 0
            if isinstance(x, (float, np.floating)): return int(x) != 0
            if isinstance(x, str):
                xs = x.strip().lower()
                if xs in truthy: return True
                if xs in falsy:  return False
                return pd.NA
            return pd.NA
        return s.map(parse).astype("boolean")

    def _apply_criteria(x: pd.DataFrame, crit: Optional[Dict[str, Any]]) -> pd.DataFrame:
        if not crit:
            return x
        mask = pd.Series(True, index=x.index)
        for k, v in crit.items():
            if k not in x.columns:
                continue
            if v is None:                        mask &= x[k].isna()
            elif isinstance(v, (list, tuple, set)): mask &= x[k].isin(list(v))
            else:                                 mask &= (x[k] == v)
        return x[mask]

    # ---------- normalize flags & ordering ----------
    df["_is_opto"] = _as_opto_bool(df[laser_col])

    order_cols = [session_col]
    if time_col in df.columns:
        order_cols.append(time_col)
    elif "start_time" in df.columns:
        order_cols.append("start_time")
    df = df.sort_values(order_cols).reset_index(drop=True)
    df["_trial_idx"] = df.groupby(session_col).cumcount()

    # Cast metric columns to nullable boolean
    for m in vis_types:
        df[m] = _as_bool_series(df[m])

    # Latent numeric
    df[latent_col] = pd.to_numeric(df[latent_col], errors="coerce")

    # Build a filled-latent specifically for the 'response' metric (per session)
    if response_latent_fill.lower() in {"ffill", "bfill", "nearest"}:
        def _fill_lat(g: pd.DataFrame) -> pd.Series:
            s = g[latent_col]
            if response_latent_fill.lower() in {"ffill"}:
                return s.ffill()
            if response_latent_fill.lower() in {"bfill"}:
                return s.bfill()
            # "nearest": simple two-pass nearest (ffill then bfill for remaining)
            return s.ffill().bfill()
        df["_latent_for_response"] = df.groupby(session_col, sort=False, group_keys=False).apply(_fill_lat)
    else:
        df["_latent_for_response"] = df[latent_col]

    # Sanity: we need at least 2 non-NA latent values to form bins (use original distribution)
    if df[latent_col].notna().sum() < 2 and df["_latent_for_response"].notna().sum() < 2:
        raise ValueError(f"Not enough non-NA values in '{latent_col}' to form bins.")

    # ---------- offsets ----------
    if isinstance(window, int):
        lo, hi = -abs(window), abs(window)
    else:
        lo, hi = window
        if lo > hi:
            lo, hi = hi, lo
    offsets = list(range(lo, hi + 1))

    # ---------- global bins (from original latent range) ----------
    latent_all = df[latent_col].dropna().values
    # If original is too sparse, fall back to filled (for response-only datasets)
    if latent_all.size < 2:
        latent_all = df["_latent_for_response"].dropna().values
    lat_min, lat_max = float(np.min(latent_all)), float(np.max(latent_all))
    if binning.lower() == "quantile":
        q = np.linspace(0, 1, bins + 1)
        edges = np.quantile(latent_all, q)
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.linspace(lat_min, lat_max, min(bins, 3))
    else:
        edges = np.linspace(lat_min, lat_max, bins + 1)
    edges = np.unique(edges)
    if edges.size < 2:
        raise ValueError("Failed to construct bin edges for latent variable.")
    bin_left = edges[:-1]
    bin_right = edges[1:]
    bin_centers = (bin_left + bin_right) / 2.0
    canonical_bins = pd.IntervalIndex.from_breaks(edges, closed="right")

    # ---------- per-session per-bin counts ----------
    rows = []
    for sess_id, g in df.groupby(session_col, sort=False):
        if g.empty:
            continue
        subj_id = str(g[subject_col].iloc[0])
        n_trials_sess = len(g)

        anchors = _apply_criteria(g[g["_is_opto"]], criteria)
        if anchors.empty:
            continue

        anchor_idx = anchors["_trial_idx"].astype(int).values

        for off in offsets:
            if off == 0:
                cand_df = anchors.copy()
            else:
                tgt = anchor_idx + off
                tgt = tgt[(tgt >= 0) & (tgt < n_trials_sess)]
                if tgt.size == 0:
                    cand_df = g.iloc[[]].copy()
                else:
                    hit = g[g["_trial_idx"].isin(tgt)].copy()
                    cand_df = hit[~hit["_is_opto"]].drop_duplicates(subset=["_trial_idx"]).copy()

            if cand_df.empty:
                continue

            # Bin for general metrics: use original latent
            cand_df["_latent_bin"] = pd.cut(
                cand_df[latent_col],
                bins=canonical_bins,
                include_lowest=True
            )
            # Bin for 'response': use the filled latent so non-response trials are included
            cand_df["_latent_bin_response"] = pd.cut(
                cand_df["_latent_for_response"],
                bins=canonical_bins,
                include_lowest=True
            )

            for metric in vis_types:
                if metric == "response":
                    # Denominator: ALL candidate trials that received a (filled) latent bin
                    tmp = pd.DataFrame({
                        "_latent_bin": cand_df["_latent_bin_response"],
                        "_hit": cand_df["response"].fillna(False).astype(bool)  # NA -> no response
                    })
                    grp = tmp.groupby("_latent_bin", observed=True)["_hit"].agg(
                        n_trials="size",
                        n_success=lambda s: int(s.sum())
                    )
                else:
                    # Denominator excludes NA (metric undefined)
                    tmp = pd.DataFrame({
                        "_latent_bin": cand_df["_latent_bin"],
                        "_hit": cand_df[metric]
                    })
                    grp = tmp.groupby("_latent_bin", observed=True)["_hit"].agg(
                        n_trials=lambda s: int(pd.Series(s).notna().sum()),
                        n_success=lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())
                    )

                # Emit rows for all bins (keep structure)
                for iv, c in zip(canonical_bins, bin_centers):
                    if iv in grp.index:
                        n_used = int(grp.loc[iv, "n_trials"])
                        n_success = int(grp.loc[iv, "n_success"])
                        rate = (n_success / n_used) if n_used > 0 else np.nan
                    else:
                        n_used = 0
                        n_success = 0
                        rate = np.nan
                    rows.append({
                        "session": str(sess_id),
                        "subject_id": subj_id,
                        "metric": metric,
                        "offset": off,
                        "bin_left": float(iv.left),
                        "bin_right": float(iv.right),
                        "bin_center": float(c),
                        "n_success": n_success,
                        "n_trials": n_used,
                        "rate": rate,  # per-session per-bin rate
                    })

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise ValueError("No data to plot. Check filters/window/criteria, latent values, and opto trials.")

    # ---------- exact pooling by counts ----------
    if line_by == "subject":
        agg = (
            summary_df
            .groupby(["subject_id", "metric", "offset", "bin_left", "bin_right", "bin_center"], as_index=False)[["n_success", "n_trials"]]
            .sum()
        )
        agg["rate"] = np.where(agg["n_trials"] > 0, agg["n_success"] / agg["n_trials"], np.nan)
        agg["session"] = agg["subject_id"]
        plot_table = agg[["session", "subject_id", "metric", "offset", "bin_center", "rate", "n_trials"]].copy()
        line_labels = sorted(plot_table["session"].unique().tolist())
    elif line_by == "all":
        agg = (
            summary_df
            .groupby(["metric", "offset", "bin_left", "bin_right", "bin_center"], as_index=False)[["n_success", "n_trials"]]
            .sum()
        )
        agg["rate"] = np.where(agg["n_trials"] > 0, agg["n_success"] / agg["n_trials"], np.nan)
        agg["session"] = "ALL"
        agg["subject_id"] = "ALL"
        plot_table = agg[["session", "subject_id", "metric", "offset", "bin_center", "rate", "n_trials"]].copy()
        line_labels = ["ALL"]
    else:
        plot_table = summary_df[["session", "subject_id", "metric", "offset", "bin_center", "rate", "n_trials"]].copy()
        line_labels = sorted(plot_table["session"].unique().tolist())

    # ---------- plotting (colors encode offset) ----------
    unique_offsets = sorted(plot_table["offset"].dropna().astype(int).unique().tolist())
    cmap = plt.get_cmap("tab10", max(1, len(unique_offsets)))
    offset_to_color = {off: cmap(i % cmap.N) for i, off in enumerate(unique_offsets)}

    figs = {}
    for metric in vis_types:
        sub = plot_table[plot_table["metric"] == metric]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=figsize)

        for off in unique_offsets:
            off_df = sub[sub["offset"] == off]
            color = offset_to_color[off]
            for label in line_labels:
                ldf = off_df[off_df["session"] == label].sort_values("bin_center")
                if ldf.empty:
                    continue
                ax.plot(
                    ldf["bin_center"].values,
                    ldf["rate"].values,
                    marker="o",
                    linewidth=1.4,
                    alpha=0.95,
                    color=color,
                )

        title_map = {"session": "Per-session", "subject": "Subject-pooled", "all": "All-data pooled"}
        ax.set_title(f"{title_map[line_by]}: {metric.replace('_',' ').title()} rate vs. {latent_col} (binned)")
        ax.set_xlabel(f"{latent_col} (bin centers)")
        ax.set_ylabel("Rate")
        if share_y:
            ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.25)

        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=offset_to_color[o], lw=2, label=f"offset {o:+d}") for o in unique_offsets]
        if handles:
            ax.legend(handles=handles, title="Shift (offset)", bbox_to_anchor=(1.02, 1), loc="upper left")

        figs[metric] = fig

    return (figs, summary_df) if return_table else figs

