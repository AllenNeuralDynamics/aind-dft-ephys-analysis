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
    line_by: str = "session",  # {"session", "subject"}
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

    line_by : {"session","subject"}, default "session"
        - "session": draw one line per session.
        - "subject": pool sessions within each subject (weighted by `n_trials`) and draw one line per subject.

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

    if line_by not in {"session", "subject"}:
        raise ValueError("line_by must be 'session' or 'subject'.")

    # ---------- optional narrowing ----------
    if subject is not None:
        df = df[df[subject_col].astype(str) == str(subject)]
    if session is not None:
        df = df[df[session_col].astype(str) == str(session)]
    if df.empty:
        raise ValueError("No data left after applying subject/session filters.")

    # ---------- helpers: robust coercions ----------
    def _as_opto_bool(s: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False)
        truthy = {"1", "true", "yes", "on", "y", "t"}
        falsy  = {"0", "false", "no", "off", "n", "f", ""}
        def parse(x):
            if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
                return False
            if isinstance(x, (bool, np.bool_)):
                return bool(x)
            if isinstance(x, (int, np.integer)):
                return x != 0
            if isinstance(x, (float, np.floating)):
                return int(x) != 0
            if isinstance(x, str):
                xs = x.strip().lower()
                if xs in truthy: return True
                if xs in falsy:  return False
                return False
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
            if isinstance(x, (bool, np.bool_)):
                return bool(x)
            if isinstance(x, (int, np.integer)):
                return x != 0
            if isinstance(x, (float, np.floating)):
                return int(x) != 0
            if isinstance(x, str):
                xs = x.strip().lower()
                if xs in truthy: return True
                if xs in falsy:  return False
                return pd.NA
            return pd.NA
        return s.map(parse).astype("boolean")

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

    # ---------- criteria helper ----------
    def _apply_criteria(x: pd.DataFrame, crit: Optional[Dict[str, Any]]) -> pd.DataFrame:
        if not crit:
            return x
        mask = pd.Series(True, index=x.index)
        for k, v in crit.items():
            if k not in x.columns:
                continue
            if v is None:
                mask &= x[k].isna()
            elif isinstance(v, (list, tuple, set)):
                mask &= x[k].isin(list(v))
            else:
                mask &= (x[k] == v)
        return x[mask]

    # ---------- compute per-session rates table ----------
    session_ids = df[session_col].dropna().astype(str).unique().tolist()
    rows = []

    for sess_id in session_ids:
        g = df[df[session_col] == sess_id].copy()
        if g.empty:
            continue

        subj_id = str(g[subject_col].iloc[0])
        n_trials = len(g)

        anchors = _apply_criteria(g[g["_is_opto"]], criteria)
        if anchors.empty:
            continue

        anchor_idx = anchors["_trial_idx"].astype(int).values

        for off in offsets:
            if off == 0:
                cand_df = anchors
            else:
                tgt = anchor_idx + off
                tgt = tgt[(tgt >= 0) & (tgt < n_trials)]
                if tgt.size == 0:
                    cand_df = g.iloc[[]]
                else:
                    hit = g[g["_trial_idx"].isin(tgt)]
                    cand_df = hit[~hit["_is_opto"]].drop_duplicates(subset=["_trial_idx"])

            for metric in vis_types:
                vals = cand_df[metric].dropna()
                n_used = int(vals.shape[0])
                rate = float(vals.mean()) if n_used > 0 else np.nan
                rows.append({
                    "session": sess_id,
                    "subject_id": subj_id,
                    "metric": metric,
                    "offset": off,
                    "rate": rate,
                    "n_trials": n_used,
                })

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise ValueError("No data to plot. Check filters/window/criteria and that opto trials exist.")

    # ---------- optionally aggregate to per-subject lines ----------
    if line_by == "subject":
        # Weighted mean across sessions per subject/metric/offset (weight = n_trials)
        subdf = summary_df.copy()
        subdf = subdf[subdf["n_trials"] > 0]
        if subdf.empty:
            raise ValueError("No non-empty bins to aggregate for subjects.")
        agg = (
            subdf
            .groupby(["subject_id", "metric", "offset"], as_index=False)
            .apply(lambda g: pd.Series({
                "rate": np.average(g["rate"].fillna(0), weights=g["n_trials"]) if g["n_trials"].sum() > 0 else np.nan,
                "n_trials": int(g["n_trials"].sum())
            }))
        )
        # For plotting consistency, fabricate a 'session' label equal to subject_id
        agg = agg.rename(columns={"subject_id": "label"})
        agg["subject_id"] = agg["label"]
        agg["session"] = agg["label"]
        plot_table = agg[["session", "subject_id", "metric", "offset", "rate", "n_trials"]].copy()
        line_labels = sorted(plot_table["session"].unique().tolist())  # equals subjects
    else:
        # per-session lines (as before)
        plot_table = summary_df.copy()
        plot_table = plot_table.rename(columns={"session": "session"})
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

        title_prefix = "Subject-pooled" if line_by == "subject" else "Per-session"
        ax.set_title(f"{title_prefix}: {metric.replace('_',' ').title()} rate vs. offset")
        ax.set_xlabel("Offset (trials) relative to opto anchors (0 = opto)")
        ax.set_ylabel("Rate")
        if share_y:
            ax.set_ylim(0.0, 1.0)
        ax.set_xticks(xticks)
        ax.grid(True, axis="y", alpha=0.25)

        # legend by subject
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=subject_to_color[s], lw=2, label=s)
            for s in subjects_present
        ]
        if handles:
            ax.legend(handles=handles, title="Subject", bbox_to_anchor=(1.02, 1), loc="upper left")

        figs[metric] = fig

    return (figs, summary_df) if return_table else figs
