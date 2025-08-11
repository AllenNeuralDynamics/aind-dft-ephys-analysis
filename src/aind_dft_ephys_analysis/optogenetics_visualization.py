import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional, Union

def plot_stay_switch_over_window(
    combined_dataframe: pd.DataFrame,
    vis_types: List[str] = ("stay", "switch", "win_stay", "lose_switch","response"),
    window: Union[int, Tuple[int, int]] = (-2, 2),
    criteria: Optional[Dict[str, Any]] = None,
    session_col: str = "session",
    subject_col: str = "subject_id",
    laser_col: str = "laser_on_trial",
    time_col: str = "trial_num",
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

    Each session is drawn as a line. Sessions from the same subject (mouse) share the same color.

    Parameters
    ----------
    combined_dataframe : pandas.DataFrame
        The per-trial, concatenated DataFrame (e.g., from `create_opto_data_frame_combined` or
        `load_opto_data_frame`). Must include:
          - boolean-like metric columns: 'stay', 'switch', 'win_stay', 'lose_switch'
          - a session identifier column (default: 'session')
          - a subject identifier column (default: 'subject_id')
          - an opto flag column (default: 'laser_on_trial') that may be 1/0, '1'/'0', True/False, etc.
          - an order key per session (default: 'trial_num'); if missing, function will fall back to
            'start_time' (if present) or row order.

    vis_types : list[str], default ("stay", "switch", "win_stay", "lose_switch")
        Which metrics to plot. Any subset of:
        ['stay', 'switch', 'win_stay', 'lose_switch'].

    window : int | tuple[int, int], default (-2, 2)
        Inclusive range of integer offsets around opto anchors to evaluate.
        - If an int n is provided → offsets = [-n, ..., -1, 0, 1, ..., n].
        - If a (lo, hi) tuple is provided → offsets = [lo, ..., hi].
        Offset 0 is always included and represents anchor opto trials.

    criteria : dict | None, default None
        Additional filters to **select opto anchors**. Keys are column names; values may be:
          - None → require NA/None in that column,
          - list/tuple/set → require value ∈ given set (via .isin),
          - anything else → require equality (==).
        If `None`, all opto trials are used as anchors.

    session_col : str, default "session"
        Column name for session identifier. Trials are grouped and offset-calculated within sessions.

    subject_col : str, default "subject_id"
        Column name for subject (mouse) identifier. Sessions from the same subject share a color.

    laser_col : str, default "laser_on_trial"
        Column indicating opto vs non-opto. May be 1/0, '1'/'0', True/False, 'true'/'false', etc.
        Internally normalized to boolean.

    time_col : str, default "trial_num"
        Column defining within-session order. If missing, the function falls back to 'start_time'
        (if present), else to the existing row order. The function assigns a per-session integer index
        '_trial_idx' and computes offsets on that index.

    share_y : bool, default True
        If True, all figures use y-limits [0, 1] to keep rate scale consistent.

    return_table : bool, default False
        If True, also returns the tidy summary table used to make the plots.

    figsize : tuple[float, float], default (6.0, 4.5)
        Figure size for each metric plot.

    Returns
    -------
    figs : dict[str, matplotlib.figure.Figure]
        Mapping from metric name to its matplotlib Figure.

    summary_df : pandas.DataFrame, optional
        Only returned when `return_table=True`. Tidy table with columns:
          ['session', 'subject_id', 'metric', 'offset', 'rate', 'n_trials']
        where 'rate' is the fraction of True values among valid trials at that offset and
        'n_trials' is the number of non-NA metric entries used at that point.

    Raises
    ------
    ValueError
        If required columns are missing or if no data can be plotted (e.g., no anchors found).

    Notes
    -----
    - Metric columns are robustly coerced to boolean with NA support, so strings like "False" will
      be parsed as False (not truthy).
    - When multiple anchors map to the same control trial at a given offset, that control trial is
      deduplicated (counted once).
    """

    df = combined_dataframe.copy()

    # ---------- validation ----------
    base_needed = {session_col, subject_col, laser_col}
    missing = sorted(base_needed - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    valid_metrics = {"stay", "switch", "win_stay", "lose_switch"}
    vis_types = list(vis_types)
    bad = sorted(set(vis_types) - valid_metrics)
    if bad:
        raise ValueError(f"Unknown vis_types: {bad}. Valid: {sorted(valid_metrics)}")
    miss_metrics = [m for m in vis_types if m not in df.columns]
    if miss_metrics:
        raise ValueError(f"Missing metric columns: {miss_metrics}")

    # ---------- helpers: robust coercions ----------
    def _as_opto_bool(s: pd.Series) -> pd.Series:
        """Coerce laser flag to boolean. Accepts 1/0, '1'/'0', True/False, 'true'/'false', 'yes'/'no', 'on'/'off'."""
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
        """Coerce metric column to pandas 'boolean' dtype (with NA)."""
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

    # sort within session by time_col (or 'start_time' or row order)
    order_cols = [session_col]
    if time_col in df.columns:
        order_cols.append(time_col)
    elif "start_time" in df.columns:
        order_cols.append("start_time")
    df = df.sort_values(order_cols).reset_index(drop=True)
    df["_trial_idx"] = df.groupby(session_col).cumcount()

    # ensure metric columns are boolean (with NA)
    for m in vis_types:
        df[m] = _as_bool_series(df[m])

    # ---------- offsets ----------
    if isinstance(window, int):
        lo, hi = -abs(window), abs(window)
    else:
        lo, hi = window
        if lo > hi:
            lo, hi = hi, lo
    offsets = list(range(lo, hi + 1))  # includes 0

    # ---------- compute rates ----------
    sessions = df[session_col].dropna().astype(str).unique().tolist()
    rows = []

    # criteria → opto anchor filter
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

    for sess in sessions:
        g = df[df[session_col] == sess].copy()
        if g.empty:
            continue

        subj = str(g[subject_col].iloc[0])
        n_trials = len(g)

        # anchors: all opto trials if criteria None; otherwise filtered opto
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
                    # control must be non-opto; dedupe by trial index
                    cand_df = hit[~hit["_is_opto"]].drop_duplicates(subset=["_trial_idx"])

            for metric in vis_types:
                vals = cand_df[metric].dropna()
                n_used = int(vals.shape[0])
                rate = float(vals.mean()) if n_used > 0 else np.nan  # True=1, False=0
                rows.append({
                    "session": sess,
                    "subject_id": subj,
                    "metric": metric,
                    "offset": off,
                    "rate": rate,
                    "n_trials": n_used,
                })

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise ValueError("No data to plot. Check your window/criteria and that opto trials exist.")

    # ---------- plot ----------
    # Color by subject (consistent across figures)
    subjects = sorted(summary_df["subject_id"].dropna().astype(str).unique().tolist())
    cmap = plt.get_cmap("tab20", max(1, len(subjects)))
    subject_to_color = {s: cmap(i % cmap.N) for i, s in enumerate(subjects)}

    all_sessions = sorted(summary_df["session"].dropna().astype(str).unique().tolist())
    xticks = sorted(summary_df["offset"].dropna().unique().tolist())

    figs = {}
    for metric in vis_types:
        sub = summary_df[summary_df["metric"] == metric]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=figsize)

        # one line per session
        for sess in all_sessions:
            s_df = sub[sub["session"] == sess].sort_values("offset")
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

        ax.set_title(f"{metric.replace('_',' ').title()} rate vs. offset")
        ax.set_xlabel("Offset (trials) relative to opto anchors (0 = opto)")
        ax.set_ylabel("Rate")
        if share_y:
            ax.set_ylim(0.0, 1.0)
        ax.set_xticks(xticks)
        ax.grid(True, axis="y", alpha=0.25)

        # legend by subject
        from matplotlib.lines import Line2D
        present_subjects = sorted(sub["subject_id"].dropna().astype(str).unique())
        handles = [Line2D([0], [0], color=subject_to_color[s], lw=2, label=s) for s in present_subjects]
        if handles:
            ax.legend(handles=handles, title="Subject", bbox_to_anchor=(1.02, 1), loc="upper left")

        figs[metric] = fig

    return (figs, summary_df) if return_table else figs
