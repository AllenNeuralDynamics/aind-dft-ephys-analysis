from typing import List, Union
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ephys_behavior import find_fit_keys_and_constructed_vars



def plot_constructed_grid_wrap(
    df: pd.DataFrame,
    constructed_vars: List[str],
    metric_types: Union[str, List[str]],
    *,
    bins: int = 50,
    dropna: bool = True,
    max_cols: int = 10,
) -> pd.DataFrame:
    """
    Plot distributions for all constructed_vars, wrapping panels so that each
    row has at most `max_cols` columns.

    For each metric_type, a separate figure is created.

    Column naming rule:
        col = f"{constructed_var}-{metric_type}"

    Parameters
    ----------
    df
        Input DataFrame.
    constructed_vars
        List of constructed variable prefixes to plot (all will be plotted).
    metric_types
        A metric type (str) or list of metric types (each metric makes one figure).
    bins
        Histogram bins.
    dropna
        Drop NaN/inf before plotting and stats.
    max_cols
        Maximum number of panels per row (wrapping width). Must be >= 1.

    Returns
    -------
    summary : pd.DataFrame
        One row per (metric_type, constructed_var) with n/mean/median and column name.
    """
    if isinstance(metric_types, str):
        metric_types = [metric_types]

    if max_cols < 1:
        raise ValueError("max_cols must be >= 1.")

    rows = []
    n_total = len(constructed_vars)
    n_rows = int(math.ceil(n_total / max_cols))
    n_cols = int(max_cols)

    for metric in metric_types:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            squeeze=False,
        )

        for idx, cv in enumerate(constructed_vars):
            r = idx // n_cols
            c = idx % n_cols
            ax = axes[r, c]

            col = f"{cv}-{metric}"

            if col not in df.columns:
                ax.set_title("missing", fontsize=9)
                ax.axis("off")
                rows.append(
                    {
                        "metric_type": metric,
                        "constructed_var": cv,
                        "column": col,
                        "n": 0,
                        "mean": np.nan,
                        "median": np.nan,
                    }
                )
                continue

            vals = df[col].to_numpy()
            if dropna:
                vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                ax.set_title("empty", fontsize=9)
                ax.axis("off")
                rows.append(
                    {
                        "metric_type": metric,
                        "constructed_var": cv,
                        "column": col,
                        "n": 0,
                        "mean": np.nan,
                        "median": np.nan,
                    }
                )
                continue

            mean_val = float(np.mean(vals))
            median_val = float(np.median(vals))

            ax.hist(vals, bins=bins)
            ax.axvline(mean_val, linestyle="--", linewidth=1)
            ax.axvline(median_val, linestyle=":", linewidth=1)

            ax.set_title(cv, fontsize=8)

            rows.append(
                {
                    "metric_type": metric,
                    "constructed_var": cv,
                    "column": col,
                    "n": int(vals.size),
                    "mean": mean_val,
                    "median": median_val,
                }
            )

        # Turn off any unused axes in the grid
        total_axes = n_rows * n_cols
        for k in range(n_total, total_axes):
            r = k // n_cols
            c = k % n_cols
            axes[r, c].axis("off")

        fig.suptitle(f"Distribution grid: {metric}", y=1.02)
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(rows)


import re
import numpy as np
import pandas as pd


# ----------------------------
# Parsing helpers
# ----------------------------
def _parse_window(constructed_var: str) -> float:
    """Extract integer window size from a constructed_var. Return NaN if not found."""
    m = re.search(r"reward_rate_window_(\d+)", constructed_var)
    return float(m.group(1)) if m else np.nan


def _parse_alpha(constructed_var: str) -> float:
    """Extract alpha (float) from a constructed_var. Return NaN if not found."""
    m = re.search(r"reward_rate_alpha_([0-9]*\.?[0-9]+)", constructed_var)
    return float(m.group(1)) if m else np.nan


def _safe_values_1d(x: np.ndarray) -> np.ndarray:
    """Convert to float array and keep finite values."""
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


# ----------------------------
# Core per-pattern selector
# ----------------------------
def best_param_by_pattern_per_row(
    df: pd.DataFrame,
    pattern: str,
    *,
    param_kind: str,           # "window" or "alpha"
    metric_suffix: str = "rsq" # e.g. "rsq"
) -> pd.DataFrame:
    """
    For each row, select the candidate (from find_fit_keys_and_constructed_vars)
    with the highest {key}-{metric_suffix}. Ties are broken by smallest parameter
    (window or alpha). Also extract best pval via {best_constructed_var}-pval.

    Returns a DataFrame with:
      - best_key
      - best_constructed_var
      - best_param (window size or alpha)
      - best_metric (best rsq)
      - best_pval
    """
    if param_kind not in {"window", "alpha"}:
        raise ValueError("param_kind must be 'window' or 'alpha'")

    keys, constructed_vars = find_fit_keys_and_constructed_vars(df, pattern)
    if len(keys) == 0:
        raise ValueError(f"No matches for pattern: {pattern!r}")

    cand = pd.DataFrame({"key": keys, "constructed_var": constructed_vars})

    if param_kind == "window":
        cand["param"] = cand["constructed_var"].apply(_parse_window)
    else:
        cand["param"] = cand["constructed_var"].apply(_parse_alpha)

    # Score column is per key (NOT per constructed_var)
    cand["score_col"] = cand["key"].map(lambda k: f"{k}-{metric_suffix}")

    # Keep only candidates whose score_col exists
    cand = cand[cand["score_col"].isin(df.columns)].reset_index(drop=True)
    if cand.empty:
        example = f"{keys[0]}-{metric_suffix}"
        raise ValueError(f"No score columns found. Example expected: {example}")

    # Build score matrix: rows=neurons, cols=candidates
    score_mat = df[cand["score_col"].tolist()].to_numpy()
    score_mat = np.where(np.isfinite(score_mat), score_mat, -np.inf)

    # Best by max score
    best_idx = np.argmax(score_mat, axis=1)
    best_score = score_mat[np.arange(score_mat.shape[0]), best_idx]

    # Tie-break: smallest param
    params = cand["param"].to_numpy()
    params_for_ties = np.where(np.isfinite(params), params, np.inf)
    tie_mask = score_mat == best_score[:, None]
    tie_params = np.where(tie_mask, params_for_ties[None, :], np.inf)
    best_idx_tie = np.argmin(tie_params, axis=1)

    is_tie = tie_mask.sum(axis=1) > 1
    best_idx_final = np.where(is_tie, best_idx_tie, best_idx)

    best_key = cand.loc[best_idx_final, "key"].to_numpy()
    best_constructed = cand.loc[best_idx_final, "constructed_var"].to_numpy()
    best_param = cand.loc[best_idx_final, "param"].to_numpy()
    best_metric = score_mat[np.arange(score_mat.shape[0]), best_idx_final]

    # Extract corresponding pval from per-constructed-var column: "{constructed}-pval"
    best_pval = np.full(df.shape[0], np.nan, dtype=float)
    for i, cv in enumerate(best_constructed):
        pcol = f"{cv}-pval"
        if pcol in df.columns:
            val = df.at[df.index[i], pcol]
            try:
                best_pval[i] = float(val)
            except Exception:
                best_pval[i] = np.nan

    return pd.DataFrame(
        {
            "best_key": best_key,
            "best_constructed_var": best_constructed,
            "best_param": best_param,
            f"best_{metric_suffix}": best_metric,
            "best_pval": best_pval,
        },
        index=df.index,
    )


# ----------------------------
# Wrapper: compute all 6 patterns and append columns
# ----------------------------
def append_best_rewardrate_params(
    df: pd.DataFrame,
    *,
    metric_suffix: str = "rsq",
    prefix: str = "best",
) -> pd.DataFrame:
    """
    Compute and append:
      - best window for running_experienced / left_reward / right_reward
      - best alpha  for ewma_experienced / ewma_left_reward / ewma_right_reward

    Appends new columns to a COPY of df and returns it.
    """
    out = df.copy()

    # Window patterns (your "model-..." should match inside "no_model-..." due to your matcher)
    window_patterns = {
        "window_experienced": "model-reward_rate_window*running_experienced",
        "window_left_reward": "model-reward_rate_window*running_left_reward",
        "window_right_reward": "model-reward_rate_window*running_right_reward",
    }

    # Alpha patterns
    alpha_patterns = {
        "alpha_experienced": "no_model-reward_rate_alpha_*-ewma_experienced",
        "alpha_left_reward": "no_model-reward_rate_alpha_*-ewma_left_reward",
        "alpha_right_reward": "no_model-reward_rate_alpha_*-ewma_right_reward",
    }

    # Compute windows
    for tag, pat in window_patterns.items():
        res = best_param_by_pattern_per_row(
            out, pat, param_kind="window", metric_suffix=metric_suffix
        )
        out[f"{prefix}_{tag}"] = res["best_param"]
        out[f"{prefix}_{tag}_{metric_suffix}"] = res[f"best_{metric_suffix}"]
        out[f"{prefix}_{tag}_pval"] = res["best_pval"]
        out[f"{prefix}_{tag}_constructed"] = res["best_constructed_var"]
        out[f"{prefix}_{tag}_key"] = res["best_key"]

    # Compute alphas
    for tag, pat in alpha_patterns.items():
        res = best_param_by_pattern_per_row(
            out, pat, param_kind="alpha", metric_suffix=metric_suffix
        )
        out[f"{prefix}_{tag}"] = res["best_param"]
        out[f"{prefix}_{tag}_{metric_suffix}"] = res[f"best_{metric_suffix}"]
        out[f"{prefix}_{tag}_pval"] = res["best_pval"]
        out[f"{prefix}_{tag}_constructed"] = res["best_constructed_var"]
        out[f"{prefix}_{tag}_key"] = res["best_key"]

    return out

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def visualize_best_params_significant(
    ds2: pd.DataFrame,
    *,
    p_thresh: float = 0.05,
    alpha_cols: List[str] = None,
    alpha_pval_cols: List[str] = None,
    window_cols: List[str] = None,
    window_pval_cols: List[str] = None,
    bins: int = 40,
) -> pd.DataFrame:
    """
    Visualize distributions of best alpha/window parameters for significant neurons.
    Titles include both significant count and total neuron count.
    """
    if alpha_cols is None:
        alpha_cols = [
            "best_alpha_experienced",
            "best_alpha_left_reward",
            "best_alpha_right_reward",
        ]
    if alpha_pval_cols is None:
        alpha_pval_cols = [c + "_pval" for c in alpha_cols]

    if window_cols is None:
        window_cols = [
            "best_window_experienced",
            "best_window_left_reward",
            "best_window_right_reward",
        ]
    if window_pval_cols is None:
        window_pval_cols = [c + "_pval" for c in window_cols]

    needed = alpha_cols + alpha_pval_cols + window_cols + window_pval_cols
    missing = [c for c in needed if c not in ds2.columns]
    if missing:
        raise ValueError(f"Missing columns in ds2: {missing[:10]}")

    n_total = ds2.shape[0]   # ⭐ total number of neurons

    summary_rows = []

    # -----------------------------
    # Alpha distributions
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    axes = axes[0]

    for i, (col, pcol) in enumerate(zip(alpha_cols, alpha_pval_cols)):
        sig_mask = ds2[pcol].to_numpy(dtype=float) < p_thresh
        vals = ds2.loc[sig_mask, col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        n_sig = int(vals.size)
        mean_val = float(np.mean(vals)) if n_sig > 0 else np.nan
        median_val = float(np.median(vals)) if n_sig > 0 else np.nan

        ax = axes[i]
        if n_sig == 0:
            ax.axis("off")
            ax.set_title(f"{col}\n(sig n=0 / total n={n_total})")
        else:
            ax.hist(vals, bins=bins)
            ax.axvline(mean_val, linestyle="--", linewidth=1)
            ax.axvline(median_val, linestyle=":", linewidth=1)
            ax.set_title(
                f"{col}\n(sig n={n_sig} / total n={n_total})"
            )
            ax.set_xlabel("alpha")
            ax.set_ylabel("count")

        summary_rows.append(
            {
                "param": col,
                "pval_col": pcol,
                "p_thresh": p_thresh,
                "n_sig": n_sig,
                "n_total": n_total,
                "mean": mean_val,
                "median": median_val,
            }
        )

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Window distributions
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    axes = axes[0]

    for i, (col, pcol) in enumerate(zip(window_cols, window_pval_cols)):
        sig_mask = ds2[pcol].to_numpy(dtype=float) < p_thresh
        vals = ds2.loc[sig_mask, col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        n_sig = int(vals.size)
        mean_val = float(np.mean(vals)) if n_sig > 0 else np.nan
        median_val = float(np.median(vals)) if n_sig > 0 else np.nan

        ax = axes[i]
        if n_sig == 0:
            ax.axis("off")
            ax.set_title(f"{col}\n(sig n=0 / total n={n_total})")
        else:
            ax.hist(vals, bins=bins)
            ax.axvline(mean_val, linestyle="--", linewidth=1)
            ax.axvline(median_val, linestyle=":", linewidth=1)
            ax.set_title(
                f"{col}\n(sig n={n_sig} / total n={n_total})"
            )
            ax.set_xlabel("window size")
            ax.set_ylabel("count")

        summary_rows.append(
            {
                "param": col,
                "pval_col": pcol,
                "p_thresh": p_thresh,
                "n_sig": n_sig,
                "n_total": n_total,
                "mean": mean_val,
                "median": median_val,
            }
        )

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(summary_rows)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


def scatter_with_linear_fit(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    title: Optional[str] = None,
    max_points: Optional[int] = 20000,
):
    """
    Scatter plot of x_col vs y_col with a linear fit line.

    Parameters
    ----------
    df : pd.DataFrame
    x_col, y_col : str
        Column names in df.
    title : str, optional
    max_points : int, optional
        If set, subsample to at most this many points for speed/legibility.
    """
    if x_col not in df.columns:
        raise ValueError(f"Missing column: {x_col}")
    if y_col not in df.columns:
        raise ValueError(f"Missing column: {y_col}")

    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = x.size
    if n == 0:
        raise ValueError(f"No finite points for {x_col} vs {y_col}")

    # Optional subsampling (keeps it responsive)
    if max_points is not None and n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
        n = x.size

    # Linear fit: y = a*x + b
    a, b = np.polyfit(x, y, 1)

    # R^2 for the fit (on plotted points)
    y_hat = a * x + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    # Plot
    plt.scatter(x, y, s=6, alpha=0.4)
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    plt.plot(x_line, a * x_line + b)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title or f"{x_col} vs {y_col}\n(n={n}, slope={a:.3g}, R²={r2:.3g})")



from typing import Literal, List

def plot_bestparam_relationships(
    ds2: pd.DataFrame,
    *,
    y: Literal["pval", "rsq"] = "pval",
    pval_transform: Optional[Literal["none", "neglog10"]] = "none",
    max_points: Optional[int] = 20000,
):
    """
    Plot scatter + linear fit for:
      - best_window_experienced / left_reward / right_reward
      - best_alpha_experienced / left_reward / right_reward

    y can be:
      - "pval"  -> uses <param>_pval
      - "rsq"   -> uses <param>_rsq

    Optional:
      - pval_transform="neglog10" plots -log10(pval) instead of raw pval.
    """
    window_bases = [
        "best_window_experienced",
        "best_window_left_reward",
        "best_window_right_reward",
    ]
    alpha_bases = [
        "best_alpha_experienced",
        "best_alpha_left_reward",
        "best_alpha_right_reward",
    ]

    bases = window_bases + alpha_bases

    # Build a working copy if we transform pvals
    df = ds2
    if y == "pval" and pval_transform == "neglog10":
        df = ds2.copy()
        for base in bases:
            pcol = f"{base}_pval"
            if pcol in df.columns:
                p = pd.to_numeric(df[pcol], errors="coerce")
                df[pcol] = -np.log10(p)

    # Plot grid: 2 rows x 3 cols (windows on top, alphas on bottom)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)

    for i, base in enumerate(window_bases):
        ax = axes[0, i]
        plt.sca(ax)
        y_col = f"{base}_{y}"
        x_col = base
        title = f"{x_col} vs {y_col}"
        scatter_with_linear_fit(df, x_col, y_col, title=title, max_points=max_points)

    for i, base in enumerate(alpha_bases):
        ax = axes[1, i]
        plt.sca(ax)
        y_col = f"{base}_{y}"
        x_col = base
        title = f"{x_col} vs {y_col}"
        scatter_with_linear_fit(df, x_col, y_col, title=title, max_points=max_points)

    if y == "pval" and pval_transform == "neglog10":
        fig.suptitle("Relationships (y = -log10(pval))", y=1.02)
    else:
        fig.suptitle(f"Relationships (y = {y})", y=1.02)

    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Optional


def boxplot_param_vs_metric_grid(
    ds2: pd.DataFrame,
    *,
    y: Literal["rsq", "pval"] = "rsq",
    sig_only: bool = False,
    p_thresh: float = 0.05,
    show_fliers: bool = False,
    max_groups: Optional[int] = None,
):
    """
    Make box plots with:
      x = best_{window/alpha}_* (discrete groups)
      y = best_{window/alpha}_*_{y} (rsq or pval)

    Panels (2x3):
      windows: experienced / left_reward / right_reward
      alphas : experienced / left_reward / right_reward

    If sig_only=True, rows are filtered by corresponding *_pval < p_thresh.

    Parameters
    ----------
    ds2 : pd.DataFrame
    y : "rsq" or "pval"
    sig_only : bool
    p_thresh : float
    show_fliers : bool
        Whether to show outliers in boxplot.
    max_groups : int, optional
        If set, keep at most this many unique x-groups (largest groups by count).
        Useful if alpha has many distinct values.
    """
    window_bases = [
        "best_window_experienced",
        "best_window_left_reward",
        "best_window_right_reward",
    ]
    alpha_bases = [
        "best_alpha_experienced",
        "best_alpha_left_reward",
        "best_alpha_right_reward",
    ]

    n_total = ds2.shape[0]

    def _panel(ax, base: str):
        x_col = base
        y_col = f"{base}_{y}"
        p_col = f"{base}_pval"

        for c in (x_col, y_col, p_col):
            if c not in ds2.columns:
                raise ValueError(f"Missing column: {c}")

        x = pd.to_numeric(ds2[x_col], errors="coerce")
        yy = pd.to_numeric(ds2[y_col], errors="coerce")
        pp = pd.to_numeric(ds2[p_col], errors="coerce")

        mask = x.notna() & yy.notna()
        if sig_only:
            mask = mask & pp.notna() & (pp < p_thresh)

        sub = pd.DataFrame({"x": x[mask], "y": yy[mask]})
        n_used = sub.shape[0]

        if n_used == 0:
            ax.axis("off")
            ax.set_title(f"{base}\n(no data after filtering)")
            return

        # Group by x and collect y arrays
        grp = sub.groupby("x")["y"].apply(lambda s: s.to_numpy())

        # Optionally reduce number of groups (by largest counts)
        if max_groups is not None and grp.shape[0] > max_groups:
            sizes = grp.apply(len).sort_values(ascending=False)
            keep = sizes.index[:max_groups]
            grp = grp.loc[keep].sort_index()

        # Sort groups by x (numeric)
        grp = grp.sort_index()

        data = list(grp.values)
        labels = [str(int(k)) if float(k).is_integer() else str(k) for k in grp.index]

        ax.boxplot(data, labels=labels, showfliers=show_fliers)
        ax.set_xlabel(base)
        ax.set_ylabel(f"{base}_{y}")

        title = f"{base}: x={base}, y={base}_{y}\n(n={n_used}/{n_total}"
        if sig_only:
            title += f", p<{p_thresh}"
        title += ")"
        ax.set_title(title, fontsize=10)

        # Rotate tick labels if too many
        if len(labels) > 10:
            ax.tick_params(axis="x", labelrotation=45)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), squeeze=False)

    # Windows (top row)
    for i, base in enumerate(window_bases):
        _panel(axes[0, i], base)

    # Alphas (bottom row)
    for i, base in enumerate(alpha_bases):
        _panel(axes[1, i], base)

    plt.tight_layout()
    plt.show()
