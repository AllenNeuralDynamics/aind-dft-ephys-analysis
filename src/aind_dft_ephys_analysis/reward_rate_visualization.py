from typing import List, Union
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
