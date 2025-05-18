import os, re
from itertools import product
from typing import List, Optional, Tuple, Union, Any, Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from general_utils import smart_read_csv
from behavior_utils   import extract_fitted_data           

_PVAL_SUFFIX = "_pval"   # suffix produced by significance_and_direction_summary_combined


# ---------------------------------------------------------------------
# Helper ── map {model → column_name} for a given variable
# ---------------------------------------------------------------------
def _model_cols(df: pd.DataFrame, variable: str) -> dict[str, str]:
    patt = re.compile(rf"^(.*?)\-{re.escape(variable)}{_PVAL_SUFFIX}$")
    return {m.group(1): col for col in df.columns if (m := patt.match(col))}


# ---------------------------------------------------------------------
# Helper ── list all behavioural variables present in df
# ---------------------------------------------------------------------
def _all_variables(df: pd.DataFrame) -> List[str]:
    patt = re.compile(r"^[^-]+?-(.+)" + re.escape(_PVAL_SUFFIX) + r"$")
    return sorted({m.group(1) for c in df.columns if (m := patt.match(c))})


# ---------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------
def plot_unitwise_pvalues_across_models(
    data: Union[str, os.PathLike, pd.DataFrame],
    variable: Optional[str] = None,
    z_scores: Optional[List[bool]] = None,
    time_windows: Optional[List[str]] = None,
    fdr_line: float = 0.05,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 120,
    lw: float = 1.0,
    alpha: float = 0.7
) -> None:
    """
    Visualise per-unit p-values across multiple correlation models.

    Typical workflow
    ----------------
    1. Run your correlation pipeline to get one or more
       ``correlations-*.csv`` files.

    2. Combine them with
           ``ephys_behavior.significance_and_direction_summary_combined(...)``
       → e.g. ``/root/capsule/results/sig_dir_all_sessions.csv``.

    3. Pass that CSV (or an already-loaded DataFrame) to
       ``plot_unitwise_pvalues_across_models``.

    What the plot shows
    -------------------
    • **One poly-line per unit** (session_id × unit_index) connecting its
      p-values across all models.

    • Panels are split by every ``time_window × z_score`` combination.

    • If *variable* is *None*, the function loops over **all** variables that
      appear in “*-<variable>_pval” columns and draws one figure per variable.

    Parameters
    ----------
    data : str | os.PathLike | pd.DataFrame
        Path to the CSV produced by
        ``ephys_behavior.significance_and_direction_summary_combined`` **or**
        the DataFrame itself (already in memory).  The CSV is read with
        ``smart_read_csv`` to preserve list/dict columns.
    variable : str | None
        Behaviour variable part of the column name.  ``None`` → auto-detect all.
    z_scores : list[bool] | None
        Which ``z_score`` values to plot.  ``None`` → auto-detect.
    time_windows : list[str] | None
        Which ``time_window`` labels to plot.  ``None`` → auto-detect.
    fdr_line : float
        Horizontal reference line (default 0.05).
    figsize, dpi : figure size / resolution.
    lw, alpha : line width and transparency of unit traces.
    """
    # 1) Load DataFrame -------------------------------------------------
    df = smart_read_csv(data) if isinstance(data, (str, os.PathLike)) else data.copy()

    # 2) Decide which variable(s) to plot ------------------------------
    variables = [variable] if variable else _all_variables(df)
    if not variables:
        raise ValueError("No '*_pval' columns found in supplied data.")

    # 3) Loop through variables, one figure each -----------------------
    for var in variables:
        mdl_cols = _model_cols(df, var)
        if not mdl_cols:
            print(f"[skip] no columns for variable '{var}'")
            continue
        models = sorted(mdl_cols)

        # Panel grid axes
        zs  = z_scores  or sorted(df["z_score"].unique())
        tws = time_windows or sorted(df["time_window"].unique())
        n_r, n_c = len(zs), len(tws)

        fig, axes = plt.subplots(
            n_r, n_c, figsize=figsize, dpi=dpi,
            sharex=True, sharey=True, squeeze=False
        )
        x_pos = range(len(models))  # categorical x coordinates

        # 4) Draw each (z_score, time_window) panel --------------------
        for (ir, z_val), (ic, tw) in product(enumerate(zs), enumerate(tws)):
            ax = axes[ir, ic]
            panel_df = df[(df["z_score"] == z_val) & (df["time_window"] == tw)]
            if panel_df.empty:
                ax.set_visible(False)
                continue

            grouped = panel_df.groupby(["session_id", "unit_index"])
            for (_, _), grp in grouped:
                y = [grp.iloc[0][mdl_cols[m]] for m in models]
                ax.plot(x_pos, y, lw=lw, alpha=alpha)

            n_units = grouped.ngroups
            ax.set_xticks(x_pos, labels=models, rotation=45, ha="right")
            ax.set_yscale("log")
            ax.axhline(fdr_line, color="red", ls="--", lw=1)
            ax.set_title(f"{var}\nwindow={tw}, z={z_val}  (n={n_units} units)")
            ax.grid(True, which="both", ls=":", alpha=0.4)
            if ic == 0:
                ax.set_ylabel("p-value (log scale)")
            if ir == n_r - 1:
                ax.set_xlabel("model")

        plt.tight_layout()
        plt.show()


