from __future__ import annotations
import os, re
import json
import itertools
from itertools import product
from typing import List, Optional, Tuple, Union, Any, Dict, Sequence
import math
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.decomposition import PCA

from general_utils import smart_read_csv
from behavior_utils   import extract_fitted_data       
from behavior_utils import extract_event_timestamps   # your helper that returns per-trial timestamps
from ephys_behavior import get_units_passed_default_qc

def plot_raster_graph(
    nwb_data: Any,
    unit_index: int,
    align_to_event: str = 'go_cue',
    time_window: List[float] = [-2, 3],
    bin_size: float = 0.05,
    fitted_data: Optional[np.ndarray] = None,
    latent_name: Optional[str] = None,
    exclude_trials: Optional[List[int]] = None,
    save_figure: bool = False,
    save_format: str = 'eps',
    save_folder: str = '/root/capsule/results',
    figure_name_prefix: Optional[str] = None,
) -> None:
    """
    Plot spike raster and PSTH for a single unit from a combined NWB file.

    This function takes a combined NWB object (containing both behavior trials and ephys unit data),
    extracts spike times for the specified unit, aligns them to a behavioral event for each trial,
    constructs a raster plot and corresponding peri-stimulus time histogram (PSTH),
    and optionally saves the figure.

    Parameters
    ----------
    nwb_data : NWB file handle
        Combined NWB object containing:
        - `trials` table or times as used by `extract_event_timestamps`.
        - `units` DynamicTable with a `spike_times` column.
    unit_index : int
        Index of the unit to plot (row in `nwb_data.units`).
    align_to_event : str, optional
        The event name to align spikes to (default 'go_cue'). Passed to `extract_event_timestamps`.
    time_window : list of two floats, optional
        [start, end] times (seconds) relative to event for alignment (default [-2, 3]).
    bin_size : float, optional
        Bin width (seconds) for PSTH histogram (default 0.05).
    fitted_data : 1D numpy.ndarray, optional
        Trial-specific values (length equal to number of trials) to sort the raster by;
        must match number of included trials if provided (default None).
    latent_name : str, optional
        Name of the latent variable in `fitted_data`; used for title annotation if sorting.
    exclude_trials : list of int, optional
        List of trial indices to exclude before sorting and plotting (default None).
    save_figure : bool, optional
        If True, saves the figure to disk (default False).
    save_format : str, optional
        File format for saving figure (e.g., 'eps', 'pdf'; default 'eps').
    save_folder : str, optional
        Directory path to save figures (default '/root/capsule/results').
    figure_name_prefix : str, optional
        Filename prefix for saved figure (omit extension). If None, filename built from unit
        index and sorting information.

    Returns
    -------
    None
        Displays the raster and PSTH plot. If `save_figure` is True, prints the saved file path.
    """
    # Create save directory if needed
    if save_figure:
        os.makedirs(save_folder, exist_ok=True)

    # 1. Retrieve event timestamps per trial
    all_times = np.array(extract_event_timestamps(nwb_data, align_to_event))
    n_trials = len(all_times)

    # 2. Exclude specified trials if any
    trials_idx = np.arange(n_trials)
    if exclude_trials:
        mask_exc = np.ones(n_trials, dtype=bool)
        mask_exc[exclude_trials] = False
        all_times = all_times[mask_exc]
        trials_idx = trials_idx[mask_exc]

    # 3. Validate fitted_data length
    if fitted_data is not None and len(fitted_data) != len(all_times):
        raise ValueError(
            f"After exclusion, fitted_data length ({len(fitted_data)}) does not match number of trials ({len(all_times)})"
        )

    # 4. Sort trials by fitted_data if provided
    if fitted_data is not None:
        sort_idx = np.argsort(fitted_data)
        all_times = all_times[sort_idx]
        trials_idx = trials_idx[sort_idx]

    # 5. Load spike times for the unit
    try:
        unit_spike_times = np.array(nwb_data.units['spike_times'][unit_index])
    except Exception:
        raise ValueError(f"Spike times not found for unit {unit_index}.")

    # 6. Align spikes and collect for PSTH
    spikes_aligned = []  # list of (trial_idx, aligned_spikes)
    all_spikes = []
    for row_idx, t0 in enumerate(all_times):
        start, end = t0 + time_window[0], t0 + time_window[1]
        mask_spk = (unit_spike_times >= start) & (unit_spike_times <= end)
        aligned = unit_spike_times[mask_spk] - t0
        spikes_aligned.append((row_idx, aligned))
        all_spikes.extend(aligned.tolist())

    num_plotted = len(all_times)

    # 7. Plotting
    fig, (ax_raster, ax_psth) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # Raster plot
    for row_idx, spks in spikes_aligned:
        ax_raster.vlines(spks, row_idx + 0.5, row_idx + 1.5, color='black')
    ax_raster.axvline(0, color='red', linestyle='--')
    ax_raster.set_ylabel('Trials')
    title = f'Raster: Unit {unit_index} aligned to {align_to_event}'
    if fitted_data is not None and latent_name:
        title += f' (sorted by {latent_name})'
    ax_raster.set_title(title)
    ax_raster.set_ylim(0.5, num_plotted + 0.5)

    # If sorted by fitted_data, annotate low/high positions
    if fitted_data is not None:
        ax_raster.text(
            time_window[1], 1, 'Low', va='center', ha='right', color='blue'
        )
        ax_raster.text(
            time_window[1], num_plotted, 'High', va='center', ha='right', color='blue'
        )

    # PSTH plot
    bins = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
    counts, _ = np.histogram(all_spikes, bins=bins)
    rates = counts / (num_plotted * bin_size)
    ax_psth.bar(bins[:-1], rates, width=bin_size, align='edge', color='black')
    ax_psth.axvline(0, color='red', linestyle='--')
    ax_psth.set_xlabel('Time from event (s)')
    ax_psth.set_ylabel('Firing rate (Hz)')
    ax_psth.set_title('PSTH')

    plt.tight_layout()

    # Save if requested
    if save_figure:
        # Build base filename
        parts = []
        if figure_name_prefix:
            parts.append(figure_name_prefix)
        parts.append(f'unit_{unit_index}')
        if fitted_data is not None and latent_name:
            parts.append(f'sorted_by_{latent_name}')
        filename = "_".join(parts) + f'.{save_format}'
        filepath = os.path.join(save_folder, filename)
        fig.savefig(filepath, format=save_format)
        print(f'Figure saved as {filepath}')

    plt.show()



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


def plot_violin_pvalues_across_models(
    data: Union[str, os.PathLike, pd.DataFrame],
    variable: Optional[str] = None,
    z_scores: Optional[List[bool]] = None,
    time_windows: Optional[List[str]] = None,
    skip_incomplete: bool = False,
    figsize: Tuple[int, int] = (8, 5),
    dpi: int = 120,
    violin_kwargs: Optional[Dict] = None
) -> None:
    """
    Create violin plots showing the full p-value distribution for each model.

    Panels are organized by combinations of `z_score` (rows) and `time_window` (columns).
    Each violin represents one model.

    Parameters
    ----------
    data : str | os.PathLike | pd.DataFrame
        CSV file path or DataFrame containing columns:
        - 'session_id', 'unit_index', 'z_score', 'time_window'
        - one or more '<model>-<variable>_pval' columns.
        When `data` is a path, it is read via `smart_read_csv` to preserve list-like columns.
    variable : str | None
        Name of the behavioural variable (suffix of p-value columns).
        If None, auto-detects all such variables and generates one figure per variable.
    z_scores : list[bool] | None
        Subset of z_score values to include. Defaults to all present in data.
    time_windows : list[str] | None
        Subset of time_window labels to include. Defaults to all present.
    skip_incomplete : bool
        If True, drop any unit whose p-values for any model are missing or non-numeric.
    figsize : tuple of (width, height)
        Figure size for each variable's set of panels.
    dpi : int
        Resolution (dots per inch) of the figure.
    violin_kwargs : dict | None
        Extra keyword arguments forwarded to `ax.violinplot`, e.g., widths.
    """
    # Load and filter data
    df = smart_read_csv(data) if isinstance(data, (str, os.PathLike)) else data.copy()
    variables = [variable] if variable else _all_variables(df)
    if not variables:
        raise ValueError("No p-value columns found.")

    for var in variables:
        # Identify model-specific p-value columns
        mdl_cols = _model_cols(df, var)
        if not mdl_cols:
            print(f"[skip] variable '{var}' has no p-value columns.")
            continue
        models = sorted(mdl_cols)

        # Determine panel layout
        zs = z_scores or sorted(df['z_score'].unique())
        tws = time_windows or sorted(df['time_window'].unique())
        n_rows, n_cols = len(zs), len(tws)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=figsize, dpi=dpi,
            sharex=False, sharey=False,
            squeeze=False
        )
        violin_kwargs = dict(violin_kwargs or {})

        # Draw each panel
        for i_row, z_val in enumerate(zs):
            for i_col, tw in enumerate(tws):
                ax = axes[i_row][i_col]
                # Subset by z_score and time_window
                panel = df[(df['z_score']==z_val) & (df['time_window']==tw)]
                if panel.empty:
                    ax.set_visible(False)
                    continue

                # Collect p-values per model
                data_for_violin = []
                for m in models:
                    vals = []
                    for _, grp in panel.groupby(['session_id', 'unit_index']):
                        raw = grp.iloc[0][mdl_cols[m]]
                        try:
                            val = float(raw)
                            if np.isnan(val):
                                if skip_incomplete:
                                    vals = []
                                    break
                            else:
                                vals.append(val)
                        except:
                            if skip_incomplete:
                                vals = []
                                break
                    data_for_violin.append(vals)

                if not any(data_for_violin):
                    ax.set_visible(False)
                    continue

                # Plot violin
                positions = np.arange(1, len(models)+1)
                parts = ax.violinplot(
                    data_for_violin,
                    positions=positions,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                    **violin_kwargs
                )
                for pc in parts['bodies']:
                    pc.set_alpha(0.6)
                if 'cmedians' in parts:
                    parts['cmedians'].set_linewidth(1.2)

                ax.set_xticks(positions, labels=models, rotation=45, ha='right')
                ax.set_ylim(0, 1)
                if i_col == 0:
                    ax.set_ylabel('p-value')
                if i_row == n_rows - 1:
                    ax.set_xlabel('model')
                ax.set_title(f"{var}\nwindow={tw}, z={z_val}")
                ax.grid(axis='y', linestyle=':', alpha=0.4)

        plt.tight_layout()
        plt.show()


def plot_fraction_significant_lines_by_session(
    data: Union[str, os.PathLike, pd.DataFrame],
    variable: Optional[str] = None,
    z_scores: Optional[List[bool]] = None,
    time_windows: Optional[List[str]] = None,
    alpha_level: float = 0.05,
    skip_incomplete: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 120,
    lw: float = 2.0,
    alpha: float = 0.8,
    session_color: str = 'C0',
    avg_color: str = 'C1',
    session_lw: float = 1.0,
    avg_lw: float = 3.0,
    session_alpha: float = 0.3
) -> None:
    """
    For each session, plot a line connecting the fraction of significant units
    (p ≤ alpha_level) across models.

    Panels are arranged by `z_score` (rows) and `time_window` (columns).

    Parameters
    ----------
    data : str | os.PathLike | pd.DataFrame
        CSV path or DataFrame with:
        - 'session_id', 'unit_index', 'z_score', 'time_window'
        - one or more '<model>-<variable>-pval' columns.
    variable : str | None
        Behaviour variable suffix. None → all detected.
    z_scores : list[bool] | None
        Subset of z_score values. Defaults to all.
    time_windows : list[str] | None
        Subset of time_window labels. Defaults to all.
    alpha_level : float
        Significance threshold (default 0.05).
    skip_incomplete : bool
        If True, exclude units with missing/non-numeric p-values.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
    lw : float
        Line width for session lines.
    alpha : float
        Line transparency for session lines.
    session_color : str
        Matplotlib color for session lines.
    avg_color : str
        Matplotlib color for the average line.
    session_lw : float
        Line width for session lines.
    avg_lw : float
        Line width for the average line.
    session_alpha : float
        Transparency for session lines.
    """
    df = smart_read_csv(data) if isinstance(data, (str, os.PathLike)) else data.copy()
    variables = [variable] if variable else _all_variables(df)
    if not variables:
        raise ValueError("No p-value columns found.")

    for var in variables:
        mdl_cols = _model_cols(df, var)
        if not mdl_cols:
            print(f"[skip] variable '{var}' has no p-value columns.")
            continue
        models = sorted(mdl_cols)
        x = np.arange(len(models))

        zs = z_scores or sorted(df['z_score'].unique())
        tws = time_windows or sorted(df['time_window'].unique())
        n_rows, n_cols = len(zs), len(tws)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=figsize, dpi=dpi,
            sharex=True, sharey=True,
            squeeze=False
        )

        for i_row, z_val in enumerate(zs):
            for i_col, tw in enumerate(tws):
                ax = axes[i_row][i_col]
                panel = df[(df['z_score'] == z_val) & (df['time_window'] == tw)]
                if panel.empty:
                    ax.set_visible(False)
                    continue

                sessions = sorted(panel['session_id'].unique())
                all_fracs = []

                for sess in sessions:
                    sess_df = panel[panel['session_id'] == sess]
                    total_units = 0
                    counts = np.zeros(len(models), dtype=int)

                    for _, grp in sess_df.groupby(['session_id', 'unit_index']):
                        raw_vals = [grp.iloc[0][mdl_cols[m]] for m in models]
                        try:
                            pvals = [float(rv) for rv in raw_vals]
                        except:
                            if skip_incomplete:
                                continue
                            pvals = [
                                np.nan if not str(rv).replace('.', '', 1).isdigit()
                                else float(rv)
                                for rv in raw_vals
                            ]
                        if skip_incomplete and any(np.isnan(p) for p in pvals):
                            continue
                        if any(np.isnan(p) for p in pvals):
                            continue

                        total_units += 1
                        counts += np.array(pvals) <= alpha_level

                    if total_units == 0:
                        continue

                    frac = counts / total_units
                    all_fracs.append(frac)
                    ax.plot(
                        x, frac,
                        lw=session_lw,
                        alpha=session_alpha,
                        color=session_color
                    )

                if all_fracs:
                    mean_frac = np.nanmean(np.vstack(all_fracs), axis=0)
                    ax.plot(
                        x, mean_frac,
                        lw=avg_lw,
                        alpha=1.0,
                        color=avg_color,
                        label='average'
                    )

                ax.set_xticks(x, labels=models, rotation=45, ha='right')
                ax.set_ylim(0, 1)
                ax.axhline(alpha_level, color='red', ls='--', lw=1)
                if i_col == 0:
                    ax.set_ylabel('fraction significant')
                if i_row == n_rows - 1:
                    ax.set_xlabel('model')
                ax.set_title(f"{var}\nwindow={tw}, z={z_val}")
                ax.grid(axis='y', linestyle=':', alpha=0.4)
                ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()



def _load_summary(source: Union[str, Path, pd.DataFrame, xr.Dataset]) -> pd.DataFrame:
    """Return the summary as a *pandas.DataFrame* with JSON cells decoded."""

    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, xr.Dataset):
        df = source.to_dataframe()
    else:  # assume path → use the project's loader that understands CSV & Zarr
        from general_utils import load_temporary_data  # local import avoids hard dep
        df = load_temporary_data(source)

    # Decode JSON columns that xarray stored as strings
    for col in df.columns:
        if df[col].dtype == object and df[col].apply(lambda x: isinstance(x, str) and x[:1] in "[{_").any():
            try:
                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            except Exception:
                pass
    return df

def scatter_summary_multi(
    combined_zarr: Union[str, Path, xr.Dataset, pd.DataFrame],
    compare_variables: Union[str, Sequence[str], Sequence[Sequence[str]]],
    *,
    time_window: Union[str, Sequence[str], None] = None,
    z_score: Union[bool, Sequence[bool], None] = None,
    figsize: Tuple[int, int] | None = None,
    dpi: int = 120,
    alpha: float = 0.7,
    s: float = 2.0,
    kw_refline: Dict[str, Any] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    x_limit: Tuple[float, float] | None = None,
    y_limit: Tuple[float, float] | None = None,
) -> None:
    """Draw **scatter plots** for every pair in *compare_variables*.

    A single figure is created for each unique `(time_window, z_score)`
    combination (after optional filters). **All units** belonging to that
    facet are plotted together.

    Parameters
    ----------
    combined_zarr
        • Path to `.zarr` or `.csv` (auto‑detected)
        • *xarray.Dataset* (already opened)
        • *pandas.DataFrame* (already in memory)
    compare_variables
        Column names you wish to correlate. Accepted forms:

        • "col_A"                     → no scatter (single column)
        • ["col_A", "col_B", ...]     → all pairwise combinations
        • [["col_A", "col_B"], [...]] → each *inner* list forms its own pair‑pool.
    time_window, z_score
        Optional filters. *None* ⇒ include **all** values present.
    figsize, dpi
        Figure size / resolution. If *None* a square grid is chosen based on
        the number of pairs.
    alpha, s
        Transparency and size for scatter points.
    kw_refline
        Extra kwargs for the y = x reference line. Default is grey dashed.
    x_label, y_label
        Optional labels for x and y axes. If None, column names are used.
    x_limit, y_limit
        Optional axis limits as tuples. If None, limits are automatically determined.
    """

    df = _load_summary(combined_zarr)

    if isinstance(compare_variables, str):
        groups: List[List[str]] = [[compare_variables]]
    elif compare_variables and isinstance(compare_variables[0], str):
        groups = [list(compare_variables)]
    else:
        groups = [list(g) for g in compare_variables]

    missing = [c for g in groups for c in g if c not in df.columns]
    if missing:
        raise KeyError(f"Column(s) not found in DataFrame: {missing}")

    def _to_set(v):
        if v is None:
            return None
        return {v} if not isinstance(v, (list, tuple, set)) else set(v)

    tw_filter = _to_set(time_window)
    z_filter = _to_set(z_score)

    tw_values = sorted(df["time_window"].unique()) if tw_filter is None else sorted(tw_filter)
    z_values = sorted(df["z_score"].unique()) if z_filter is None else sorted(z_filter)

    kw_refline = kw_refline or {"lw": 1, "color": "gray", "ls": "--"}

    for tw, zs in itertools.product(tw_values, z_values):
        subset = df[(df["time_window"] == tw) & (df["z_score"] == zs)]
        if subset.empty:
            continue

        for g_idx, g_cols in enumerate(groups):
            pairs = list(itertools.combinations(g_cols, 2))
            if not pairs:
                continue

            n_pairs = len(pairs)
            n_cols = math.ceil(math.sqrt(n_pairs))
            n_rows = math.ceil(n_pairs / n_cols)
            fig_w, fig_h = figsize or (4 * n_cols, 4 * n_rows)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
            axes = np.ravel(axes)

            for ax, (x_col, y_col) in zip(axes, pairs):
                x = subset[x_col].astype(float).values
                y = subset[y_col].astype(float).values
                valid = ~(np.isnan(x) | np.isnan(y))
                if not valid.any():
                    ax.set_visible(False)
                    continue
                ax.scatter(x[valid], y[valid], alpha=alpha, s=s)
                lim_lo = np.nanmin(np.concatenate([x[valid], y[valid]]))
                lim_hi = np.nanmax(np.concatenate([x[valid], y[valid]]))
                ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], **kw_refline)

                if x_limit:
                    ax.set_xlim(*x_limit)
                else:
                    ax.set_xlim(lim_lo, lim_hi)

                if y_limit:
                    ax.set_ylim(*y_limit)
                else:
                    ax.set_ylim(lim_lo, lim_hi)

                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel(x_label or x_col, fontsize=8)
                ax.set_ylabel(y_label or y_col, fontsize=8)

            for ax in axes[n_pairs:]:
                ax.set_visible(False)

            fig.suptitle(
                f"time_window={tw} • z_score={zs} • group {g_idx+1} (units={len(subset)})",
                fontsize=11
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()


def fraction_significant_multi(
    combined_zarr: Union[str, Path, xr.Dataset, pd.DataFrame],
    compare_variables: Sequence[str],
    *,
    variable_labels: Sequence[str] | None = None,
    time_window: Union[str, Sequence[str], None] = None,
    z_score: Union[bool, Sequence[bool], None] = None,
    alpha_level: float = 0.05,
    figsize: Tuple[int, int] | None = None,
    dpi: int = 120,
    session_lw: float = 1.0,
    session_alpha: float = 0.3,
    session_color: str = "C0",
    avg_color: str = "C1",
    avg_lw: float = 3.0,
) -> None:
    """Plot per-session **fraction of significant units** across variables.

    One figure is produced for each `(time_window, z_score)` combination. Each
    line corresponds to a session; the thick line is the across-session mean.

    Parameters
    ----------
    compare_variables
        List of column names that **must end with `-pval`**. Fractions are
        computed as  (# units with p <= alpha_level) / (total units) for each
        session and variable.
    variable_labels
        Optional custom labels for the x-axis. Must match length of compare_variables.
    """

    if not compare_variables:
        raise ValueError("`compare_variables` must contain at least one column name")

    if any(not v.endswith("-pval") for v in compare_variables):
        raise ValueError("All compare_variables must have the suffix '-pval'")

    if variable_labels and len(variable_labels) != len(compare_variables):
        raise ValueError("`variable_labels` must have the same length as `compare_variables`")

    df = _load_summary(combined_zarr)

    missing = [v for v in compare_variables if v not in df.columns]
    if missing:
        raise KeyError(f"Column(s) not in DataFrame: {missing}")

    def _to_set(v):
        if v is None:
            return None
        return {v} if not isinstance(v, (list, tuple, set)) else set(v)

    tw_filter = _to_set(time_window)
    z_filter = _to_set(z_score)

    tw_values = sorted(df["time_window"].unique()) if tw_filter is None else sorted(tw_filter)
    z_values = sorted(df["z_score"].unique()) if z_filter is None else sorted(z_filter)

    x = np.arange(len(compare_variables))

    for tw, zs in itertools.product(tw_values, z_values):
        subset = df[(df["time_window"] == tw) & (df["z_score"] == zs)]
        if subset.empty:
            continue

        sessions = subset["session_id"].unique()
        fig_w, fig_h = figsize if figsize else (6, 4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        all_fracs = []
        for sess in sessions:
            rows = subset[subset["session_id"] == sess]
            total_units = len(rows)
            if total_units == 0:
                continue
            fracs = [np.sum(rows[col].astype(float) <= alpha_level) / total_units for col in compare_variables]
            all_fracs.append(fracs)
            ax.plot(x, fracs, lw=session_lw, alpha=session_alpha, color=session_color)

        if all_fracs:
            mean_frac = np.nanmean(all_fracs, axis=0)
            ax.plot(x, mean_frac, lw=avg_lw, color=avg_color, label="mean")

        ax.axhline(alpha_level, color="red", ls="--", lw=1)
        labels = variable_labels if variable_labels else compare_variables
        ax.set_xticks(x, labels=labels, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction significant")
        ax.set_xlabel("variable")
        ax.set_title(f"Fraction significant (time_window={tw}, z_score={zs})")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()



def fit_and_plot_by_session(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    session_col: str = "session_id",
    x_label: str | None = None,
    y_label: str | None = None
) -> dict[str, tuple[float, float]]:
    """
    For each row in `df`, fit a simple linear regression y = slope*x + intercept
    using the list-like values in columns `x_col` and `y_col`, then plot the
    scatter and fitted line in its own figure.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns `x_col` and `y_col` with list-like numeric data per row,
        and a `session_col` identifying each row.
    x_col : str
        Column name for the predictor (x) values.
    y_col : str
        Column name for the response (y) values.
    session_col : str
        Column name for the session identifier.
    x_label : str or None
        Label for the x-axis (defaults to x_col if None).
    y_label : str or None
        Label for the y-axis (defaults to y_col if None).

    Returns
    -------
    dict
        Mapping from session identifier to (slope, intercept).
    """
    results: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        session = row[session_col]
        x = np.array(row[x_col], dtype=float)
        y = np.array(row[y_col], dtype=float)
        # filter out NaNs
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]
        if len(x_clean) < 2:
            # skip if insufficient data
            continue

        # fit line
        slope, intercept = np.polyfit(x_clean, y_clean, 1)
        results[session] = (slope, intercept)

        # plot
        plt.figure()
        plt.scatter(x_clean, y_clean)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line)
        plt.xlabel(x_label or x_col)
        plt.ylabel(y_label or y_col)
        plt.title(f"{session}: y = {slope:.3f} x + {intercept:.3f}")
        plt.show()
    return results


def fit_and_plot_all_sessions(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    session_col: str = "session_id",
    plot_dots: bool = True,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Fit and plot regression lines for all sessions on a single figure.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing session identifiers and two list-like columns.
    x_col : str
        Column name for the predictor values (list per row).
    y_col : str
        Column name for the response values (list per row).
    session_col : str, default "session_id"
        Column name for the session identifier.
    plot_dots : bool, default True
        Whether to scatter plot individual data points.
    x_label : str, optional
        Label for the x-axis (defaults to x_col if None).
    y_label : str, optional
        Label for the y-axis (defaults to y_col if None).

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Mapping from session identifier to (slope, intercept).
    """
    fit_results: Dict[str, Tuple[float, float]] = {}

    plt.figure()
    for _, row in df.iterrows():
        session = row[session_col]
        x = np.array(row[x_col], dtype=float)
        y = np.array(row[y_col], dtype=float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]
        if len(x_clean) < 2:
            continue

        # Fit line
        slope, intercept = np.polyfit(x_clean, y_clean, 1)
        fit_results[session] = (slope, intercept)

        # Scatter points if requested
        if plot_dots:
            plt.scatter(x_clean, y_clean, s=10, alpha=0.5)

        # Regression line
        x_vals = np.array([x_clean.min(), x_clean.max()])
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, linewidth=2)

    num_sessions = len(fit_results)
    plt.title(f"Regression lines for {num_sessions} sessions")
    plt.xlabel(x_label or x_col)
    plt.ylabel(y_label or y_col)
    plt.tight_layout()
    plt.show()

    return fit_results


def plot_angle_fraction_polar(
    ds: pd.DataFrame,
    *,
    filter_region: str,
    time_window: str,
    col_x: str,
    col_y: str,
    col_pval_x: str,
    col_pval_y: str,
    alpha_level: float = 0.05,
    include: str = "both",
    # one of {"both", "x", "y", "any", "none", "all"}
    # "both" = X- and Y-significant; "x" = X-significant only; "y" = Y-significant only
    # "any"  = X or Y significant;  "none" = neither significant; "all" = all points (ignores p-values)
    angle_bin_deg: float = 15.0,
    normalize: str = "selected",
    # one of {"selected","overall"}:
    #   "selected" → fraction within the selected subset (denominator = #points that pass masks)
    #   "overall"  → fraction out of all filtered points before the p-value mask
    start_angle_deg: float = 0.0,    # rotate binning origin if desired
    show_reference_diagonals: bool = True,
    figsize: tuple[float,float] = (6.5, 6.5),
    dpi: int = 120,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Make a circular (polar) bar plot of the fraction of neurons by polar angle
    in the (col_x, col_y) space.

    Filtering and significance semantics mirror `plot_diagonal_significance`.

    Parameters
    ----------
    ds
        Full DataFrame containing at least col_x, col_y, p-value columns, 'time_window', and 'brain_region'.
    filter_region
        '' for all regions, '!MD' for non-MD, or an exact region code to include.
    time_window
        Exact time window value to include (e.g., '-1_0').
    col_x, col_y
        Names of the X and Y statistic columns (e.g., t-values).
    col_pval_x, col_pval_y
        Column names for the corresponding p-values.
    alpha_level
        P-value threshold for significance.
    include
        Which set of units to include based on significance ("both","x","y","any","none","all").
    angle_bin_deg
        Angular bin width in degrees (e.g., 15 → 24 bins).
    normalize
        "selected": bar heights sum to 1 over the selected subset.
        "overall" : bar heights sum to (selected / overall) over the full filtered set.
    start_angle_deg
        Rotate the 0° direction for binning (useful if you want bins centered on 45°, etc.).
    show_reference_diagonals
        If True, draw reference rays at ±45° and 180° offsets.
    figsize, dpi
        Figure size / resolution.

    Returns
    -------
    fig, ax, table
        Matplotlib figure/axes, and a table with counts/fractions per angular bin.
    """
    # ------------------------
    # 1) Region / window filter and basic cleaning
    # ------------------------
    if filter_region == '':
        filtered = ds[ds['time_window'] == time_window].copy()
        region_label = 'all regions'
    elif filter_region == '!MD':
        filtered = ds[(ds['brain_region'] != 'MD') & (ds['time_window'] == time_window)].copy()
        region_label = 'non-MD regions'
    else:
        filtered = ds[(ds['brain_region'] == filter_region) & (ds['time_window'] == time_window)].copy()
        region_label = filter_region

    needed = [col_x, col_y, col_pval_x, col_pval_y]
    filtered = filtered.dropna(subset=needed)
    if filtered.empty:
        raise ValueError("No rows remain after filtering or NaN removal.")

    # ------------------------
    # 2) Significance masks
    # ------------------------
    mask_x    = filtered[col_pval_x] <= alpha_level
    mask_y    = filtered[col_pval_y] <= alpha_level
    mask_both = mask_x & mask_y
    mask_any  = mask_x | mask_y
    mask_none = ~(mask_x | mask_y)

    if include == "both":
        sel_mask = mask_both
        sel_name = "Both-sig"
    elif include == "x":
        sel_mask = mask_x & ~mask_y
        sel_name = "X-only-sig"
    elif include == "y":
        sel_mask = mask_y & ~mask_x
        sel_name = "Y-only-sig"
    elif include == "any":
        sel_mask = mask_any
        sel_name = "Any-sig"
    elif include == "none":
        sel_mask = mask_none
        sel_name = "Non-sig"
    elif include == "all":
        sel_mask = np.ones(len(filtered), dtype=bool)
        sel_name = "All"
    else:
        raise ValueError("`include` must be one of {'both','x','y','any','none','all'}")

    selected = filtered[sel_mask]
    if selected.empty:
        raise ValueError("Selection mask resulted in zero rows.")

    # ------------------------
    # 3) Angles (0..2π)
    # ------------------------
    x = filtered[col_x].to_numpy(dtype=float)
    y = filtered[col_y].to_numpy(dtype=float)
    theta_all = np.arctan2(y, x)               # (-π, π]
    theta_all = np.mod(theta_all, 2*np.pi)     # [0, 2π)

    xs = selected[col_x].to_numpy(dtype=float)
    ys = selected[col_y].to_numpy(dtype=float)
    theta_sel = np.arctan2(ys, xs)
    theta_sel = np.mod(theta_sel, 2*np.pi)

    # Rotate angles if requested
    rot = np.deg2rad(start_angle_deg)
    theta_all = np.mod(theta_all - rot, 2*np.pi)
    theta_sel = np.mod(theta_sel - rot, 2*np.pi)

    # ------------------------
    # 4) Bin edges and counts
    # ------------------------
    if angle_bin_deg <= 0 or angle_bin_deg > 360:
        raise ValueError("`angle_bin_deg` must be in (0, 360].")
    n_bins = int(np.ceil(360.0 / angle_bin_deg))
    bin_edges = np.linspace(0.0, 2*np.pi, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    counts_sel, _ = np.histogram(theta_sel, bins=bin_edges)
    denom = counts_sel.sum() if normalize == "selected" else len(theta_all)  # selected vs overall
    fractions = counts_sel.astype(float) / max(denom, 1)

    # Build results table
    table = pd.DataFrame({
        "bin_start_deg": np.rad2deg(bin_edges[:-1]),
        "bin_end_deg":   np.rad2deg(bin_edges[1:]),
        "bin_center_deg": np.rad2deg(bin_centers),
        "count": counts_sel,
        "fraction": fractions
    })

    # ------------------------
    # 5) Plot (polar bar plot)
    # ------------------------
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="polar")

    bars = ax.bar(
        bin_centers, fractions,
        width=bin_width,
        align="center",
        edgecolor="k",
        linewidth=0.8,
        alpha=0.85
    )

    # Style tweaks
    ax.set_theta_direction(1)          # clockwise increases
    ax.set_theta_zero_location("E")     # 0° (bin start_angle) to the right
    ax.set_title(
        f"{region_label}, time_window = {time_window}\n"
        f"{sel_name} • bins={n_bins} (Δ={angle_bin_deg:.1f}°) • "
        f"normalize={normalize}",
        va="bottom"
    )

    # Optional reference diagonals at ±45° (and opposite)
    if show_reference_diagonals:
        ref_degs = np.array([45, 135, 225, 315], dtype=float) - start_angle_deg
        for d in ref_degs:
            th = np.deg2rad(d % 360)
            ax.plot([th, th], [0, ax.get_rmax()], ls="--", lw=1.0, color="gray", alpha=0.8)

    # Radial grid and label percentages at peaks
    ax.set_rlabel_position(90)  # put radial labels on top
    ax.grid(alpha=0.3, ls=":")
    # Annotate a few highest bins
    if fractions.size:
        top_idx = np.argsort(fractions)[-min(5, len(fractions)):]
        for i in top_idx:
            if fractions[i] <= 0:
                continue
            th = bin_centers[i]
            r  = fractions[i]
            ax.text(th, r + 0.02 * (ax.get_rmax() or 1), f"{100*r:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()
    return fig, ax, table



def plot_diagonal_significance(
    ds: pd.DataFrame,
    filter_region: str,
    time_window: str,
    col_x: str,
    col_y: str,
    col_pval_x: str,
    col_pval_y: str,
    diagonal: str = 'negative',           # default aligned with docstring
    angle_tolerance_deg: float = 10.0,
    xlim: tuple = (-30, 30),
    ylim: tuple = (-30, 30),
    point_size: float = 5.0,
    alpha_level: float = 0.05,
    fit_oval: bool = True,               # New parameter to fit an oval to the data
    fit_oval_only_both_sig: bool = True  # New parameter to fit oval only on Both-sig points
):
    """
    Filter a DataFrame by region and time window, compute significance masks
    for X and Y variables, and plot diagonal panels as specified.

    Parameters
    ----------
    ds : pandas.DataFrame
        The full dataset containing at least the columns specified.
    filter_region : str
        Region filter: '' for all regions, '!MD' for non-MD, or a specific region code.
    time_window : str
        The time_window value to select (e.g. '-1_0').
    col_x, col_y : str
        Column names for the X and Y statistics (e.g. t-values).
    col_pval_x, col_pval_y : str
        Column names for the corresponding p-values.
    diagonal : str, optional
        Which diagonal to plot: 'both', 'negative', or 'positive'. Default is 'negative'.
    angle_tolerance_deg : float, optional
        Tolerance around the diagonal in degrees (default=10°).
    xlim : tuple, optional
        X-axis limits (default=(-30, 30)).
    ylim : tuple, optional
        Y-axis limits (default=(-30, 30)).
    point_size : float, optional
        Marker size for all scatter points (default=5).
    alpha_level: float, optional
        The p value used to determine the significant neuron.
    fit_oval : bool, optional
        If True, fit and plot an oval (ellipse) to the scatter plot (default=False).
    fit_oval_only_both_sig : bool, optional
        If True, fit the oval using only the Both-sig points (default=False).
    """
    # Backward-compatibility shim if caller still passes p_value via keywords
    if 'p_value' in locals():
        alpha_level = locals().get('p_value', alpha_level)

    # 1) Region + time window filter
    if filter_region == '':
        filtered = ds[ds['time_window'] == time_window].copy()
        region_label = 'all regions'
    elif filter_region == '!MD':
        filtered = ds[(ds['brain_region'] != 'MD') & (ds['time_window'] == time_window)].copy()
        region_label = 'non-MD regions'
    else:
        filtered = ds[(ds['brain_region'] == filter_region) & (ds['time_window'] == time_window)].copy()
        region_label = filter_region

    # 2) Drop rows with NaNs in any required columns
    needed = [col_x, col_y, col_pval_x, col_pval_y]
    filtered = filtered.dropna(subset=needed)
    if filtered.empty:
        print("No rows after filtering / NaN removal.")
        return None, None

    # 3) Significance masks (use <=)
    mask_x    = filtered[col_pval_x] <= alpha_level
    mask_y    = filtered[col_pval_y] <= alpha_level
    mask_both = mask_x & mask_y
    mask_none = ~(mask_x | mask_y)

    total = len(filtered)

    # 4) Angle-based banding
    angle_tol = np.deg2rad(angle_tolerance_deg)
    max_tol = np.deg2rad(44.9)
    if angle_tol >= max_tol:
        angle_tol = max_tol

    angles = np.arctan2(filtered[col_y].values, filtered[col_x].values)

    def in_band(center):
        delta = np.abs(np.arctan2(np.sin(angles - center), np.cos(angles - center)))
        return delta < angle_tol

    band_neg = mask_both & (in_band(-np.pi/4) | in_band(3*np.pi/4))
    band_pos = mask_both & (in_band(np.pi/4) | in_band(-3*np.pi/4))

    # 5) Counts/fractions
    count_none = int(mask_none.sum())
    count_x    = int(mask_x.sum())
    count_y    = int(mask_y.sum())
    count_both = int(mask_both.sum())
    count_neg_band = int(band_neg.sum())
    count_pos_band = int(band_pos.sum())

    # Sign fractions by sign
    cnt_x_pos = int((mask_x & (filtered[col_x] > 0)).sum())
    cnt_x_neg = int((mask_x & (filtered[col_x] < 0)).sum())
    frac_x_pos = cnt_x_pos / count_x if count_x else 0.0
    frac_x_neg = cnt_x_neg / count_x if count_x else 0.0

    cnt_y_pos = int((mask_y & (filtered[col_y] > 0)).sum())
    cnt_y_neg = int((mask_y & (filtered[col_y] < 0)).sum())
    frac_y_pos = cnt_y_pos / count_y if count_y else 0.0
    frac_y_neg = cnt_y_neg / count_y if count_y else 0.0

    # 6) Which panels to draw
    draw_neg = diagonal in ('both', 'negative')
    draw_pos = diagonal in ('both', 'positive')
    n_panels = int(draw_neg) + int(draw_pos)

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]
    ax_idx = 0
    x_vals = np.linspace(*xlim, 200)

    def fit_ellipse(x, y):
        # Center the data before applying PCA
        data = np.column_stack((x, y))
        center = np.mean(data, axis=0)
        data_centered = data - center  # Center the data by subtracting the mean
        
        # Fit an ellipse to the centered data using PCA
        pca = PCA(n_components=2)
        pca.fit(data_centered)
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_

        # The angle of rotation
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

        # Ellipse parameters
        t = np.linspace(0, 2 * np.pi, 100)
        ellipse = np.array([np.cos(t), np.sin(t)])
        ellipse[0, :] *= np.sqrt(eigenvalues[0])
        ellipse[1, :] *= np.sqrt(eigenvalues[1])

        # Rotate the ellipse based on the eigenvectors
        ellipse_rot = np.dot(eigenvectors.T, ellipse)

        # Correcting axis vectors to ensure they are 2D
        major_axis = eigenvectors[:, 0] * np.sqrt(eigenvalues[0]) * 2  # Major axis along the first eigenvector
        minor_axis = eigenvectors[:, 1] * np.sqrt(eigenvalues[1]) * 2  # Minor axis along the second eigenvector

        # Ensure that major_axis and minor_axis are treated as 2D vectors, not scalars
        major_axis = np.array([major_axis[0], major_axis[1]])  # Create a 2D vector for major axis
        minor_axis = np.array([minor_axis[0], minor_axis[1]])  # Create a 2D vector for minor axis

        return ellipse_rot + center[:, None], major_axis + center, minor_axis + center, angle, center


    def draw_panel(ax: plt.Axes, negative: bool):
        # Scatter layers
        ax.scatter(filtered[col_x], filtered[col_y], color='lightgray', s=point_size,
                   label=f'Non-sig (n={count_none}; {count_none/total:.1%})')
        ax.scatter(filtered.loc[mask_x, col_x], filtered.loc[mask_x, col_y],
                   color='blue', s=point_size,
                   label=f'X-sig (n={count_x}; {count_x/total:.1%})')
        ax.scatter(filtered.loc[mask_y, col_x], filtered.loc[mask_y, col_y],
                   color='green', s=point_size,
                   label=f'Y-sig (n={count_y}; {count_y/total:.1%})')
        ax.scatter(filtered.loc[mask_both, col_x], filtered.loc[mask_both, col_y],
                   color='red', s=point_size,
                   label=f'Both-sig (n={count_both}; {count_both/total:.1%})')

        # Diagonal band overlay
        band_mask = band_neg if negative else band_pos
        cnt_band  = count_neg_band if negative else count_pos_band
        ax.scatter(filtered.loc[band_mask, col_x], filtered.loc[band_mask, col_y],
                   color='purple', s=point_size,
                   label=f'Diag band (n={cnt_band}; {cnt_band/total:.1%})')

        # Diagonal line and tolerance boundaries
        center = -np.pi/4 if negative else np.pi/4
        diag_y = -x_vals if negative else x_vals
        ax.plot(x_vals, diag_y, '--', color='k', label='y = -x' if negative else 'y = +x')
        b_hi = np.tan(center + angle_tol) * x_vals
        b_lo = np.tan(center - angle_tol) * x_vals
        ax.plot(x_vals, b_hi, ':', color='k')
        ax.plot(x_vals, b_lo, ':', color='k')

        # Panel-specific annotation
        if negative:
            title = f"Negative Diagonal (±{angle_tolerance_deg}°)"
        else:
            title = f"Positive Diagonal (±{angle_tolerance_deg}°)"

        stats_text = (
            f"X-sig: + {cnt_x_pos} ({frac_x_pos:.1%}), − {cnt_x_neg} ({frac_x_neg:.1%})\n"
            f"Y-sig: + {cnt_y_pos} ({frac_y_pos:.1%}), − {cnt_y_neg} ({frac_y_neg:.1%})\n"
        )

        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(col_x)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                va='top', fontsize='small',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
        ax.legend(loc='upper right', fontsize='small')

        # Fit and plot ellipse if requested
        if fit_oval:
            data_for_ellipse = filtered
            if fit_oval_only_both_sig:
                data_for_ellipse = filtered.loc[mask_both]  # Only use Both-sig points
            ellipse_rot, major_axis, minor_axis, angle, center = fit_ellipse(data_for_ellipse[col_x], data_for_ellipse[col_y])
            ax.plot(ellipse_rot[0, :], ellipse_rot[1, :], color='orange', lw=2, label="Fitted Ellipse")
            
            # Plot Major and Minor Axes
            ax.quiver(center[0], center[1], major_axis[0], major_axis[1], angles='xy', scale_units='xy', scale=1, color='blue', label="Major Axis")
            ax.quiver(center[0], center[1], minor_axis[0], minor_axis[1], angles='xy', scale_units='xy', scale=1, color='green', label="Minor Axis")

            # Add angle of rotation to the legend
            ax.legend(loc='upper left', fontsize='small', title=f"Rotation: {np.degrees(angle):.1f}°")
            ax.set_aspect('equal', adjustable='box')

    if draw_neg:
        draw_panel(axes[ax_idx], negative=True)
        axes[ax_idx].set_ylabel(col_y)
        ax_idx += 1
    if draw_pos:
        draw_panel(axes[ax_idx], negative=False)

    plt.suptitle(
        f"{region_label}, time_window = {time_window}\n"
        "Gray=non-sig, Blue=X-sig, Green=Y-sig, Red=both-sig; Purple=diagonal band & both-sig",
        fontsize=14
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
    return fig, axes



def plot_projection(
    proj_df: pd.DataFrame,
    trial_idx: Optional[int] = None,
    average: bool = False,
    baseline_window: Optional[Tuple[float, float]] = None,
    n_quantiles: int = 4,
    figsize: Tuple[float, float] = (8, 4),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    unit_number: Optional[int] = None
) -> plt.Axes:
    """
    Plot projected PSTH(s) from a proj_df DataFrame.

    Parameters
    ----------
    proj_df : pd.DataFrame
        Output of `project_psth_per_trial`.
    trial_idx : int, optional
        If provided, plots only that trial's projection.
    average : bool, default False
        If True and trial_idx is None, plots the mean ±1 SD across all trials.
    baseline_window : tuple, optional
        (start, end) in seconds relative to event. If provided, groups trials
        by the mean projection in this window into `n_quantiles` and plots
        one line per quantile.
    n_quantiles : int, default 4
        Number of quantile groups when `baseline_window` is specified.
    figsize : tuple, default (8,4)
        Figure size (width, height).
    title : str, optional
        Plot title. If None, a default is chosen.
    ax : plt.Axes, optional
        Existing axes to draw on.
    unit_number : int, optional
        If provided, appends "(n_units=<unit_number>)" to the title.

    Returns
    -------
    ax : plt.Axes
        Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    bins = proj_df["psth_bins"].iat[0]
    all_proj = np.stack(proj_df.projection.values, axis=0)

    # prepare title suffix if unit_number is given
    suffix = f" (n_units={unit_number})" if unit_number is not None else ""

    # Single-trial override
    if trial_idx is not None:
        row = proj_df[proj_df.trial_idx == trial_idx]
        if row.empty:
            raise ValueError(f"No trial_idx = {trial_idx}")
        y = row.projection.iloc[0]
        ax.plot(bins, y, color="C0", lw=2, label=f"Trial {trial_idx}")
        base_title = title or f"Projected PSTH, trial {trial_idx}"

    # Quantile grouping override
    elif baseline_window is not None:
        start, end = baseline_window
        mask = (bins >= start) & (bins < end)
        baseline_mean = all_proj[:, mask].mean(axis=1)
        groups = pd.qcut(baseline_mean, q=n_quantiles, labels=False, duplicates='drop')
        for q in range(groups.max() + 1):
            grp_mask = groups == q
            mean_q = all_proj[grp_mask].mean(axis=0)
            count = int(grp_mask.sum())
            ax.plot(bins, mean_q, lw=2, label=f"Q{q+1} (n={count})")
        base_title = title or f"Projection by {n_quantiles} quantiles\n(baseline {start}-{end}s)"

    # Average across all trials
    elif average:
        mean = all_proj.mean(axis=0)
        sd = all_proj.std(axis=0)
        ax.plot(bins, mean, color="C1", lw=2, label="Mean")
        ax.fill_between(bins, mean - sd, mean + sd, color="C1", alpha=0.3, label="±1 SD")
        base_title = title or "Mean projected PSTH ±1 SD"

    # Plot each trial
    else:
        for y in all_proj:
            ax.plot(bins, y, color="gray", alpha=0.4)
        base_title = title or "Projected PSTH, all trials"

    # set title with optional unit count suffix
    ax.set_title(base_title + suffix)
    ax.set_xlabel("Time (s) relative to event")
    ax.set_ylabel("Projection")
    ax.legend()
    ax.grid(True)
    return ax


def plot_session_spike_raster(
    nwb_data: Any,
    *,
    time_window: Tuple[float, float],
    probes: Optional[Sequence[Union[int, str]]] = None,  # match values in units['device_name']
    unit_indices: Optional[Sequence[int]] = None,
    max_units: Optional[int] = 200000,
    events: Optional[Dict[str, Union[str, Sequence[float], np.ndarray]]] = None,
    event_styles: Optional[Dict[str, Dict[str, Any]]] = None,
    line_length: float = 0.8,
    linewidth: float = 0.9,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    save_figure: bool = False,
    save_folder: str = "/root/capsule/results",
    filename: Optional[str] = None,
) -> List[int]:
    """
    Plot a session-wide spike raster for **QC-passed** units within an absolute
    session time window. Uses `units['device_name']` as the probe label.

    Behavior with `probes`:
      • If `probes` is provided with **multiple** values → units are **sorted by probe**
        (in the order given), spikes are colored **per-probe**, and a legend is shown.
      • If `probes` has a single value → only that probe is shown (single color).
      • If `probes` is None → all QC units are shown (single color).

    Parameters
    ----------
    nwb_data : Any
        NWB object with a `units` DynamicTable having 'spike_times' and ideally 'device_name'.
    time_window : (float, float)
        Absolute session-time window in seconds to visualize.
    probes : sequence[int|str] | None
        Optional filter. Exact matches against `units['device_name']` (compared as strings).
        If multiple provided, also controls group order and colors.
    unit_indices : sequence[int] | None
        Optional extra subset of units to intersect with the QC set.
    max_units : int | None
        If provided and >0, limit the number of plotted units after filtering/sorting.
    events : dict[str, str | arraylike] | None
        Event label → either behavior event name (string) or array of absolute times (seconds).
    event_styles : dict[str, dict] | None
        Matplotlib kwargs per event label.
    line_length : float
        Vertical tick length for each spike (y-axis units).
    linewidth : float
        Tick line width.
    figsize : (int, int)
        Figure size.
    dpi : int
        Figure DPI.
    save_figure : bool
        Save the figure if True.
    save_folder : str
        Output folder to save figure.
    filename : str | None
        Base filename (without extension). Auto-generated if None.

    Returns
    -------
    list[int]
        Ordered list of plotted unit indices (after QC, probe filtering, and sorting).
    """
    # --- basic checks ---
    if not hasattr(nwb_data, "units"):
        raise ValueError("`nwb_data` must have a `units` table.")
    tbl = nwb_data.units
    if "spike_times" not in getattr(tbl, "colnames", []):
        raise ValueError("`nwb_data.units` must contain a 'spike_times' column.")

    t_start, t_end = float(time_window[0]), float(time_window[1])
    if t_end <= t_start:
        raise ValueError("`time_window` must have end > start.")

    # --- QC filter (and optional explicit subset) ---
    qc_pass: set[int] = set(map(int, get_units_passed_default_qc(nwb_data)))
    if unit_indices is not None:
        qc_pass &= set(map(int, unit_indices))
    if not qc_pass:
        print("No units pass QC (and filters).")
        return []

    # --- Build metadata: (unit_index, device_name) for QC units ---
    def _devname(i: int) -> Optional[str]:
        if "device_name" in tbl.colnames:
            try:
                v = tbl["device_name"][i]
                return None if v is None else str(v)
            except Exception:
                return None
        return None

    meta: List[Tuple[int, Optional[str]]] = [(i, _devname(i)) for i in qc_pass]

    # --- Optional filter to provided probes ---
    probe_order: List[str] = []
    if probes:
        probe_order = [str(p) for p in probes]
        keep = set(probe_order)
        meta = [m for m in meta if (m[1] is not None and m[1] in keep)]

    if not meta:
        print("No QC-passed units left after probe filter.")
        return []

    # --- Sorting & coloring by probe if multiple probes provided ---
    multi_probe = probes is not None and len(probe_order) > 1
    if multi_probe:
        order_index = {p: k for k, p in enumerate(probe_order)}
        meta.sort(key=lambda m: (order_index.get(m[1], 1_000_000), m[0]))
        default_cycle = list(rcParams['axes.prop_cycle'].by_key().get('color', [])) or [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf"
        ]
        probe_colors: Dict[str, str] = {p: default_cycle[i % len(default_cycle)] for i, p in enumerate(probe_order)}
    else:
        meta.sort(key=lambda m: m[0])  # single probe / no filter → order by unit index
        probe_colors = {}

    # --- Truncate if requested ---
    candidates: List[int] = [m[0] for m in meta]
    devnames:   List[Optional[str]] = [m[1] for m in meta]
    if isinstance(max_units, int) and max_units > 0:
        candidates = candidates[:max_units]
        devnames   = devnames[:max_units]

    # --- Gather spikes within window ---
    unit_spike_lists: List[np.ndarray] = []
    for i in candidates:
        try:
            st = np.asarray(tbl["spike_times"][i], dtype=float)
        except Exception:
            st = np.asarray([])
        mask = (st >= t_start) & (st <= t_end)
        unit_spike_lists.append(st[mask])

    # --- Events (absolute times) ---
    if events is None:
        events = {}
    styles_default: Dict[str, Any] = {"color": "red", "ls": "--", "lw": 1.0, "alpha": 0.9, "zorder": 1}
    event_styles = event_styles or {}
    event_times_map: Dict[str, np.ndarray] = {}
    for name, spec in events.items():
        if isinstance(spec, str):
            try:
                arr = np.asarray(extract_event_timestamps(nwb_data, spec), dtype=float)
            except Exception:
                arr = np.asarray([], dtype=float)
        else:
            arr = np.asarray(spec, dtype=float)
        event_times_map[name] = arr[(arr >= t_start) & (arr <= t_end)]

    # --- Draw ---
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    y_base = np.arange(1, len(candidates) + 1)

    # Draw spikes as colored vertical ticks; color by probe if multi-probe
    for y0, spikes, dn in zip(y_base, unit_spike_lists, devnames):
        if spikes.size == 0:
            continue
        tick_color = probe_colors.get(dn, "black") if multi_probe else "black"
        for t in spikes:
            ax.vlines(t, y0 - line_length / 2, y0 + line_length / 2,
                      linewidth=linewidth, color=tick_color)

    # Draw group separators if multiple probes
    if multi_probe:
        for i in range(1, len(devnames)):
            if devnames[i] != devnames[i-1]:
                ax.axhline(i + 0.5, color="gray", ls=":", lw=0.8, alpha=0.7)

    # Draw events
    for name, times in event_times_map.items():
        if times.size == 0:
            continue
        style = styles_default.copy()
        style.update(event_styles.get(name, {}))
        for t in times:
            ax.axvline(t, **style)

    # Axes labels/limits
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(0.5, len(candidates) + 0.5)
    ax.set_xlabel("Time (s, absolute session)")
    ax.set_ylabel("Units (QC-passed)")

    # Y tick labels (unit index + device_name)
    labels = [f"{u}" + (f" · {dn}" if dn else "") for u, dn in zip(candidates, devnames)]
    step = max(1, len(labels) // 25)
    ax.set_yticks(y_base[::step])
    ax.set_yticklabels(labels[::step], fontsize=8)

    title_bits = [
        f"Session raster [{t_start:.3f}, {t_end:.3f}] s",
        f"QC units={len(candidates)}"
    ]
    if probes:
        title_bits.append(f"device_name ∈ {list(map(str, probes))}")
    ax.set_title(" · ".join(title_bits))

    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # ======================
    # Legend OUTSIDE figure
    # ======================
    legend_handles: List[Line2D] = []

    # Probe legend (if multiple probes)
    if multi_probe:
        for p in probe_order:
            legend_handles.append(Line2D([0], [0], color=probe_colors[p], lw=2, label=str(p)))

    # Event legend proxies (use same style as drawn)
    for name, style_src in event_times_map.items():
        # Only add if the event exists in window
        if style_src.size == 0:
            continue
        style = styles_default.copy()
        style.update(event_styles.get(name, {}))
        legend_handles.append(
            Line2D([0], [0],
                   color=style.get("color", "red"),
                   linestyle=style.get("ls", "--"),
                   linewidth=style.get("lw", 1.0),
                   alpha=style.get("alpha", 0.9),
                   label=name)
        )

    if legend_handles:
        # Place legend outside on the right
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            title="Legend",
            fontsize=9
        )
        # Leave some room on the right for the legend
        plt.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        plt.tight_layout()

    # Save (ensure external legend is included)
    if save_figure:
        os.makedirs(save_folder, exist_ok=True)
        if filename is None:
            sess = getattr(nwb_data, "session_id", "session")
            filename = f"session_raster_QC_{sess}_{t_start:.2f}_{t_end:.2f}"
        out_path = os.path.join(save_folder, f"{filename}.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved: {out_path}")

    plt.show()
    return candidates