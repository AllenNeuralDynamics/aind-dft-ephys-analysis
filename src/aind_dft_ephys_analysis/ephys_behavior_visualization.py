import os, re
from itertools import product
from typing import List, Optional, Tuple, Union, Any, Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from general_utils import smart_read_csv
from behavior_utils   import extract_fitted_data       
from behavior_utils import extract_event_timestamps   # your helper that returns per-trial timestamps

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
        - one or more '<model>-<variable>_pval' columns.
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

