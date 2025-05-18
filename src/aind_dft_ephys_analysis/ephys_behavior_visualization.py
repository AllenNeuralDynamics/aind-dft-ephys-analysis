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


