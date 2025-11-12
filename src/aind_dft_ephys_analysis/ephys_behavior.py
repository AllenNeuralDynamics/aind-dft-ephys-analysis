from __future__ import annotations
import os
import glob
import json
from pathlib import Path
from collections import defaultdict
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Optional, Union, Tuple, Dict
import multiprocessing


from behavior_utils import extract_event_timestamps, find_trials
from general_utils import extract_session_name_core
from nwb_utils import NWBUtils
from general_utils import smart_read_csv
from aind_spurious_correlation import methods

from ephys_utils import append_units_locations


def get_units_passed_default_qc(nwb_data: Any) -> np.ndarray:
    """
    Retrieves the indices of units in a combined NWB dataset that have passed
    the automated default QC checks and are not labeled as 'noise'.

    The `default_qc` flag is precomputed based on the
    following metrics:

    - **presence_ratio** (float): Fraction of the recording session during which the unit was active.
      Units must have `presence_ratio >= 0.8`.
    - **isi_violations_ratio** (float): Proportion of inter-spike intervals violating the refractory period.
      Units must have `isi_violations_ratio <= 0.5`.
    - **amplitude_cutoff** (float): Estimated fraction of missed spikes due to detection threshold.
      Units must have `amplitude_cutoff <= 0.1`.

    After computing these, the `default_qc` field should be True. Additionally, units
    labeled as 'noise' by `decoder_label` are excluded.

    Parameters
    ----------
    nwb_data : NWB file handle
        Combined NWB object containing a `units` table with columns:
        `default_qc` and `decoder_label`.

    Returns
    -------
    indices : numpy.ndarray
        1D array of integer indices of units where `default_qc` is True
        and `decoder_label` != 'noise'.
    """
    tbl = nwb_data.units
    default_qc = np.array(tbl['default_qc'].data)
    labels = np.array(tbl['decoder_label'].data)

    # Mask for QC-passed, non-noise units
    mask = ((default_qc == True) | (default_qc == 'True')) & (labels != 'noise')

    # Return indices
    indices = np.nonzero(mask)[0]
    print(f"Number of units passing QC: {len(indices)}")
    return indices


def get_the_mean_firing_rate(
    nwb_data: Any,
    unit_index: Union[int, List[int], None] = None,
    align_to_event: str = 'go_cue',
    time_windows: List[List[float]] = [[-1, 0], [0, 2]],
    z_score: Union[bool, List[bool]] = False
) -> pd.DataFrame:
    """
    Calculate mean firing rates per trial for specified units over one or more time windows
    aligned to a behavioral event, and return a tidy DataFrame with session, unit, window,
    and a list of firing rates per trial.

    Parameters
    ----------
    nwb_data : NWB file handle
        Combined NWB object containing `session_id`, behavior trials, and units spike_times.
    unit_index : int, list of int, or None
        Unit index or list of indices to process. If None, uses all units passing default QC.
    align_to_event : str
        Event name to align spikes to (default 'go_cue').
    time_windows : list of [start, end]
        List of time windows (in seconds relative to event) for computing firing rates.
        For example, `[[−2, 1], [1, 2], [0.2, 3, −1, 0]]` specifies three time windows.
        If a window contains four elements, like `[0.2, 3, −1, 0]`, it means that two separate mean firing rates will be calculated:
        - The mean for the first range `[0.2, 3]` and
        - The mean for the second range `[-1, 0]`.
        The first mean is then subtracted by the second mean.
    z_score : bool
        If True, normalize each unit's firing rates (per window) across trials to z-scores.

    Returns
    -------
    firing_rate_df : pd.DataFrame
        A DataFrame with columns:
          - session_id: the NWB session identifier
          - unit_index: integer unit index
          - time_window: string label of the form "start_end"
          - rates: list of mean firing rates (spikes/sec) per trial
    """
    # append units locations to nwb_data
    session_id = extract_session_name_core(getattr(nwb_data, 'session_id', None))
    nwb_data = append_units_locations(nwb_data, session_id)

    all_times = np.array(extract_event_timestamps(nwb_data, align_to_event))
    n_trials = len(all_times)

    if unit_index is None:
        units_to_process = list(get_units_passed_default_qc(nwb_data))
    elif isinstance(unit_index, list):
        units_to_process = unit_index
    else:
        units_to_process = [unit_index]

    if isinstance(z_score, bool):
        z_score = [z_score]

    rows = []
    for u in units_to_process:
        spikes = np.array(nwb_data.units['spike_times'][u])
        loc = nwb_data.units["ccf_location"][u] or {}
        region = loc.get("brain_region", "")

        for time_window in time_windows:
            if len(time_window) == 4:
                # Handle four-element time window
                first_window = [time_window[0], time_window[1]]
                second_window = [time_window[2], time_window[3]]
                duration_1 = first_window[1] - first_window[0]
                duration_2 = second_window[1] - second_window[0]

                rates_1 = []
                rates_2 = []
                for t0 in all_times:
                    cnt1 = np.sum((spikes >= t0 + first_window[0]) & (spikes <= t0 + first_window[1]))
                    cnt2 = np.sum((spikes >= t0 + second_window[0]) & (spikes <= t0 + second_window[1]))
                    rates_1.append(cnt1 / duration_1 if duration_1 > 0 else np.nan)
                    rates_2.append(cnt2 / duration_2 if duration_2 > 0 else np.nan)

                rates = np.array(rates_1) - np.array(rates_2)
            else:
                # Handle normal two-element time windows
                start_offset, end_offset = time_window
                duration = end_offset - start_offset
                rates = []
                for t0 in all_times:
                    cnt = np.sum((spikes >= t0 + start_offset) & (spikes <= t0 + end_offset))
                    rates.append(cnt / duration if duration > 0 else np.nan)

            orig_rates = np.array(rates)
            for flag in z_score:
                rates_to_store = orig_rates.copy()
                if flag:
                    m, sd = np.nanmean(rates_to_store), np.nanstd(rates_to_store)
                    rates_to_store = (rates_to_store - m) / sd
                    rates_to_store = np.nan_to_num(rates_to_store)
                rows.append({
                    'session_id': session_id,
                    'unit_index': u,
                    "align_to_event": align_to_event,
                    'time_window': f"{time_window[0]}_{time_window[1]}" if len(time_window) == 2 else f"{time_window[0]}_{time_window[1]}_{time_window[2]}_{time_window[3]}",
                    'z_score': flag,
                    "brain_region": region,
                    "ccf_location": loc,
                    'rates': rates_to_store.tolist()
                })

    firing_rate_df = pd.DataFrame(
        rows,
        columns=[
            "session_id",
            "unit_index",
            "align_to_event",
            "time_window",
            "z_score",
            "brain_region",
            "ccf_location",
            "rates",
        ],
    )
    return firing_rate_df


def get_the_mean_firing_rate_combined(
    session_names: List[str],
    align_to_event: str = 'go_cue',
    time_windows: List[List[float]] = [[-1, 0], [0.3, 2],[0.3,2,-1,0]],
    z_score: Union[bool, List[bool]] = [True,False],
    save_folder: str = '/root/capsule/results',
    save_name: str = 'combined_firing_rates.csv',
    save_result: bool = False
) -> pd.DataFrame:
    """
    Computes mean firing rates for multiple sessions, combines them into one DataFrame,
    and optionally saves the result to disk.

    Parameters
    ----------
    session_names : list of str
        List of session identifiers to process.
    align_to_event : str
        Event name to align spikes to (default 'go_cue').
    time_windows : list of [start, end]
        List of time windows (in seconds relative to event) for computing firing rates.
        For example, `[[−2, 1], [1, 2], [0.2, 3, −1, 0]]` specifies three time windows.
        If a window contains four elements, like `[0.2, 3, −1, 0]`, it means that two separate mean firing rates will be calculated:
        - The mean for the first range `[0.2, 3]` and
        - The mean for the second range `[-1, 0]`.
        The first mean is then subtracted by the second mean.
    z_score : bool or list of bool
        Whether to normalize firing rates per window across trials within each session.
    save_folder : str
        Directory path to save the combined DataFrame (default '/root/capsule/results').
    save_name : str
        Filename for the saved combined DataFrame (default 'combined_firing_rates.csv').
    save_result : bool
        If True, saves the combined DataFrame to disk (default False).

    Returns
    -------
    combined_df : pd.DataFrame
        Concatenated DataFrame of firing rates across all sessions.
    """
    all_dfs = []
    for sess in session_names:
        print(f"Processing session {sess}...")
        nwb_data, tag = NWBUtils.combine_nwb(session_name=sess)
        if tag in ['none_loaded', 'behavior_only']:
            print(f"Warning: could not load session {sess}, skipping.")
            continue
        df = get_the_mean_firing_rate(
            nwb_data,
            align_to_event=align_to_event,
            time_windows=time_windows,
            z_score=z_score
        )
        try:
            nwb_data.io.close()
        except Exception:
            pass
        all_dfs.append(df)

    if not all_dfs:
        combined_df = pd.DataFrame(columns=[
            "session_id",
            "unit_index",
            "align_to_event",
            "time_window",
            "z_score",
            "brain_region",
            "ccf_location",
            "rates",
        ])
    else:
        combined_df = pd.concat(all_dfs, ignore_index=True)

    if save_result:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, save_name)
        combined_df.to_csv(save_path, index=False)
        print(f"Combined DataFrame saved to {save_path}")

    return combined_df


# Helper for parallel processing of each row
def _process_row_task(
    task: Tuple[
        int,                    # Row index in df_firing
        Dict[str, Any],         # One row of df_firing converted to dict
        pd.DataFrame,           # Behavior summary dataframe (indexed by session_id)
        List[str],              # List of correlation model names to run
        List[str],              # List of behavior variable names to correlate
        set,                    # Set of session_ids to skip (invalid or NaN)
        int                     # Total number of rows/tasks – for progress logging
    ]
) -> Tuple[int, Dict[str, Any]]:
    """
    Worker function executed in parallel to compute correlations for **one**
    (session, unit) row of *df_firing* against the requested behavior variables.

    Parameters
    ----------
    task : tuple
        A 7-element tuple with:
        0. **row_idx** (int) – Index of the current row in *df_firing*.
        1. **firing_row** (dict) – Row data (already `to_dict()`-ed) containing
           `session_id`, `unit_index`, and a trial-length list of firing **rates**.
        2. **beh_summary** (*pd.DataFrame*) – Behavior dataframe indexed by
           `session_id`; each column is a variable list per trial.
        3. **model_names** (List[str]) – Correlation/regression model names
           (e.g. `"simple_LR"`, `"ARMA_model"`).
        4. **beh_variables** (List[str]) – Names of the behavior variables to
           correlate with firing rates.
        5. **invalid_sessions** (set[str]) – Sessions that should be skipped
           because *all* requested variables are missing/NaN.
        6. **total_tasks** (int) – Total number of rows, used only for
           human-readable progress messages.

    Returns
    -------
    Tuple[int, Dict[str, Any]]
        *(row_idx, results_dict)* where **results_dict** maps
        `"{model}-{variable}" → result-dictionary` (or is empty if the row was
        skipped).
    """
    # ---------------------------------------------------------------------
    #  Unpack the tuple into descriptive local names
    # ---------------------------------------------------------------------
    (
        row_idx,
        firing_row,
        beh_summary,
        model_names,
        beh_variables,
        invalid_sessions,
        total_tasks,
    ) = task

    session_id: str = firing_row['session_id']
    unit_idx: Any  = firing_row.get('unit_index', 'unknown')

    # Prepare container for all model/variable results generated for this row
    results: Dict[str, Any] = {}

    # ---------------------------------------------------------------------
    #  Skip rows whose session lacks usable behavior data
    # ---------------------------------------------------------------------
    if session_id in invalid_sessions or session_id not in beh_summary.index:
        return row_idx, results  # Empty result – caller will leave columns as None

    # ---------------------------------------------------------------------
    #  Extract data needed for correlation
    # ---------------------------------------------------------------------
    rates: np.ndarray = np.asarray(firing_row['rates'], dtype=float)
    n_trials: int     = len(rates)

    # ---------------------------------------------------------------------
    #  Iterate over every (model, variable) pair requested
    # ---------------------------------------------------------------------
    for model_name in model_names:
        if not hasattr(methods, model_name):
            # Gracefully ignore typos / unsupported models
            continue
        regression_fn = getattr(methods, model_name)

        for beh_var in beh_variables:
            column_name = f"{model_name}-{beh_var}"
            print(f"Processing row {row_idx + 1}/{total_tasks}: "
                  f"session {session_id}, unit {unit_idx} → "
                  f"model '{model_name}', variable '{beh_var}'")

            # Pull the behavior vector and its no-response mask
            beh_series = beh_summary.at[session_id, beh_var]
            mask_col   = get_mask_trial_type(beh_var)  # e.g. 'no_response_goCue'
            mask_trials    = beh_summary.at[session_id, mask_col]

            # Build boolean mask to exclude no-response trials
            valid_mask      = np.ones(n_trials, dtype=bool)
            valid_mask[mask_trials] = False
            rates_clean     = rates[valid_mask]

            # Sanity-check alignment between behavior and firing vectors
            if (
                beh_series is None
                or (isinstance(beh_series, float) and np.isnan(beh_series))
                or not isinstance(beh_series, (list, tuple))
                or len(beh_series) != len(rates_clean)
            ):
                # Skip mis-aligned or missing data
                continue

            # -----------------------------------------------------------------
            #  Dispatch to the selected model and store its outputs
            # -----------------------------------------------------------------
            if model_name == 'simple_LR':
                res = regression_fn(rates_clean, beh_series, behavior_name=column_name)
                results[column_name] = {
                    'slope':     res.params.get(column_name, np.nan),
                    'intercept': res.params.get('const', np.nan),
                    'r_squared': getattr(res, 'rsquared_adj', np.nan),
                    'p_value':   res.pvalues.get(column_name, np.nan)
                }

            elif model_name == 'ARMA_model':
                res = regression_fn(rates_clean, beh_series,
                                    behavior_name=column_name,
                                    AR_p=3, MA_q=0)
                results[column_name] = {
                    'aic':     getattr(res, 'aic', np.nan),
                    'bic':     getattr(res, 'bic', np.nan),
                    'llf':     getattr(res, 'llf', np.nan),
                    'params':  res.params.to_dict(),
                    'pvalues': res.pvalues.to_dict(),
                }

            elif model_name == 'ARDL_model':
                res = regression_fn(rates_clean, beh_series,
                                    behavior_name=column_name,
                                    y_lag=5, x_order=0)
                results[column_name] = {
                    'aic':     getattr(res, 'aic', np.nan),
                    'bic':     getattr(res, 'bic', np.nan),
                    'llf':     getattr(res, 'llf', np.nan),
                    'params':  res.params.to_dict(),
                    'pvalues': res.pvalues.to_dict(),
                }

            elif model_name in ('cyclic_shift', 'linear_shift'):
                res = regression_fn(rates_clean, beh_series,
                                    behavior_name=column_name,
                                    ensemble_size=100, min_shift=3)
                results[column_name] = res  # These models already return a dict

            elif model_name == 'phase_randomization':
                res = regression_fn(rates_clean, beh_series,
                                    behavior_name=column_name,
                                    ensemble_size=100)
                results[column_name] = res

    # Return the row index (so the caller knows where to write) and the dict
    return row_idx, results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 1. WORKER – fit ONE model on ONE (session, unit) firing-rate row        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _multi_row_task(
    task: Tuple[
        int,                    # 0. row index in df_firing
        Dict[str, Any],         # 1. firing_row.to_dict()
        pd.DataFrame,           # 2. beh_summary (indexed by session_id)
        List[str],              # 3. variables (behaviour columns)
        str,                    # 4. model_name
        Dict[str, Any],         # 5. model_kwargs
        int,                    # 6. trial_shift  (can be 0, ±1, ±2, …)
        int,                    # 7. group_idx  (variable-group index)
        Union[int]              # 8. correlation_direction (0 or 1)
    ]
) -> Tuple[int, str, int, int, object]:
    """
    Pool-worker helper. Returns a quadruple:

        (row_idx, model_name, group_idx, result_object)

    • *result_object*  –
        statsmodels results instance … on success  
        None                           … if this row is skipped  
        {'ERROR': <msg>}               … if the fit raised an Exception
    """
    (row_idx, firing_row, beh_summary,
     variables, model_name, model_kwargs,
     shift, group_idx, correlation_direction) = task

    session_id: str     = firing_row["session_id"]
    rates_all           = np.asarray(firing_row["rates"], dtype=float)

    # ─────────────  quick exit if behaviour missing  ─────────────
    if session_id not in beh_summary.index:
        return row_idx, model_name, group_idx, shift, None

    beh_row = beh_summary.loc[session_id, variables]
    if beh_row.isna().all():
        return row_idx, model_name, group_idx, shift, None

    # ─────────────  build valid-trial mask (no-response)  ─────────
    n_trials           = len(rates_all)
    mask_valid         = np.ones(n_trials, dtype=bool)
    for v in variables:
        mask_col = get_mask_trial_type(v)
        if mask_col in beh_summary.columns:
            mask_valid[beh_summary.at[session_id, mask_col]] = False

    if not mask_valid.any():
        return row_idx, model_name, group_idx, shift, None

    rates_valid = rates_all[mask_valid]

    # ─────────────  per-variable shift & trimming  ────────────────
    beh_vectors = {}
    rates_trim  = None
    for v in variables:
        vec_full   = np.asarray(beh_row[v])
        fr_trim, bv_trim = _pairwise_apply_shift(rates_valid, vec_full, shift)

        if rates_trim is None:                  # first variable decides length
            rates_trim = fr_trim
        else:                                   # keep arrays equal length
            min_len   = min(len(rates_trim), len(fr_trim))
            rates_trim = rates_trim[:min_len]
            bv_trim    = bv_trim[:min_len]
        beh_vectors[v] = bv_trim

    if rates_trim.size == 0:
        return row_idx, model_name, group_idx, shift, correlation_direction, {"ERROR": "rates_trim size is 0"}

    # ─────────────  dispatch to the selected model  ───────────────
    try:
        model_fn = getattr(methods, model_name)
        if correlation_direction == 0:
            res = model_fn(
                fr_ts=rates_trim,
                behavior_ts=beh_vectors,
                **model_kwargs
            )
        elif correlation_direction == 1:
            if isinstance(beh_vectors, dict):
                if len(beh_vectors) == 1:
                    beh_vectors = list(beh_vectors.values())[0]
                else:
                    raise ValueError("Multiple behavior vectors detected, expected only one.")
            # Call the model function
            #print(f'beh_vectors: {beh_vectors}')
            #print(f'beh_vectors size: {len(beh_vectors)}')
            #print(f'rates_trim: {rates_trim}')
            #print(f'rates_trim size: {len(rates_trim)}')
            if len(beh_vectors) != len(rates_trim):
                print('beh_vectors size does not equal to the rates_trim size')
            res = model_fn(
                fr_ts=beh_vectors,
                behavior_ts={'firing_rate':rates_trim},
                **model_kwargs
            )

        res_serial = _stats_to_dict(res)
        res_serial["trial_shift"] = shift
        res_serial["fit_parameters"] = dict(model_kwargs)
        res_serial["fit_variables"]  = list(beh_vectors.keys())
        res_serial["correlation_direction"] = correlation_direction

        return row_idx, model_name, group_idx, shift, correlation_direction, res_serial

    except Exception as e:
        return row_idx, model_name, group_idx, shift, correlation_direction, {"ERROR": str(e)}



def _stats_to_dict(res: Any) -> Dict[str, Any]:
    """
    Convert a *statsmodels* results object – OLS, ARIMA/ARMA, ARDL, etc. –
    into a compact JSON-serialisable dict.

    • All attributes are queried with *getattr* and default to NaN if missing,
      so the same helper works for every model class.
    • For OLS / simple_LR, fields such as ``rsq``, ``rsq_adj``, ``f_stat``,
      and ``f_pvalue`` will be populated, while ARMA/ARDL keep their AIC/BIC.
    • Heavy arrays (residuals, fittedvalues, …) are intentionally omitted.
    """
    out = {
        # Model diagnostics / information criteria
        "aic":       getattr(res, "aic",       np.nan),
        "bic":       getattr(res, "bic",       np.nan),
        "hqic":      getattr(res, "hqic",      np.nan),
        "llf":       getattr(res, "llf",       np.nan),

        # Sample size & variance
        "nobs":      getattr(res, "nobs",      np.nan),
        "df_model":  getattr(res, "df_model",  np.nan),
        "df_resid":  getattr(res, "df_resid",  np.nan),
        "sigma2":    getattr(res, "sigma2",    np.nan),   # OLS only

        # Goodness-of-fit (OLS / simple_LR)
        "rsq":       getattr(res, "rsquared",       np.nan),
        "rsq_adj":   getattr(res, "rsquared_adj",   np.nan),
        "f_stat":    getattr(res, "fvalue",         np.nan),
        "f_pvalue":  getattr(res, "f_pvalue",       np.nan),

        # Coefficients & inference
        "params":    getattr(res, "params",   pd.Series(dtype=float)).to_dict(),
        "bse":       getattr(res, "bse",      pd.Series(dtype=float)).to_dict(),
        "tvalues":   getattr(res, "tvalues",  pd.Series(dtype=float)).to_dict(),
        "pvalues":   getattr(res, "pvalues",  pd.Series(dtype=float)).to_dict(),

        # Confidence intervals (if available → DataFrame → nested dict)
        "conf_int": (
            getattr(res, "conf_int", lambda: None)()
            .to_dict(orient="index")
            if callable(getattr(res, "conf_int", None)) else {}
        ),
    }

    return out


def _df_to_xarray(df: pd.DataFrame) -> xr.Dataset:
    """Return an *xarray.Dataset* with non‑primitive cells JSON‑encoded."""
    enc = df.copy()
    complex_mask = enc.applymap(lambda x: isinstance(x, (dict, list, tuple, type(None))))
    complex_cols = complex_mask.any(axis=0).index[complex_mask.any(axis=0)].tolist()
    for col in complex_cols:
        enc[col] = enc[col].apply(json.dumps)
    return xr.Dataset.from_dataframe(enc)



# ─────────────────────────────────────────────────────────────────────────────
# 0)  Utility – align two trial-length vectors with a given lag
# ─────────────────────────────────────────────────────────────────────────────
def _pairwise_apply_shift(firing: np.ndarray,
                          behaviour: np.ndarray,
                          shift: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (firing_trimmed, behaviour_trimmed) so that

        firing[t]   ↔   behaviour[t + shift]

    • Positive *shift*  →  look-ahead  (behaviour is *later* in time)
    • Negative *shift*  →  look-back   (behaviour is *earlier* in time)
    """
    if shift == 0:
        return firing, behaviour

    if shift > 0:
        # Drop last *shift* trials from firing, first *shift* from behaviour
        return firing[:-shift], behaviour[shift:]
    else:  # shift < 0
        # Drop first |shift| trials from firing, last |shift| from behaviour
        return firing[-shift:], behaviour[:shift]   # same as behaviour[:-|shift|]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 2. MAIN FRONT-END ── now accepts nested *variable* groups               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def correlate_firing_latent_multiple_variable(
    df_firing: pd.DataFrame,
    df_behavior: pd.DataFrame,
    variables: Union[List[str], List[List[str]]],
    correlation_model: Union[str, List[str], Tuple[str, ...]] = ["ARDL_model","simple_LR"],
    model_kwargs: Optional[Dict[str, Dict[str, Any]]] = {
            "simple_LR":          {"add_constant": True},
            "ARMA_model":         {"ar_order": 3, "ma_order": 0},
            "ARDL_model":         {"y_lags": 3, "x_order": 0},
        },
    trial_shifts: Union[int, List[int], Tuple[int, ...]] = [-1,0,1],  
    n_jobs: Optional[int] = None,
    save_folder: str = "/root/capsule/scratch",
    save_name: str = "correlations_multi",
    save_result: bool = False,
    save_format: str = "zarr", 
    exclude_columns: Optional[List[str]] = ["rates"],
    correlation_direction: List[int] = [0,1],
) -> pd.DataFrame | xr.Dataset:
    """
    Fit **one multivariate model per (session, unit)** for **each** requested
    model family, in parallel.

    ▶ Trial exclusion is done *per variable* using  
      ``mask_col = get_mask_trial_type(variable)``.

    ▷ Adds one new column per model:

        f"{model}-MULTI"

      whose cells hold the full statsmodels results object (or an error dict).    Parameters
    ----------
    df_firing
        Table from ``get_the_mean_firing_rate_combined`` – must include
        'session_id' and list-like 'rates'.
    df_behavior
        Behaviour summary table – must include 'session_id' and *variables*.
    variables
        • Flat *List[str]*  → one multivariate regression containing **all** vars.  
        • Nested *List[List[str]]*  → one regression **per inner list** (variable group).
          The results for each group are stored **in order** in a *list* inside the
          “<model>-MULTI” column.
    correlation_model
        One model name **or** a list/tuple, e.g.
            "ARDL_model"
            ["ARMA_model", "ARDL_model", "simple_LR"]
        Supported models:
            - "ARMA_model"
            - "ARDL_model"
            - "simple_LR"
    model_kwargs
        model_kwargs={
            "ARMA_model": {"ar_order": 2, "ma_order": 1},
            "ARDL_model": {"y_lags": 3, "x_order": 2}
        }
    n_jobs
        Worker count for ``multiprocessing.Pool``.  
        ``None`` → use ``multiprocessing.cpu_count()``.
    save_folder / save_name / save_result
        If *save_result* is True, write the augmented DataFrame to disk.
    save_format
        Default is zarr.  # "csv" | "netcdf" | "zarr"
    exclude_columns
        Optional list of columns to discard from the *result* DataFrame –
        useful to avoid serialising large arrays such as raw firing rate
        time‑series.
    trial_shifts : int | list[int] | tuple[int, ...]
        • 0     → align same trial (default, identical to old behaviour)  
        • 1     → behaviour at *t + 1* vs. firing at *t* (look-ahead)  
        • -2    → behaviour two trials **earlier** than firing  
        • [-2,0,+2] → run all three lags and store them side-by-side.
    correlation_direction : int or list[int]
        The direction(s) of the regression:
        - 0: regress the behavior variable to the neural activity
        - 1: regress the neural activity to the behavior
        If a list is provided, both 0 and 1 will be applied for each trial shift.
    Returns
    -------
    pd.DataFrame
        Copy of *df_firing* with one additional column per model.
        ───────────────────────────────────────────────────────────────────────────────
        STRUCTURE OF  result["correlation_results"]  PER ROW
        ───────────────────────────────────────────────────────────────────────────────
        row  =  (session_id × unit_index × time_window × z_score)

        "correlation_results" : dict
            ├── key = <model_name>                 # "simple_LR", "ARDL_model", …
            │
            │   value = list[ list[ dict | None ] ]
            │            ▲      ▲
            │            │      │
            │            │      └─ inner-index  s   → trial-shift **trial_shifts[s]**
            │            │         • length   == len(trial_shifts)
            │            │         • entry    = None  → fit skipped / failed
            │            │
            │            └─ outer-index  g   → variable-group **g**
            │               • length   == n_var_groups                       (order preserved)
            │
            │               EXAMPLE     trial_shifts = [-2, -1, 0, +1]
            │               ┌────────────────────────────────────────────┐
            │               │  row_dict["ARDL_model"]                    │
            │               │     ├─ g = 0  → list of 4 shifts           │
            │               │     │             • idx 0  shift -2        │
            │               │     │             • idx 1  shift -1        │
            │               │     │             • idx 2  shift  0        │
            │               │     │             • idx 3  shift +1        │
            │               │     └─ g = 1  → list of 4 shifts           │
            │               └────────────────────────────────────────────┘
            │
            └─ Each non-None element is a serialised *statsmodels* fit:
                {
                    "aic":          … ,          # ARMA / ARDL
                    "bic":          … ,
                    "hqic":         … ,
                    "llf":          … ,
                    "sigma2":       … ,          # OLS only
                    "rsq":          … ,          # OLS only
                    "rsq_adj":      … ,
                    "f_stat":       … ,
                    "f_pvalue":     … ,

                    "params":       {β₀, β₁, …},
                    "pvalues":      {β₀, β₁, …},
                    "tvalues":      {β₀, β₁, …},
                    "bse":          {β₀, β₁, …},
                    "conf_int":     {β₀: [low, high], …},

                    "fit_parameters": {kwargs actually used},
                    "fit_variables":  [var₁, var₂, …],
                    "trial_shift":    <int>,        #   NEW FIELD
                }

        Quick-access patterns
        ─────────────────────
        row_dict = df.loc[i, "correlation_results"]

        # 1) all ARDL fits for **variable-group g = 1**
        fits_g1 = row_dict["ARDL_model"][1]          # list over shifts

        # 2) fit at shift 0 for that group
        fit = fits_g1[ trial_shifts.index(0) ]

        # 3) coefficient of “RPE”
        beta_rpe = fit["params"].get("RPE", np.nan)

        # 4) its p-value
        p_rpe = fit["pvalues"].get("RPE", np.nan)

    """
    # ──────────────────────────────────────────────────────────────────
    # 0) NORMALISE  – model list  &  variable-group list
    # ──────────────────────────────────────────────────────────────────

    if isinstance(trial_shifts, (int, np.integer)):
        trial_shifts = [int(trial_shifts)]
    else:
        trial_shifts = [int(s) for s in trial_shifts]

    models = [correlation_model] if isinstance(correlation_model, str) else list(correlation_model)
    unknown = [m for m in models if not hasattr(methods, m)]
    if unknown:
        raise ValueError(f"Unknown model helper(s) in `methods`: {unknown}")

    # Accept both flat and nested lists
    if not variables:
        raise ValueError("`variables` must not be empty")

    if isinstance(variables[0], (list, tuple)):
        var_groups: List[List[str]] = [list(g) for g in variables]            # nested given
    else:
        var_groups = [list(variables)]                                        # single group

    model_kwargs = model_kwargs or {}
    exclude_columns = list(exclude_columns or [])

    # ──────────────────────────────────────────────────────────────────
    # 1) PREP  – behaviour table & output DataFrame
    # ──────────────────────────────────────────────────────────────────
    beh_summary = df_behavior.set_index("session_id")
    result      = df_firing.copy().reset_index(drop=True)

    # Pre-allocate list-typed columns (one list per row that we will append to)
    result["correlation_results"] = [{} for _ in range(len(result))]

    # ──────────────────────────────────────────────────────────────────
    # 2) BUILD TASK LIST  – one task PER (row, model, var_group)
    # ──────────────────────────────────────────────────────────────────
    tasks: List[Tuple] = []
    for idx, row in result.iterrows():
        row_dict = row.to_dict()
        for m in models:
            kwargs = model_kwargs.get(m, {})
            for g_idx, group in enumerate(var_groups):
                for shift in trial_shifts:
                    for direction in correlation_direction:
                        tasks.append(
                            (idx, row_dict, beh_summary, group, m, kwargs, shift, g_idx, direction)
                        )

    # ──────────────────────────────────────────────────────────────────
    # 3) MULTIPROCESS POOL
    # ──────────────────────────────────────────────────────────────────
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=n_jobs) as pool:
        total = len(tasks)
        for done, (row_idx, model_name, g_idx, shift, direction, res_obj) in enumerate(
            pool.imap_unordered(_multi_row_task, tasks), start=1
        ):
            store = result.at[row_idx, "correlation_results"]

            # Ensure the model is initialized in the store
            if model_name not in store:
                store[model_name] = [
                    [None] * len(trial_shifts)       # One list per shift
                    for _ in range(len(var_groups))  # One per variable-group
                ]

            # Ensure that the group index is initialized
            if g_idx >= len(store[model_name]):
                store[model_name].append([None] * len(trial_shifts))  # Append new group if needed

            # Ensure that the shift index is initialized
            shift_idx = trial_shifts.index(shift)  # This is the key line to ensure shift_idx is set correctly

            if shift_idx >= len(store[model_name][g_idx]):
                store[model_name][g_idx].append([None] * len(correlation_direction))  # Append new shift if needed

            # Ensure the direction is initialized properly, even if it is None
            if store[model_name][g_idx][shift_idx] is None:
                store[model_name][g_idx][shift_idx] = [None] * len(correlation_direction)  # Initialize with None

            # Now assign the result object to the direction
            store[model_name][g_idx][shift_idx][direction] = res_obj

            if done % 100 == 0 or done == total:
                print(f"[multi-var] progress: {done}/{total} fits completed")

    # ──────────────────────────────────────────────────────────────────
    # 4) DROP LARGE COLUMNS & OPTIONAL SAVE
    # ──────────────────────────────────────────────────────────────────
    result_out = (
        result.drop(columns=[c for c in exclude_columns if c in result.columns], errors="ignore")
    )

    ds = _df_to_xarray(result_out)  # needed even if we eventually return DataFrame

    if save_result:
        os.makedirs(save_folder, exist_ok=True)
        base = os.path.join(save_folder, save_name)

        if save_format == "csv":
            result_out.to_csv(f"{base}.csv", index=False)
            print(f"[multi-var] results saved to {base}.csv")
        elif save_format == "netcdf":
            ds.to_netcdf(f"{base}.nc")
            print(f"[multi-var] results saved to {base}.nc (NetCDF)")
        else:  # default "zarr"
            ds.to_zarr(f"{base}.zarr", mode="w")
            print(f"[multi-var] results saved to {base}.zarr (Zarr)")

    # Return the same type as before for backward compatibility
    return ds if save_format in ("netcdf", "zarr") else result_out


def get_mask_trial_type(var: str) -> str:
    """
    Return the name of the behavior column indicating which trials to exclude
    for this particular variable.
    """
    # You can extend this mapping if some variables use different exclusion columns.
    mapping = {
        # e.g. 'movement_time': 'no_movement_trials',
        # 'reaction_time': 'no_response_trials',
    }
    #return mapping.get(var, 'no_response_trials')
    return 'no_response_trials'


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SIGNIFICANCE / COEFFICIENT / T-STAT SUMMARY --  MULTI-VARIATE PIPELINE  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def correlation_results_summary(
    corr_df_or_ds: Union[pd.DataFrame, xr.Dataset, str, Path],
    meta_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Summarise the nested ``correlation_results`` column created by
    ``correlate_firing_latent_multiple_variable`` and iterate through the direction layer as well.

    NEW BEHAVIOUR  ───────────────────────────────────────────────────────────
    • The *group index* **g#** is now defined by
          (fit_variables, trial_shift)
      so a model fitted with the same variable set at different shifts
      receives different group numbers.
      This makes it trivial to plot coefficients across shifts without
      accidentally conflating them.

    COLUMN NAMING SCHEME  (hyphens everywhere)
    ──────────────────────────────────────────────────────────────────────────
        <model>-<var>-g<i>-s<k>-d<j>-pval   (minimum p across lags)
        <model>-<var>-g<i>-s<k>-d<j>-coef   (β coefficient, first lag)
        <model>-<var>-g<i>-s<k>-d<j>-tval   (t-statistic)

        <model>-g<i>-s<k>-aic          (global metrics per fit)
        <model>-g<i>-s<k>-bic
        <model>-g<i>-s<k>-rsq
        …

        fit_metadata  – dict keyed by "<model>-g<i>-s<k>-d<j>" containing
            • fit_parameters   • fit_variables   • trial_shift
    """
    # 1) Load input → df_raw  (exact code you asked for)
    if isinstance(corr_df_or_ds, (str, Path)) and str(corr_df_or_ds).endswith(".zarr"):
        ds = xr.open_zarr(corr_df_or_ds, consolidated=False)
        df_raw = ds.to_dataframe()
        for c in df_raw.columns:
            if df_raw[c].dtype == object:
                try:
                    df_raw[c] = df_raw[c].apply(json.loads)
                except Exception:
                    pass
    elif isinstance(corr_df_or_ds, xr.Dataset):
        df_raw = corr_df_or_ds.to_dataframe()
        for c in df_raw.columns:
            if df_raw[c].dtype == object:
                try:
                    df_raw[c] = df_raw[c].apply(json.loads)
                except Exception:
                    pass
    elif isinstance(corr_df_or_ds, pd.DataFrame):
        df_raw = corr_df_or_ds.copy()
    else:
        raise TypeError("Unsupported input type for `corr_df_or_ds`")

    if "correlation_results" not in df_raw.columns:
        raise ValueError("Input missing 'correlation_results' column")

    # 2) Metadata scaffold
    if meta_cols is None:
        meta_cols = ["session_id", "unit_index", "time_window", "z_score", "ccf_location", "brain_region"]

    meta_df = df_raw[meta_cols].copy()
    meta_df["fit_metadata"] = [{} for _ in range(len(meta_df))]

    # 3) Collect new columns row-by-row in dicts (no fragmentation)
    new_rows: List[Dict[str, Any]] = [{} for _ in range(len(meta_df))]
    group_index_map: Dict[str, Dict[Tuple[Any, ...], int]] = defaultdict(dict)

    for row_idx, row in df_raw.iterrows():
        corr_dict: Dict[str, List[List[Dict[str, Any]]]] = row["correlation_results"]
        meta_blob: Dict[str, Any] = {}
        values = new_rows[row_idx]

        for model, group_list in corr_dict.items():
            for shift_list in (group_list or []):
                for direction_list in (shift_list or []):
                    if direction_list is None:
                        print(f"Skipping empty direction_list for model: {model}, shift_list: {shift_list}")
                        continue  # Skip to the next valid direction_list
                    
                    # Iterate over the direction layer
                    for fit in direction_list:
                        if not fit or "params" not in fit:
                            continue
                        direction = int(fit.get("correlation_direction"))
                        shift_val = int(fit.get("trial_shift", 0))
                        shift_tag = f"s{shift_val:+d}".replace("+", "")
                        var_tuple = tuple(fit.get("fit_variables", []))
                        group_key = var_tuple      # shift included
                        if group_key not in group_index_map[model]:
                            group_index_map[model][group_key] = len(group_index_map[model])
                        g_idx = group_index_map[model][group_key]

                        params: Dict[str, float] = fit["params"]
                        pvalues: Dict[str, float] = fit["pvalues"]
                        tvalues: Dict[str, float] = fit.get("tvalues", {})

                        # ── behaviour-specific metrics ─────────────────
                        base_vars = {k.split(".")[0] for k in params if k.split(".")[0] != "const"}
                        for var in base_vars:
                            lag_keys = [k for k in params if k.startswith(var)]
                            if not lag_keys:
                                continue
                            beta = params[lag_keys[0]]
                            p_min = min(pvalues.get(k, np.nan) for k in lag_keys)
                            t_val = tvalues.get(lag_keys[0], np.nan)

                            values[f"{model}-{var}-g{g_idx}-{shift_tag}-d{direction}-pval"] = p_min
                            values[f"{model}-{var}-g{g_idx}-{shift_tag}-d{direction}-coef"] = beta
                            values[f"{model}-{var}-g{g_idx}-{shift_tag}-d{direction}-tval"] = t_val

                        # ── global fit metrics ─────────────────────────
                        for metric in ("rsq", "rsq_adj", "aic", "bic", "hqic",
                                    "llf", "sigma2", "f_stat", "f_pvalue"):
                            values[f"{model}-g{g_idx}-{shift_tag}-d{direction}-{metric}"] = fit.get(metric, np.nan)

                        # ── metadata per fit ───────────────────────────
                        meta_key = f"{model}-g{g_idx}-{shift_tag}-d{direction}"
                        meta_blob[meta_key] = {
                            "fit_parameters": fit.get("fit_parameters", {}),
                            "fit_variables": list(var_tuple),
                            "trial_shift": shift_val,
                            "direction": direction,
                        }

        meta_df.at[row_idx, "fit_metadata"] = meta_blob

    # 4) One big concat → no fragmentation
    wide_df = pd.DataFrame(new_rows, index=meta_df.index, dtype=float)
    result = pd.concat([meta_df, wide_df], axis=1)

    return result





# ──────────────────────────────────────────────────────────────────────────
# 2)  BATCH WRAPPER  →  CSV  *or*  Zarr
# ──────────────────────────────────────────────────────────────────────────
def correlation_results_summary_combined(
    paths: Optional[List[Union[str, Path]]] = None,
    search_folder: str = "/root/capsule/scratch",
    pattern: str = "correlations_multi*.zarr",
    save_folder: str = "/root/capsule/scratch",
    save_name: str = "sig_dir_all_sessions",
    save_result: bool = False,
    save_format: str = "zarr",
    meta_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Batch version of ``correlation_results_summary`` that accepts
    **multiple correlation result folders (.zarr)** produced by
    ``correlate_firing_latent_multiple_variable``,
    converts each into a significance / direction table, and concatenates
    the lot.

    ─────────────────────────────────────────────────────────────────────
    PARAMETERS
    ─────────────────────────────────────────────────────────────────────
    paths : list[str | pathlib.Path] | None
        Explicit list of Zarr directories (or legacy CSVs) to process.
        • If *None*, files are discovered with *pattern* inside *search_folder*.
        • Mixed lists are allowed – every entry is auto-routed to the correct
          parser (multi-var Zarr → new routine, legacy CSV → old routine).

    search_folder : str
        Directory that is searched when *paths* is **None**.

    pattern : str
        Glob pattern used for auto-discovery.  **Default**:
            ``'correlations_multi*.zarr'``

    save_folder, save_name, save_result
        Control optional saving of the concatenated DataFrame.
        • When *save_result* is True the table is written as CSV:
            ``<save_folder>/<save_name>``
    save_format
        • *save_format*  may be **'csv'** (default) or **'zarr'**.  
          When 'zarr' is chosen the combined table is written to  
          ``<save_folder>/<save_name>.zarr`` in *consolidated* mode.
    meta_cols : list[str] | None
        Metadata columns to preserve verbatim.  If *None* the default set
        ``['session_id', 'unit_index', 'time_window', 'z_score','ccf_location','brain_region']`` is used.

    ─────────────────────────────────────────────────────────────────────
    RETURNS
    ─────────────────────────────────────────────────────────────────────
    pd.DataFrame
        Combined significance / direction table from **all** supplied result
        files / folders.  Rows are sorted by:

            source_file  ·  session_id  ·  unit_index
    """

    # ........................................................................
    # 1)  Discover / validate file list
    # ........................................................................
    if paths is None:
        paths = sorted(Path(search_folder).expanduser().glob(pattern))
    if not paths:
        raise FileNotFoundError("No correlation result files discovered.")

    print(f"Found {len(paths)} correlation result file(s)/folder(s):")
    for p in paths:
        print(f"  • {p}")

    # ........................................................................
    # 2)  Per-file conversion
    # ........................................................................
    frames = []
    for i, p in enumerate(paths, start=1):
        print(f"\n[ {i}/{len(paths)} ]  Processing {p} …")
        tbl = correlation_results_summary(p, meta_cols=meta_cols)
        tbl.insert(0, "source_file", Path(p).name)
        frames.append(tbl)

    combined = (
        pd.concat(frames, ignore_index=True)
          .sort_values(["source_file", "session_id", "unit_index"])
          .reset_index(drop=True)
    )

    # ........................................................................
    # 3)  Optional save
    # ........................................................................
    if save_result:
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        if save_format == "csv":
            out = Path(save_folder) / f"{save_name}.csv"
            combined.to_csv(out, index=False)
            print(f"[multi-var] combined summary written to {out}")

        elif save_format == "zarr":
            out = Path(save_folder) / f"{save_name}.zarr"

            # JSON-encode list / dict cells so xarray can serialise them
            enc = combined.copy()
            for col in enc.columns:
                if enc[col].apply(lambda x: isinstance(x, (dict, list, tuple))).any():
                    enc[col] = enc[col].apply(json.dumps)

            xr.Dataset.from_dataframe(enc).to_zarr(out, mode="w", consolidated=True)
            print(f"[multi-var] combined summary written to {out}")

        else:
            raise ValueError("`save_format` must be 'csv' or 'zarr'")

    return combined


def extract_columns_by_filters(
    corr_df: pd.DataFrame,
    col_names: Union[str, List[str]],
    filters: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Select and return one or more columns from the correlation results
    DataFrame based on an arbitrary set of equality filters.

    Parameters
    ----------
    corr_df : pd.DataFrame
        The wide-format correlation results table. Must include any columns
        you intend to filter on plus the target `col_names`.
    col_names : str or list of str
        Name or list of names of the column(s) to extract, e.g.
        'simple_LR-...-pval' or
        ['simple_LR-...-pval', 'ARDL_model-...-coef'].
    filters : dict, optional
        A mapping {column_name: value} that each row must match. For example:
            {
              'source_file': 'sig_dir_all_sessions.zarr',
              'time_window': '-1_0',
              'z_score': True
            }
        If `None` or empty, no filtering is applied.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the requested columns from all rows
        matching the filters.  If you pass a single string as `col_names`,
        the result will still be a one-column DataFrame.

    Raises
    ------
    ValueError
        If any filter key or any requested column name is not present in corr_df.
    """
    # normalize col_names to list
    names = [col_names] if isinstance(col_names, str) else list(col_names)

    # apply filters
    df = corr_df
    if filters:
        for key, val in filters.items():
            if key not in df.columns:
                raise ValueError(f"Filter column '{key}' not found in DataFrame")
            df = df[df[key] == val]

    # check requested columns exist
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise ValueError(f"Requested column(s) not found: {missing}")

    return df[names]

def project_psth_per_trial(
    nwb_data: Any,
    corr_df: pd.DataFrame,
    filters: Dict[str, Any] = {"time_window": "-1_0", "z_score": False},
    coef_col: str = "simple_LR-ForagingCompareThreshold-value-1-g2-s0-d0-coef",
    trial_type: str = "switch_trial",
    align_to_event: str = "go_cue",
    time_window: Tuple[float, float] = (-3.0, 5),
    bin_size: float = 0.05,
    coef_direction: Optional[str] = "both",
    p_value: Optional[float] = 0.05
) -> pd.DataFrame:
    """
    Compute a per-trial PSTH for each unit, then project onto a coefficient vector
    or onto the p-values (if `p_value` thresholding is used).

    Parameters
    ----------
    nwb_data : Any
        An NWB file handle containing an ephys `units` DynamicTable with a
        'spike_times' column.
    corr_df : pd.DataFrame
        Wide-format correlation summary DataFrame (must include `unit_index`
        and the coefficient & p-value columns).
    filters : Dict[str, Any], optional
        Equality filters passed to `extract_columns_by_filters` to select
        subset of units. Defaults to {"time_window": "-1_0", "z_score": False}.
    coef_col : str, optional
        Name of the coefficient column in `corr_df`. Must end with "-coef".
    trial_type : str, optional
        Trial-type string passed to `find_trials`. Defaults to "switch_trial".
    align_to_event : str, optional
        Event name to align spikes to. Defaults to "go_cue".
    time_window : Tuple[float, float], optional
        Start and end (sec, relative to event) of PSTH window. Defaults to (-3, 5).
    bin_size : float, optional
        Width of each PSTH bin in seconds. Defaults to 0.05.
    coef_direction : str, optional
        One of:
          - "positive": keep only units with coef_col > 0
          - "negative": keep only units with coef_col < 0
          - "both" (default) or None: keep all units
    p_value : float, optional
        If set, threshold on the corresponding p-value column (derived by
        replacing '-coef' with '-pval' in coef_col). Only units with pval < p_value
        are kept, and weights = their p-values.

    Returns
    -------
    pd.DataFrame
        One row per selected trial with columns:
          - source_file, trial_idx, event_time,
          - psth_bins (np.ndarray), projection (np.ndarray),
          - filters, trial_type, align_to_event,
          - time_window, bin_size, coef_col,
          - coef_direction, p_value
    """
    session_base = Path(nwb_data.session_id).stem

    # Validate coef_col and derive pval column
    if not coef_col.endswith("-coef"):
        raise ValueError(f"coef_col must end with '-coef', got {coef_col!r}")
    pval_col = coef_col.replace("-coef", "-pval")

    # 1) extract unit → coef & p-val mapping
    col_names = ["unit_index", coef_col, pval_col, "source_file"]
    coef_tbl = extract_columns_by_filters(
        corr_df=corr_df,
        col_names=col_names,
        filters=filters
    )
    coef_tbl = coef_tbl[coef_tbl["source_file"].str.contains(session_base)]

    # 1a) apply coefficient-direction filter
    if coef_direction == "positive":
        coef_tbl = coef_tbl[coef_tbl[coef_col] > 0]
    elif coef_direction == "negative":
        coef_tbl = coef_tbl[coef_tbl[coef_col] < 0]
    elif coef_direction in (None, "both"):
        pass
    else:
        raise ValueError("coef_direction must be 'positive', 'negative', 'both', or None")

    # 1b) apply p-value threshold and switch to p-value weights if needed
    weight_col = coef_col
    if p_value is not None:
        coef_tbl = coef_tbl[coef_tbl[pval_col] < p_value]
        weight_col = pval_col

    # If no units remain, return empty DataFrame with correct columns
    out_cols = [
        "source_file", "trial_idx", "event_time",
        "psth_bins", "projection",
        "n_units",
        "filters", "trial_type", "align_to_event",
        "time_window", "bin_size", "coef_col",
        "coef_direction", "p_value"
    ]

    if coef_tbl.empty:
        return pd.DataFrame(columns=out_cols)

    # Build weight map and unit list
    coef_map = dict(zip(coef_tbl["unit_index"], coef_tbl[weight_col]))
    units_to_use = sorted(coef_map.keys())

    # 2) find trials & event times
    all_event_times = extract_event_timestamps(nwb_data, align_to_event)
    keep_trials = find_trials(nwb_data, trial_type)
    event_times = np.array(all_event_times)[keep_trials]

    # 3) define PSTH bins
    start, end = time_window
    edges = np.arange(start, end + bin_size, bin_size)
    centers = edges[:-1] + bin_size / 2.0
    n_bins = len(centers)

    # 4) build 3D PSTH array (units × trials × bins)
    n_units = len(units_to_use)
    n_trials = len(keep_trials)
    psth = np.zeros((n_units, n_trials, n_bins), dtype=float)
    tbl = nwb_data.units

    for ui, u in enumerate(units_to_use):
        spikes = np.array(tbl["spike_times"][u])
        for ti, t_idx in enumerate(keep_trials):
            t0 = all_event_times[t_idx]
            rel = spikes - t0
            counts, _ = np.histogram(rel, bins=edges)
            psth[ui, ti, :] = counts / bin_size

    # 5) project across units using chosen weights
    coefs = np.array([coef_map[u] for u in units_to_use])[:, None, None]
    proj = np.sum(psth * coefs, axis=0)  # shape (n_trials, n_bins)

    # 6) assemble output DataFrame
    rows: List[Dict[str, Any]] = []
    for ti, t_idx in enumerate(keep_trials):
        rows.append({
            "source_file":    coef_tbl["source_file"].iat[0],
            "trial_idx":      t_idx,
            "event_time":     event_times[ti],
            "psth_bins":      centers,
            "projection":     proj[ti, :],
            "n_units":        n_units,
            "filters":        filters,
            "trial_type":     trial_type,
            "align_to_event": align_to_event,
            "time_window":    time_window,
            "bin_size":       bin_size,
            "coef_col":       coef_col,
            "coef_direction": coef_direction,
            "p_value":        p_value
        })

    return pd.DataFrame(rows, columns=out_cols)



