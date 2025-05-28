import os
import glob
import json
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Optional, Union, Tuple, Dict
import multiprocessing


from behavior_utils import extract_event_timestamps 
from general_utils import extract_session_name_core
from nwb_utils import NWBUtils
from general_utils import smart_read_csv
from aind_spurious_correlation import methods


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
    time_windows: List[List[float]] = [[-1, 0],[0,2]],
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
        Windows (seconds relative to event) for computing firing rates; e.g., [[-2,1],[1,2]].
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
    session_id = extract_session_name_core(getattr(nwb_data, 'session_id', None))
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
        for start_offset, end_offset in time_windows:
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
                    'time_window': f"{start_offset}_{end_offset}",
                    'z_score': flag,
                    'rates': rates_to_store.tolist()
                })

    firing_rate_df = pd.DataFrame(rows, columns=['session_id', 'unit_index', 'time_window','z_score', 'rates'])
    return firing_rate_df

def get_the_mean_firing_rate_combined(
    session_names: List[str],
    align_to_event: str = 'go_cue',
    time_windows: List[List[float]] = [[-1, 0], [0.3, 2]],
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
        Windows (seconds relative to event) for computing firing rates.
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
        combined_df = pd.DataFrame(columns=['session_id', 'unit_index', 'time_window', 'z_score', 'rates'])
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
        int,                    # 0. row_idx  – index of this row in df_firing
        Dict[str, Any],         # 1. firing_row – row.to_dict(); incl. session_id, rates …
        pd.DataFrame,           # 2. beh_summary – behaviour table, indexed by session_id
        List[str],              # 3. variables – behaviour columns to include
        str,                    # 4. model_name – "ARMA_model", "ARDL_model", …
        Dict[str, Any],         # 5. model_kwargs – extra kwargs forwarded to that model
        int                     # 6. group_idx   – sequential index of the var-group
    ]
) -> Tuple[int, str, int, object]:
    """
    Pool-worker helper.  Returns a quadruple:

        (row_idx, model_name, group_idx, result_object)

    • *result_object*  –
        statsmodels results instance   … on success  
        None                           … if this row is skipped  
        {'ERROR': <msg>}               … if the fit raised an Exception
    """
    (row_idx, firing_row, beh_summary,
     variables, model_name, model_kwargs, group_idx) = task

    session_id: str   = firing_row["session_id"]
    rates: np.ndarray = np.asarray(firing_row["rates"], dtype=float)
    n_trials: int     = len(rates)

    # ──────────────────────────────────────────────────────────────────
    # 1) QUICK EXITS  – no behaviour data, or all NaN
    # ──────────────────────────────────────────────────────────────────
    if session_id not in beh_summary.index:
        return row_idx, model_name, group_idx, None

    beh_row = beh_summary.loc[session_id, variables]

    if beh_row.isna().all():
        return row_idx, model_name, group_idx, None

    # ──────────────────────────────────────────────────────────────────
    # 2) BUILD BEHAVIOUR DICT  +  GLOBAL VALID-TRIAL MASK
    # ──────────────────────────────────────────────────────────────────
    beh_dict   = {v: beh_row[v] for v in variables}
    mask_valid = np.ones(n_trials, dtype=bool)

    for v in variables:
        mask_col = get_mask_trial_type(v)
        if mask_col in beh_summary.columns:
            mask_valid[beh_summary.at[session_id, mask_col]] = False
        # mask_valid &= ~pd.isna(beh_dict[v])   # keep if desired

    if not mask_valid.any():
        return row_idx, model_name, group_idx, None

    rates_clean = rates[mask_valid]

    # ──────────────────────────────────────────────────────────────────
    # 3) FIT THE REQUESTED MODEL
    # ──────────────────────────────────────────────────────────────────
    try:
        model_fn = getattr(methods, model_name)
        res = model_fn(
            fr_ts=rates_clean,
            behavior_ts=beh_dict,
            **model_kwargs
        )
        res_clean = _stats_to_dict(res)
        res_clean["fit_parameters"] = dict(model_kwargs)
        res_clean["fit_variables"] = variables
        return row_idx, model_name, group_idx, res_clean
    except Exception as e:
        return row_idx, model_name, group_idx, {"ERROR": str(e)}




def _stats_to_dict(res: Any) -> Dict[str, Any]:
    """Serialize a *statsmodels* result object to a compact, JSON‑safe dict.

    The helper is purposely generic – it tries to pull a wide set of common
    attributes if they exist and falls back to *NaN* when not. Heavy arrays
    (e.g. residuals) are omitted to keep file size under control.
    """
    return {
        # Information criteria & log‑likelihood
        "aic":       getattr(res, "aic", np.nan),
        "bic":       getattr(res, "bic", np.nan),
        "hqic":      getattr(res, "hqic", np.nan),
        "llf":       getattr(res, "llf", np.nan),
        # Model / sample size
        "nobs":      getattr(res, "nobs", np.nan),
        "df_model":  getattr(res, "df_model", np.nan),
        "df_resid":  getattr(res, "df_resid", np.nan),
        "sigma2":    getattr(res, "sigma2", np.nan),
        # Goodness‑of‑fit (if available, e.g. ARDL)
        "rsq":       getattr(res, "rsquared", np.nan),
        "rsq_adj":   getattr(res, "rsquared_adj", np.nan),
        # Coefficients & inference
        "params":    getattr(res, "params", pd.Series(dtype=float)).to_dict(),
        "bse":       getattr(res, "bse", pd.Series(dtype=float)).to_dict(),
        "tvalues":   getattr(res, "tvalues", pd.Series(dtype=float)).to_dict(),
        "pvalues":   getattr(res, "pvalues", pd.Series(dtype=float)).to_dict(),
        # Confidence intervals if present (DataFrame → nested dict)
        "conf_int": (
            getattr(res, "conf_int", lambda: None)()
            .to_dict(orient="index")
            if callable(getattr(res, "conf_int", None)) else {}
        ),
    }


def _df_to_xarray(df: pd.DataFrame) -> xr.Dataset:
    """Return an *xarray.Dataset* with non‑primitive cells JSON‑encoded."""
    enc = df.copy()
    complex_mask = enc.applymap(lambda x: isinstance(x, (dict, list, tuple, type(None))))
    complex_cols = complex_mask.any(axis=0).index[complex_mask.any(axis=0)].tolist()
    for col in complex_cols:
        enc[col] = enc[col].apply(json.dumps)
    return xr.Dataset.from_dataframe(enc)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 2. MAIN FRONT-END ── now accepts nested *variable* groups               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def correlate_firing_latent_multiple_variable(
    df_firing: pd.DataFrame,
    df_behavior: pd.DataFrame,
    variables: Union[List[str], List[List[str]]],
    correlation_model: Union[str, List[str], Tuple[str, ...]] = "ARDL_model",
    model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    n_jobs: Optional[int] = None,
    save_folder: str = "/root/capsule/results",
    save_name: str = "correlations_multi",
    save_result: bool = False,
    save_format: str = "zarr", 
    exclude_columns: Optional[List[str]] = ["rates"],
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
            "ARMA_model"
            ["ARMA_model", "ARDL_model"]
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
    Returns
    -------
    pd.DataFrame
        Copy of *df_firing* with one additional column per model.
    """
    # ──────────────────────────────────────────────────────────────────
    # 0) NORMALISE  – model list  &  variable-group list
    # ──────────────────────────────────────────────────────────────────
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
    for m in models:
        result[f"{m}-MULTI"] = [[] for _ in range(len(result))]

    # ──────────────────────────────────────────────────────────────────
    # 2) BUILD TASK LIST  – one task PER (row, model, var_group)
    # ──────────────────────────────────────────────────────────────────
    tasks: List[Tuple] = []
    for idx, row in result.iterrows():
        row_dict = row.to_dict()
        for m in models:
            kwargs = model_kwargs.get(m, {})
            for g_idx, group in enumerate(var_groups):
                tasks.append((idx, row_dict, beh_summary, group, m, kwargs, g_idx))

    # ──────────────────────────────────────────────────────────────────
    # 3) MULTIPROCESS POOL
    # ──────────────────────────────────────────────────────────────────
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=n_jobs) as pool:
        total = len(tasks)
        for done, (row_idx, model_name, g_idx, res_obj) in enumerate(
            pool.imap_unordered(_multi_row_task, tasks), start=1
        ):
            # Append result in the correct order
            result.at[row_idx, f"{model_name}-MULTI"].append(res_obj)

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


def correlate_firing_latent(
    df_firing: pd.DataFrame,
    df_behavior: pd.DataFrame,
    variables: List[str],
    correlation_model: Union[str, List[str]] = ('simple_LR', 'ARMA_model'),
    save_folder: str = '/root/capsule/results',
    save_name: str = 'correlations.csv',
    save_result: bool = False
) -> pd.DataFrame:
    """
    For each entry in df_firing, regress its firing rates against each specified
    behavior time series under one or more models. Appends one new column per
    (model, variable) pair; each cell is a dict with keys
    'slope', 'intercept', 'r2', 'pvalue' (for simple_LR) or other results for other models.

    Parameters
    ----------
    df_firing : pd.DataFrame
        Must contain ['session_id','unit_index','time_window','z_score','rates'].
        'rates' is a list-like of floats (one per trial). Output from get_the_mean_firing_rate_combined.
    df_behavior : pd.DataFrame
        Must contain 'session_id', each name in `variables` as list-like, and
        variable-specific no-response trial lists. Output from generate_behavior_summary_combined.
    variables : List[str]
        Names of the behavior columns to correlate.
    correlation_model : Union[str, List[str]]
        Model name or list of names of functions in aind_spurious_correlation.methods.
        Supported models:
          - 'simple_LR'
          - 'ARMA_model'
          - 'ARDL_model'
          - 'cyclic_shift'
          - 'linear_shift'
          - 'phase_randomization'
    save_folder : str
        Directory to write the CSV if `save_result` is True.
    save_name : str
        Filename (with .csv) to use when saving.
    save_result : bool
        If True, writes the returned DataFrame to disk.

    Returns
    -------
    pd.DataFrame
        Copy of df_firing with additional dict-valued columns. Column names:
        `{model}_{variable}`.
    """
    # ---------------------------------------------------------------------
    # 1. Normalize arguments
    # ---------------------------------------------------------------------
    models = [correlation_model] if isinstance(correlation_model, str) else list(correlation_model)

    # ---------------------------------------------------------------------
    # 2. Identify sessions whose *all* requested variables are missing/NaN
    # ---------------------------------------------------------------------
    beh = df_behavior.set_index('session_id')
    invalid_sessions: set[str] = {
        sess for sess, row in beh[variables].iterrows()
        if all((val is None) or (isinstance(val, float) and np.isnan(val)) for val in row)
    }

    # ---------------------------------------------------------------------
    # 3. Prepare output frame
    # ---------------------------------------------------------------------
    result = df_firing.copy().reset_index(drop=True)
    for model in models:
        for var in variables:
            result[f"{model}-{var}"] = None

    # ---------------------------------------------------------------------
    # 4. Build tasks for parallel execution
    # ---------------------------------------------------------------------
    tasks = []
    total_tasks = len(result)
    for i, row in result.iterrows():
        tasks.append((i, row.to_dict(), beh, models, variables, invalid_sessions, total_tasks))

    # ---------------------------------------------------------------------
    # 5. Execute in parallel with progress tracking
    # ---------------------------------------------------------------------
    n_jobs = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=n_jobs) as pool:
        for count, (i, row_results) in enumerate(pool.imap_unordered(_process_row_task, tasks), start=1):
            print(f"Progress: {count}/{total_tasks} tasks completed")
            for col, val in row_results.items():
                result.at[i, col] = val

    # ---------------------------------------------------------------------
    # 6. Optional save to disk
    # ---------------------------------------------------------------------
    if save_result:
        os.makedirs(save_folder, exist_ok=True)
        out_path = os.path.join(save_folder, save_name)
        result.to_csv(out_path, index=False)
        print(f"Correlation results saved to {out_path}")

    return result


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


def significance_and_direction_summary(
    corr_df: pd.DataFrame,
    p_threshold: Union[float, List[float]] = 0.05,   # kept for API-compat but no longer used
    meta_cols: List[str] | None = None
) -> pd.DataFrame:
    """
    Return a **trimmed DataFrame** that keeps only
    
        ┌─────────────────────────────────────────────────────────────┐
        │ session_id | unit_index | time_window | z_score |           │
        │ <orig>_pval | <orig>_dir     (for every correlation column) │
        └─────────────────────────────────────────────────────────────┘
    
    where *<orig>* is each original correlation column in ``corr_df``.
    
    * ``<orig>_pval``  – extracted *p*-value (``NaN`` if it cannot be found)  
    * ``<orig>_dir``   – the sign of the behavioural coefficient:
        'positive' · 'negative' · 'none' · 'unknown'
    
    ────────────────────────────────────────────────────────────────────
    HOW *p*-VALUE & COEFFICIENT (β) ARE EXTRACTED
    ────────────────────────────────────────────────────────────────────
    
    ╭────────────────────┬────────────────────────────────────────────────╮
    │ Model family       │ Extraction rule                                │
    ├────────────────────┼────────────────────────────────────────────────┤
    │ simple_LR          │ {'slope', 'p_value'}                           │
    │                    │   pval = cell['p_value']                       │
    │                    │   β    = cell['slope']                         │
    │                    │                                                │
    │ ARMA_model         │ {'params', 'pvalues'}                          │
    │                    │   pval = cell['pvalues'][col_name]             │
    │                    │   β    = cell['params'][col_name]              │
    │                    │                                                │
    │ ARDL_model         │ same keys but lagged '<col>.L0', '.L1', …      │
    │                    │   pval = **min** p across matching lag keys    │
    │                    │   β    = first matching coefficient            │
    │                    │                                                │
    │ phase_randomization│ cell is a percentile (0–100)                   │
    │ cyclic_shift       │   pval = 2·min(pct, 100−pct) / 100             │
    │ linear_shift       │ same as above; β reported as NaN → 'unknown'   │
    ╰────────────────────┴────────────────────────────────────────────────╯
    
    Direction (dir) is assigned as:
        • 'positive'  if β > 0  
        • 'negative'  if β < 0  
        • 'none'      if β == 0 or β is NaN  
        • 'unknown'   if β is not available
    
    Parameters
    ----------
    corr_df : pd.DataFrame
        Output of ``correlate_firing_latent`` – a table of correlation results
        (dicts or numeric values) plus identifying metadata.
    p_threshold : float | list[float], optional
        Ignored in this version (left for backward-compatibility).
    meta_cols : list[str] | None, optional
        Metadata columns to preserve verbatim.  Defaults to
        ``['session_id', 'unit_index', 'time_window', 'z_score']``.
    
    Returns
    -------
    pd.DataFrame
        The four metadata columns +
        one ``_pval`` and one ``_dir`` column for every correlation metric.
    """
    # ─────────────────────────────────────────────────────────────── #
    # 1) Figure out which columns belong to metadata and which are    #
    #    correlation results.                                         #
    # ─────────────────────────────────────────────────────────────── #
    if meta_cols is None:
        meta_cols = ["session_id", "unit_index", "time_window", "z_score"]

    # Start the output with metadata only.
    summary = corr_df[meta_cols].copy()

    # Every non-meta, non-'rates' column is treated as a correlation fit.
    corr_cols = [
        col for col in corr_df.columns
        if col not in meta_cols and col != "rates"
    ]

    # ─────────────────────────────────────────────────────────────── #
    # 2) Helper: given a single cell + its column name, return        #
    #       (p_value, beta)                                           #
    # ─────────────────────────────────────────────────────────────── #
    def _extract_p_beta(cell: object, col_name: str) -> tuple[float, float]:
        """
        Handle all model families and return a tuple:
            (p_value, beta)
        beta may be NaN if not available.
        """
        # — simple_LR —------------------------------------------------
        if isinstance(cell, dict) and {"slope", "p_value"} <= cell.keys():
            return cell["p_value"], cell["slope"]

        # — ARMA / ARDL —---------------------------------------------
        if isinstance(cell, dict) and {"params", "pvalues"} <= cell.keys():
            params, pvals = cell["params"], cell["pvalues"]

            if col_name in params:            # ARMA: exact key present
                return pvals.get(col_name, np.nan), params[col_name]

            # ARDL: need to look for lagged keys that start with col_name
            lag_keys = [k for k in params if k.startswith(col_name)]
            if lag_keys:
                p_min = min(pvals.get(k, np.nan) for k in lag_keys)
                return p_min, params[lag_keys[0]]

            return np.nan, np.nan             # no matching key

        # — permutation / shift — numeric percentile —----------------
        if isinstance(cell, (float, int, np.floating)):
            percentile = float(cell)          # 0–100
            p_two_tailed = 2 * min(percentile, 100 - percentile) / 100.0
            return p_two_tailed, np.nan       # beta unknown

        # — fallback — unknown cell type —----------------------------
        return np.nan, np.nan

    # ─────────────────────────────────────────────────────────────── #
    # 3) Loop through each correlation column and append two new cols #
    # ─────────────────────────────────────────────────────────────── #
    for col in corr_cols:
        # Vectorised extraction of p-values and betas for this column
        p_series, beta_series = zip(*corr_df[col].apply(_extract_p_beta, col_name=col))
        p_array   = np.asarray(p_series, dtype=float)
        beta_array = np.asarray(beta_series, dtype=float)

        # Store p-values
        summary[f"{col}_pval"] = p_array

        # Determine direction as per rules
        dir_labels = np.where(
            np.isnan(beta_array), "unknown",
            np.where(beta_array > 0, "positive",
                     np.where(beta_array < 0, "negative", "none"))
        )
        summary[f"{col}_dir"] = dir_labels

    return summary

def significance_and_direction_summary_combined(
    csv_paths: Optional[List[str]] = None,
    search_folder: str = '/root/capsule/results',
    pattern: str = 'correlations-*.csv',
    save_folder: str = '/root/capsule/results',
    save_name: str = 'sig_dir_all_sessions.csv',
    save_result: bool = False,
    p_threshold: Union[float, List[float]] = 0.05,
    meta_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Batch version of ``significance_and_direction_summary`` that accepts
    multiple correlation CSVs, reads them with *smart_read_csv*, converts
    each into a significance / direction table, and concatenates the lot.

    Parameters
    ----------
    csv_paths : list[str] | None
        Explicit list of CSV files to process.  If None, files are discovered
        with *pattern* inside *search_folder*.
    search_folder : str
        Directory used when *csv_paths* is not supplied.
    pattern : str
        Glob pattern for auto-discovery (default 'correlations_ephys-*.csv').
    save_folder, save_name, save_result
        Controls for optional saving of the combined DataFrame.
    p_threshold, meta_cols
        Forwarded to ``significance_and_direction_summary``.

    Returns
    -------
    pd.DataFrame
        Combined significance / direction table from all input files.
    """
    # ────────────────────────────────────────────────
    # 1) Resolve which CSVs to load
    # ────────────────────────────────────────────────
    if csv_paths is None:
        csv_paths = sorted(
            str(p) for p in Path(search_folder).expanduser().glob(pattern)
        )

    if not csv_paths:
        raise FileNotFoundError(
            "No correlation CSV files found – "
            "check *csv_paths*, *search_folder*, or *pattern*."
        )

    print(f"Found {len(csv_paths)} correlation file(s):")
    for p in csv_paths:
        print(f"  – {p}")

    # ────────────────────────────────────────────────
    # 2) Convert each CSV → summary DataFrame
    # ────────────────────────────────────────────────
    summaries = []
    for p in csv_paths:
        corr_df = smart_read_csv(p)                      # ←  CHANGED LINE
        summary_df = significance_and_direction_summary(
            corr_df,
            p_threshold=p_threshold,
            meta_cols=meta_cols
        )
        summary_df.insert(0, 'source_file', os.path.basename(p))
        summaries.append(summary_df)

    combined = (
        pd.concat(summaries, ignore_index=True)
          .sort_values(['source_file', 'session_id', 'unit_index'])
          .reset_index(drop=True)
    )

    # ────────────────────────────────────────────────
    # 3) Optional save
    # ────────────────────────────────────────────────
    if save_result:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        out_path = Path(save_folder) / save_name
        combined.to_csv(out_path, index=False)
        print(f"Combined summary saved to {out_path}")

    return combined


