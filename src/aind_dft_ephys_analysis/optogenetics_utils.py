import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections.abc import Iterable
from typing import List, Any, Union, Optional, Dict

from behavior_utils import extract_fitted_data
from nwb_utils import NWBUtils

def create_opto_data_frame(nwb_data: Any) -> pd.DataFrame:
    """
    Create a per-trial DataFrame from `nwb_data.intervals['trials']`, append the session
    name and session-level metadata, and attach latent variables from the model
    `QLearning_L1F1_CK1_softmax` **only to trials with a response** (animal_response != 2).
    For no-response trials, latent columns are set to `None`.

    Additionally, add four boolean columns describing stay/switch patterns:
      - stay:        current choice equals previous choice (both trials are responses)
      - switch:      current choice differs from previous choice (both trials are responses)
      - win_stay:    previous trial rewarded AND stay
      - lose_switch: previous trial unrewarded AND switch

    If the previous trial is a no-response, all four flags are False for the current trial.

    Parameters
    ----------
    nwb_data : object
        An NWB object with:
        - `.intervals['trials']` (TimeIntervals/DynamicTable convertible to DataFrame)
        - `.session_id` (str) — if it ends with ".json", the suffix is removed
        - `.scratch['metadata']` (DynamicTable) with exactly one row of session metadata

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per trial. Columns include:
        - all trial fields from the NWB `trials` table,
        - `session` (cleaned from `nwb_data.session_id`),
        - all columns from `nwb_data.scratch['metadata']`,
        - `stay`, `switch`, `win_stay`, `lose_switch` (booleans),
        - latent-variable columns named like
          `QLearning_L1F1_CK1_softmax-<latent_name>` for each requested latent,
          where rows with no response contain `None`.

    Raises
    ------
    ValueError
        - If `nwb_data.scratch['metadata']` does not contain exactly one row.
        - If any latent series cannot be extracted.
        - If any latent series length differs from the number of response trials
          (i.e., `sum(animal_response != 2)`).

    Notes
    -----
    Latents are fetched via `extract_fitted_data(...)` for the model alias
    `QLearning_L1F1_CK1_softmax`. They are assumed to exclude no-response trials,
    so each latent series must have length equal to the number of response trials.
    """
    # --- trials table ---
    trials_tbl = nwb_data.intervals['trials']
    df = trials_tbl.to_dataframe().copy()
    df.index.name = 'trial_id'
    n_trials = len(df)

    # explicit 0-based trial counter as a column
    df['trial_num'] = np.arange(n_trials, dtype=int)

    # --- session name ---
    session = getattr(nwb_data, 'session_id', '')
    if isinstance(session, str) and session.endswith('.json'):
        session = session[:-5]
    df['session'] = session

    # --- animal ID ---
    subject_id=nwb_data.subject.subject_id
    df['subject_id'] = subject_id
    # --- metadata ---
    meta_df = nwb_data.scratch['metadata'].to_dataframe().copy()
    if len(meta_df) != 1:
        raise ValueError("Expected metadata table to have exactly 1 row.")
    for k, v in meta_df.iloc[0].to_dict().items():
        df[k] = v

    # --- response & reward arrays from the NWB trials table ---
    # 0=left, 1=right, 2=no-response
    resp = nwb_data.trials['animal_response'][:]
    # Boolean flag: True if the trial has a response (0/1), False if no-response (2)
    df['response'] = (resp != 2).astype(bool)
    # reward status for each trial (True if either side rewarded)
    rewardedL = nwb_data.trials['rewarded_historyL'][:]
    rewardedR = nwb_data.trials['rewarded_historyR'][:]
    rewarded = np.logical_or(rewardedL, rewardedR).astype(bool)

    # --- compute stay/switch and win_stay/lose_switch ---
    # previous-trial arrays (roll by 1; set first entry to invalid defaults)
    prev_resp = np.roll(resp, 1)
    prev_resp[0] = 2  # mark as no-response for t=0 so all flags become False

    prev_rew = np.roll(rewarded, 1)
    prev_rew[0] = False

    valid_prev = (prev_resp != 2)     # previous trial must be a response
    valid_curr = (resp != 2)          # current trial must be a response

    stay_arr = (resp == prev_resp) & valid_prev & valid_curr
    switch_arr = (resp != prev_resp) & valid_prev & valid_curr

    win_stay_arr = prev_rew & stay_arr
    lose_switch_arr = (~prev_rew) & switch_arr & valid_prev  # valid_prev already in switch_arr

    # attach columns (bool dtype)
    df['stay'] = stay_arr
    df['switch'] = switch_arr
    df['win_stay'] = win_stay_arr
    df['lose_switch'] = lose_switch_arr

    # valid only when both previous and current trials are responses
    valid_pair = (prev_resp != 2) & (resp != 2)
    valid_pair_win_prev = valid_pair & prev_rew
    valid_pair_lose_prev = valid_pair & (~prev_rew)

    df['stay']        = pd.Series(stay_arr, index=df.index).where(valid_pair, pd.NA).astype('boolean')
    df['switch']      = pd.Series(switch_arr, index=df.index).where(valid_pair, pd.NA).astype('boolean')
    df['win_stay']    = pd.Series(win_stay_arr, index=df.index).where(valid_pair_win_prev, pd.NA).astype('boolean')
    df['lose_switch'] = pd.Series(lose_switch_arr, index=df.index).where(valid_pair_lose_prev, pd.NA).astype('boolean')


    # --- response mask (valid trials) for latent placement ---
    responses = nwb_data.trials['animal_response'][:]
    valid_mask = (responses != 2)
    valid_idx = np.where(valid_mask)[0]
    n_valid = int(valid_mask.sum())

    # --- latents to fetch ---
    model_alias = 'QLearning_L1F1_CK1_softmax'
    latent_names: List[str] = [
        'deltaQ', 'deltaQ-1', 'deltaQ+1',
        'sumQ', 'sumQ-1', 'sumQ+1',
        'right_choice_probability', 'right_choice_probability-1', 'right_choice_probability+1',
        'left_choice_probability',  'left_choice_probability-1',  'left_choice_probability+1',
        'RPE', 'RPE-1', 'RPE+1',
        'QL', 'QL-1', 'QL+1',
        'QR', 'QR-1', 'QR+1',
        'chosenQ', 'chosenQ-1', 'chosenQ+1',
        'unchosenQ', 'unchosenQ-1', 'unchosenQ+1',
        'reward', 'reward-1', 'reward+1',
        'choice', 'choice-1', 'choice+1',
    ]

    # --- other parameters ---
    df['ITI_delay_sum'] = nwb_data.trials['goCue_start_time'][:]-nwb_data.trials['start_time'][:]

    full_session_name = getattr(nwb_data, 'session_id', None)

    for ln in latent_names:
        arr = extract_fitted_data(
            nwb_behavior_data=nwb_data,
            session_name=full_session_name,
            model_alias=model_alias,
            latent_name=ln
        )
        if arr is None:
            raise ValueError(f"Latent '{ln}' could not be extracted for model '{model_alias}'.")

        # Enforce length == number of response trials
        if len(arr) != n_valid:
            raise ValueError(
                f"Latent '{ln}' length ({len(arr)}) != number of response trials ({n_valid})."
            )

        # Build column: None for no-response, values for valid trials
        col = [None] * n_trials
        values = arr.tolist() if hasattr(arr, "tolist") else list(arr)
        for pos, trial_i in enumerate(valid_idx):
            col[trial_i] = values[pos]

        # Set dtype=object to preserve None
        df[f'{model_alias}-{ln}'] = pd.Series(col, dtype=object)

    return df


def find_unique_combinations(
    df: pd.DataFrame,
    columns: List[str],
    count_col: str = "n_trials",
    include_na: bool = True,
) -> pd.DataFrame:
    """
    Find unique value combinations in the given columns and count trials per combo,
    plus the number of unique sessions and unique mice contributing to each combo.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame (e.g., from create_opto_data_frame).
        Should contain 'session' and 'subject_id' to populate n_session and n_mice.
    columns : list of str
        Column names to check for unique combinations.
    count_col : str, default "n_trials"
        Name of the count column in the output.
    include_na : bool, default True
        If True, include groups where at least one column is NA/None.
        If False, drop rows with NA in any of the specified columns before grouping.

    Returns
    -------
    pd.DataFrame
        One row per unique combination with:
          - '{count_col}': total rows in the group
          - 'n_session'  : number of unique sessions contributing to the group
          - 'n_mice'     : number of unique subject_id values contributing to the group
    """
    # Validate columns
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    work = df[columns].copy()
    if not include_na:
        work = work.dropna(subset=columns)

    # We'll need session/subject_id to compute n_session/n_mice if available
    have_session = 'session' in df.columns
    have_subject = 'subject_id' in df.columns

    # Build a working frame that includes grouping cols and optional keys
    extra_keys = []
    if have_session:
        work = work.join(df['session'])
        extra_keys.append('session')
    if have_subject:
        work = work.join(df['subject_id'])
        extra_keys.append('subject_id')

    # Group and aggregate
    gb = work.groupby(columns, dropna=include_na)

    counts = gb.size().rename(count_col)

    if have_session:
        n_session = gb['session'].nunique(dropna=True).rename('n_session')
    else:
        n_session = pd.Series(pd.NA, index=counts.index, dtype='Int64', name='n_session')

    if have_subject:
        n_mice = gb['subject_id'].nunique(dropna=True).rename('n_mice')
    else:
        n_mice = pd.Series(pd.NA, index=counts.index, dtype='Int64', name='n_mice')

    out = (
        pd.concat([counts, n_session, n_mice], axis=1)
          .reset_index()
          .sort_values(by=count_col, ascending=False, kind="stable")
          .reset_index(drop=True)
    )
    return out




def find_unique_stimulation(
    df: pd.DataFrame,
    columns: List[str] = [
        'laser_on_trial', 'laser_wavelength', 'laser_location', 'laser_1_power',
        'laser_2_power', 'laser_on_probability', 'laser_duration', 'laser_start',
        'laser_start_offset', 'laser_end', 'laser_end_offset', 'laser_protocol',
        'laser_frequency', 'laser_rampingdown', 'laser_pulse_duration',
        'session_wide_control', 'fraction_of_session', 'session_start_with',
        'session_alternation'
    ],
    strict: bool = False
) -> pd.DataFrame:
    """
    Return unique stimulation parameter combinations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from create_opto_data_frame.
    columns : list of str
        Stimulation-related columns to consider.
    strict : bool
        If True, raise an error when any requested column is missing.
        If False, silently ignore missing columns and use the intersection.

    Returns
    -------
    pd.DataFrame
        Unique combinations across the available columns (no duplicates).
    """
    # Determine which requested columns are present
    present = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]

    if strict and missing:
        raise ValueError(f"Missing columns: {missing}")

    if not present:
        # Nothing to compute
        return pd.DataFrame(columns=[])

    # Use your helper to get unique combos
    return find_unique_combinations(df, present)

def _collect_nwb_files(sources: Union[str, Path, Iterable[Union[str, Path]]]) -> List[Path]:
    """
    Normalize input into a list of .nwb file Paths.
    - If a folder is provided, collect all *.nwb in it.
    - If a list/tuple is provided, accept file Paths (and optionally folders).
    """
    files: List[Path] = []

    def _is_iterable_but_not_str(x):
        return isinstance(x, Iterable) and not isinstance(x, (str, bytes, Path))

    if _is_iterable_but_not_str(sources):
        for item in sources:
            p = Path(item)
            if p.is_dir():
                files.extend(sorted(p.glob("*.nwb")))
            elif p.is_file() and p.suffix.lower() == ".nwb":
                files.append(p)
            else:
                print(f"Warning: Skipping non-existent or non-NWB path: {p}")
    else:
        p = Path(sources)
        if p.is_dir():
            files = sorted(p.glob("*.nwb"))
        elif p.is_file() and p.suffix.lower() == ".nwb":
            files = [p]
        else:
            raise FileNotFoundError(f"Input is neither a folder nor a valid .nwb file: {p}")

    # dedupe & ensure existence
    files = sorted({f.resolve() for f in files if f.exists()})
    if not files:
        raise FileNotFoundError("No NWB files found to process.")
    return files


def create_opto_data_frame_combined(
    sources: Union[str, Path, Iterable[Union[str, Path]]] = "/root/capsule/data/optogenetics_nwb",
    save_path: Optional[Union[str, Path]] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Build and (optionally) save a combined per-trial DataFrame from one or more NWB files,
    with progress output.

    Parameters
    ----------
    sources : str | Path | Iterable[str|Path], default "/root/capsule/data/optogenetics_nwb"
        - A folder containing .nwb files, OR
        - A single .nwb path, OR
        - A list/tuple of .nwb paths (and/or folders).
    save_path : str | Path, optional
        If provided, write the combined CSV to this path. If a directory is given,
        the file 'combined_opto_data_frame.csv' will be created inside it.
    show_progress : bool, default True
        If True, display a progress bar (tqdm if available) or simple prints.

    Returns
    -------
    pandas.DataFrame
        Trials from all sessions concatenated (row-wise).
    """
    # Collect files
    nwb_files = _collect_nwb_files(sources)
    total = len(nwb_files)

    # Progress helper
    pbar = None
    use_tqdm = False
    if show_progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=total, desc="Processing NWB files", unit="file")
            use_tqdm = True
        except Exception:
            print(f"Processing {total} NWB file(s)...")

    combined: List[pd.DataFrame] = []
    for i, nwb_path in enumerate(nwb_files, start=1):
        if show_progress and not use_tqdm:
            print(f"[{i}/{total}] {nwb_path}")

        nwb_data = NWBUtils.read_behavior_nwb(nwb_full_path=str(nwb_path))
        if nwb_data is None:
            msg = f"Warning: Could not read NWB file {nwb_path}, skipping."
            if use_tqdm:
                pbar.write(msg)  # keep tqdm bar intact
            else:
                print(msg)
            if use_tqdm:
                pbar.update(1)
            continue

        try:
            df = create_opto_data_frame(nwb_data)
            combined.append(df)
        except Exception as e:
            msg = f"Error processing {nwb_path}: {e}"
            if use_tqdm:
                pbar.write(msg)
            else:
                print(msg)
        finally:
            try:
                nwb_data.io.close()
            except Exception:
                pass

        if use_tqdm:
            pbar.update(1)

    if use_tqdm and pbar is not None:
        pbar.close()

    if not combined:
        raise ValueError("No valid DataFrames were created from the input NWB files.")

    combined_df = pd.concat(combined, ignore_index=True)

    if save_path is not None:
        save_path = Path(save_path)
        # If a directory is given, place default filename inside it
        if save_path.suffix.lower() != ".csv":
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / "combined_opto_data_frame.csv"
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)

        combined_df.to_csv(save_path, index=False)
        if show_progress:
            print(f"Combined DataFrame saved to {save_path}")

    return combined_df


def load_opto_data_frame(
    csv_path: Union[str, Path] = "/root/capsule/results/combined_opto_data_frame.csv"
) -> pd.DataFrame:
    """
    Load a saved optogenetics combined CSV and normalize missing values so they
    match the in-memory representation from `create_opto_data_frame_combined`
    (i.e., use Python `None` in object columns instead of 'nan'/'None' strings).

    Parameters
    ----------
    csv_path : str | Path
        Path to the saved CSV file.

    Returns
    -------
    pandas.DataFrame
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read as strings; don't auto-convert to NaN so we can normalize ourselves
    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,   # don't treat 'NA', 'NaN', etc. as NaN automatically
        na_values=[]             # disable built-in NA parsing
    )

    NULL_STRINGS = {"", "None", "none", "NULL", "null", "NaN", "nan", "NAN", "<NA>", "<na>"}

    # Normalize null-like entries to actual None
    for col in df.columns:
        df[col] = df[col].map(lambda x: None if x in NULL_STRINGS else x)

    return df



def find_unique_values_by_conditions(
    df: pd.DataFrame,
    conditions: Dict[str, Any],
    output_column: str,
    dropna: bool = True,
) -> List[Any]:
    """
    Filter rows by a set of equality/isin conditions and return unique values
    from `output_column`.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame (e.g., from create_opto_data_frame).
    conditions : dict
        Mapping of column -> value(s) to match. Each value can be:
          - a single value (== comparison),
          - an iterable of values (isin comparison),
          - None (match NA/None in that column).
        Example: {"laser_start": "Go cue", "laser_end": "Trial start"}
    output_column : str
        Column whose unique values to return from the matched rows.
    dropna : bool, default True
        Whether to drop NA/None in the output before taking uniques.

    Returns
    -------
    list
        List of unique values in `output_column` among rows that satisfy all conditions.

    Raises
    ------
    ValueError
        If any referenced column does not exist.
    """
    # Validate columns
    cols_needed = set(conditions.keys()) | {output_column}
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # Build boolean mask
    mask = pd.Series(True, index=df.index)
    for col, val in conditions.items():
        if val is None:
            mask &= df[col].isna()
        elif isinstance(val, (list, tuple, set)):
            mask &= df[col].isin(list(val))
        else:
            mask &= (df[col] == val)

    series = df.loc[mask, output_column]
    if dropna:
        series = series.dropna()

    return series.drop_duplicates().tolist()
