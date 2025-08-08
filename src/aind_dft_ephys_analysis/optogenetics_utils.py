import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections.abc import Iterable
from typing import List, Any, Union, Optional

from behavior_utils import extract_fitted_data
from nwb_utils import NWBUtils

def create_opto_data_frame(nwb_data: Any) -> pd.DataFrame:
    """
    Create a per-trial DataFrame from `nwb_data.intervals['trials']`, append the session
    name and session-level metadata, and attach latent variables from the model
    `QLearning_L1F1_CK1_softmax` **only to trials with a response** (animal_response != 2).
    For no-response trials, latent columns are set to `None`.

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
    trials = nwb_data.intervals['trials']
    df = trials.to_dataframe().copy()
    df.index.name = 'trial_id'
    n_trials = len(df)

    # --- session name ---
    session = getattr(nwb_data, 'session_id', '')
    if isinstance(session, str) and session.endswith('.json'):
        session = session[:-5]
    df['session'] = session

    # --- metadata ---
    meta_df = nwb_data.scratch['metadata'].to_dataframe().copy()
    if len(meta_df) != 1:
        raise ValueError("Expected metadata table to have exactly 1 row.")
    for k, v in meta_df.iloc[0].to_dict().items():
        df[k] = v

    # --- response mask (valid trials) ---
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
        # Ensure list-like
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
    Find unique value combinations in the given columns and count trials per combo.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame (e.g., from create_opto_data_frame).
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
        One row per unique combination with a '{count_col}' column.
    """
    # Validate columns
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    work = df[columns].copy()
    if not include_na:
        work = work.dropna(subset=columns)

    # Group and count
    out = (
        work.groupby(columns, dropna=include_na)
            .size()
            .reset_index(name=count_col)
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
    save_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Build and (optionally) save a combined per-trial DataFrame from one or more NWB files.

    Parameters
    ----------
    sources : str | Path | Iterable[str|Path]
        - A folder containing .nwb files, OR
        - A single .nwb path, OR
        - A list/tuple of .nwb paths (and/or folders).
    save_path : str | Path, optional
        If provided, write the combined CSV to this path.

    Returns
    -------
    pandas.DataFrame
        Trials from all sessions concatenated (row-wise).
    """
    nwb_files = _collect_nwb_files(sources)

    combined: List[pd.DataFrame] = []
    for nwb_path in nwb_files:
        print(f"Processing {nwb_path}...")
        nwb_data = NWBUtils.read_behavior_nwb(nwb_full_path=str(nwb_path))
        if nwb_data is None:
            print(f"Warning: Could not read NWB file {nwb_path}, skipping.")
            continue

        try:
            df = create_opto_data_frame(nwb_data)
            combined.append(df)
        except Exception as e:
            print(f"Error processing {nwb_path}: {e}")
        finally:
            try:
                nwb_data.io.close()
            except Exception:
                pass

    if not combined:
        raise ValueError("No valid DataFrames were created from the input NWB files.")

    combined_df = pd.concat(combined, ignore_index=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(save_path, index=False)
        print(f"Combined DataFrame saved to {save_path}")

    return combined_df

def load_opto_data_frame(
    csv_path: Union[str, Path] = "/root/capsule/results/combined_opto_data_frame.csv"
) -> pd.DataFrame:
    """
    Load a saved optogenetics combined CSV file back into a DataFrame.

    This restores None values in latent variable columns where NaN was saved,
    so the DataFrame matches the output format of `create_opto_data_frame_combined`.

    Parameters
    ----------
    csv_path : str | Path
        Path to the saved CSV file from `create_opto_data_frame_combined`.

    Returns
    -------
    pandas.DataFrame
        DataFrame ready for downstream analysis.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV (all as object so we can restore None where needed)
    df = pd.read_csv(csv_path, dtype=object)

    # Replace 'nan' strings or float NaN with None
    df = df.where(pd.notna(df), None)

    return df
