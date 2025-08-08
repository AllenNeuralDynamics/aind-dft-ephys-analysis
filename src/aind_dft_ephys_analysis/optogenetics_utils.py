from behavior_utils import extract_fitted_data
import pandas as pd
from typing import List, Any

import numpy as np
import pandas as pd
from typing import List, Any
from behavior_utils import extract_fitted_data

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



def find_unique_combinations(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Find all unique value combinations in the given columns of a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data (e.g., from create_opto_data_frame).
    columns : list of str
        Column names to check for unique combinations.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing unique combinations (no duplicates).
    """
    # Validate columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    # Get unique combinations
    unique_combinations = df[columns].drop_duplicates().reset_index(drop=True)
    
    return unique_combinations


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

