import pandas as pd
from typing import List

def create_opto_data_frame(nwb_data: object) -> pd.DataFrame:
    """
    Create a DataFrame from nwb_data.intervals['trials'] with trial_id as rows,
    and append session name plus metadata fields from nwb_data.scratch['metadata'].
    
    Parameters
    ----------
    nwb_data : object
        An NWB object with:
        - .intervals['trials'] (TimeIntervals)
        - .session_id (string)
        - .scratch['metadata'] (DynamicTable)
    
    Returns
    -------
    pd.DataFrame
        Each row is a trial; columns are trial fields + session info + metadata fields.
    """
    # Extract trials table as DataFrame
    trials = nwb_data.intervals['trials']
    df = trials.to_dataframe().copy()
    df.index.name = 'trial_id'
    
    # Get session name without ".json"
    session = getattr(nwb_data, 'session_id', '')
    if isinstance(session, str) and session.endswith('.json'):
        session = session[:-5]
    
    # Add session column
    df['session'] = session
    
    # Extract metadata table as DataFrame (assume single row)
    meta_df = nwb_data.scratch['metadata'].to_dataframe().copy()
    if len(meta_df) == 1:
        # Turn single-row metadata into dict
        meta_dict = meta_df.iloc[0].to_dict()
        # Add each metadata field as a column
        for k, v in meta_dict.items():
            df[k] = v
    else:
        raise ValueError("Expected metadata table to have exactly 1 row.")
    
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
