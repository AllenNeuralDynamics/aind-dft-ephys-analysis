import re
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional,Any,Union

def extract_session_name_core(session_name: str) -> str | None:
    """
    Extracts the core session name from a given session name string.

    The core session name follows the pattern:
    "XXXXXX_YYYY-MM-DD_HH-MM-SS", where:
      - XXXXXX: A 6-digit identifier (e.g., subject ID)
      - YYYY-MM-DD: The date (year, month, day)
      - HH-MM-SS: The time (hour, minute, second)

    Parameters:
    session_name (str): The full session name containing additional prefixes or suffixes.

    Returns:
    str | None: The extracted core session name if found, otherwise None.
    """

    # Regular expression to match the core session name pattern
    pattern = r'(\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'

    # Search for the pattern in the given session name
    match = re.search(pattern, session_name)

    # Return the matched core session name if found, otherwise return None
    return match.group(1) if match else None


def sort_sessions_by_animal_and_date(session_list: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Given a list of session folder names, this function extracts the animal ID and the datetime
    from each folder name, sorts the sessions by animal ID (numerically) and then by datetime,
    and groups them by animal ID.
    
    Expected session name patterns:
      "ecephys_706893_2024-05-28_15-15-38"
      "ecephys_706893_2024-05-28_15-15-38_sorted_2024-05-28_16-00-00"
      "behavior_123456_2024-05-28_15-15-38"
      "behavior_123456_2024-05-28_15-15-38_processed_2024-05-28_16-05-00"
    
    Returns:
      Tuple containing:
        - A sorted list of session names.
        - A dictionary mapping animal IDs (as strings) to a list of sorted session names.
    """
    parsed_sessions = []
    
    for session in session_list:
        parts = session.split('_')
        # Ensure there are at least 4 parts: prefix, animal_id, date, time, ...
        if len(parts) < 4:
            continue
        
        animal_id = parts[1]
        date_str = parts[2]  # e.g., "2024-05-28"
        time_str = parts[3]  # e.g., "15-15-38"
        
        # Convert time string from "15-15-38" to "15:15:38" for proper datetime parsing.
        time_str_formatted = time_str.replace('-', ':')
        try:
            dt = datetime.strptime(f"{date_str} {time_str_formatted}", "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            # Skip this session if datetime parsing fails.
            continue
        
        parsed_sessions.append((animal_id, dt, session))
    
    # Sort sessions first by animal ID (numerically) then by datetime.
    parsed_sessions.sort(key=lambda x: (int(x[0]), x[1]))
    
    sorted_sessions = [session for _, _, session in parsed_sessions]
    
    # Group sessions by animal ID.
    sessions_by_animal = defaultdict(list)
    for animal_id, _, session in parsed_sessions:
        sessions_by_animal[animal_id].append(session)
    
    return sorted_sessions, dict(sessions_by_animal)

def find_ephys_sessions(
    folder_path: str = '/root/capsule/data/'
) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    """
    Discover and sort ephys session folders in a directory.

    Scans the given `folder_path` for subdirectories starting with 'ecephys_'.
    Then:
      1. Sorts all found sessions by animal ID and datetime.
      2. Groups sorted sessions by animal ID.
      3. Identifies spike-sorted sessions (folder name contains 'sorted').

    Args:
        folder_path: Base path containing ephys session folders.

    Returns:
        A tuple of:
        - all_sessions: Sorted list of all ephys session folder names.
        - sessions_by_animal: Dict mapping animal IDs to their session lists.
        - spike_sorted_sessions: List of folder names that include 'sorted'.
    """
    root = Path(folder_path)
    session_list = [d.name for d in root.iterdir() if d.is_dir() and d.name.startswith('ecephys_')]

    all_sessions, sessions_by_animal = sort_sessions_by_animal_and_date(session_list)
    spike_sorted_sessions = [s for s in all_sessions if 'sorted' in s]

    return all_sessions, sessions_by_animal, spike_sorted_sessions


def find_behavior_sessions(
    folder_path: str = '/root/capsule/data/'
) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    """
    Discover and sort behavior session folders in a directory.

    Scans the given `folder_path` for subdirectories starting with 'behavior_'.
    Then:
      1. Sorts all found sessions by animal ID and datetime.
      2. Groups sorted sessions by animal ID.
      3. Identifies processed sessions (folder name contains 'processed').

    Args:
        folder_path: Base path containing behavior session folders.

    Returns:
        A tuple of:
        - all_sessions: Sorted list of all behavior session folder names.
        - sessions_by_animal: Dict mapping animal IDs to their session lists.
        - processed_behavior_sessions: List of folder names that include 'processed'.
    """
    root = Path(folder_path)
    session_list = [d.name for d in root.iterdir() if d.is_dir() and d.name.startswith('behavior_')]

    all_sessions, sessions_by_animal = sort_sessions_by_animal_and_date(session_list)
    processed_behavior_sessions = [s for s in all_sessions if 'processed' in s]

    return all_sessions, sessions_by_animal, processed_behavior_sessions

def format_session_name(session_name):
    """
    Formats the session name to ensure it follows a standard format.
    
    This function:
    1. Ensures the session name ends with the '.nwb' extension.
    2. Removes the 'ecephys_' prefix if present.

    Parameters:
    - session_name (str): The original session name to be formatted.

    Returns:
    - str: The formatted session name.
    """
    # Step 0: extract the core session name
    session_name=extract_session_name_core(session_name)
    # Step 1: Ensure that session_name ends with '.nwb'
    if not session_name.endswith('.nwb'):
        # If the session name does not already have '.nwb', append it.
        session_name += '.nwb'
    
    # Step 2: Remove the 'ecephys_' prefix if it exists
    if session_name.startswith('ecephys_'):
        # Replace the 'ecephys_' prefix with an empty string
        session_name = session_name.replace('ecephys_', '')
    
    # Return the formatted session name
    return session_name


def smart_read_csv(
    filepath: Union[str, Path],
    object_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame, automatically parsing any columns that
    contain Python literal structures (lists or dicts).

    The function will:
      1. If `object_columns` is None, read the first row of the file to detect
         columns whose string values start with '[' or '{'.
      2. Use `ast.literal_eval` on those columns when reading the full CSV.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file to read.
    object_columns : List[str], optional
        Explicit list of column names to parse as Python objects. If None,
        detection is performed automatically.

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed list/dict columns.
    """
    # Ensure filepath is string
    path_str = str(filepath)

    # Detect object columns if not provided
    if object_columns is None:
        sample = pd.read_csv(path_str, nrows=1, dtype=str)
        object_columns = [
            col for col, val in sample.iloc[0].items()
            if isinstance(val, str) and val.strip().startswith(('[' ,'{' ))
        ]

    # Build converters for pandas
    converters = {col: ast.literal_eval for col in object_columns}

    # Read full CSV with converters applied
    return pd.read_csv(path_str, converters=converters)