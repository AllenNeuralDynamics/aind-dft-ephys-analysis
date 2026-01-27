from __future__ import annotations

# ------------------------------
# Standard library
# ------------------------------
import ast
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ------------------------------
# Third-party libraries
# ------------------------------
import numpy as np
import pandas as pd
import xarray as xr



def extract_ID_Date(session_name: str) -> Optional[Tuple[str, str]]:
    """
    Extract the animal ID and date string from a session name.

    Looks for the pattern: _<6-digit-ID>_<YYYY-MM-DD>_
    Examples:
      "ecephys_706893_2024-05-28_15-15-38"     -> ("706893", "2024-05-28")
      "behavior_123456_2024-12-01_processed"  -> ("123456", "2024-12-01")

    Parameters
    ----------
    session_name : str
        The session folder (or file) name.

    Returns
    -------
    Optional[Tuple[str, str]]
        (animal_id, date_str) if matched; otherwise None.
    """
    pattern = r'(\d{6})_(\d{4}-\d{2}-\d{2})'
    match = re.search(pattern, session_name)
    if not match:
        return None

    animal_id, date_str = match.group(1), match.group(2)
    return animal_id, date_str



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
    object_columns: Optional[List[str]] = None,
    bool_columns: Optional[List[str]] = None,
    n_rows: Optional[int] = None,
    auto_numeric: bool = True,
) -> pd.DataFrame:
    """
    Read a CSV into a DataFrame, safely parsing columns that contain Python
    literal structures (lists or dicts) and converting obvious boolean and
    numeric columns.

    Parsing rules
    -------------
    - Object columns are parsed into Python objects:
        * Lists:
            - Supports numeric lists containing nan/inf/-inf
            - Falls back to literal_eval for non-numeric lists
        * Dicts:
            - Uses ast.literal_eval (safe)
    - Boolean columns are converted from TRUE/FALSE/True/False.
    - Remaining columns can be auto-converted to numeric.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file to read.
    object_columns : list[str], optional
        Columns to parse as Python objects. If None, the first row is inspected
        for values beginning with '[' or '{'.
    bool_columns : list[str], optional
        Columns to coerce to bool. If None, detection is based on the first row
        containing the literal strings TRUE / FALSE (case-insensitive).
    n_rows : int, optional
        If given, only read that many rows (handy for a quick preview).
    auto_numeric : bool, default True
        When True, attempts to convert the remaining non-object / non-bool
        columns to numeric dtypes.

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame with lists/dicts, booleans, and numeric columns
        converted to native Python / pandas types where possible.
    """
    path_str = str(filepath)

    # ------------------------------------------------------------------
    # 1) Inspect first row to auto-detect object / bool columns
    # ------------------------------------------------------------------
    sample = pd.read_csv(path_str, nrows=1, dtype=str)

    if object_columns is None:
        object_columns = [
            col for col, val in sample.iloc[0].items()
            if isinstance(val, str) and val.strip().startswith(("[", "{"))
        ]

    if bool_columns is None:
        bool_columns = []
        for col in sample.columns:
            v = sample.at[0, col]
            if isinstance(v, str):
                s = v.strip().upper()
                if s in {"TRUE", "FALSE"}:
                    bool_columns.append(col)

    # ------------------------------------------------------------------
    # 2) Regex helpers:
    #    - strip np.float64(...)
    #    - support nan/inf tokens
    # ------------------------------------------------------------------
    float64_re = re.compile(r"np\.float64\(\s*([^)]+?)\s*\)")
    nan_token_re = re.compile(r"(?<![A-Za-z0-9_])nan(?![A-Za-z0-9_])", flags=re.IGNORECASE)
    posinf_token_re = re.compile(r"(?<![A-Za-z0-9_])inf(?![A-Za-z0-9_])", flags=re.IGNORECASE)
    neginf_token_re = re.compile(r"(?<![A-Za-z0-9_])-inf(?![A-Za-z0-9_])", flags=re.IGNORECASE)

    def _try_parse_numeric_list(s: str):
        """
        Fast path for numeric lists (including nan/inf):
          "[1.0, nan, 2, -inf]" -> list[float]
        Returns None if it doesn't look parseable as numeric list.
        """
        st = s.strip()
        if not (st.startswith("[") and st.endswith("]")):
            return None

        body = st[1:-1].strip()
        if body == "":
            return []

        # np.fromstring can parse 'nan'/'inf' tokens (case-insensitive).
        # It will stop early if there are non-numeric tokens.
        arr = np.fromstring(body, sep=",", dtype=float)

        # Heuristic: ensure we didn't silently parse only a prefix.
        # Count commas as a proxy for expected number of elements.
        expected_n = body.count(",") + 1
        if arr.size == expected_n:
            return arr.tolist()

        return None

    def _safe_parse(val):
        """
        Safely parse list/dict-like strings.
        - First: remove np.float64 wrappers.
        - Then: for lists, try numeric fast-path (handles nan/inf).
        - Else: make nan/inf parseable for literal_eval, then literal_eval.
        - If parsing fails, return original value.
        """
        if not isinstance(val, str):
            return val

        s = val.strip()
        if not (s.startswith("[") or s.startswith("{")):
            return val

        clean = float64_re.sub(r"\1", s)

        # Fast numeric list parse (best for your reward-rate columns)
        if clean.startswith("["):
            parsed = _try_parse_numeric_list(clean)
            if parsed is not None:
                return parsed

        # Make NaN/Inf parseable by literal_eval for mixed-type lists/dicts
        clean2 = nan_token_re.sub('float("nan")', clean)
        clean2 = neginf_token_re.sub('float("-inf")', clean2)
        clean2 = posinf_token_re.sub('float("inf")', clean2)

        try:
            return ast.literal_eval(clean2)
        except (ValueError, SyntaxError):
            return val

    # ------------------------------------------------------------------
    # 3) Build converters mapping for read_csv
    # ------------------------------------------------------------------
    converters = {col: _safe_parse for col in object_columns}
    converters.update({
        col: (lambda x: str(x).strip().upper() == "TRUE") for col in bool_columns
    })

    # ------------------------------------------------------------------
    # 4) Read the CSV with converters
    # ------------------------------------------------------------------
    df = pd.read_csv(path_str, converters=converters, nrows=n_rows)

    # ------------------------------------------------------------------
    # 5) Optionally convert remaining columns to numeric
    # ------------------------------------------------------------------
    if auto_numeric:
        other_cols = df.columns.difference(list(object_columns) + list(bool_columns))
        for col in other_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

    return df



def load_temporary_data(
    path: Union[str, Path],
    *,
    decode_json: bool = True,
) -> pd.DataFrame:
    """
    • Accepts both **CSV** and **Zarr** formats.  
    • Automatically JSON-decodes cells that were string-encoded when saving
      (``fit_metadata`` **and** any column whose values look like dicts / lists).

    Parameters
    ----------
    path : str | pathlib.Path
        Path to ``*.csv`` **or** ``*.zarr`` directory.
    decode_json : bool, default **True**
        When *True*, try to ``json.loads`` every object-dtype column whose first
        non-null value starts with “{” or “[”.
        Set to *False* if you prefer to keep those columns as raw strings.

    Returns
    -------
    pandas.DataFrame
        A fully-decoded DataFrame identical to what was in memory before saving.

    Examples
    --------
    Read back the table produced by
    ``significance_and_direction_summary_combined_multi`` **or**
    ``significance_and_direction_summary_multi``.

    >>> df = load_temporary_data("/root/capsule/results/sig_dir_all_sessions.csv")
    >>> df.filter(like="_pval").head()

    >>> df = load_temporary_data("/root/capsule/results/sig_dir_all_sessions.zarr")
    >>> df.loc[0, "fit_metadata"]["ARDL_model_g0"]
    {'fit_parameters': {'y_lags': 3, 'x_order': 0},
     'fit_variables': ['RPE', 'total_value']}
    """
    path = Path(path).expanduser()

    # ──────────────────────────────────────────────────────────────────
    # 1)  Dispatch by file extension
    # ──────────────────────────────────────────────────────────────────
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)

    elif path.suffix.lower() == ".zarr":
        # consolidated=False for full compatibility with non-consolidated stores
        ds = xr.open_zarr(path, consolidated=False)
        df = ds.to_dataframe()

    else:
        raise ValueError("`path` must point to a .csv file *or* a .zarr directory")

    # ──────────────────────────────────────────────────────────────────
    # 2)  Optional JSON decode
    # ──────────────────────────────────────────────────────────────────
    if decode_json:
        for col in df.columns:
            if df[col].dtype == object:
                # peek at first non-null value
                first = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
                if isinstance(first, str) and first.strip()[:1] in ("{", "["):
                    try:
                        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                    except Exception:
                        # leave column untouched if decoding fails
                        pass

    return df


