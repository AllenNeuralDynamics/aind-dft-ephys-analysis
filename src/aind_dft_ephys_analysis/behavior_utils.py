# ============================================================
# Standard library
# ============================================================
import io
import json
import math
import os
import time
from contextlib import redirect_stderr, redirect_stdout
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ============================================================
# Third-party libraries
# ============================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests.exceptions import ConnectionError

# ============================================================
# Project / Local imports
# ============================================================
from aind_analysis_arch_result_access.han_pipeline import (
    get_mle_model_fitting as _orig_get_mle_model_fitting,
)
from general_utils import (
    extract_ID_Date,
    extract_session_name_core,
    format_session_name,
    smart_read_csv,
)
from model_fitting import fit_q_learning_model
from nwb_utils import NWBUtils



def silent_get_mle_model_fitting(*args, **kwargs):
    """
    Call get_mle_model_fitting but suppress any output to stdout and stderr.
    """
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        return _orig_get_mle_model_fitting(*args, **kwargs)

get_mle_model_fitting = silent_get_mle_model_fitting


def extract_event_timestamps(
    nwb_behavior_data: Any,
    event_name: str,
    lick_time_window: Optional[float] = None,
    before_go_cue_lick_time_window: Optional[List[float]] = None,
    quiet_window: Optional[List[float]] = None
) -> List[float]:
    """
    Extract event timestamps from an NWB behavior dataset for various event types.

    Parameters
    ----------
    nwb_behavior_data : NWB file handle
        The NWB behavior object containing trials and acquisition modules.
    event_name : str
        The name of the event to extract timestamps for. Supported event names include:

        * 'go_cue': Go cue onset times.
        * 'left_lick': Left lick event times.
        * 'right_lick': Right lick event times.
        * 'lick': Combined left and right lick event times.
        * 'reward': Reward delivery event times (left and right).
        * 'reward_left': Left reward delivery event times.
        * 'reward_right': Right reward delivery event times.
        * 'trial_start': Start times of trials.
        * 'trial_end': End times of trials.
        * 'reward_go_cue_start': Go cue onset times where a reward was delivered in that trial.
        * 'no_reward_go_cue_start': Go cue onset times where no reward was delivered in that trial.
        * 'no_response_go_cue_start': Go cue onset times where the animal did not respond (animal_response == 2).

        * 'after_go_cue_first_lick': The first lick after the go cue (any trial).
        * 'after_go_cue_first_left_lick': The first lick after the go cue if and only if that first lick was on the left side.
        * 'after_go_cue_first_right_lick': The first lick after the go cue if and only if that first lick was on the right side.
        * 'after_go_cue_first_lick_reward': The first lick after the go cue in rewarded trials (regardless of side).
        * 'after_go_cue_first_lick_no_reward': The first lick after the go cue in no-reward trials (regardless of side).
        * 'after_go_cue_first_left_lick_reward': The first left lick after the go cue in rewarded trials, but only if it's the first lick of the trial.
        * 'after_go_cue_first_right_lick_reward': The first right lick after the go cue in rewarded trials, but only if it's the first lick of the trial.
        * 'after_go_cue_first_left_lick_no_reward': The first left lick after the go cue in no-reward trials, but only if it's the first lick of the trial.
        * 'after_go_cue_first_right_lick_no_reward': The first right lick after the go cue in no-reward trials, but only if it's the first lick of the trial.

        * 'before_go_cue_lick': All licks (left and right) in [goCue + start_offset, goCue + end_offset], where offsets come from before_go_cue_lick_time_window.
        * 'before_go_cue_lick_left': Only left licks in that same interval.
        * 'before_go_cue_lick_right': Only right licks in that same interval.

        * 'quiet_lick': Any lick (left or right) that has no other lick in [t + quiet_before, t + quiet_after].
        * 'quiet_left_lick': A left lick that has no other lick (left or right) in that window.
        * 'quiet_right_lick': A right lick that has no other lick (left or right) in that window.

    lick_time_window : float, optional
        Time window (seconds) after go cue for first-lick events (default: 1.0).
    before_go_cue_lick_time_window : list of float, optional
        [start_offset, end_offset] for licks before go cue (default: [-3.0, -2.0]).
    quiet_window : list of float, optional
        [quiet_before, quiet_after] window to identify isolated licks (default: [-0.5, 0.5]).

    Returns
    -------
    List[float]
        Timestamps for the specified event.

    Raises
    ------
    ValueError
        If an unsupported event_name is provided or required acquisition is missing.
    """
    # Initialize default windows if not supplied
    if before_go_cue_lick_time_window is None:
        before_go_cue_lick_time_window = [-3.0, -2.0]
    if quiet_window is None:
        quiet_window = [-0.5, 0.5]
    # Default lick_time_window when needed
    if lick_time_window is None:
        lick_time_window = 1.0

    # Shorthand access to trials table and acquisition modules
    trials = nwb_behavior_data.trials
    acq = nwb_behavior_data.acquisition

    # Handle simple events
    if event_name == 'go_cue':
        return trials['goCue_start_time'][:].tolist()

    if event_name in ('left_lick', 'right_lick'):
        key = f"{event_name}_time"
        if key not in acq:
            raise ValueError(f"Missing acquisition: {key}")
        return acq[key].timestamps[:].tolist()

    if event_name == 'lick':
        left = acq['left_lick_time'].timestamps[:]
        right = acq['right_lick_time'].timestamps[:]
        return np.concatenate((left, right)).tolist()

    if event_name == 'reward':
        left = acq['left_reward_delivery_time'].timestamps[:]
        right = acq['right_reward_delivery_time'].timestamps[:]
        return np.concatenate((left, right)).tolist()

    if event_name in ('reward_left', 'reward_right'):
        side = 'left' if 'left' in event_name else 'right'
        key = f"{side}_reward_delivery_time"
        if key not in acq:
            raise ValueError(f"Missing acquisition: {key}")
        return acq[key].timestamps[:].tolist()

    if event_name == 'trial_start':
        return trials['start_time'][:].tolist()

    if event_name == 'trial_end':
        return trials['stop_time'][:].tolist()

    # Handle trial-based cue events
    go_times = trials['goCue_start_time'][:]
    rewardedL = trials['rewarded_historyL'][:]
    rewardedR = trials['rewarded_historyR'][:]
    rewarded = np.logical_or(rewardedL, rewardedR)
    responses = trials['animal_response'][:]

    if event_name == 'reward_go_cue_start':
        return go_times[rewarded].tolist()

    if event_name == 'no_reward_go_cue_start':
        mask = np.logical_and(~rewarded, responses != 2)
        return go_times[mask].tolist()

    if event_name == 'no_response_go_cue_start':
        return go_times[responses == 2].tolist()

    # After-go-cue first-lick events
    if event_name.startswith('after_go_cue_first'):
        left_ts = acq['left_lick_time'].timestamps[:]
        right_ts = acq['right_lick_time'].timestamps[:]
        out = []
        for i, gc in enumerate(go_times):
            if 'reward' in event_name and not rewarded[i]:
                continue
            if 'no_reward' in event_name and rewarded[i]:
                continue
            maskL = (left_ts >= gc) & (left_ts < gc + lick_time_window)
            maskR = (right_ts >= gc) & (right_ts < gc + lick_time_window)
            licks = np.concatenate((left_ts[maskL], right_ts[maskR]))
            if licks.size == 0:
                continue
            first_lick = np.min(licks)
            if 'left' in event_name and first_lick not in left_ts:
                continue
            if 'right' in event_name and first_lick not in right_ts:
                continue
            out.append(first_lick)
        return out

    # Before-go-cue licks
    if event_name.startswith('before_go_cue_lick'):
        left_ts = acq['left_lick_time'].timestamps[:]
        right_ts = acq['right_lick_time'].timestamps[:]
        start_off, end_off = before_go_cue_lick_time_window
        out = []
        for gc in go_times:
            start, end = gc + start_off, gc + end_off
            if 'left' in event_name:
                out.extend(left_ts[(left_ts >= start) & (left_ts < end)].tolist())
            elif 'right' in event_name:
                out.extend(right_ts[(right_ts >= start) & (right_ts < end)].tolist())
            else:
                out.extend(
                    np.concatenate((
                        left_ts[(left_ts >= start) & (left_ts < end)],
                        right_ts[(right_ts >= start) & (right_ts < end)])
                    ).tolist()
                )
        return sorted(out)

    # Quiet licks: no neighbor in window
    if event_name.startswith('quiet_lick'):
        left_ts = acq['left_lick_time'].timestamps[:]
        right_ts = acq['right_lick_time'].timestamps[:]
        all_licks = np.sort(np.concatenate((left_ts, right_ts)))
        before, after = quiet_window
        candidates = (
            left_ts if 'left' in event_name else
            right_ts if 'right' in event_name else
            all_licks
        )
        out = []
        for t in candidates:
            neighbors = all_licks[(all_licks >= t + before) & (all_licks <= t + after)]
            if neighbors.size <= 1:
                out.append(t)
        return sorted(out)

    raise ValueError(f"Unsupported event '{event_name}'")



def get_fitted_model_names(
    session_name: str,
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> Optional[List[str]]:
    """
    Retrieve all available fitted model aliases for a given NWB session.

    This function includes retry logic for transient network failures
    such as ConnectionError or RemoteDisconnected.

    Parameters
    ----------
    session_name : str
        The NWB session name (e.g., "744329_2024-11-25_12-13-37.nwb").

    max_retries : int, optional (default=3)
        Maximum number of retry attempts if a network-related error occurs.

    base_delay : float, optional (default=2.0)
        Base delay in seconds used for exponential backoff.
        Actual sleep time = base_delay * (2 ** (attempt - 1)).

    Returns
    -------
    Optional[List[str]]
        List of unique model alias names if available.
        Returns None if no fitting results exist.

    Raises
    ------
    ValueError
        If session_name is empty or cannot be parsed.

    Exception
        Re-raises the final exception if all retries fail.
    """

    # ------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------
    if not session_name:
        raise ValueError("The 'session_name' parameter cannot be empty.")

    # ------------------------------------------------------------
    # Parse subject_id and session_date from session_name
    # ------------------------------------------------------------
    subject_id, session_date = extract_ID_Date(session_name) or (None, None)

    if subject_id is None or session_date is None:
        raise ValueError(
            f"Could not parse subject ID & date from '{session_name}'"
        )

    # ------------------------------------------------------------
    # Attempt to fetch model-fitting results with retry logic
    # ------------------------------------------------------------
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            # Call external data source (likely network-dependent)
            df = get_mle_model_fitting(
                subject_id=subject_id,
                session_date=session_date,
            )

            # If successful, exit retry loop
            break

        except (ConnectionError, RemoteDisconnected) as e:
            # Store exception for potential re-raise
            last_exception = e

            print(
                f"[Attempt {attempt}/{max_retries}] "
                f"Network failure for session {session_name}: {e}"
            )

            # If not last attempt, wait using exponential backoff
            if attempt < max_retries:
                sleep_time = base_delay * (2 ** (attempt - 1))
                time.sleep(sleep_time)
            else:
                # All retries exhausted — re-raise last exception
                raise

        except Exception:
            # Do NOT retry unknown errors
            # This avoids masking real logic or data bugs
            raise

    # ------------------------------------------------------------
    # Handle empty or missing results
    # ------------------------------------------------------------
    if df is None or df.empty:
        print(
            f"No model-fitting results found "
            f"for subject {subject_id} on {session_date}"
        )
        return None

    # ------------------------------------------------------------
    # Determine which column contains model alias
    # Different versions of pipeline may store alias differently
    # ------------------------------------------------------------
    if "agent_alias" in df.columns:
        alias_col = "agent_alias"

    elif "analysis_results.fit_settings.agent_alias" in df.columns:
        alias_col = "analysis_results.fit_settings.agent_alias"

    else:
        raise ValueError(
            "Could not find alias column in fit-results DataFrame"
        )

    # ------------------------------------------------------------
    # Return unique alias names
    # dropna() removes invalid entries
    # unique() ensures no duplicates
    # ------------------------------------------------------------
    return df[alias_col].dropna().unique().tolist()





def get_fitted_latent(
    session_name: str,
    model_alias: Optional[str] = None,
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> Union[pd.DataFrame, Dict[str, Any], None]:
    """
    Retrieve fitted latent variables (and parameters) for a specific model
    in a given NWB session, with retry logic for transient network failures.

    If model_alias is None:
        Returns the full fit-results DataFrame.
    If model_alias is provided:
        Returns a dict with keys:
          - "params": fitted parameters (usually a dict-like object)
          - "fitted_latent_variables": latent variables (dict/arrays/lists)
          - "results": the full selected row (pandas Series)

    Parameters
    ----------
    session_name : str
        NWB session name, e.g. "744329_2024-11-25_12-13-37.nwb".
    model_alias : Optional[str]
        Model alias to filter for. If None, returns full DataFrame.
    max_retries : int
        Max retry attempts for network-related errors when fetching results.
    base_delay : float
        Base delay for exponential backoff:
            sleep = base_delay * (2 ** (attempt - 1))

    Returns
    -------
    Union[pandas.DataFrame, Dict[str, Any], None]
        - DataFrame if model_alias is None
        - Dict if model_alias is provided and found
        - None if no data found (empty results or alias not found)

    Raises
    ------
    ValueError
        If session_name is empty or alias column cannot be determined.
    Exception
        Re-raises the final exception if network failures persist after retries.
    """

    # ------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------
    if not session_name:
        raise ValueError("The 'session_name' parameter cannot be empty.")

    # ------------------------------------------------------------
    # Parse subject ID and session date from the session name
    # ------------------------------------------------------------
    subject_id, session_date = extract_ID_Date(session_name) or (None, None)
    if subject_id is None or session_date is None:
        # Keep this as a soft failure to match your original behavior
        print(f"Could not parse subject ID & date from '{session_name}'")
        return None

    # ------------------------------------------------------------
    # Fetch all fits for this subject & date with retry logic
    # Only retry on network-related errors. Do not retry other exceptions.
    # ------------------------------------------------------------
    last_exception = None
    df = None

    for attempt in range(1, max_retries + 1):
        try:
            df = get_mle_model_fitting(subject_id=subject_id, session_date=session_date)
            break  # Success

        except (ConnectionError, RemoteDisconnected) as e:
            last_exception = e
            print(
                f"[Attempt {attempt}/{max_retries}] "
                f"Network failure fetching fits for {session_name}: {e}"
            )

            # Exponential backoff before retrying
            if attempt < max_retries:
                sleep_time = base_delay * (2 ** (attempt - 1))
                time.sleep(sleep_time)
            else:
                # Retries exhausted: re-raise to make failure explicit
                raise

        except Exception:
            # Any other error is likely not transient; surface it immediately
            raise

    # ------------------------------------------------------------
    # Handle missing/empty results
    # ------------------------------------------------------------
    if df is None or df.empty:
        print(f"No model-fitting results for subject {subject_id} on {session_date}")
        return None

    # ------------------------------------------------------------
    # If no alias requested, return the entire DataFrame
    # ------------------------------------------------------------
    if model_alias is None:
        return df

    # ------------------------------------------------------------
    # Determine which column contains the model alias
    # Support multiple possible column names for compatibility.
    # ------------------------------------------------------------
    if "agent_alias" in df.columns:
        alias_col = "agent_alias"
    elif "analysis_results.fit_settings.agent_alias" in df.columns:
        alias_col = "analysis_results.fit_settings.agent_alias"
    else:
        raise ValueError("Could not find alias column in fit-results DataFrame")

    # ------------------------------------------------------------
    # Filter to the requested model alias
    # ------------------------------------------------------------
    sel = df[df[alias_col] == model_alias]
    if sel.empty:
        print(f"No entries for alias '{model_alias}' in fit results")
        return None

    # ------------------------------------------------------------
    # Select the first matching row
    # If you expect multiple fits per alias, you can change the selection logic
    # (e.g., choose best log-likelihood if such a column exists).
    # ------------------------------------------------------------
    row = sel.iloc[0]

    # ------------------------------------------------------------
    # Extract fields with compatibility fallbacks
    # Some pipelines may store these under different keys.
    # ------------------------------------------------------------
    # Primary expected columns
    params_col_candidates = ["params", "fitted_params", "analysis_results.params"]
    latent_col_candidates = ["latent_variables", "fitted_latent_variables", "analysis_results.latent_variables"]

    params_val = None
    for c in params_col_candidates:
        if c in sel.columns:
            params_val = row[c]
            break

    latent_val = None
    for c in latent_col_candidates:
        if c in sel.columns:
            latent_val = row[c]
            break

    if params_val is None or latent_val is None:
        print(
            "Fit-results DataFrame missing required columns for output. "
            f"Found params={params_val is not None}, latent={latent_val is not None}. "
            f"Available columns: {list(sel.columns)}"
        )
        return None

    # ------------------------------------------------------------
    # Return a structured output
    # - "results" gives full row for downstream debugging/metadata
    # ------------------------------------------------------------
    return {
        "params": params_val,
        "fitted_latent_variables": latent_val,
        "results": row,
    }




def extract_fitted_data(
    nwb_behavior_data: Any,
    fitted_latent: Optional[Dict[str, Any]] = None,
    session_name: Optional[str] = None,
    model_alias: Optional[str] = None,
    latent_name: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Extract a chosen derived time series from a behavioral model’s fitted latent variables.

    You must supply **either**:
      - `fitted_latent`: the dict returned by `get_fitted_latent(...)`, `model_alias` and latent_name**or**
      - all three of `session_name`, `model_alias`, and `latent_name`, so that this function can fetch it.

    Parameters
    ----------
    nwb_behavior_data : Any
        An open NWB object (from NWBHDF5IO.read() or NWBZarrIO.read()). Must contain `.trials` with columns:
          - 'rewarded_historyL', 'rewarded_historyR', 'animal_response'
        (needed for RPE and related series).

    fitted_latent : dict, optional
        If provided, this dict should have come from:
            get_fitted_latent(session_name, model_alias)
        containing at least 'fitted_latent_variables'.
        If None, this function will fetch it using the other arguments.

    session_name : str, optional
        NWB session identifier, e.g. "744329_2024-11-25_12-13-37.nwb".
        Required if `fitted_latent` is None.

    model_alias : str, optional
        Alias of the fitted model, e.g. "QLearning_L1F1_CK1_softmax".
        Required if `fitted_latent` is None.
        Special case:
          - "no_model" with latent_name starting with "reward_rate_#":
            behavior-only reward-rate computed from the trials table.

    latent_name : str, optional
       Which derived series to return. One of:
          - 'deltaQ'                    → ΔQ (Q₁ − Q₀) after update
          - 'deltaQ-1'                  → ΔQ (after update) with last trial dropped and first-trial value replaced by 0
          - 'deltaQ+1'                  → ΔQ (after update) with first trial dropped and last-trial value replaced by 0
          - 'sumQ'                      → ΣQ (Q₁ + Q₀) after update
          - 'sumQ-1'                    → ΣQ (after update) with last trial dropped and first-trial value replaced by 0
          - 'sumQ+1'                    → ΣQ (after update) with first trial dropped and last-trial value replaced by 0
          - 'right_choice_probability'  → choice_prob[1] after update
          - 'right_choice_probability-1'→ choice_prob[1] with last entry dropped and first entry replaced by 0
          - 'right_choice_probability+1'→ choice_prob[1] with first entry dropped and last entry replaced by 0
          - 'left_choice_probability'   → choice_prob[0] after update
          - 'left_choice_probability-1' → choice_prob[0] with last entry dropped and first entry replaced by 0
          - 'left_choice_probability+1' → choice_prob[0] with first entry dropped and last entry replaced by 0
          - 'RPE'                       → reward_prediction_error 
          - 'RPE-1'                     → RPE with last valid entry dropped and first valid entry replaced by 0
          - 'RPE+1'                     → RPE with first valid entry dropped and last valid entry replaced by 0
          - 'QL'                        → Q for option 0 after update
          - 'QL-1'                      → Q₀ (after update) with last trial dropped and first-trial value replaced by 0
          - 'QL+1'                      → Q₀ (after update) with first trial dropped and last-trial value replaced by 0
          - 'QR'                        → Q for option 1 after update
          - 'QR-1'                      → Q₁ (after update) with last trial dropped and first-trial value replaced by 0
          - 'QR+1'                      → Q₁ (after update) with first trial dropped and last-trial value replaced by 0
          - 'chosenQ'                   → Q of the chosen option (after choice)
          - 'chosenQ-1'                 → Chosen-Q with last valid entry dropped and first valid entry replaced by 0
          - 'chosenQ+1'                 → Chosen-Q with first valid entry dropped and last valid entry replaced by 0
          - 'unchosenQ'                 → Q of the unchosen option (after update)
          - 'unchosenQ-1'               → Unchosen-Q (after update) with last valid entry dropped and first valid entry replaced by 0
          - 'unchosenQ+1'               → Unchosen-Q (after update) with first valid entry dropped and last valid entry replaced by 0
          - 'reward'                    → Returns 1 for rewarded trials, 0 for unrewarded trials (no trimming)
          - 'reward-1'                  → Reward series with last valid entry dropped and first valid entry replaced by 0
          - 'reward+1'                  → Reward series with first valid entry dropped and last valid entry replaced by 0
          - 'choice'                    → Animal's choice per trial (0=left, 1=right) with no trimming, excludes no-response trials
          - 'choice-1'                  → Choice series with last valid entry dropped and first entry replaced by 0 (keep length constant)
          - 'choice+1'                  → Choice series with first valid entry dropped and last entry replaced by 0 (keep length constant)
          - 'value'                     → For model ForagingCompareThreshold.
          - 'value-1'                   → For model ForagingCompareThreshold (with last trial dropped and first-trial value replaced by 0).
          - 'value+1'                   → For model ForagingCompareThreshold (with first trial dropped and last-trial value replaced by 0).
          - 'reward_rate_N' (with model_alias == "no_model"):
                Empirical reward rate over the previous N valid trials,
                computed as (# rewarded) / (# rewarded + # unrewarded),
                excluding no-response trials. The current trial is not included
                in the window; early trials use as many previous valid trials
                as available, and if there are none the rate is set to 0.

    Returns
    -------
    np.ndarray or None
        A 1-D float array for the requested latent series, or None if unsupported.
        All returned arrays (except None) have the same length as the number of
        valid trials, unless otherwise noted.
    """
    try:
        # ------------------------------------------------------------------
        # Special case: behavior-only reward rate (no_model)
        # ------------------------------------------------------------------
        if model_alias == 'no_model':
            pass
        if (
            model_alias == "no_model"
            and latent_name is not None
            and latent_name.startswith("reward_rate_")
        ):
            # Parse window size from latent_name, e.g. "reward_rate_10"
            try:
                window_size = int(latent_name.split("_")[-1])
            except Exception:
                print(f"Invalid reward_rate latent_name: {latent_name}")
                return None

            trials = nwb_behavior_data.trials
            rewardedL = trials['rewarded_historyL'][:]
            rewardedR = trials['rewarded_historyR'][:]
            responses = trials['animal_response'][:]

            # Exclude no-response trials (animal_response == 2), as in 'reward'
            valid_mask = (responses != 2)
            rewarded = (rewardedL | rewardedR).astype(int)[valid_mask]

            n_valid = len(rewarded)
            reward_rate = np.zeros(n_valid, dtype=float)

            # For each valid trial index i, compute reward rate over the
            # previous `window_size` valid trials (excluding trial i itself).
            for i in range(n_valid):
                start = max(0, i - window_size)
                window_rewards = rewarded[start:i]  # previous valid trials only
                if window_rewards.size == 0:
                    reward_rate[i] = 0.0
                else:
                    reward_rate[i] = float(window_rewards.sum()) / float(window_rewards.size)

            return reward_rate

        # ------------------------------------------------------------------
        # 1) Validate inputs and fetch fit if needed (model-based latents)
        # ------------------------------------------------------------------
        if fitted_latent is None:
            if session_name is None or model_alias is None or latent_name is None:
                raise ValueError(
                    "If fitted_latent is not provided, session_name, model_alias, and latent_name are required"
                )
            fit = get_fitted_latent(session_name, model_alias)
        else:
            if latent_name is None:
                raise ValueError("latent_name must be provided when using fitted_latent directly")
            fit = fitted_latent

        # 2) Ensure fetched fit is valid
        if not fit or 'fitted_latent_variables' not in fit:
            return None

        FL = fit['fitted_latent_variables']

        # Extract Q-values arrays if available
        if 'q_value' in FL:
            q0_full = np.array(FL['q_value'][0])
            q1_full = np.array(FL['q_value'][1])
        else:
            q0_full = q1_full = None

        if 'value' in FL:
            value_full = np.array(FL['value'])

        # Helper: apply trimming based on suffix
        def _trim_series(arr: np.ndarray, base: str, suffix: str) -> np.ndarray:
            """
            Trim `arr` according to suffix:
            - ''   : drop first element (base behavior for QL/QR, etc.)
            - '-1' : drop last element
            - '+1' : drop first two elements
            """
            if suffix == '':
                return arr[1:]
            elif suffix == '-1':
                # drop last valid trial; first trial is already initialized
                return arr[:-1]
            elif suffix == '+1':
                # drop the first trial, and append a 0 at the end
                trimmed = arr[2:]
                return np.append(trimmed, 0)
            else:
                raise ValueError(f"Unknown suffix '{suffix}' for {base}")

        # Parse suffix if present
        base_name = latent_name
        suffix = ''
        if latent_name is not None and (latent_name.endswith('-1') or latent_name.endswith('+1')):
            if latent_name.endswith('-1'):
                base_name, suffix_part = latent_name.rsplit('-', 1)
                suffix = '-' + suffix_part
            else:
                base_name, suffix_part = latent_name.rsplit('+', 1)
                suffix = '+' + suffix_part

        # ------------------------------------------------------------------
        # 3) Compute requested series (model-based)
        # ------------------------------------------------------------------

        # ----- value for ForagingCompareThreshold -----
        if base_name == 'value' and model_alias == 'ForagingCompareThreshold':
            return _trim_series(value_full, 'value', suffix)

        # ----- deltaQ -----
        if base_name == 'deltaQ':
            if q0_full is None or q1_full is None:
                return None
            diff = q1_full - q0_full
            return _trim_series(diff, 'deltaQ', suffix)

        # ----- sumQ -----
        if base_name == 'sumQ':
            if q0_full is None or q1_full is None:
                return None
            total = q1_full + q0_full
            return _trim_series(total, 'sumQ', suffix)

        # ----- choice probabilities -----
        if base_name in ('right_choice_probability', 'left_choice_probability'):
            cp = np.array(FL['choice_prob'][1]) if base_name == 'right_choice_probability' else np.array(FL['choice_prob'][0])
            if suffix == '':
                return cp
            elif suffix == '-1':
                # drop last valid trial, then prepend 0
                trimmed = cp[:-1]
                return np.insert(trimmed, 0, 0)
            elif suffix == '+1':
                # drop first valid trial, then append 0
                trimmed = cp[1:]
                return np.append(trimmed, 0)
            else:
                return None

        # ----- Reward Prediction Error (RPE) -----
        if base_name == 'RPE' and model_alias == 'ForagingCompareThreshold':
            trials = nwb_behavior_data.trials
            rewardedL = trials['rewarded_historyL'][:]
            rewardedR = trials['rewarded_historyR'][:]
            responses = trials['animal_response'][:]

            # Drop last trial from value arrays before computing
            value = value_full[:-1]
            valid = responses != 2
            rewarded = (rewardedL | rewardedR).astype(int)[valid]

            rpe_full = rewarded - value

            if suffix == '':
                return rpe_full
            elif suffix == '-1':
                trimmed = rpe_full[:-1]
                return np.insert(trimmed, 0, 0)
            elif suffix == '+1':
                trimmed = rpe_full[1:]
                return np.append(trimmed, 0)
            else:
                return None

        if base_name == 'RPE' and model_alias != 'ForagingCompareThreshold':
            trials = nwb_behavior_data.trials
            rewardedL = trials['rewarded_historyL'][:]
            rewardedR = trials['rewarded_historyR'][:]
            responses = trials['animal_response'][:]

            # Drop last trial from Q arrays before computing
            q0 = q0_full[:-1]
            q1 = q1_full[:-1]
            valid = responses != 2
            rewarded = (rewardedL | rewardedR).astype(int)[valid]
            resp_valid = responses[valid]

            rpe_full = np.where(resp_valid == 0, rewarded - q0, rewarded - q1)

            if suffix == '':
                return rpe_full
            elif suffix == '-1':
                # drop last valid trial, then append a 0 to the begining to keep length consistent
                trimmed = rpe_full[:-1]
                return np.insert(trimmed, 0, 0)
            elif suffix == '+1':
                # drop first valid trial, then prepend a 0 to the last trial to keep length consistent
                trimmed = rpe_full[1:]
                return np.append(trimmed, 0)
            else:
                return None

        # ----- QL and QR -----
        if base_name in ('QL', 'QR'):
            arr_full = q0_full if base_name == 'QL' else q1_full
            return _trim_series(arr_full, base_name, suffix)

        # ----- chosenQ and unchosenQ -----
        if base_name in ('chosenQ', 'unchosenQ'):
            trials = nwb_behavior_data.trials
            rewardedL = trials['rewarded_historyL'][:]
            rewardedR = trials['rewarded_historyR'][:]
            responses = trials['animal_response'][:]

            # Drop last trial from Q arrays
            q0 = q0_full[:-1]
            q1 = q1_full[:-1]
            valid_mask = (responses != 2)
            resp_valid = responses[valid_mask]

            chosen = np.where(resp_valid == 0, q0, q1)
            unchosen = np.where(resp_valid == 0, q1, q0)
            series_full = chosen if base_name == 'chosenQ' else unchosen

            if suffix == '':
                return series_full
            elif suffix == '-1':
                # drop last valid trial, then append a 0 to the begining to keep length consistent
                trimmed = series_full[:-1]
                return np.insert(trimmed, 0, 0)
            elif suffix == '+1':
                # drop first valid trial, then prepend a 0 to the last trial to keep length consistent
                trimmed = series_full[1:]
                return np.append(trimmed, 0)
            else:
                return None

        # ----- reward -----
        if base_name == 'reward':
            trials = nwb_behavior_data.trials
            rewardedL = trials['rewarded_historyL'][:]
            rewardedR = trials['rewarded_historyR'][:]
            responses = trials['animal_response'][:]

            valid_mask = (responses != 2)
            rewarded = (rewardedL | rewardedR).astype(int)[valid_mask]

            if suffix == '':
                return rewarded
            elif suffix == '-1':
                # drop last valid trial, then append a 0 to the begining to keep length consistent
                trimmed = rewarded[:-1]
                return np.insert(trimmed, 0, 0)
            elif suffix == '+1':
                # drop first valid trial, then prepend a 0 to the last trial to keep length consistent
                trimmed = rewarded[1:]
                return np.append(trimmed, 0)
            else:
                return None

        # ----- choice -----
        if base_name == 'choice':
            # pull the raw choice vector (0=left, 1=right, 2=no‐response)
            all_resp = nwb_behavior_data.trials['animal_response'][:]

            # exclude no‐response trials
            valid_mask = all_resp != 2
            resp = all_resp[valid_mask]

            # now apply suffix‐based trimming
            if suffix == '':
                return resp
            elif suffix == '-1':
                # drop last valid trial, prepend a 0 to keep length consistent
                trimmed = resp[:-1]
                return np.insert(trimmed, 0, 0)
            elif suffix == '+1':
                # drop first valid trial, append a 0
                trimmed = resp[1:]
                return np.append(trimmed, 0)
            else:
                return None


        # Unsupported latent_name
        return None

    except Exception as e:
        # print your custom message plus the exception’s own message
        print(f"Can't extract {latent_name} from {model_alias}: {e}")
        return None


def find_trials(
    nwb_behavior_data: Any,
    trial_type: Union[str, List[str]] = 'no_response'
) -> List[int]:
    """
    Return trial indices matching one or more trial_type values.

    Parameters
    ----------
    nwb_behavior_data : Any
        An NWB behavior object with a `.trials` table containing at least:
          - 'animal_response'          (0=left, 1=right, 2=no‐response)
          - 'rewarded_historyL'        (boolean array)
          - 'rewarded_historyR'        (boolean array)

    trial_type : str or list of str, optional (default: 'no_response')
        One or more of the following:

        **Standard types**
        - 'no_response'      : trials where animal_response == 2
        - 'response'         : trials where animal_response != 2
        - 'rewarded'         : trials where either side was rewarded
        - 'unrewarded'       : trials where no reward and response != 2
        - 'left_rewarded'    : trials where left side was rewarded
        - 'right_rewarded'   : trials where right side was rewarded
        - 'left_choice'      : left choice trials
        - 'right_choice'     : right choice trials

        **Switch-based types**
        - 'switch_trial'             : any switch between consecutive trials
        - 'switch_trial_reward'      : switch trials that **are** rewarded
        - 'switch_trial_noreward'    : switch trials that **aren't** rewarded
        - 'switch_LR'                : switch from left → right
        - 'switch_LR_reward'         : left→right **and** rewarded
        - 'switch_LR_noreward'       : left→right **and not** rewarded
        - 'switch_RL'                : switch from right → left
        - 'switch_RL_reward'         : right→left **and** rewarded
        - 'switch_RL_noreward'       : right→left **and not** rewarded

    Returns
    -------
    List[int]
        Zero-based indices of trials matching the specified type(s).
    """
    # If given a list, union results for each type recursively
    if isinstance(trial_type, (list, tuple)):
        idx_set = set()
        for t in trial_type:
            idx_set.update(find_trials(nwb_behavior_data, t))
        return sorted(idx_set)

    trials = nwb_behavior_data.trials
    resp = trials['animal_response'][:]
    rewardedL = trials['rewarded_historyL'][:]
    rewardedR = trials['rewarded_historyR'][:]
    rewarded = np.logical_or(rewardedL, rewardedR)

    # Handle all switch‐based trial types (including reward variants)
    switch_types = {
        'switch_trial',
        'switch_trial_reward',
        'switch_trial_noreward',
        'switch_LR',
        'switch_LR_reward',
        'switch_LR_noreward',
        'switch_RL',
        'switch_RL_reward',
        'switch_RL_noreward',
    }
    if trial_type in switch_types:
        switch_indices: List[int] = []
        for idx in range(1, len(resp)):
            prev, curr = resp[idx-1], resp[idx]
            # skip if either is no-response
            if prev == 2 or curr == 2:
                continue

            is_switch = (curr != prev)
            is_LR = (prev == 0 and curr == 1)
            is_RL = (prev == 1 and curr == 0)
            is_rew = bool(rewarded[idx])

            if trial_type == 'switch_trial' and is_switch:
                switch_indices.append(idx)
            elif trial_type == 'switch_trial_reward' and is_switch and is_rew:
                switch_indices.append(idx)
            elif trial_type == 'switch_trial_noreward' and is_switch and not is_rew:
                switch_indices.append(idx)
            elif trial_type == 'switch_LR' and is_LR:
                switch_indices.append(idx)
            elif trial_type == 'switch_LR_reward' and is_LR and is_rew:
                switch_indices.append(idx)
            elif trial_type == 'switch_LR_noreward' and is_LR and not is_rew:
                switch_indices.append(idx)
            elif trial_type == 'switch_RL' and is_RL:
                switch_indices.append(idx)
            elif trial_type == 'switch_RL_reward' and is_RL and is_rew:
                switch_indices.append(idx)
            elif trial_type == 'switch_RL_noreward' and is_RL and not is_rew:
                switch_indices.append(idx)

        return switch_indices

    # Standard trial types
    if trial_type == 'no_response':
        return np.where(resp == 2)[0].tolist()
    if trial_type == 'left_choice':
        return np.where(resp == 0)[0].tolist()
    if trial_type == 'right_choice':
        return np.where(resp == 1)[0].tolist()
    elif trial_type == 'response':
        return np.where(resp != 2)[0].tolist()
    elif trial_type == 'rewarded':
        return np.where(rewarded)[0].tolist()
    elif trial_type == 'unrewarded':
        mask = np.logical_and(~rewarded, resp != 2)
        return np.where(mask)[0].tolist()
    elif trial_type == 'left_rewarded':
        return np.where(rewardedL)[0].tolist()
    elif trial_type == 'right_rewarded':
        return np.where(rewardedR)[0].tolist()
    else:
        raise ValueError(f"Unsupported trial_type '{trial_type}'")


def generate_behavior_summary(
    nwb_data: Any,
    model_alias:  Union[str, List[str]] = [
        'ForagingCompareThreshold',
        'QLearning_L1F1_CK1_softmax',
        'QLearning_L2F1_softmax',
        'QLearning_L2F1_CK1_softmax',
        'QLearning_L2F1_CKfull_softmax',
        'q_learning_Y1',
        'no_model',
        'QLearning_L1F0_CKfull_softmax',
        'QLearning_L1F1_CKfull_softmax'
    ],
    latent_names: Optional[List[str]] = None,
    trial_types : Optional[List[str]]  = None
) -> pd.DataFrame:
    """
    Build a one-row behavioral summary for a single session.

    This version avoids repeatedly calling `get_fitted_latent`:
    for each model alias (except 'no_model' and the locally-fitted
    'q_learning_Y1'), we call `get_fitted_latent(...)` **once** and
    cache the result, then reuse that cached `fitted_latent` for all
    requested latent_names via the `fitted_latent=` argument of
    `extract_fitted_data`.

    If *model_alias* includes "q_learning_Y1", the function fits that model
    locally with `fit_q_learning_model(...)` and uses its latent variables
    instead of calling the remote API.
    """
    # ------------------------------------------------------------------
    # 0. Session identifier
    # ------------------------------------------------------------------
    full_session_name = getattr(nwb_data, 'session_id', None)
    session_id = extract_session_name_core(full_session_name) or full_session_name

    # ------------------------------------------------------------------
    # 1. Normalise inputs
    # ------------------------------------------------------------------
    aliases = [model_alias] if isinstance(model_alias, str) else list(model_alias)

    if latent_names is None:
        latent_names = [
            'deltaQ',
            'deltaQ-1',
            'deltaQ+1',
            'sumQ',
            'sumQ-1',
            'sumQ+1',
            'right_choice_probability',
            'right_choice_probability-1',
            'right_choice_probability+1',
            'left_choice_probability',
            'left_choice_probability-1',
            'left_choice_probability+1',
            'RPE',
            'RPE-1',
            'RPE+1',
            'QL',
            'QL-1',
            'QL+1',
            'QR',
            'QR-1',
            'QR+1',
            'chosenQ',
            'chosenQ-1',
            'chosenQ+1',
            'unchosenQ',
            'unchosenQ-1',
            'unchosenQ+1',
            'reward',
            'reward-1',
            'reward+1',
            'choice',
            'choice-1',
            'choice+1',
            'value',
            'value-1',
            'value+1',
        ]


    # Adding different reward rates
    window=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    alpha=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.75,0.8,0.85,0.9,0.95]
    latent_names += [f"reward_rate_window_{i}" for i in window]
    latent_names += [f"reward_rate_alpha_{i}" for i in alpha]

    if trial_types is None:
        trial_types = [
            'no_response', 'response', 'rewarded', 'unrewarded',
            'left_rewarded', 'right_rewarded',
            'switch_trial',
            'switch_trial_reward',
            'switch_trial_noreward',
            'switch_LR',
            'switch_LR_reward',
            'switch_LR_noreward',
            'switch_RL',
            'switch_RL_reward',
            'switch_RL_noreward',
            'left_choice',
            'right_choice',
        ]

    # ------------------------------------------------------------------
    # 2. Caches
    # ------------------------------------------------------------------
    # For locally fitted models (currently only 'q_learning_Y1')
    local_fit_cache: Dict[str, Dict[str, Any]] = {}

    # For remotely fetched fits (other aliases except 'no_model')
    remote_fit_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    # Reward-rate caches (per session) to avoid recomputation
    rr_running_cache: Dict[int, Dict[str, Any]] = {}
    rr_ewma_cache: Dict[float, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 3. Assemble summary
    # ------------------------------------------------------------------
    summary: Dict[str, Any] = {'session_id': session_id}

    for alias in aliases:
        # --------------------------------------------------------------
        # 3a. Obtain (or compute) fitted_latent for this alias ONCE
        # --------------------------------------------------------------
        if alias == 'q_learning_Y1':
            # Fit only once per session for this alias
            if alias not in local_fit_cache:
                fit_dict = fit_q_learning_model(nwb_data, model_name=alias)
                local_fit_cache[alias] = fit_dict
            fit_source = 'local'

        elif alias == 'no_model':
            # Behavior-only quantities (reward_rate_N, etc.)
            fit_source = 'no_model'

        else:
            # Remote / archived model fits
            if alias not in remote_fit_cache:
                # This calls get_fitted_latent(session_name, alias) ONCE
                remote_fit_cache[alias] = get_fitted_latent(
                    session_name=full_session_name,
                    model_alias=alias
                )
            fit_source = 'remote'

        # --------------------------------------------------------------
        # 3b. Add each requested latent series
        # --------------------------------------------------------------
        for ln in latent_names:
            # ----------------------------------------------------------
            # Reward-rate special latents (behavior-only; not model-based)
            # ----------------------------------------------------------
            if (ln.startswith("reward_rate_window_") or ln.startswith("reward_rate_alpha_")) and alias == 'no_model':
                # These are behavior-only; we compute them once per (window/alpha)
                # and store three output series as separate columns.

                if ln.startswith("reward_rate_window_"):
                    # Parse window size
                    try:
                        w = int(ln.split("reward_rate_window_")[-1])
                    except Exception:
                        # If parse fails, set columns to None and move on
                        for k in ("running_left_reward", "running_right_reward", "running_experienced"):
                            summary[f"{alias}-{ln}-{k}"] = None
                        continue

                    # Compute (or reuse) running reward-rate family
                    if w not in rr_running_cache:
                        rr_running_cache[w] = compute_all_reward_rates(
                            nwb_data=nwb_data,
                            window=w,
                            alpha=None,
                            include_noresponse=False,
                            metrics="running",
                            include_raw=False,
                            drop_noresponse_trials=True,
                        )

                    rr = rr_running_cache[w]
                    out_keys = ("running_left_reward", "running_right_reward", "running_experienced")

                    for k in out_keys:
                        col_name_rr = f"{alias}-{ln}-{k}"
                        values_rr = rr.get(k, None)
                        summary[col_name_rr] = values_rr.tolist() if values_rr is not None else None

                    # Skip the normal latent extraction path for this ln
                    continue

                # Else: reward_rate_alpha_
                try:
                    a = float(ln.split("reward_rate_alpha_")[-1])
                except Exception:
                    for k in ("ewma_left_reward", "ewma_right_reward", "ewma_experienced"):
                        summary[f"{alias}-{ln}-{k}"] = None
                    continue

                # Compute (or reuse) EWMA reward-rate family
                if a not in rr_ewma_cache:
                    rr_ewma_cache[a] = compute_all_reward_rates(
                        nwb_data=nwb_data,
                        window=None,
                        alpha=a,
                        include_noresponse=False,
                        metrics="ewma",
                        include_raw=False,
                        drop_noresponse_trials=True,
                    )

                rr = rr_ewma_cache[a]
                out_keys = ("ewma_left_reward", "ewma_right_reward", "ewma_experienced")

                for k in out_keys:
                    col_name_rr = f"{alias}-{ln}-{k}"
                    values_rr = rr.get(k, None)
                    summary[col_name_rr] = values_rr.tolist() if values_rr is not None else None

                continue


            col_name = f"{alias}-{ln}"
            # ---- Local fit ('q_learning_Y1') ----
            if fit_source == 'local':
                fit_dict = local_fit_cache.get(alias)
                if not fit_dict:
                    summary[col_name] = None
                    continue

                values = extract_fitted_data(
                    nwb_behavior_data=nwb_data,
                    fitted_latent=fit_dict,
                    latent_name=ln,
                )

            # ---- Behavior-only, no model fit ('no_model') ----
            elif fit_source == 'no_model':
                values = extract_fitted_data(
                    nwb_behavior_data=nwb_data,
                    session_name=full_session_name,
                    model_alias='no_model',
                    latent_name=ln,
                )

            # ---- Remote fit (cached result from get_fitted_latent) ----
            else:  # fit_source == 'remote'
                fit_dict = remote_fit_cache.get(alias)
                if not fit_dict:
                    # Could not fetch / parse this alias for this session
                    summary[col_name] = None
                    continue

                values = extract_fitted_data(
                    nwb_behavior_data=nwb_data,
                    fitted_latent=fit_dict,
                    model_alias=alias,
                    latent_name=ln,
                )

            summary[col_name] = values.tolist() if values is not None else None

    # ------------------------------------------------------------------
    # 4. Add trial-type index lists
    # ------------------------------------------------------------------
    for tt in trial_types:
        summary[f"{tt}_trials"] = find_trials(nwb_data, tt)

    # ------------------------------------------------------------------
    # 5. Return single-row DataFrame
    # ------------------------------------------------------------------
    return pd.DataFrame([summary])




def generate_behavior_summary_combined(
    session_names: List[str],
    model_alias:  Union[str, List[str]] = ['ForagingCompareThreshold','QLearning_L1F1_CK1_softmax', 'QLearning_L2F1_softmax', 'QLearning_L2F1_CK1_softmax','QLearning_L2F1_CKfull_softmax','q_learning_Y1','no_model', 'QLearning_L1F0_CKfull_softmax','QLearning_L1F1_CKfull_softmax'],
    latent_names: Optional[List[str]] = None,
    trial_types: Optional[List[str]] = None,
    save_folder: str = '/root/capsule/results',
    save_name: str = 'combined_behavior_summary.csv',
    save_result: bool = False
) -> pd.DataFrame:
    """
    Generate and optionally save combined behavior summaries across sessions.

    Parameters
    ----------
    session_names : list of str
        NWB session identifiers.
    model_alias : str or list of str
        Alias(es) for the fitted model(s) to summarize.
    latent_names : list of str, optional
        Which latent variables to include (defaults to all supported).
    trial_types : list of str, optional
        Which trial categories to flag (defaults to key types).
    save_folder : str
        Directory in which to save the combined CSV if requested.
    save_name : str
        Filename for saving the combined summary CSV.
    save_result : bool
        If True, save the combined summary to disk.

    Returns
    -------
    pd.DataFrame
        One row per session with summary columns.
    """
    all_summaries: List[pd.DataFrame] = []
    for sess in session_names:
        nwb_data = NWBUtils.read_ophys_or_behavior_nwb(
            session_name=sess
        )
        if nwb_data is None:
            print(f"Warning: could not load session {sess}, skipping.")
            continue
        summary_df = generate_behavior_summary(
            nwb_data=nwb_data,
            model_alias=model_alias,
            latent_names=latent_names,
            trial_types=trial_types
        )
        try:
            nwb_data.io.close()
        except Exception:
            pass
        all_summaries.append(summary_df)

    if all_summaries:
        combined_df = pd.concat(all_summaries, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    if save_result:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, save_name)
        combined_df.to_csv(save_path, index=False)
        print(f"Combined summary saved to {save_path}")

    return combined_df


def load_and_combine_csvs(
    folder: Union[str, Path],
    pattern: str = "behavior_summary-*.csv"
) -> pd.DataFrame:
    """
    Load all CSV files in `folder` matching `pattern` and concatenate them into one DataFrame.
    Empty files are skipped gracefully.

    Parameters
    ----------
    folder : str or Path
        Directory in which to look for CSV files.
    pattern : str, optional
        Glob pattern to match filenames (default "behavior_summary-*.csv").

    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame containing all rows from all matched files.

    Raises
    ------
    FileNotFoundError
        If no files matching the pattern are found in `folder`.
    ValueError
        If none of the matched files could be loaded (all empty or invalid).
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching pattern {pattern!r} in folder {folder}")

    dfs = []
    for file in files:
        try:
            df = smart_read_csv(str(file))
            dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")
        except Exception as e:
            print(f"Warning: failed to read {file}: {e}")

    if not dfs:
        raise ValueError(f"Found {len(files)} files matching pattern, but none could be loaded.")

    combined = pd.concat(dfs, ignore_index=True)
    return combined

# Example usage:
# combined_df = load_and_combine_csvs("/root/capsule/results", "behavior_summary-*.csv")
# print(combined_df.shape)




####################################################################################################################################################

# ============================================================
# Reward-rate metrics for 2-lickspout dynamic foraging (NWB)
#
# Standardized NWB trial fields (assumed fixed)
# ---------------------------------------------
#   - nwb_data.trials['rewarded_historyL'][:] : bool
#   - nwb_data.trials['rewarded_historyR'][:] : bool
#   - nwb_data.trials['animal_response'][:]   : int
#       0 = left, 1 = right, 2 = noresponse
#
# Global convention used throughout
# ---------------------------------
# Every function has include_noresponse:
#   - include_noresponse=False:
#       Exclude noresponse (choice==2) trials from the metric.
#       Means: those trials are treated as NaN / skipped; EWMA holds value.
#   - include_noresponse=True:
#       Include noresponse trials as 0-reward contributions.
#       Means: omissions reduce reward rate.
# ============================================================


# ------------------------------
# Basic extractors (standard keys)
# ------------------------------

def _as_bool_trials(nwb_data, key: str) -> np.ndarray:
    """Load a boolean trial column from nwb_data.trials."""
    return np.asarray(nwb_data.trials[key][:], dtype=bool)


def _as_int_trials(nwb_data, key: str) -> np.ndarray:
    """Load an integer trial column from nwb_data.trials."""
    return np.asarray(nwb_data.trials[key][:], dtype=int)


def get_choices(nwb_data) -> np.ndarray:
    """
    Get per-trial choices from the standardized column 'animal_response'.

    Returns
    -------
    choices : (n_trials,) int ndarray
      0 = left, 1 = right, 2 = noresponse
    """
    choices = _as_int_trials(nwb_data, "animal_response")
    if choices.ndim != 1:
        raise ValueError(f"Expected 1D choices vector, got shape={choices.shape}")
    return choices


def responded_mask(choices: np.ndarray) -> np.ndarray:
    """Boolean mask for trials with an actual response (left/right)."""
    choices = np.asarray(choices, dtype=int)
    return (choices == 0) | (choices == 1)


# ------------------------------
# Reward vectors with omission handling
# ------------------------------

def get_reward_vectors(nwb_data, *, include_noresponse: bool = False) -> Dict[str, np.ndarray]:
    """
    Get canonical reward vectors with consistent omission handling.

    Definitions (before omission handling)
    --------------------------------------
    rL[t]   = 1 if left reward delivered on trial t else 0
    rR[t]   = 1 if right reward delivered on trial t else 0
    rAny[t] = 1 if reward delivered on either side on trial t else 0

    Omission handling
    -----------------
    If include_noresponse=False:
      reward values at choice==2 trials are set to NaN (so means ignore them).
    If include_noresponse=True:
      reward values at choice==2 trials are set to 0 (so omissions reduce rates).

    Returns
    -------
    dict:
      'left'  -> (n_trials,) float ndarray
      'right' -> (n_trials,) float ndarray
      'any'   -> (n_trials,) float ndarray
    """
    rL = _as_bool_trials(nwb_data, "rewarded_historyL").astype(float)
    rR = _as_bool_trials(nwb_data, "rewarded_historyR").astype(float)
    if rL.shape != rR.shape:
        raise ValueError(f"rewarded_historyL shape {rL.shape} != rewarded_historyR shape {rR.shape}")

    rAny = ((rL > 0) | (rR > 0)).astype(float)

    choices = get_choices(nwb_data)
    if choices.shape[0] != rAny.shape[0]:
        raise ValueError("choices length must match reward vectors length")

    noresp = (choices == 2)
    if include_noresponse:
        rL[noresp] = 0.0
        rR[noresp] = 0.0
        rAny[noresp] = 0.0
    else:
        rL[noresp] = np.nan
        rR[noresp] = np.nan
        rAny[noresp] = np.nan

    return {"left": rL, "right": rR, "any": rAny}


def get_experienced_reward(nwb_data, *, include_noresponse: bool = False) -> np.ndarray:
    """
    Experienced reward per trial = reward on the chosen side.

    Definition
    ----------
    experienced[t] =
      rL[t] if choice[t] == 0
      rR[t] if choice[t] == 1
      NaN   if choice[t] == 2 and include_noresponse=False
      0     if choice[t] == 2 and include_noresponse=True

    Returns
    -------
    experienced : (n_trials,) float ndarray
    """
    # Use base reward vectors with omissions included as 0, then apply experienced mapping
    r_base = get_reward_vectors(nwb_data, include_noresponse=True)
    choices = get_choices(nwb_data)

    exp = np.full_like(r_base["any"], np.nan, dtype=float)
    exp[choices == 0] = r_base["left"][choices == 0]
    exp[choices == 1] = r_base["right"][choices == 1]

    if include_noresponse:
        exp[choices == 2] = 0.0

    return exp


# ------------------------------
# Metric 1: Global reward rate
# ------------------------------

def reward_rate_global(
    reward: np.ndarray,
    *,
    include_noresponse: bool,
    choices: Optional[np.ndarray] = None,
) -> float:
    """
    Global reward rate = mean(reward) over selected trials.

    Noresponse handling
    -------------------
    - include_noresponse=True:
        Include all trials as they appear in `reward` (typically 0 on omissions).
    - include_noresponse=False:
        Exclude omission trials. Requires `choices` (or pass reward with NaNs on omissions).

    Notes
    -----
    If you generated `reward` using get_reward_vectors(..., include_noresponse=False),
    omission trials are NaN and np.nanmean will already exclude them. In that case,
    passing choices is optional, but supported for clarity.

    Returns
    -------
    float
    """
    reward = np.asarray(reward, dtype=float)

    if not include_noresponse:
        if choices is not None:
            m = responded_mask(choices)
            reward = reward[m]
        # If choices is None, we rely on NaNs (if present) and nanmean below.

    if reward.size == 0:
        return float("nan")
    return float(np.nanmean(reward))


# ------------------------------
# Metric 2: Running reward rate
# ------------------------------
def reward_rate_running(
    reward: np.ndarray,
    *,
    window: int = 20,
    causal: bool = True,
    include_noresponse: bool,
    choices: Optional[np.ndarray] = None,
    init: float = 0.0,
) -> np.ndarray:
    """
    Sliding-window reward rate with explicit initialization.

    Definition (causal=True)
    ------------------------
    rate[0] = init
    rate[t] = mean(reward[max(0, t-window):t])  (past-only; excludes current trial)

    Noresponse handling
    -------------------
    - include_noresponse=True:
        Window mean includes omissions as they appear in `reward` (often 0).
    - include_noresponse=False:
        Exclude omissions from each window mean.
        Requires `choices`, OR reward encoded with NaNs on omissions.

    Returns
    -------
    rate : (n_trials,) float ndarray
    """
    reward = np.asarray(reward, dtype=float)
    n = reward.size
    rate = np.full(n, np.nan, dtype=float)

    # Explicit initialization (matches EWMA semantics)
    if n > 0:
        rate[0] = float(init)

    if (not include_noresponse) and (choices is not None):
        choices = np.asarray(choices, dtype=int)

    for t in range(1, n):
        if causal:
            lo = max(0, t - window)
            hi = t
        else:
            lo = max(0, t - window // 2)
            hi = min(n, t + window // 2 + 1)

        if hi <= lo:
            rate[t] = rate[t - 1]
            continue

        x = reward[lo:hi]

        if include_noresponse:
            rate[t] = float(np.nanmean(x)) if x.size > 0 else rate[t - 1]
        else:
            if choices is None:
                rate[t] = float(np.nanmean(x)) if x.size > 0 else rate[t - 1]
            else:
                m = responded_mask(choices[lo:hi])
                vals = x[m]
                rate[t] = float(np.nanmean(vals)) if vals.size > 0 else rate[t - 1]

    return rate


# ------------------------------
# Metric 3: EWMA reward rate
# ------------------------------

def reward_rate_ewma(
    reward: np.ndarray,
    *,
    alpha: float = 0.05,
    init: float = 0.0,
    include_noresponse: bool,
    choices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    EWMA reward rate (past-only).

    Definition
    ----------
    rr[0] = init
    rr[t] = (1 - alpha) * rr[t-1] + alpha * reward[t-1]   if update trial is included
    rr[t] = rr[t-1]                                       if update trial is excluded

    Noresponse handling
    -------------------
    - include_noresponse=True:
        EWMA updates every trial using reward[t-1] (omissions typically contribute 0).
    - include_noresponse=False:
        EWMA does NOT update on omission trials (holds value constant).
        Requires choices, unless omission trials are encoded as NaN in reward, in which
        case NaNs also trigger "hold constant".

    Returns
    -------
    rr : (n_trials,) float ndarray
    """
    reward = np.asarray(reward, dtype=float)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")

    if (not include_noresponse) and (choices is not None):
        choices = np.asarray(choices, dtype=int)

    n = reward.size
    rr = np.zeros(n, dtype=float)
    rr[0] = float(init)

    for t in range(1, n):
        prev = rr[t - 1]
        x = reward[t - 1]  # past-only update source

        if include_noresponse:
            rr[t] = (1.0 - alpha) * prev + alpha * (0.0 if np.isnan(x) else x)
        else:
            if choices is None:
                # If x is NaN (e.g., omission), hold rr constant
                rr[t] = prev if np.isnan(x) else (1.0 - alpha) * prev + alpha * x
            else:
                # Only update if previous trial had a response
                if choices[t - 1] in (0, 1) and np.isfinite(x):
                    rr[t] = (1.0 - alpha) * prev + alpha * x
                else:
                    rr[t] = prev

    return rr


def alpha_from_half_life(half_life_trials: float) -> float:
    """
    Convert a desired half-life (trials) to EWMA alpha:
      alpha = 1 - 2^(-1/H)
    """
    if half_life_trials <= 0:
        raise ValueError("half_life_trials must be > 0")
    return 1.0 - 2.0 ** (-1.0 / half_life_trials)


# ------------------------------
# One-call wrapper
# ------------------------------
def compute_all_reward_rates(
    nwb_data,
    *,
    window: Optional[int] = None,
    alpha: Optional[float] = None,
    include_noresponse: bool = True,
    metrics: Union[str, Sequence[str]] = "all",
    include_raw: bool = True,
    drop_noresponse_trials: bool = False,
) -> Dict[str, Any]:
    """
    Compute reward-rate metrics with configurable output families.

    Parameters
    ----------
    window : int or None
        Window length for running reward rate (required only if "running" is requested).
    alpha : float or None
        EWMA smoothing parameter (required only if "ewma" is requested).
    include_noresponse : bool
        Omission handling (consistent with get_reward_vectors / get_experienced_reward).
    metrics : str or sequence of str
        Which metric families to compute/return. Supported:
          - "global"
          - "running"
          - "ewma"
          - "all" (default) = global + running + ewma
    include_raw : bool
        If True, return raw vectors (reward_any/left/right, experienced_reward, choices, responded_mask).
    drop_noresponse_trials : bool
        If True, drop no-response trials (choice==2) from the *array-valued* outputs of:
          - running_* arrays
          - ewma_* arrays

        Notes
        -----
        - This does NOT affect the scalar global_* metrics (they are already controlled by include_noresponse).
        - This also does NOT change how the metrics are computed; it only filters the returned arrays.
        - If you need raw vectors to be dropped as well for alignment, you can easily extend the same mask
          to raw outputs (see the commented block near the end).

    Returns
    -------
    dict
        Dictionary containing requested metrics.
    """
    # ------------------------------------------------------------
    # Normalize metrics selection
    # ------------------------------------------------------------
    if isinstance(metrics, str):
        metrics_set = {metrics.lower()}
    else:
        metrics_set = {str(m).lower() for m in metrics}

    if "all" in metrics_set:
        metrics_set = {"global", "running", "ewma"}

    allowed = {"global", "running", "ewma"}
    unknown = sorted(metrics_set - allowed)
    if unknown:
        raise ValueError(f"Unknown metrics={unknown}. Allowed: {sorted(allowed)} or 'all'.")

    # ------------------------------------------------------------
    # Validate required parameters *only if needed*
    # ------------------------------------------------------------
    if "running" in metrics_set:
        if window is None or window <= 0:
            raise ValueError("`window` must be a positive int when 'running' metrics are requested.")

    if "ewma" in metrics_set:
        if alpha is None or not (0.0 < alpha <= 1.0):
            raise ValueError("`alpha` must be in (0, 1] when 'ewma' metrics are requested.")

    # ------------------------------------------------------------
    # Shared base vectors
    # ------------------------------------------------------------
    choices = get_choices(nwb_data)
    rewards = get_reward_vectors(nwb_data, include_noresponse=include_noresponse)
    exp = get_experienced_reward(nwb_data, include_noresponse=include_noresponse)

    resp_mask = responded_mask(choices)  # True for choice in {0,1}, False for choice==2

    out: Dict[str, Any] = {
        "n_trials": int(rewards["any"].size),
        "include_noresponse": bool(include_noresponse),
        "drop_noresponse_trials": bool(drop_noresponse_trials),
    }

    # ------------------------------------------------------------
    # Global metrics
    # ------------------------------------------------------------
    if "global" in metrics_set:
        out["global_any"] = reward_rate_global(
            rewards["any"], include_noresponse=include_noresponse, choices=choices
        )
        out["global_left_reward"] = reward_rate_global(
            rewards["left"], include_noresponse=include_noresponse, choices=choices
        )
        out["global_right_reward"] = reward_rate_global(
            rewards["right"], include_noresponse=include_noresponse, choices=choices
        )
        out["global_experienced"] = reward_rate_global(
            exp, include_noresponse=include_noresponse, choices=choices
        )

    # ------------------------------------------------------------
    # Running metrics (past-only)
    # ------------------------------------------------------------
    if "running" in metrics_set:
        out["running_any"] = reward_rate_running(
            rewards["any"],
            window=window,
            causal=True,
            include_noresponse=include_noresponse,
            choices=choices,
        )
        out["running_left_reward"] = reward_rate_running(
            rewards["left"],
            window=window,
            causal=True,
            include_noresponse=include_noresponse,
            choices=choices,
        )
        out["running_right_reward"] = reward_rate_running(
            rewards["right"],
            window=window,
            causal=True,
            include_noresponse=include_noresponse,
            choices=choices,
        )
        out["running_experienced"] = reward_rate_running(
            exp,
            window=window,
            causal=True,
            include_noresponse=include_noresponse,
            choices=choices,
        )

    # ------------------------------------------------------------
    # EWMA metrics (past-only)
    # ------------------------------------------------------------
    if "ewma" in metrics_set:
        out["ewma_any"] = reward_rate_ewma(
            rewards["any"],
            alpha=alpha,
            init=0.0,
            include_noresponse=include_noresponse,
            choices=choices,
        )
        out["ewma_left_reward"] = reward_rate_ewma(
            rewards["left"],
            alpha=alpha,
            init=0.0,
            include_noresponse=include_noresponse,
            choices=choices,
        )
        out["ewma_right_reward"] = reward_rate_ewma(
            rewards["right"],
            alpha=alpha,
            init=0.0,
            include_noresponse=include_noresponse,
            choices=choices,
        )
        out["ewma_experienced"] = reward_rate_ewma(
            exp,
            alpha=alpha,
            init=0.0,
            include_noresponse=include_noresponse,
            choices=choices,
        )

    # ------------------------------------------------------------
    # Optionally drop no-response trials from running/ewma arrays
    # ------------------------------------------------------------
    if drop_noresponse_trials:
        # Only filter array-valued outputs from running_* and ewma_* families.
        for k in list(out.keys()):
            if k.startswith("running_") or k.startswith("ewma_"):
                v = out.get(k, None)
                if isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == resp_mask.shape[0]:
                    out[k] = v[resp_mask]

        out["n_trials_after_drop"] = int(resp_mask.sum())

    # ------------------------------------------------------------
    # Raw vectors (optional)
    # ------------------------------------------------------------
    if include_raw:
        out["reward_any"] = rewards["any"]
        out["reward_left"] = rewards["left"]
        out["reward_right"] = rewards["right"]
        out["experienced_reward"] = exp
        out["choices"] = choices
        out["responded_mask"] = resp_mask

        # If you ever want raw vectors also dropped when drop_noresponse_trials=True,
        # uncomment the block below to keep everything aligned by default:
        #
        # if drop_noresponse_trials:
        #     out["reward_any"] = out["reward_any"][resp_mask]
        #     out["reward_left"] = out["reward_left"][resp_mask]
        #     out["reward_right"] = out["reward_right"][resp_mask]
        #     out["experienced_reward"] = out["experienced_reward"][resp_mask]
        #     out["choices"] = out["choices"][resp_mask]
        #     out["responded_mask"] = out["responded_mask"][resp_mask]

    return out





# ------------------------------
# Example usage
# ------------------------------
# alpha = alpha_from_half_life(half_life_trials=20)
# rates = compute_all_reward_rates(nwb_data, window=20, alpha=alpha, include_noresponse=False)
# print("Global any-reward rate:", rates["global_any"])






def alpha_from_half_life(H: float) -> float:
    """
    Convert a desired half-life (in trials) into EWMA alpha.

    What “half-life” means
    ----------------------
    Half-life H means: the influence (weight) of an observation decays by 50%
    after H trials.

    In EWMA, the relative influence of an observation k trials ago (compared to the most recent)
    decays geometrically as (1 - alpha)^k. So the half-life condition is:

        (1 - alpha)^H = 1/2

    Solve for alpha:

        1 - alpha = 2^(-1/H)
        alpha = 1 - 2^(-1/H)

    Interpretation
    --------------
    - Smaller alpha -> longer half-life (slower forgetting; smoother estimate).
    - Larger alpha  -> shorter half-life (faster adaptation; noisier estimate).

    Parameters
    ----------
    H : float
        Half-life in trials. Must be > 0.

    Returns
    -------
    alpha : float
        EWMA smoothing parameter in (0, 1).
    """
    if H <= 0:
        raise ValueError("H must be > 0")
    return 1.0 - 2.0 ** (-1.0 / H)


def alpha_from_mean_lag(L: float) -> float:
    """
    Convert a desired mean lag (expected age of evidence, in trials) into EWMA alpha.

    What “mean lag” means
    ---------------------
    Consider EWMA weights over past trials:

        w_k = alpha * (1 - alpha)^k,   k = 0, 1, 2, ...

    where k=0 is the most recent past trial, k=1 is two trials back, etc.

    The expected lag (mean age) of the evidence under this distribution is:

        E[k] = (1 - alpha) / alpha

    Setting E[k] = L and solving:

        (1 - alpha) / alpha = L
        1 - alpha = L * alpha
        1 = (L + 1) * alpha
        alpha = 1 / (L + 1)

    How to use this for “uses last N trials”
    ----------------------------------------
    A hard window of N trials has an average lag roughly at the midpoint:

        L ≈ (N - 1) / 2

    So if you believe the animal’s “center of mass” of evidence is around the middle
    of the last N trials, this mapping is appropriate.

    Interpretation
    --------------
    - This does NOT force “most weight inside last N trials”.
    - Instead, it places the *average age* of evidence near your chosen value.

    Parameters
    ----------
    L : float
        Mean lag in trials. Must be >= 0.

    Returns
    -------
    alpha : float
        EWMA smoothing parameter in (0, 1].
    """
    if L < 0:
        raise ValueError("L must be >= 0")
    return 1.0 / (L + 1.0)


def alpha_from_mass_within_last_N(N: int, p: float) -> float:
    """
    Choose alpha so that a fraction p of total EWMA weight lies within the last N trials.

    What “mass within last N” means
    -------------------------------
    The EWMA weights form a geometric distribution over lags:

        w_k = alpha * (1 - alpha)^k

    The cumulative weight in the most recent N trials is:

        sum_{k=0..N-1} w_k = 1 - (1 - alpha)^N

    We choose alpha so that:

        1 - (1 - alpha)^N = p

    Solve:

        (1 - alpha)^N = 1 - p
        1 - alpha = (1 - p)^(1/N)
        alpha = 1 - (1 - p)^(1/N)

    Interpretation
    --------------
    This is the most literal translation of:
      “the animal mainly uses the last N trials”
    because it lets you state exactly how dominant the last N trials are.

    Examples (N=5)
    --------------
    - p = 0.90 means 90% of the total EWMA weight is within last 5 trials.
    - p = 0.95 means EWMA is very close to a 5-trial hard window.

    Parameters
    ----------
    N : int
        Number of recent trials to capture.
    p : float
        Desired cumulative weight in (0, 1). Typical: 0.80, 0.90, 0.95.

    Returns
    -------
    alpha : float
        EWMA smoothing parameter in (0, 1).
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    return 1.0 - (1.0 - p) ** (1.0 / N)


def effective_window(alpha: float) -> float:
    """
    Rule-of-thumb “effective window length” for EWMA.

    What it means
    -------------
    EWMA uses infinitely many past trials, but the contribution of older trials becomes tiny.
    A useful heuristic is:

        effective_window ≈ 1 / alpha

    Interpretation
    --------------
    - alpha = 0.05 -> ~20-trial effective window
    - alpha = 0.10 -> ~10-trial effective window
    - alpha = 0.33 -> ~3-trial effective window

    This is not an exact identity; it is a widely used approximation.

    Returns
    -------
    float
    """
    if alpha <= 0:
        return float("inf")
    return 1.0 / alpha


def alpha_table_for_trial_memory(
    N: int,
    *,
    mass_levels: List[float] = (0.80, 0.90, 0.95),
    print_table: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build a table of alpha choices when you believe the animal uses ~N recent trials.

    What the table contains
    -----------------------
    Three families of mappings from “uses N trials” to EWMA alpha:

    1) Half-life = N trials
       - Meaning: influence halves after N trials.
       - Use when: you want a characteristic decay timescale around N, not a hard window.

    2) Mean lag = (N - 1) / 2
       - Meaning: the *average age* of evidence is the midpoint of the last N trials.
       - Use when: you think the animal’s evidence is centered around the middle of last N,
         but older trials may still have non-trivial influence.

    3) p% mass in last N trials (for each p in mass_levels)
       - Meaning: exactly p fraction of total EWMA weight lies within last N trials.
       - Use when: you literally mean “mostly last N trials” and want a transparent,
         quantifiable statement.

    Recommended default when you say “uses last N trials”
    -----------------------------------------------------
    If your statement implies older-than-N trials are largely negligible,
    pick p=0.90 or p=0.95 in the mass-within-last-N mapping.

    Parameters
    ----------
    N : int
        “Memory length” in trials (e.g., N=5).
    mass_levels : list of float
        Fractions p for mass-within-last-N mapping.
    print_table : bool
        Whether to print a readable summary.

    Returns
    -------
    table : list of dict
        Each row includes:
          - interpretation (string)
          - meaning (string)
          - alpha (float)
          - effective_window (float)
    """
    if N <= 0:
        raise ValueError("N must be > 0")

    table: List[Dict[str, Any]] = []

    # 1) Half-life mapping
    a_half = alpha_from_half_life(float(N))
    table.append(
        {
            "interpretation": f"Half-life = {N} trials",
            "meaning": (
                f"Influence decays by 50% after {N} trials. "
                "Good when you want a characteristic decay timescale ~N."
            ),
            "alpha": a_half,
            "effective_window": effective_window(a_half),
        }
    )

    # 2) Mean-lag mapping (midpoint of N-trial window)
    L = (N - 1) / 2.0
    a_lag = alpha_from_mean_lag(L)
    table.append(
        {
            "interpretation": f"Mean lag = {L:.1f} trials",
            "meaning": (
                f"Sets the expected age of evidence E[k] to {L:.1f} trials "
                f"(midpoint of a {N}-trial window). Older trials still contribute."
            ),
            "alpha": a_lag,
            "effective_window": effective_window(a_lag),
        }
    )

    # 3) Mass-within-last-N mappings
    for p in mass_levels:
        a_mass = alpha_from_mass_within_last_N(N, float(p))
        table.append(
            {
                "interpretation": f"{int(p * 100)}% mass in last {N} trials",
                "meaning": (
                    f"Ensures {int(p * 100)}% of EWMA total weight lies within the most recent "
                    f"{N} trials. Most literal interpretation of 'uses last {N} trials'."
                ),
                "alpha": a_mass,
                "effective_window": effective_window(a_mass),
            }
        )

    if print_table:
        print(f"\nAlpha choices when the animal uses the previous {N} trials:\n")
        for row in table:
            a = row["alpha"]
            print(
                f"- {row['interpretation']}\n"
                f"  meaning: {row['meaning']}\n"
                f"  alpha = {a:.3f}   "
                f"(effective window ~ {row['effective_window']:.1f} trials)\n"
            )

        # Provide a transparent "recommended" entry (90% mass if present, else max mass)
        if len(mass_levels) > 0:
            p_rec = 0.90 if (0.90 in mass_levels) else max(mass_levels)
            a_rec = alpha_from_mass_within_last_N(N, p_rec)
            print(
                f"Recommended alpha when you mean 'mostly last {N} trials' "
                f"({int(p_rec * 100)}% mass): {a_rec:.3f}\n"
            )

    return table


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
# table = alpha_table_for_trial_memory(N=5, mass_levels=[0.80, 0.90, 0.95], print_table=True)
# alpha_for_model = alpha_from_mass_within_last_N(5, 0.90)






def plot_reward_rates_vs_value_all_models(
    *,
    nwb_data: Any,
    window: Union[int, Sequence[int]] = 8,
    alpha: Union[float, Sequence[float]] = 0.2,
    include_noresponse: bool = False,
    model_aliases: Optional[Sequence[str]] = None,
    drop_last_latent: bool = True,
    point_size: float = 6.0,
    alpha_points: float = 0.5,
    max_points: Optional[int] = None,
    random_state: int = 0,
    figsize_per_panel: Tuple[float, float] = (4.0, 3.2),
    warn_on_length_mismatch: bool = True,
) -> None:
    """
    Scatter plots of selected reward-rate metrics vs model value signals.

    Reward-rate metrics plotted (fixed set)
    --------------------------------------
    - ewma_experienced
    - ewma_left_reward
    - ewma_right_reward
    - running_experienced
    - running_left_reward
    - running_right_reward

    Model-specific y-axis
    ---------------------
    - QLearning* models: sumQ = q_value[0] + q_value[1]
    - ForagingCompareThreshold: fitted_latent_variables['value']

    Window/alpha sweeps
    -------------------
    - window can be an int or a list of ints
    - alpha can be a float or a list of floats
    The function plots all combinations of (window, alpha), one figure per combination.

    Alignment behavior (practical + transparent)
    --------------------------------------------
    - Reward-rate arrays have length = n_trials.
    - Fitted latents (q_value/value) are assumed to have noresponse trials removed upstream,
      so we drop noresponse trials from reward-rate arrays using:
          responded = (animal_response != 2)

    - If lengths still mismatch between reward-rate (after masking) and latent:
        * We DO NOT error.
        * We trim both to min_len (to ensure plotting works).
        * If warn_on_length_mismatch=True, we print a warning that tells you
          exactly which model/metric/combination mismatched and what min_len was used.

    This preserves your “it works” behavior while still surfacing hidden alignment issues.
    """
    # ------------------------------------------------------------
    # Normalize window and alpha to lists
    # ------------------------------------------------------------
    if isinstance(window, (list, tuple, np.ndarray)):
        windows = [int(w) for w in window]
    else:
        windows = [int(window)]

    if isinstance(alpha, (list, tuple, np.ndarray)):
        alphas = [float(a) for a in alpha]
    else:
        alphas = [float(alpha)]

    if any(w <= 0 for w in windows):
        raise ValueError(f"All window values must be positive integers. Got: {windows}")
    if any((a <= 0.0 or a > 1.0) for a in alphas):
        raise ValueError(f"All alpha values must be in (0, 1]. Got: {alphas}")

    # ------------------------------------------------------------
    # Default models
    # ------------------------------------------------------------
    if model_aliases is None:
        model_aliases = [
            "QLearning_L1F1_CK1_softmax",
            "QLearning_L2F1_softmax",
            "QLearning_L2F1_CKfull_softmax",
            "ForagingCompareThreshold",
            "QLearning_L2F1_CK1_softmax",
        ]

    # ------------------------------------------------------------
    # Reward-rate keys to plot (fixed)
    # ------------------------------------------------------------
    rr_keys = [
        "ewma_experienced",
        "ewma_left_reward",
        "ewma_right_reward",
        "running_experienced",
        "running_left_reward",
        "running_right_reward",
    ]

    # ------------------------------------------------------------
    # Drop noresponse trials to align to fitted latents
    # ------------------------------------------------------------
    choices = np.asarray(nwb_data.trials["animal_response"][:], dtype=int)
    responded = choices != 2

    # ------------------------------------------------------------
    # RNG for optional subsampling
    # ------------------------------------------------------------
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------
    # Cache fitted latents per model (avoid refetch per combo)
    # ------------------------------------------------------------
    model_latents: Dict[str, Dict[str, Any]] = {}
    for model_alias in model_aliases:
        try:
            fit = get_fitted_latent(session_name=nwb_data.session_id, model_alias=model_alias)
            latents = fit.get("fitted_latent_variables", {})

            if model_alias == "ForagingCompareThreshold":
                if "value" in latents:
                    y = np.asarray(latents["value"], dtype=float)
                    y_name = "value"
                else:
                    y = None
                    y_name = "value"
            else:
                if "q_value" in latents and len(latents["q_value"]) >= 2:
                    qL = np.asarray(latents["q_value"][0], dtype=float)
                    qR = np.asarray(latents["q_value"][1], dtype=float)
                    y = qL + qR
                    y_name = "sumQ"
                else:
                    y = None
                    y_name = "q_value"

            if y is not None and drop_last_latent and y.size > 0:
                y = y[:-1]

            model_latents[model_alias] = {"y": y, "y_name": y_name}
        except Exception as e:
            model_latents[model_alias] = {"y": None, "y_name": "latent", "error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------
    # Loop over all (window, alpha) combinations
    # ------------------------------------------------------------
    for w in windows:
        for a in alphas:
            reward_rates = compute_all_reward_rates(
                nwb_data=nwb_data,
                window=w,
                alpha=a,
                include_noresponse=include_noresponse,
            )

            # Ensure requested keys exist (at least one)
            present_keys = [k for k in rr_keys if k in reward_rates]
            if len(present_keys) == 0:
                raise ValueError(
                    f"None of the requested reward-rate metrics were found. "
                    f"Requested: {rr_keys}. Available: {sorted(list(reward_rates.keys()))}"
                )

            use_keys = present_keys
            n_rows = len(model_aliases)
            n_cols = len(use_keys)

            fig_w = max(6.0, figsize_per_panel[0] * n_cols)
            fig_h = max(3.5, figsize_per_panel[1] * n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
            fig.suptitle(
                f"Reward-rate vs latent: window={w}, alpha={a}, include_noresponse={include_noresponse}",
                y=1.02,
            )

            for i, model_alias in enumerate(model_aliases):
                y = model_latents[model_alias].get("y", None)
                y_name = model_latents[model_alias].get("y_name", "latent")

                if y is None:
                    for j in range(n_cols):
                        ax = axes[i, j]
                        ax.axis("off")
                        if j == 0:
                            err = model_latents[model_alias].get("error", f"missing {y_name}")
                            ax.text(0.02, 0.5, f"{model_alias}\n{err}", va="center")
                    continue

                for j, rr_key in enumerate(use_keys):
                    ax = axes[i, j]

                    rr_full = np.asarray(reward_rates[rr_key], dtype=float)  # length == n_trials
                    x = rr_full[responded]  # drop noresponse trials to match latent convention

                    # If lengths mismatch, warn and trim to make plotting work (your desired behavior).
                    if x.size != y.size:
                        min_len = min(x.size, y.size)
                        if warn_on_length_mismatch:
                            print(
                                f"[WARN] Length mismatch: model='{model_alias}', metric='{rr_key}', "
                                f"window={w}, alpha={a}: x={x.size}, y={y.size}. "
                                f"Trimming to min_len={min_len}."
                            )
                        x2 = x[:min_len]
                        y2 = y[:min_len]
                    else:
                        x2 = x
                        y2 = y

                    # Remove NaN/inf values (e.g., early running window can be NaN)
                    finite = np.isfinite(x2) & np.isfinite(y2)
                    x2 = x2[finite]
                    y2 = y2[finite]

                    # Optional subsampling for readability
                    if max_points is not None and x2.size > max_points:
                        idx = rng.choice(x2.size, size=max_points, replace=False)
                        x2 = x2[idx]
                        y2 = y2[idx]

                    ax.scatter(x2, y2, s=point_size, alpha=alpha_points)

                    if i == 0:
                        ax.set_title(rr_key, fontsize=10)
                    if j == 0:
                        ax.set_ylabel(f"{model_alias}\n{y_name}", fontsize=10)
                    else:
                        ax.set_ylabel("")

                    if i == n_rows - 1:
                        ax.set_xlabel(rr_key)

            plt.tight_layout()
            plt.show()
