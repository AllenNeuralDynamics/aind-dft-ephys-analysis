import os       
import json   
import requests   
from typing import Optional, List, Any, Dict ,Union

import numpy as np   
import pandas as pd

from general_utils import format_session_name
from general_utils import extract_session_name_core
from nwb_utils import NWBUtils
from model_fitting import fit_q_learning_model

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
    url: str = "https://api.allenneuraldynamics-test.org/v1/behavior_analysis/mle_fitting"
) -> List[str]:
    """
    Retrieves fitted model(s) from the specified URL for a given session name.

    If the provided session name does not end with ".nwb", the function will
    automatically append ".nwb" to it.

    Args:
        session_name (str):
            The NWB session name. If the provided session name does not end with
            ".nwb", ".nwb" will be appended to it.
        url (str):
            The endpoint to request model fitting results from.
            Defaults to "https://api.allenneuraldynamics-test.org/v1/behavior_analysis/mle_fitting".

    Returns:
        List[str]:
            A list of fitted model aliases found in the response JSON.

    Raises:
        ValueError:
            If `session_name` is an empty string.
        requests.exceptions.RequestException:
            If the HTTP request fails due to network, timeout, or other errors.
    """
    # Validate the session_name parameter
    if not session_name:
        raise ValueError("The 'session_name' parameter cannot be empty.")

    session_name = format_session_name(session_name)

    # Construct the query parameters
    filter_payload = {"nwb_name": session_name}
    projection_payload = {
        "analysis_results.fit_settings.agent_alias": 1,
        "_id": 0,
    }

    try:
        # Make the GET request
        response = requests.get(
            url,
            params={
                "filter": json.dumps(filter_payload),
                "projection": json.dumps(projection_payload),
            },
            timeout=10  # seconds
        )
        # Raise an HTTPError if the response was unsuccessful
        response.raise_for_status()

        # Parse the JSON data
        data = response.json()

        # Extract the model aliases
        fitted_models = [
            item["analysis_results"]["fit_settings"]["agent_alias"]
            for item in data
        ]
        return fitted_models

    except requests.exceptions.RequestException as e:
        # Handle potential network or HTTP errors
        print(f"Error during the request: {e}")
        return

def get_fitted_latent(
    session_name: str,
    model_alias: str,
    url: str = "https://api.allenneuraldynamics-test.org/v1/behavior_analysis/mle_fitting",
    timeout: Optional[int] = 10
) -> Dict[str, Any]:
    """
    Retrieve the fitted latent variables (and parameters) for a specific model
    in a given NWB session.

    Args:
        session_name (str): 
            The NWB session name. For example, "744329_2024-11-25_12-13-37.nwb".
        model_alias (str): 
            The alias of the model for which to retrieve the fitted latent variables.
            For example, "QLearning_L1F1_CK1_softmax".
        url (str): 
            The endpoint to request data from.
            Defaults to "https://api.allenneuraldynamics-test.org/v1/behavior_analysis/mle_fitting".
        timeout (int, optional):
            The request timeout in seconds. Defaults to 10.

    Returns:
        Dict[str, Any]:
            A dictionary containing at least:
            - "params": model-fitted parameters
            - "fitted_latent_variables": time series or other latent variables

    Raises:
        ValueError:
            If the JSON response is empty or does not have the expected structure.
        requests.exceptions.RequestException:
            If the HTTP request fails due to a network error, timeout, or other issues.
    """
    # Validate the session_name parameter
    if not session_name:
        raise ValueError("The 'session_name' parameter cannot be empty.")

    session_name = format_session_name(session_name)

    # Build the filter and projection payloads
    filter_payload = {
        "nwb_name": session_name,
        "analysis_results.fit_settings.agent_alias": model_alias,
    }
    projection_payload = {
        "analysis_results.params": 1,
        "analysis_results.fitted_latent_variables": 1,
        "_id": 0,
    }

    try:
        response = requests.get(
            url,
            params={
                "filter": json.dumps(filter_payload),
                "projection": json.dumps(projection_payload),
            },
            timeout=timeout,
        )
        # Raise an HTTPError if the response was unsuccessful
        response.raise_for_status()

        data = response.json()
        if not data:
            print(f"The response returned an empty list. No data was found for session:{session_name} model:{model_alias}.")
            return 

        # Expect only one record for this session and model
        record_dict = data[0]

        # Extract fitted parameters and latent variables
        analysis_results = record_dict.get("analysis_results", {})
        if "params" not in analysis_results or "fitted_latent_variables" not in analysis_results:
            raise ValueError(
                "The expected keys ('params' and 'fitted_latent_variables') were not found in the response."
            )

        return {
            "params": analysis_results["params"],
            "fitted_latent_variables": analysis_results["fitted_latent_variables"]
        }

    except requests.exceptions.RequestException as e:
        print(f"HTTP Request Error: {e}")
        return

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
      - `fitted_latent`: the dict returned by `get_fitted_latent(...)`, **or**
      - all three of `session_name`, `model_alias`, and `latent_name`, so that this function can fetch it.

    Parameters
    ----------
    nwb_behavior_data : Any
        An open NWB object (from NWBHDF5IO.read() or NWBZarrIO.read()).
        Must contain `.trials` with columns:
          - 'rewarded_historyL', 'rewarded_historyR', 'animal_response'
        (needed only for latent_name='RPE', 'chosen_q', 'unchosen_q').

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

    latent_name : str, optional
        Which derived series to return. One of:
          - 'q_value_difference'       → Q₁ − Q₀, skipping trial 0
          - 'total_value'              → Q₁ + Q₀, skipping trial 0
          - 'right_choice_probability' → choice_prob[1] (all trials)
          - 'left_choice_probability'  → choice_prob[0] (all trials)
          - 'RPE'                      → reward_prediction_error on valid trials
          - 'QL'                       → Q for option 0 (all trials)
          - 'QR'                       → Q for option 1 (all trials)
          - 'chosen_q'                 → Q of the chosen option on valid trials
          - 'unchosen_q'               → Q of the unchosen option on valid trials
        Required if `fitted_latent` is None, **and** always needed to choose the output.

    Returns
    -------
    np.ndarray or None
        A 1-D float array for the requested latent series, or None if unsupported.
    """
    # 1) Validate inputs and fetch if needed
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
    q0 = np.array(FL['q_value'][0])
    q1 = np.array(FL['q_value'][1])

    # 3) Compute requested series
    if latent_name == 'q_value_difference':
        return (q1 - q0)[1:]

    if latent_name == 'total_value':
        return (q1 + q0)[1:]

    if latent_name == 'right_choice_probability':
        return np.array(FL['choice_prob'][1])

    if latent_name == 'left_choice_probability':
        return np.array(FL['choice_prob'][0])

    if latent_name == 'RPE':
        trials = nwb_behavior_data.trials
        rewardedL = trials['rewarded_historyL'][:]
        rewardedR = trials['rewarded_historyR'][:]
        responses = trials['animal_response'][:]

        # drop initial trial
        q0 = q0[1:]
        q1 = q1[1:]

        valid = responses != 2
        rewarded = (rewardedL | rewardedR).astype(int)[valid]
        resp     = responses[valid]

        # reward prediction error = reward − Q(choice)
        return np.where(resp == 0, rewarded - q0, rewarded - q1)

    if latent_name == 'QL':
        # drop initial trial
        return q0[1:]

    if latent_name == 'QR':
        # drop initial trial
        return q1[1:]

    if latent_name == 'chosen_q':
        trials = nwb_behavior_data.trials
        rewardedL = trials['rewarded_historyL'][:]
        rewardedR = trials['rewarded_historyR'][:]
        responses = trials['animal_response'][:]

        # drop initial trial
        q0 = q0[1:]
        q1 = q1[1:]

        valid = responses != 2
        rewarded = (rewardedL | rewardedR).astype(int)[valid]
        resp     = responses[valid]

        return np.where(resp == 0, q0, q1)

    if latent_name == 'unchosen_q':
        trials = nwb_behavior_data.trials
        rewardedL = trials['rewarded_historyL'][:]
        rewardedR = trials['rewarded_historyR'][:]
        responses = trials['animal_response'][:]

        # drop initial trial
        q0 = q0[1:]
        q1 = q1[1:]

        valid = responses != 2
        rewarded = (rewardedL | rewardedR).astype(int)[valid]
        resp     = responses[valid]

        return np.where(resp == 0, q1, q0)

    # Unsupported latent_name
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
        NWB object with .trials table containing:
          - 'rewarded_historyL', 'rewarded_historyR', 'animal_response'
    trial_type : str or list of str, optional
        One or more of:
          - 'no_response'      : trials where animal_response == 2
          - 'response'         : trials where animal_response != 2
          - 'rewarded'         : trials where either L or R was rewarded
          - 'unrewarded'       : trials where no reward and response != 2
          - 'left_rewarded'    : trials where left side was rewarded
          - 'right_rewarded'   : trials where right side was rewarded
        Default is 'no_response'. If a list is provided, results are unioned.

    Returns
    -------
    List[int]
        Sorted list of zero-based trial indices matching the type(s).
    """
    # If given a list, union together each single-type result
    if isinstance(trial_type, (list, tuple)):
        idx_set = set()
        for t in trial_type:
            idx_set.update(find_trials(nwb_behavior_data, t))
        return sorted(idx_set)

    # single-trial_type logic
    trials = nwb_behavior_data.trials
    resp = trials['animal_response'][:]
    rewardedL = trials['rewarded_historyL'][:]
    rewardedR = trials['rewarded_historyR'][:]
    rewarded = np.logical_or(rewardedL, rewardedR)

    if trial_type == 'no_response':
        return np.where(resp == 2)[0].tolist()
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
    model_alias:  Union[str, List[str]] = ['QLearning_L1F1_CK1_softmax', 'QLearning_L2F1_softmax', 'QLearning_L2F1_CK1_softmax'],
    latent_names: Optional[List[str]] = None,
    trial_types : Optional[List[str]]  = None
) -> pd.DataFrame:
    """
    Build a one-row behavioral summary for a single session.

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
            'q_value_difference',
            'total_value',
            'right_choice_probability',
            'RPE',
            'chosen_q',
            'unchosen_q',
            'QL',
            'QR',
        ]

    if trial_types is None:
        trial_types = [
            'no_response', 'response', 'rewarded', 'unrewarded',
            'left_rewarded', 'right_rewarded'
        ]

    # ------------------------------------------------------------------
    # 2. Cache for any locally fitted model
    # ------------------------------------------------------------------
    local_fit_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # 3. Assemble summary
    # ------------------------------------------------------------------
    summary: dict[str, Any] = {'session_id': session_id}

    for alias in aliases:
        # --------------------------------------------------------------
        # 3a. Obtain (or compute) fitted_latent for this alias
        # --------------------------------------------------------------
        if alias == 'q_learning_Y1':
            # Fit only once per session
            if alias not in local_fit_cache:
                fit_dict = fit_q_learning_model(nwb_data, model_name=alias)
                local_fit_cache[alias] = fit_dict
            fit_source = 'local'   # just for clarity / debugging
        else:
            fit_source = 'remote'

        # --------------------------------------------------------------
        # 3b. Add each requested latent series
        # --------------------------------------------------------------
        for ln in latent_names:
            if fit_source == 'local':
                values = extract_fitted_data(
                    nwb_behavior_data=nwb_data,
                    fitted_latent=local_fit_cache[alias],
                    latent_name=ln
                )
            else:
                values = extract_fitted_data(
                    nwb_behavior_data=nwb_data,
                    session_name=full_session_name,
                    model_alias=alias,
                    latent_name=ln
                )

            col_name = f"{alias}-{ln}"
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
    model_alias:  Union[str, List[str]] = ['QLearning_L1F1_CK1_softmax', 'QLearning_L2F1_softmax', 'QLearning_L2F1_CK1_softmax'],
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
