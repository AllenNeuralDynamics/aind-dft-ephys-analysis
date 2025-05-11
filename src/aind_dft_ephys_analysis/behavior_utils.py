import os       
import json     
from typing import Optional, List, Any, Dict  

import numpy as np        
import requests           


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
            print("The response returned an empty list. No data was found.")
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
