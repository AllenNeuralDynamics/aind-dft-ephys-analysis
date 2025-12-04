import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple


def _get_lick_times(ts) -> np.ndarray:
    """
    Extract lick timestamps from an NWB TimeSeries object.

    Parameters
    ----------
    ts : TimeSeries
        NWB TimeSeries containing lick timestamps (assumed in `.timestamps`).

    Returns
    -------
    np.ndarray
        Array of lick timestamps (1D, in seconds).
    """
    return np.asarray(ts.timestamps[:])


def _first_lick_latency_per_trial(
    go_times: np.ndarray,
    left_licks: np.ndarray,
    right_licks: np.ndarray,
    max_latency: Optional[float] = 2.0,
) -> np.ndarray:
    """
    Compute the latency of the first lick (left or right) after the go cue.

    For each trial i, this function finds the earliest lick timestamp
    (left or right) that occurs at or after go_times[i]. If max_latency is
    not None, only licks within [go_times[i], go_times[i] + max_latency]
    are considered. If no lick is found in this window, the latency is NaN.

    Parameters
    ----------
    go_times : np.ndarray
        Go cue timestamps for each trial (shape: [n_trials]).
    left_licks : np.ndarray
        Timestamps of all left-lick events (1D array).
    right_licks : np.ndarray
        Timestamps of all right-lick events (1D array).
    max_latency : float or None, optional
        Maximum allowed latency window (seconds) to consider a lick valid.
        If None, no upper bound is applied. Default is 2.0.

    Returns
    -------
    np.ndarray
        Per-trial first lick latencies (shape: [n_trials]).
        Each entry is:
            earliest_lick_time - go_time
        or NaN if no lick occurs within the allowed window.
    """
    n_trials = len(go_times)
    latencies = np.full(n_trials, np.nan)

    for i, t0 in enumerate(go_times):
        # index of first lick >= go cue
        li = np.searchsorted(left_licks, t0, side="left")
        ri = np.searchsorted(right_licks, t0, side="left")

        left_after = left_licks[li:]
        right_after = right_licks[ri:]

        if max_latency is not None:
            left_after = left_after[left_after - t0 <= max_latency]
            right_after = right_after[right_after - t0 <= max_latency]

        all_after = np.concatenate([left_after, right_after])
        if all_after.size > 0:
            latencies[i] = np.min(all_after) - t0

    return latencies


def compute_behavior_qc_from_nwb(
    nwb_data: Any,
    response_latency_window: Optional[float] = 2.0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Compute behavior QC metrics from an NWB behavior file.

    This function summarizes trial structure, choices, rewards, reward volumes,
    and licking behavior for a single session. It assumes a dynamic-foraging-
    like task where the animal makes left/right choices and may lick during
    ITI and delay periods.

    Metrics are not split by opto vs non-opto trials; only a session-level
    indicator of optogenetic trials is provided.

    Parameters
    ----------
    nwb_data : Any
        NWB behavior file loaded with NWBUtils. Required fields:

        trials:
            - 'animal_response'   : int
                0 = left choice
                1 = right choice
                2 = no response
            - 'laser_on_trial'    : int
                0 = laser off
                1 = laser on
            - 'rewarded_historyL' : bool
                True if trial was rewarded when animal chose left.
            - 'rewarded_historyR' : bool
                True if trial was rewarded when animal chose right.
            - 'reward_size_left'  : float
                Reward size available on the left side for this trial
                (same units as water volume, e.g. µL).
            - 'reward_size_right' : float
                Reward size available on the right side for this trial.
            - 'start_time'        : float
                Trial start time (seconds).
            - 'delay_start_time'  : float
                Delay period start time (seconds). May be NaN.
            - 'goCue_start_time'  : float
                Go cue time (seconds).

        acquisition:
            - 'left_lick_time'    : TimeSeries
                Timestamps of left-lick events.
            - 'right_lick_time'   : TimeSeries
                Timestamps of right-lick events.

    response_latency_window : float or None, optional
        Maximum window (in seconds) after the go cue to search for the first
        lick when computing first-lick latency. If None, no upper bound is
        applied. Default is 2.0 seconds.

    Returns
    -------
    metrics : dict
        Summary QC metrics for the session. Keys are sorted alphabetically.

        Choice composition:
        -------------------
        - 'choice_fraction_left'  : float
            Fraction of trials where animal_response == 0 (left choice).
        - 'choice_fraction_right' : float
            Fraction of trials where animal_response == 1 (right choice).

        Licking during delay:
        ---------------------
        - 'early_lick_fraction' : float
            Fraction of trials with at least one lick (left or right) during
            the delay period (where delay exists).

        First-lick latencies:
        ---------------------
        - 'first_lick_latency_all'        : list of float
        - 'mean_first_lick_latency'       : float
        - 'mean_first_lick_latency_left'  : float
        - 'mean_first_lick_latency_right' : float

        Optogenetic information:
        ------------------------
        - 'is_opto_session' : float (0.0 or 1.0)
        - 'n_opto_trials'   : float
        - 'n_trials'        : float

        Lick fractions:
        ---------------
        - 'lick_fraction_iti'         : float
        - 'lick_fraction_iti_left'    : float
        - 'lick_fraction_iti_right'   : float
        - 'lick_fraction_delay'       : float
        - 'lick_fraction_delay_left'  : float
        - 'lick_fraction_delay_right' : float

        Trial timing:
        -------------
        - 'mean_delay_time' : float
        - 'mean_iti_time'   : float

        Reward-related measures:
        ------------------------
        - 'reward_fraction'        : float
            Fraction of all trials that are rewarded.
        - 'reward_fraction_left'   : float
            Fraction of left-choice trials (animal_response == 0)
            that are rewarded.
        - 'reward_fraction_right'  : float
            Fraction of right-choice trials (animal_response == 1)
            that are rewarded.
        - 'reward_volume_total'    : float
            Total reward volume delivered across all trials. On each trial i,
            reward volume is:
                reward_size_left[i]  if animal chose left and was rewarded, or
                reward_size_right[i] if animal chose right and was rewarded,
                0 otherwise.
        - 'reward_volume_left'     : float
            Total reward volume delivered on trials where the animal chose
            left and was rewarded (sum over those trials only).
        - 'reward_volume_right'    : float
            Total reward volume delivered on trials where the animal chose
            right and was rewarded.

        Response rate:
        --------------
        - 'response_rate' : float
            Fraction of trials where the animal made a choice
            (animal_response != 2).

        Win–stay / lose–switch:
        ------------------------
        - 'win_stay_rate'
        - 'win_stay_rate_left'
        - 'win_stay_rate_right'
        - 'lose_switch_rate'
        - 'lose_switch_rate_left'
        - 'lose_switch_rate_right'

    trial_df : pandas.DataFrame
        Per-trial behavior summary with columns:
        - 'animal_response'
        - 'responded'
        - 'laser_on_trial'
        - 'reward'             (bool)
        - 'reward_volume'      (float, volume delivered on that trial)
        - 'start_time'
        - 'delay_start_time'
        - 'goCue_start_time'
        - 'iti_time'
        - 'delay_time'
        - 'first_lick_latency'
        - 'left_licks_iti', 'right_licks_iti', 'total_licks_iti'
        - 'left_licks_delay', 'right_licks_delay', 'total_licks_delay'
        - 'early_lick_trial'
    """

    trials = nwb_data.trials

    # ------------------------
    # Read trial-level fields
    # ------------------------
    animal_response = np.asarray(trials["animal_response"])
    laser_on_trial = np.asarray(trials["laser_on_trial"])
    rewarded_historyL = np.asarray(trials["rewarded_historyL"], dtype=bool)
    rewarded_historyR = np.asarray(trials["rewarded_historyR"], dtype=bool)
    reward_size_left = np.asarray(trials["reward_size_left"], dtype=float)
    reward_size_right = np.asarray(trials["reward_size_right"], dtype=float)
    start_times = np.asarray(trials["start_time"])
    delay_start_times = np.asarray(trials["delay_start_time"])
    go_times = np.asarray(trials["goCue_start_time"])

    n_trials = len(animal_response)

    # ------------------------
    # Opto detection
    # ------------------------
    is_opto_session = bool(np.any(laser_on_trial == 1))
    n_opto_trials = int(np.sum(laser_on_trial == 1))

    # ------------------------
    # Response rate (overall)
    # responded[i] is True if the trial had a left or right choice.
    # ------------------------
    responded = animal_response != 2
    response_rate = responded.mean() if n_trials > 0 else np.nan

    # ------------------------
    # Reward (overall)
    # reward[i] = rewarded_historyL[i] if choice is left,
    #            = rewarded_historyR[i] if choice is right,
    #            = False otherwise (no response).
    # ------------------------
    reward = np.zeros(n_trials, dtype=bool)
    left_mask_choice = responded & (animal_response == 0)
    right_mask_choice = responded & (animal_response == 1)

    reward[left_mask_choice] = rewarded_historyL[left_mask_choice]
    reward[right_mask_choice] = rewarded_historyR[right_mask_choice]

    reward_fraction = reward.mean() if n_trials > 0 else np.nan

    # ------------------------
    # Reward volume per trial
    # reward_volume[i] is non-zero only if the trial is rewarded.
    # For left-choice rewarded trials: reward_size_left[i]
    # For right-choice rewarded trials: reward_size_right[i]
    # ------------------------
    reward_volume = np.zeros(n_trials, dtype=float)

    left_reward_mask = (animal_response == 0) & reward
    right_reward_mask = (animal_response == 1) & reward

    reward_volume[left_reward_mask] = reward_size_left[left_reward_mask]
    reward_volume[right_reward_mask] = reward_size_right[right_reward_mask]

    # Total volumes (sum over trials)
    reward_volume_total = float(np.nansum(reward_volume))
    reward_volume_left = float(np.nansum(reward_volume[animal_response == 0]))
    reward_volume_right = float(np.nansum(reward_volume[animal_response == 1]))

    # ------------------------
    # First lick latency (overall)
    # ------------------------
    left_licks = _get_lick_times(nwb_data.acquisition["left_lick_time"])
    right_licks = _get_lick_times(nwb_data.acquisition["right_lick_time"])

    first_lick_latency = _first_lick_latency_per_trial(
        go_times,
        left_licks,
        right_licks,
        max_latency=response_latency_window,
    )
    mean_first_lick_latency = (
        float(np.nanmean(first_lick_latency))
        if np.any(~np.isnan(first_lick_latency))
        else np.nan
    )

    # ------------------------
    # Choice-specific fractions, reward, and latency
    # ------------------------
    choice_left_mask = animal_response == 0
    choice_right_mask = animal_response == 1

    if n_trials > 0:
        choice_fraction_left = float(np.mean(choice_left_mask))
        choice_fraction_right = float(np.mean(choice_right_mask))
    else:
        choice_fraction_left = np.nan
        choice_fraction_right = np.nan

    # Reward fraction conditional on left/right choice
    if np.any(choice_left_mask):
        reward_fraction_left = float(np.mean(reward[choice_left_mask]))
    else:
        reward_fraction_left = np.nan

    if np.any(choice_right_mask):
        reward_fraction_right = float(np.mean(reward[choice_right_mask]))
    else:
        reward_fraction_right = np.nan

    # First-lick latency conditional on left/right choice
    if np.any(choice_left_mask):
        mean_first_lick_latency_left = float(
            np.nanmean(first_lick_latency[choice_left_mask])
        )
    else:
        mean_first_lick_latency_left = np.nan

    if np.any(choice_right_mask):
        mean_first_lick_latency_right = float(
            np.nanmean(first_lick_latency[choice_right_mask])
        )
    else:
        mean_first_lick_latency_right = np.nan

    # ------------------------
    # Win–stay / lose–switch (overall and side-specific)
    # ------------------------
    prev_choice = animal_response[:-1]
    curr_choice = animal_response[1:]
    prev_reward = reward[:-1]

    # Only include pairs where both trials have valid choices (not 2)
    valid_pair = (prev_choice != 2) & (curr_choice != 2)

    win_mask = valid_pair & prev_reward       # previous trial rewarded
    lose_mask = valid_pair & (~prev_reward)   # previous trial not rewarded

    # Overall win–stay: repeat choice after rewarded trial
    if np.any(win_mask):
        win_stay_rate = float(
            np.mean(curr_choice[win_mask] == prev_choice[win_mask])
        )
    else:
        win_stay_rate = np.nan

    # Overall lose–switch: switch choice after non-rewarded trial
    if np.any(lose_mask):
        lose_switch_rate = float(
            np.mean(curr_choice[lose_mask] != prev_choice[lose_mask])
        )
    else:
        lose_switch_rate = np.nan

    # Side-specific win–stay
    win_mask_left = win_mask & (prev_choice == 0)
    win_mask_right = win_mask & (prev_choice == 1)

    if np.any(win_mask_left):
        win_stay_rate_left = float(
            np.mean(curr_choice[win_mask_left] == 0)
        )
    else:
        win_stay_rate_left = np.nan

    if np.any(win_mask_right):
        win_stay_rate_right = float(
            np.mean(curr_choice[win_mask_right] == 1)
        )
    else:
        win_stay_rate_right = np.nan

    # Side-specific lose–switch
    lose_mask_left = lose_mask & (prev_choice == 0)
    lose_mask_right = lose_mask & (prev_choice == 1)

    if np.any(lose_mask_left):
        lose_switch_rate_left = float(
            np.mean(curr_choice[lose_mask_left] != 0)
        )
    else:
        lose_switch_rate_left = np.nan

    if np.any(lose_mask_right):
        lose_switch_rate_right = float(
            np.mean(curr_choice[lose_mask_right] != 1)
        )
    else:
        lose_switch_rate_right = np.nan

    # ------------------------
    # ITI time per trial
    # ITI window per trial i:
    #   [start_time[i], delay_start_time[i])  if delay_start_time[i] is not NaN
    #   [start_time[i], goCue_start_time[i])  otherwise.
    # ------------------------
    iti_time = np.where(
        ~np.isnan(delay_start_times),
        delay_start_times - start_times,
        go_times - start_times,
    )

    # ------------------------
    # Delay time per trial
    # Delay window per trial i:
    #   [delay_start_time[i], goCue_start_time[i])  if delay_start_time[i] not NaN
    #   no delay window, delay_time[i] = 0.0        otherwise.
    # ------------------------
    delay_time = np.where(
        ~np.isnan(delay_start_times),
        go_times - delay_start_times,
        0.0,
    )

    mean_iti = float(np.nanmean(iti_time)) if n_trials > 0 else np.nan
    mean_delay = float(np.nanmean(delay_time)) if n_trials > 0 else np.nan

    # ------------------------
    # Licks during ITI and delay
    # ------------------------
    left_licks_iti = np.zeros(n_trials, dtype=int)
    right_licks_iti = np.zeros(n_trials, dtype=int)
    left_licks_delay = np.zeros(n_trials, dtype=int)
    right_licks_delay = np.zeros(n_trials, dtype=int)
    early_lick_trial = np.zeros(n_trials, dtype=bool)

    for i in range(n_trials):
        # ITI window
        iti_start = start_times[i]
        if not np.isnan(delay_start_times[i]):
            iti_end = delay_start_times[i]
        else:
            iti_end = go_times[i]

        li_start = np.searchsorted(left_licks, iti_start, side="left")
        li_end = np.searchsorted(left_licks, iti_end, side="left")
        left_licks_iti[i] = li_end - li_start

        ri_start = np.searchsorted(right_licks, iti_start, side="left")
        ri_end = np.searchsorted(right_licks, iti_end, side="left")
        right_licks_iti[i] = ri_end - ri_start

        # Delay window (if present)
        if not np.isnan(delay_start_times[i]):
            delay_start = delay_start_times[i]
            delay_end = go_times[i]

            ld_start = np.searchsorted(left_licks, delay_start, side="left")
            ld_end = np.searchsorted(left_licks, delay_end, side="left")
            left_licks_delay[i] = ld_end - ld_start

            rd_start = np.searchsorted(right_licks, delay_start, side="left")
            rd_end = np.searchsorted(right_licks, delay_end, side="left")
            right_licks_delay[i] = rd_end - rd_start

            # Early lick trial: any lick in delay window
            if (left_licks_delay[i] + right_licks_delay[i]) > 0:
                early_lick_trial[i] = True
        else:
            left_licks_delay[i] = 0
            right_licks_delay[i] = 0
            early_lick_trial[i] = False

    total_licks_iti = left_licks_iti + right_licks_iti
    total_licks_delay = left_licks_delay + right_licks_delay

    if n_trials > 0:
        # ITI lick fractions
        lick_fraction_iti = float(np.mean(total_licks_iti > 0))
        lick_fraction_iti_left = float(np.mean(left_licks_iti > 0))
        lick_fraction_iti_right = float(np.mean(right_licks_iti > 0))

        # Delay lick fractions
        lick_fraction_delay = float(np.mean(total_licks_delay > 0))
        lick_fraction_delay_left = float(np.mean(left_licks_delay > 0))
        lick_fraction_delay_right = float(np.mean(left_licks_delay > 0))

        # Early lick fraction (any delay-period licking)
        early_lick_fraction = float(early_lick_trial.mean())
    else:
        lick_fraction_iti = np.nan
        lick_fraction_iti_left = np.nan
        lick_fraction_iti_right = np.nan
        lick_fraction_delay = np.nan
        lick_fraction_delay_left = np.nan
        lick_fraction_delay_right = np.nan
        early_lick_fraction = np.nan

    # ------------------------
    # Per-trial table
    # ------------------------
    trial_df = pd.DataFrame({
        "animal_response": animal_response,
        "responded": responded,
        "laser_on_trial": laser_on_trial,
        "reward": reward,
        "reward_volume": reward_volume,
        "start_time": start_times,
        "delay_start_time": delay_start_times,
        "goCue_start_time": go_times,
        "iti_time": iti_time,
        "delay_time": delay_time,
        "first_lick_latency": first_lick_latency,
        "left_licks_iti": left_licks_iti,
        "right_licks_iti": right_licks_iti,
        "total_licks_iti": total_licks_iti,
        "left_licks_delay": left_licks_delay,
        "right_licks_delay": right_licks_delay,
        "total_licks_delay": total_licks_delay,
        "early_lick_trial": early_lick_trial,
    })

    # ------------------------
    # Summary metrics (unsorted)
    # ------------------------
    metrics_unsorted: Dict[str, Any] = {
        "choice_fraction_left": choice_fraction_left,
        "choice_fraction_right": choice_fraction_right,
        "early_lick_fraction": early_lick_fraction,
        "first_lick_latency_all": first_lick_latency.tolist(),
        "is_opto_session": float(is_opto_session),
        "lick_fraction_delay": lick_fraction_delay,
        "lick_fraction_delay_left": lick_fraction_delay_left,
        "lick_fraction_delay_right": lick_fraction_delay_right,
        "lick_fraction_iti": lick_fraction_iti,
        "lick_fraction_iti_left": lick_fraction_iti_left,
        "lick_fraction_iti_right": lick_fraction_iti_right,
        "lose_switch_rate": lose_switch_rate,
        "lose_switch_rate_left": lose_switch_rate_left,
        "lose_switch_rate_right": lose_switch_rate_right,
        "mean_delay_time": mean_delay,
        "mean_first_lick_latency": mean_first_lick_latency,
        "mean_first_lick_latency_left": mean_first_lick_latency_left,
        "mean_first_lick_latency_right": mean_first_lick_latency_right,
        "mean_iti_time": mean_iti,
        "n_opto_trials": float(n_opto_trials),
        "n_trials": float(n_trials),
        "response_rate": float(response_rate),
        "reward_fraction": reward_fraction,
        "reward_fraction_left": reward_fraction_left,
        "reward_fraction_right": reward_fraction_right,
        "reward_volume_left": reward_volume_left,
        "reward_volume_right": reward_volume_right,
        "reward_volume_total": reward_volume_total,
        "win_stay_rate": win_stay_rate,
        "win_stay_rate_left": win_stay_rate_left,
        "win_stay_rate_right": win_stay_rate_right,
    }

    # Sort metrics by key (alphabetical order) for easier inspection
    metrics: Dict[str, Any] = dict(
        sorted(metrics_unsorted.items(), key=lambda kv: kv[0])
    )

    return metrics, trial_df
