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
    max_latency: Optional[float] = 1,
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
        If None, no upper bound is applied. Default is 1.0.

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
    response_latency_window: Optional[float] = 1,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Compute behavior QC metrics from an NWB behavior file.

    This function summarizes trial structure, choices, rewards, reward volumes,
    timing (ITI and delay), and licking behavior for a single session. It assumes
    a dynamic-foraging-like task where the animal makes left/right choices and
    may lick during ITI and delay periods.

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
        applied. Default is 1.0 second.

    Returns
    -------
    metrics : dict
        Session-level summary metrics. Keys (alphabetical):

        Choice composition:
        -------------------
        - 'choice_fraction_left' : float
            Fraction of trials with a left choice (animal_response == 0).
        - 'choice_fraction_right' : float
            Fraction of trials with a right choice (animal_response == 1).

        Early licking:
        --------------
        - 'early_lick_fraction' : float
            Fraction of trials with any lick (left or right) in the delay window.
            Trials without a delay (delay_start_time is NaN) always count as False.

        First-lick latencies:
        ---------------------
        - 'first_lick_latency_all' : list of float
            First-lick latencies (s) for all trials, NaN where no lick is found
            within the response_latency_window.
        - 'first_lick_latency_all_left' : list of float
            First-lick latencies for left-choice trials (animal_response == 0).
        - 'first_lick_latency_all_right' : list of float
            First-lick latencies for right-choice trials (animal_response == 1).
        - 'mean_first_lick_latency' : float
            Mean first-lick latency over all trials, ignoring NaN.
        - 'mean_first_lick_latency_left' : float
            Mean first-lick latency restricted to left-choice trials, ignoring NaN.
        - 'mean_first_lick_latency_right' : float
            Mean first-lick latency restricted to right-choice trials, ignoring NaN.

        Optogenetic information:
        ------------------------
        - 'is_opto_session' : float
            1.0 if any trial has laser_on_trial == 1, otherwise 0.0.
        - 'n_opto_trials' : float
            Number of trials with laser_on_trial == 1.
        - 'n_trials' : float
            Total number of trials.

        Lick fractions:
        ---------------
        All of these are fractions over trials (0–1), based on whether there is
        at least one lick in the corresponding window.

        - 'lick_fraction_iti' : float
            Fraction of trials with any lick (left or right) during ITI.
        - 'lick_fraction_iti_left' : float
            Fraction of trials with at least one left lick during ITI.
        - 'lick_fraction_iti_right' : float
            Fraction of trials with at least one right lick during ITI.
        - 'lick_fraction_delay' : float
            Fraction of trials with any lick (left or right) during delay.
        - 'lick_fraction_delay_left' : float
            Fraction of trials with at least one left lick during delay.
        - 'lick_fraction_delay_right' : float
            Fraction of trials with at least one right lick during delay.

        Trial timing:
        -------------
        - 'mean_iti_time' : float
            Mean ITI duration (s) across trials.
            ITI for trial i:
              delay_start_time[i] - start_time[i] if delay_start_time is not NaN,
              otherwise goCue_start_time[i] - start_time[i].
        - 'mean_delay_time' : float
            Mean delay duration (s) across trials.
            Delay for trial i:
              goCue_start_time[i] - delay_start_time[i] if delay_start_time not NaN,
              otherwise 0.0.

        Reward-related measures:
        ------------------------
        - 'reward_fraction' : float
            Fraction of all trials that are rewarded.
        - 'reward_fraction_left' : float
            Fraction of left-choice trials that are rewarded.
        - 'reward_fraction_right' : float
            Fraction of right-choice trials that are rewarded.
        - 'reward_volume_total' : float
            Total reward volume delivered in the session. For each rewarded trial,
            reward_size_left or reward_size_right is added depending on choice.
        - 'reward_volume_left' : float
            Sum of reward_volume over trials where animal_response == 0.
        - 'reward_volume_right' : float
            Sum of reward_volume over trials where animal_response == 1.

        Response rate:
        --------------
        - 'response_rate' : float
            Fraction of trials where the animal made a choice
            (animal_response != 2).

        Win–stay / lose–switch:
        ------------------------
        These are computed over consecutive trial pairs (i, i+1) where both
        trials have valid choices (animal_response != 2).

        - 'win_stay_rate' : float
            Fraction of pairs where trial i is rewarded and the choice on
            trial i+1 repeats the choice on trial i.
        - 'win_stay_rate_left' : float
            Same as above, restricted to pairs where trial i choice is left.
        - 'win_stay_rate_right' : float
            Same as above, restricted to pairs where trial i choice is right.
        - 'lose_switch_rate' : float
            Fraction of pairs where trial i is unrewarded and the choice on
            trial i+1 switches to the other side.
        - 'lose_switch_rate_left' : float
            Same as above, restricted to pairs where trial i choice is left.
        - 'lose_switch_rate_right' : float
            Same as above, restricted to pairs where trial i choice is right.

    trial_df : pandas.DataFrame
        Per-trial DataFrame with one row per trial, containing:

        Basic trial info:
        -----------------
        - 'animal_response' : int
            0 = left choice, 1 = right choice, 2 = no response.
        - 'responded' : bool
            True if animal_response != 2.
        - 'laser_on_trial' : int
            0 = laser off, 1 = laser on.
        - 'start_time' : float
            Trial start time (s).
        - 'delay_start_time' : float
            Delay start time (s), may be NaN for no-delay trials.
        - 'goCue_start_time' : float
            Go cue time (s).

        Reward per trial:
        -----------------
        - 'reward' : bool
            True if the trial is rewarded, based on choice and rewarded_historyL/R.
        - 'reward_volume' : float
            Reward volume delivered on that trial (0 if not rewarded).

        Timing per trial:
        -----------------
        - 'iti_time' : float
            ITI duration (s) for that trial:
              delay_start_time - start_time if delay_start_time not NaN,
              otherwise goCue_start_time - start_time.
        - 'delay_time' : float
            Delay duration (s) for that trial:
              goCue_start_time - delay_start_time if delay_start_time not NaN,
              otherwise 0.0.

        First-lick latency:
        -------------------
        - 'first_lick_latency' : float
            Latency from go cue to first lick (left or right) within the
            response_latency_window, NaN if no lick in the window.

        Licks in ITI:
        -------------
        - 'left_licks_iti' : int
            Number of left licks in the ITI window.
        - 'right_licks_iti' : int
            Number of right licks in the ITI window.
        - 'total_licks_iti' : int
            left_licks_iti + right_licks_iti.

        Licks in delay:
        ---------------
        - 'left_licks_delay' : int
            Number of left licks in the delay window (0 if no delay).
        - 'right_licks_delay' : int
            Number of right licks in the delay window (0 if no delay).
        - 'total_licks_delay' : int
            left_licks_delay + right_licks_delay.

        Delay-lick flag:
        ----------------
        - 'early_lick_trial' : bool
            True if there is at least one lick (left or right) in the delay
            window for this trial; False otherwise.

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
    # ------------------------
    responded = animal_response != 2
    response_rate = responded.mean() if n_trials > 0 else np.nan

    # ------------------------
    # Reward (overall)
    # ------------------------
    reward = np.zeros(n_trials, dtype=bool)
    left_mask_choice = responded & (animal_response == 0)
    right_mask_choice = responded & (animal_response == 1)

    reward[left_mask_choice] = rewarded_historyL[left_mask_choice]
    reward[right_mask_choice] = rewarded_historyR[right_mask_choice]

    reward_fraction = reward.mean() if n_trials > 0 else np.nan

    # ------------------------
    # Reward volume per trial
    # ------------------------
    reward_volume = np.zeros(n_trials, dtype=float)

    left_reward_mask = (animal_response == 0) & reward
    right_reward_mask = (animal_response == 1) & reward

    reward_volume[left_reward_mask] = reward_size_left[left_reward_mask]
    reward_volume[right_reward_mask] = reward_size_right[right_reward_mask]

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

    if np.any(choice_left_mask):
        reward_fraction_left = float(np.mean(reward[choice_left_mask]))
    else:
        reward_fraction_left = np.nan

    if np.any(choice_right_mask):
        reward_fraction_right = float(np.mean(reward[choice_right_mask]))
    else:
        reward_fraction_right = np.nan

    if np.any(choice_left_mask):
        mean_first_lick_latency_left = float(
            np.nanmean(first_lick_latency[choice_left_mask])
        )
        first_lick_latency_all_left = first_lick_latency[choice_left_mask].tolist()
    else:
        mean_first_lick_latency_left = np.nan
        first_lick_latency_all_left = []

    if np.any(choice_right_mask):
        mean_first_lick_latency_right = float(
            np.nanmean(first_lick_latency[choice_right_mask])
        )
        first_lick_latency_all_right = first_lick_latency[choice_right_mask].tolist()
    else:
        mean_first_lick_latency_right = np.nan
        first_lick_latency_all_right = []

    # ------------------------
    # Win–stay / lose–switch (overall and side-specific)
    # ------------------------
    prev_choice = animal_response[:-1]
    curr_choice = animal_response[1:]
    prev_reward = reward[:-1]

    valid_pair = (prev_choice != 2) & (curr_choice != 2)

    win_mask = valid_pair & prev_reward
    lose_mask = valid_pair & (~prev_reward)

    if np.any(win_mask):
        win_stay_rate = float(
            np.mean(curr_choice[win_mask] == prev_choice[win_mask])
        )
    else:
        win_stay_rate = np.nan

    if np.any(lose_mask):
        lose_switch_rate = float(
            np.mean(curr_choice[lose_mask] != prev_choice[lose_mask])
        )
    else:
        lose_switch_rate = np.nan

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
    # ITI and delay times
    # ------------------------
    iti_time = np.where(
        ~np.isnan(delay_start_times),
        delay_start_times - start_times,
        go_times - start_times,
    )

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

            if (left_licks_delay[i] + right_licks_delay[i]) > 0:
                early_lick_trial[i] = True
        else:
            left_licks_delay[i] = 0
            right_licks_delay[i] = 0
            early_lick_trial[i] = False

    total_licks_iti = left_licks_iti + right_licks_iti
    total_licks_delay = left_licks_delay + right_licks_delay

    if n_trials > 0:
        lick_fraction_iti = float(np.mean(total_licks_iti > 0))
        lick_fraction_iti_left = float(np.mean(left_licks_iti > 0))
        lick_fraction_iti_right = float(np.mean(right_licks_iti > 0))

        lick_fraction_delay = float(np.mean(total_licks_delay > 0))
        lick_fraction_delay_left = float(np.mean(left_licks_delay > 0))
        lick_fraction_delay_right = float(np.mean(right_licks_delay > 0))

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
        "first_lick_latency_all_left": first_lick_latency_all_left,
        "first_lick_latency_all_right": first_lick_latency_all_right,
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

    metrics: Dict[str, Any] = dict(
        sorted(metrics_unsorted.items(), key=lambda kv: kv[0])
    )

    return metrics, trial_df


