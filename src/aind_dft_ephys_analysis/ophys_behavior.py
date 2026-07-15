import os
import glob
import random
import itertools
import numpy as np
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO
from behavior_utils import get_fitted_model_names,get_fitted_latent
from aind_dynamic_foraging_basic_analysis.plot import plot_foraging_session
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter1d
from aind_spurious_correlation import methods
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
from model_fitting_deprecated import ModelFitting
from typing import Optional, Tuple


class OphysBehavior(ModelFitting):
    def __init__(self, session_name, folder_path='/root/capsule/data/'):
        """
        Initializes the OphysBehavior instance for reading ephys and behavior NWB data.

        Parameters:
        - session_name (str): The session name of the ephys NWB file (e.g. '764791_2025-01-15_13-08-58').
        - folder_path (str): The folder path where the ephys and behavior NWB files are stored (default is '/root/capsule/data/').
        """
        self.session_name = session_name
        self.folder_path = folder_path
        self.nwb_ophys_data = None
        self.fitted_latent = {}
        self.internal_fitted_model_names = {}  # Dictionary to hold fitted results keyed by model name.
        self.read_ophys_nwb()
        self.get_fitted_model_names()
        self.get_fitted_latent()
        # copy the nwb_ophys_data to nwb_ophys_data
        self.nwb_behavior_data=self.nwb_ophys_data

    def align_to_event(self,
                    data_name: str = None,
                    channel: str = 'G', 
                    fiber: (str, int) = '1', 
                    processed_type: str = 'P', 
                    fit_methods: str = 'bright',
                    time_window: list = [-3, 6],
                    event_time_stamps_to_align: (list, None) = None,
                    event_name: str = 'go_cue',
                    bin_size: float = 0.05,
                    lick_time_window: float = 1,
                    before_go_cue_lick_time_window: list = [-3, -2],
                    quiet_window: list = [-0.5, 0.5]
                    ) -> np.ndarray:
        """
        Aligns fluorescence data to specified event timestamps and bins it.

        - Extracts data using the specified parameters (`channel`, `fiber`, `processed_type`, `fit_methods`).
        - Aligns the fluorescence signal to `event_time_stamps_to_align` within the given `time_window`.
        - Bins the data at the specified `bin_size` to compute firing rates (like PSTH).
        - Stores the binned data under `self.aligned[data_name][event_name]` for future reference.

        Parameters:
        - data_name (str): The field used to extract the fiber data. If not provided, it will be extracted from channel, fiber, processed_type and fit_methods
        - channel (str): The fluorescence channel.
            - 'G' : Green channel
            - 'R' : Red channel
            - 'Iso' : Isosbestic control channel
        - fiber (str or int): The fiber number, either '1' or '2'.
        - processed_type (str): The type of preprocessing applied.
            - 'P', 'p', 'preprocessed' : Preprocessed data
            - 'D', 'd', 'dff' : Motion-corrected data
        - fit_methods (str): The method used to correct signal quenching.
            - 'bright' : Brightness correction
            - 'exp' : Exponential fitting
            - 'poly' : Polynomial fitting
        - time_window (list): The time window (in seconds) to extract data around each event (default: [-3, 3]).
        - event_time_stamps_to_align (list or None): List of timestamps to align data to. If `None`, no alignment is performed.
        - event_name (str): Name of the event for reference. It will extract event timestamps automatically when the event_time_stamps_to_align is not provided. 
        - bin_size (float): Bin size (in seconds) for binning the fluorescence signal.
        - lick_time_window (float): The time window (in seconds) after go cue to consider the first lick.
        - before_go_cue_lick_time_window (list): For 'before_go_cue_lick...' events, the time window (in seconds)
        - quiet_window (float): For 'quiet_lick' events, how many seconds of no licking must follow the lick for it to be called “quiet.”

        Returns:
        - np.ndarray: A 2D matrix (trials x bins) containing the binned fluorescence data.

        Raises:
        - ValueError: If `event_time_stamps_to_align` is empty or invalid.
        """

        # Validate event timestamps
        if event_time_stamps_to_align is None or len(event_time_stamps_to_align) == 0:
            event_time_stamps_to_align = self.extract_event_timestamps(event_name,lick_time_window,before_go_cue_lick_time_window,quiet_window)
        
        if event_time_stamps_to_align is None or len(event_time_stamps_to_align) == 0:
            raise ValueError("Invalid event_time_stamps_to_align. Must provide a non-empty list of event timestamps.")

        # Generate data key using name_mapper
        if data_name is None:
            data_name = self.name_mapper(channel=channel, fiber=fiber, processed_type=processed_type, fit_methods=fit_methods)

        # Ensure data exists in the NWB file
        if data_name not in self.nwb_ophys_data.acquisition:
            print(f"Data '{data_name}' not found in NWB acquisition.")
            print(f"Looking for the data in processing['fiber_photometry'].")
            if data_name in self.nwb_ophys_data.processing['fiber_photometry'].data_interfaces:
                print(f"Found data in processing['fiber_photometry'].")
                current_data=self.nwb_ophys_data.processing['fiber_photometry']
            else:
                raise ValueError(f"Data '{data_name}' not found in NWB acquisition.")
        else:
            current_data=self.nwb_ophys_data.acquisition

        # Extract fluorescence data and timestamps
        data_values = current_data[data_name].data[:]
        data_timestamps = current_data[data_name].timestamps[:]

        # Define bin edges based on time window and bin size
        bin_edges = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
        num_bins = len(bin_edges) - 1  # Number of bins

        # Initialize matrix to store binned data
        aligned_matrix = np.zeros((len(event_time_stamps_to_align), num_bins))

        # Align data to each event timestamp
        for i, event_time in enumerate(event_time_stamps_to_align):
            # Find the closest index in the timestamp array
            event_idx = np.argmin(np.abs(data_timestamps - event_time))

            # Get the indices corresponding to the desired time window
            start_time = event_time + time_window[0]
            end_time = event_time + time_window[1]

            # Extract data in the window
            valid_indices = np.where((data_timestamps >= start_time) & (data_timestamps <= end_time))[0]
            valid_times = data_timestamps[valid_indices] - event_time
            valid_data = data_values[valid_indices]

            # Bin data
            binned_data, _ = np.histogram(valid_times, bins=bin_edges, weights=valid_data)

            # Normalize by bin width to get mean signal per bin
            aligned_matrix[i, :] = binned_data / bin_size

        # Ensure `self.aligned` exists and has the necessary structure
        if not hasattr(self, "aligned"):
            self.aligned = {}

        if data_name not in self.aligned:
            self.aligned[data_name] = {}

        if 'event_name' not in self.aligned[data_name]:
            self.aligned[data_name][event_name] = {}
        # Store the binned data
        self.aligned[data_name][event_name]['data'] = aligned_matrix
        self.aligned[data_name][event_name]['time_window'] = time_window
        self.aligned[data_name][event_name]['bin_size'] = bin_size

        return aligned_matrix

    def extract_event_timestamps(
        self,
        event_name: str,
        lick_time_window: float = 1.0,
        before_go_cue_lick_time_window: list = [-3, -2],
        quiet_window: list = [-0.5, 0.5]
    ) -> list:
        """
        Extracts event timestamps from the NWB file based on the specified event name.

        This version uses multiple parameters to control different lick time windows:

        Parameters:
        - event_name (str): The name of the event to extract timestamps for. Supported event names include:
            - 'go_cue': Go cue onset times.
            - 'left_lick': Left lick event times.
            - 'right_lick': Right lick event times.
            - 'lick': Combined left and right lick event times.
            - 'reward': Reward delivery event times (left and right).
            - 'reward_left': Left reward delivery event times.
            - 'reward_right': Right reward delivery event times.
            - 'trial_start': Start times of trials.
            - 'trial_end': End times of trials.
            - 'reward_go_cue_start': Go cue onset times where a reward was delivered in that trial.
            - 'no_reward_go_cue_start': Go cue onset times where no reward was delivered in that trial.
            - 'no_response_go_cue_start': Go cue onset times where the animal did not respond (animal_response == 2).

            - 'after_go_cue_first_lick': The first lick after the go cue (any trial).
            - 'after_go_cue_first_left_lick': The first lick after the go cue if and only if that first lick was on the left side.
            - 'after_go_cue_first_right_lick': The first lick after the go cue if and only if that first lick was on the right side.
            - 'after_go_cue_first_lick_reward': The first lick after the go cue in rewarded trials (regardless of side).
            - 'after_go_cue_first_lick_no_reward': The first lick after the go cue in no-reward trials (regardless of side).
            - 'after_go_cue_first_left_lick_reward': The first left lick after the go cue in rewarded trials, but only if it's the first lick of the trial.
            - 'after_go_cue_first_right_lick_reward': The first right lick after the go cue in rewarded trials, but only if it's the first lick of the trial.
            - 'after_go_cue_first_left_lick_no_reward': The first left lick after the go cue in no-reward trials, but only if it's the first lick of the trial.
            - 'after_go_cue_first_right_lick_no_reward': The first right lick after the go cue in no-reward trials, but only if it's the first lick of the trial.

            - 'before_go_cue_lick': All licks (left and right) in [goCue + start_offset, goCue + end_offset), where the offsets come from `before_go_cue_lick_time_window`.
            - 'before_go_cue_lick_left': Only left licks in that same interval.
            - 'before_go_cue_lick_right': Only right licks in that same interval.

            - 'quiet_lick': Any lick (left or right) that has no other lick in `[t + quiet_before, t + quiet_after]`.
            - 'quiet_left_lick': A left lick that has no other lick (left or right) in that window.
            - 'quiet_right_lick': A right lick that has no other lick (left or right) in that window.

        - lick_time_window (float): For 'after_go_cue_first...' events, the time window (in seconds) after go cue.  
        Licks in `[goCue, goCue + lick_time_window)` will be considered. Defaults to `1.0`.

        - before_go_cue_lick_time_window (list): A two-element list `[start_offset, end_offset]` (both typically negative)  
        for 'before_go_cue_lick...' events. Licks in `[goCue + start_offset, goCue + end_offset)` will be included.  
        Defaults to `[-3, -2]`.

        - quiet_window (list): A two-element list `[quiet_before, quiet_after]` used by 'quiet_lick' events. A lick at time `t` is quiet  
        if there is no other lick in `[t + quiet_before, t + quiet_after]`. Typically `quiet_before <= 0 < quiet_after`.  
        Defaults to `[-0.5, 0.5]`.

        Returns:
        - list: A flat list of absolute timestamps corresponding to the specified event.

        Important:
        - If the very first lick in a trial is right, that trial does not count for a 'first_left_lick' event (and vice versa).

        Raises:
        - ValueError: If the provided event name is not found in the NWB trials.
        """

        # 1. Handle simpler events (no trial-based logic).
        if event_name == 'go_cue':
            return self.nwb_ophys_data.trials.goCue_start_time[:].tolist()
        elif event_name == 'left_lick':
            return self.nwb_ophys_data.acquisition['left_lick_time'].timestamps[:].tolist()
        elif event_name == 'right_lick':
            return self.nwb_ophys_data.acquisition['right_lick_time'].timestamps[:].tolist()
        elif event_name == 'lick':
            return (
                self.nwb_ophys_data.acquisition['left_lick_time'].timestamps[:].tolist() +
                self.nwb_ophys_data.acquisition['right_lick_time'].timestamps[:].tolist()
            )
        elif event_name == 'reward':
            return (
                self.nwb_ophys_data.acquisition['right_reward_delivery_time'].timestamps[:].tolist() +
                self.nwb_ophys_data.acquisition['left_reward_delivery_time'].timestamps[:].tolist()
            )
        elif event_name == 'reward_left':
            return self.nwb_ophys_data.acquisition['left_reward_delivery_time'].timestamps[:].tolist()
        elif event_name == 'reward_right':
            return self.nwb_ophys_data.acquisition['right_reward_delivery_time'].timestamps[:].tolist()
        elif event_name == 'trial_start':
            return self.nwb_ophys_data.trials.start_time[:].tolist()
        elif event_name == 'trial_end':
            return self.nwb_ophys_data.trials.stop_time[:].tolist()

        # 2. Identify rewarded vs. no_reward trials
        rewarded_historyL = self.nwb_ophys_data.trials['rewarded_historyL'][:]  # boolean array
        rewarded_historyR = self.nwb_ophys_data.trials['rewarded_historyR'][:]  # boolean array
        rewarded_trials = np.logical_or(rewarded_historyL, rewarded_historyR)

        # Get animal responses for each trial (0: left, 1: right, 2: no response)
        animal_response = self.nwb_ophys_data.trials.animal_response[:]

        if event_name == 'reward_go_cue_start':
            # Return all rewarded trials (including any with no response, if present)
            return self.nwb_ophys_data.trials.goCue_start_time[:][rewarded_trials].tolist()
        elif event_name == 'no_reward_go_cue_start':
            # Exclude trials with no response (animal_response == 2)
            valid_trials = np.logical_and(np.logical_not(rewarded_trials), animal_response != 2)
            return self.nwb_ophys_data.trials.goCue_start_time[:][valid_trials].tolist()
        elif event_name == 'no_response_go_cue_start':
            # Include only trials with no animal response
            no_response_trials = (animal_response == 2)
            return self.nwb_ophys_data.trials.goCue_start_time[:][no_response_trials].tolist()

        # 3. For 'after_go_cue_first...' events: pick FIRST lick in [go_cue, go_cue + lick_time_window)
        if event_name.startswith('after_go_cue_first'):
            go_cue_times = self.nwb_ophys_data.trials.goCue_start_time[:]
            left_lick_times = self.nwb_ophys_data.acquisition['left_lick_time'].timestamps[:]
            right_lick_times = self.nwb_ophys_data.acquisition['right_lick_time'].timestamps[:]

            first_licks_all = []

            for i, go_cue in enumerate(go_cue_times):
                trial_is_rewarded = rewarded_trials[i]

                # If event name has '_reward' but trial not rewarded => skip
                if 'reward' in event_name and 'no_reward' not in event_name:
                    if not trial_is_rewarded:
                        continue
                # If event name has '_no_reward' but trial is rewarded => skip
                elif 'no_reward' in event_name:
                    if trial_is_rewarded:
                        continue

                # Gather all licks within [go_cue, go_cue + lick_time_window]
                # We'll find the earliest lick overall, then decide if it's left or right.
                l_after_go = left_lick_times[
                    (left_lick_times >= go_cue) &
                    (left_lick_times < go_cue + lick_time_window)
                ]
                r_after_go = right_lick_times[
                    (right_lick_times >= go_cue) &
                    (right_lick_times < go_cue + lick_time_window)
                ]
                licks_after_go = np.concatenate((l_after_go, r_after_go))

                if licks_after_go.size == 0:
                    continue

                earliest_lick = np.min(licks_after_go)

                # check side
                earliest_is_left = (earliest_lick in l_after_go)
                earliest_is_right = (earliest_lick in r_after_go)

                # skip if mismatch
                if 'left_lick' in event_name and not earliest_is_left:
                    continue
                if 'right_lick' in event_name and not earliest_is_right:
                    continue

                first_licks_all.append(earliest_lick)

            return first_licks_all

        # 4. For 'before_go_cue_lick...' events: gather ALL licks in [go_cue + start_offset, go_cue + end_offset)
        if event_name.startswith('before_go_cue_lick'):
            go_cue_times = self.nwb_ophys_data.trials.goCue_start_time[:]
            left_lick_times = self.nwb_ophys_data.acquisition['left_lick_time'].timestamps[:]
            right_lick_times = self.nwb_ophys_data.acquisition['right_lick_time'].timestamps[:]

            start_offset, end_offset = before_go_cue_lick_time_window
            before_licks_all = []

            for go_cue in go_cue_times:
                start_t = go_cue + start_offset
                end_t = go_cue + end_offset

                if 'left' in event_name:
                    l_licks = left_lick_times[
                        (left_lick_times >= start_t) & (left_lick_times < end_t)
                    ]
                    before_licks_all.extend(l_licks.tolist())
                elif 'right' in event_name:
                    r_licks = right_lick_times[
                        (right_lick_times >= start_t) & (right_lick_times < end_t)
                    ]
                    before_licks_all.extend(r_licks.tolist())
                else:
                    # combined
                    l_licks = left_lick_times[
                        (left_lick_times >= start_t) & (left_lick_times < end_t)
                    ]
                    r_licks = right_lick_times[
                        (right_lick_times >= start_t) & (right_lick_times < end_t)
                    ]
                    combined = np.concatenate((l_licks, r_licks))
                    before_licks_all.extend(combined.tolist())

            return sorted(before_licks_all)

        # 5. For 'quiet_lick' events
        if event_name.startswith('quiet_lick'):
            """
            A lick is 'quiet' if no other lick is in [t + quiet_before, t + quiet_after].
            We'll plot each candidate lick (gray), highlight the quiet ones (red),
            and draw lines showing the quiet window for the quiet ones.
            """
            left_lick_times = self.nwb_ophys_data.acquisition['left_lick_time'].timestamps[:]
            right_lick_times = self.nwb_ophys_data.acquisition['right_lick_time'].timestamps[:]
            all_licks = np.sort(np.concatenate((left_lick_times, right_lick_times)))

            # Determine candidate licks (side-specific or all)
            if 'left' in event_name:
                candidate_licks = np.sort(left_lick_times)
            elif 'right' in event_name:
                candidate_licks = np.sort(right_lick_times)
            else:
                candidate_licks = all_licks

            quiet_before, quiet_after = quiet_window
            quiet_licks = []

            for t in candidate_licks:
                quiet_start = t + quiet_before
                quiet_end = t + quiet_after

                left_idx = np.searchsorted(all_licks, quiet_start, side='left')
                right_idx = np.searchsorted(all_licks, quiet_end, side='left')
                slice_licks = all_licks[left_idx : right_idx]

                # If no other lick in that slice, or if the only lick is 't', it's quiet
                if len(slice_licks) == 0:
                    quiet_licks.append(t)
                elif len(slice_licks) == 1 and slice_licks[0] == t:
                    quiet_licks.append(t)
                else:
                    unique_licks = np.unique(slice_licks)
                    if len(unique_licks) == 1 and unique_licks[0] == t:
                        quiet_licks.append(t)
                    # else, not quiet

            return sorted(quiet_licks)

        # fallback
        raise ValueError(
            f"Unsupported event name '{event_name}'. Available options: "
            "'go_cue', 'left_lick', 'right_lick', 'lick', 'reward', 'reward_left', 'reward_right', "
            "'trial_start', 'trial_end', 'reward_go_cue_start', 'no_reward_go_cue_start', 'no_response_go_cue_start', "
            "'after_go_cue_first_lick', 'after_go_cue_first_left_lick', 'after_go_cue_first_right_lick', "
            "'after_go_cue_first_lick_reward', 'after_go_cue_first_lick_no_reward', "
            "'after_go_cue_first_left_lick_reward', 'after_go_cue_first_right_lick_reward', "
            "'after_go_cue_first_left_lick_no_reward', 'after_go_cue_first_right_lick_no_reward', "
            "'before_go_cue_lick', 'before_go_cue_lick_left', 'before_go_cue_lick_right', "
            "'quiet_lick', 'quiet_left_lick', 'quiet_right_lick'."
        )



    def name_mapper(self, 
                    channel: str = 'G', 
                    fiber: (str, int) = '1', 
                    processed_type: str = 'P', 
                    fit_methods: str = 'bright') -> str:
        """
        Constructs a standardized filename based on the given parameters.

        Parameters:
        - channel (str): The fluorescence channel.
            - 'G' : Green channel
            - 'R' : Red channel
            - 'Iso' : Isosbestic control channel
        - fiber (str or int): The fiber number, either '1' or '2'.
        - processed_type (str): The type of preprocessing applied.
            - 'P', 'p', 'preprocessed' : Preprocessed data
            - 'D', 'd', 'dff' : motion-corrected data
        - fit_methods (str): The method used to correct signal quenching.
            - 'bright' : Brightness correction
            - 'exp' : Exponential fitting
            - 'poly' : Polynomial fitting

        Returns:
        - str: A standardized string name following the format:
                "{channel}_{fiber}_{processed_type}-{fit_methods}"

        Raises:
        - ValueError: If an invalid value is provided for any of the parameters.
        """

        # Validate channel
        valid_channels = {'G', 'R', 'Iso'}
        if channel not in valid_channels:
            raise ValueError(f"Invalid channel '{channel}'. Must be one of {valid_channels}.")

        # Validate fiber
        if str(fiber) not in {'1', '2'}:
            raise ValueError(f"Invalid fiber '{fiber}'. Must be '1' or '2'.")

        # Validate processed_type
        processed_mapping = {'P': 'preprocessed', 'p': 'preprocessed', 'preprocessed': 'preprocessed',
                            'D': 'dff', 'd': 'dff', 'dff': 'dff'}
        if processed_type not in processed_mapping:
            raise ValueError(f"Invalid processed_type '{processed_type}'. Must be one of {list(processed_mapping.keys())}.")

        # Validate fit_methods
        valid_fit_methods = {'bright', 'exp', 'poly'}
        if fit_methods not in valid_fit_methods:
            raise ValueError(f"Invalid fit_methods '{fit_methods}'. Must be one of {valid_fit_methods}.")

        # Construct name
        processed_str = processed_mapping[processed_type]
        name = f"{channel}_{fiber}_{processed_str}-{fit_methods}"

        return name

    def plot_heat_map(self, 
                  data_name: (str, list) = 'G_1_preprocessed-bright',
                  event_name: (str, list) = 'go_cue',
                  cmap: str = 'viridis',
                  value_range: Optional[Tuple[float, float]] = None,
                  behavior_model: Optional[str] = 'q_learning_Y1',
                  align_by_latent_name: Optional[str] = None,
                  ) -> None:
        """
        Plots heatmaps of aligned fluorescence data for all combinations of data_name and event_name.
        The subplots are arranged so that each row corresponds to one data channel (data_name) and
        each column corresponds to an event (event_name).

        If `align_by_latent_name` is provided, each heatmap's rows are reordered by ascending
        values of that latent variable (taken from the fitted model `behavior_model`) so that
        lower-latent trials appear at the bottom and higher-latent trials at the top of the heatmap.

        Parameters
        ----------
        data_name : str or list of str
            The key(s) used to extract fluorescence data.
        event_name : str or list of str
            The event name(s) to which the data are aligned.
        cmap : str
            The matplotlib colormap to use.
        value_range : tuple of float or None
            If provided, (vmin, vmax) for color-clipping each row's plots. If None, each row's
            color scale is automatically adjusted to the global min and max of that row.
        behavior_model : str or None
            Name of the fitted behavior model to extract latent data from. Only needed if
            `align_by_latent_name` is not None.
        align_by_latent_name : str or None
            The name of the latent variable to use for reordering the trials in ascending order.
            If None, the trials are left in the original alignment order.

        Returns
        -------
        None
            Displays the heatmap plots.
        """
        # Ensure inputs are lists.
        if not isinstance(data_name, list):
            data_names = [data_name]
        else:
            data_names = data_name

        if not isinstance(event_name, list):
            event_names = [event_name]
        else:
            event_names = event_name

        n_rows = len(data_names)
        n_cols = len(event_names)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        
        # This will hold the imshow objects for each row so we can unify color ranges per row.
        im_list_by_row = [[] for _ in range(n_rows)]

        trials = self.nwb_behavior_data.intervals['trials'][:]
        choice_history = trials['animal_response'].to_numpy()
        valid_mask = choice_history != 2  # Exclude trials with no response

        for i, dn in enumerate(data_names):
            # ---------------------------------------------------------------------
            # If align_by_latent_name is specified, fetch and store the latent data
            # for reordering. We'll do it once per data channel (not per event).
            # This presumes each row in the aligned matrix is a single trial in order!
            # ---------------------------------------------------------------------
            if align_by_latent_name is not None:
                if behavior_model == 'q_learning_Y1':
                    self.fit_q_learning_model()
                latent_values = self.extract_fitted_data(model_name=behavior_model, 
                                                        latent_name=align_by_latent_name)
                latent_values = np.array(latent_values)  # Ensure it's a numpy array.

            for j, en in enumerate(event_names):
                ax = axs[i, j]

                # Ensure aligned data exist; if not, compute alignment on-the-fly.
                if not (hasattr(self, "aligned") and dn in self.aligned and en in self.aligned[dn]):
                    print(f"Aligned data not found for {dn} and event {en}. Computing alignment...")
                    self.align_to_event(data_name=dn, event_name=en)
                
                aligned_data = self.aligned[dn][en]
                aligned_matrix = aligned_data["data"]
                time_window = aligned_data["time_window"]
                
                if align_by_latent_name is not None:
                    # 2. Exclude no-response trials using animal responses from behavior data.
                    aligned_matrix = aligned_matrix[valid_mask, :]

                # If there's no valid data, skip.
                if aligned_matrix is None or aligned_matrix.size == 0:
                    print(f"Error: No valid data for {dn} and event {en}.")
                    continue

                # ---------------------------------------------------------------------
                # If we have a latent variable, and the number of trials matches,
                # reorder the rows by ascending latent value.
                # ---------------------------------------------------------------------
                if align_by_latent_name is not None:
                    if latent_values.shape[0] == aligned_matrix.shape[0]:
                        # Sort indices by ascending latent value
                        sort_idx = np.argsort(latent_values)
                        aligned_matrix = aligned_matrix[sort_idx, :]
                    else:
                        print(f"Warning: Cannot reorder by latent '{align_by_latent_name}' "
                            f"for data '{dn}' & event '{en}' because length mismatch "
                            f"(latent={len(latent_values)} vs. matrix={aligned_matrix.shape[0]}).")

                # Prepare time bins for the x-axis
                num_bins = aligned_matrix.shape[1]
                time_bins = np.linspace(time_window[0], time_window[1], num_bins)

                # Clip data if value_range is provided (applied after sorting).
                if value_range is not None:
                    data_to_plot = np.clip(aligned_matrix, *value_range)
                else:
                    data_to_plot = aligned_matrix

                # Plot the heatmap
                im = ax.imshow(data_to_plot, aspect='auto', cmap=cmap, interpolation='none',
                            origin='lower', extent=[time_bins[0], time_bins[-1], 0, aligned_matrix.shape[0]])
                # Draw a vertical line at time=0
                ax.axvline(0, color='red', linestyle='--', linewidth=2, label=en)
                ax.set_xlabel("Time (s) from Event")
                ax.set_ylabel("Trials")
                title_str = f"{dn}\n{en}"
                if align_by_latent_name is not None:
                    title_str += f"\n(sorted by {align_by_latent_name})"
                ax.set_title(title_str)
                ax.legend(loc='upper left')

                # Append the image to our list for color-range unification
                im_list_by_row[i].append(im)

                # Draw a colorbar for each subplot
                fig.colorbar(im, ax=ax)

        # ------------------------------------------------------------------------------
        # For each data channel (row), unify the color range across all columns 
        # if value_range is not provided. We compute the global min/max across subplots.
        # ------------------------------------------------------------------------------
        for i in range(n_rows):
            if value_range is None:
                row_vmin, row_vmax = np.inf, -np.inf
                for im in im_list_by_row[i]:
                    data = im.get_array()
                    row_vmin = min(row_vmin, np.nanmin(data))
                    row_vmax = max(row_vmax, np.nanmax(data))
                current_range = (row_vmin, row_vmax)
            else:
                current_range = value_range

            for im in im_list_by_row[i]:
                im.set_clim(current_range)

        plt.tight_layout()
        plt.show()



    def plot_psth(self, 
                data_name: (str, list) = 'G_1_preprocessed-bright', 
                event_name: (str, list) = 'go_cue', 
                smooth: float = 0.1,
                y_range: Optional[list] = None) -> None:
        """
        Plots peri-stimulus time histograms (PSTHs) of fluorescence data for all
        combinations of data_name and event_name. The subplots are arranged so that
        every row corresponds to a given data channel (data_name) and all subplots in
        that row (i.e. for different events) share the same y-axis range.
        
        If y_range is provided, that range is applied; otherwise, after plotting the
        method computes the global y-range for each data channel and updates the plots.

        Parameters:
        data_name (str or list of str): The key(s) for the fluorescence data.
        event_name (str or list of str): The event name(s) to which the data are aligned.
        smooth (float): Standard deviation (in seconds) for Gaussian smoothing; if 0, no smoothing.
        y_range (list, optional): A two-element list [ymin, ymax] to use for all subplots in a row.
                                    If None, the global range is computed per data_name.
        
        Returns:
        None (displays the PSTH plots).
        """
        # Wrap inputs as lists if necessary.
        if not isinstance(data_name, list):
            data_names = [data_name]
        else:
            data_names = data_name

        if not isinstance(event_name, list):
            event_names = [event_name]
        else:
            event_names = event_name

        n_rows = len(data_names)
        n_cols = len(event_names)

        # Create a grid of subplots with each row corresponding to one data_name.
        # We'll use sharey='row' so that subplots in the same row are linked.
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey='row')
        
        # If there's only one row or one column, make sure axs is 2D.
        if n_rows == 1 and n_cols == 1:
            axs = np.array([[axs]])
        elif n_rows == 1:
            axs = np.array([axs])
        elif n_cols == 1:
            axs = np.array([[ax] for ax in axs])
        
        # For each data channel (row) and each event (column)...
        for i, dn in enumerate(data_names):
            for j, en in enumerate(event_names):
                ax = axs[i, j]
                # Ensure aligned data exist for the combination; if not, compute it.
                if not (hasattr(self, "aligned") and dn in self.aligned and en in self.aligned[dn]):
                    print(f"Aligned data not found for {dn} and event {en}. Computing alignment...")
                    self.align_to_event(data_name=dn, event_name=en)
                aligned_data = self.aligned[dn][en]
                aligned_matrix = aligned_data["data"]
                time_window = aligned_data["time_window"]
                bin_size = aligned_data["bin_size"]
                
                if aligned_matrix is None or aligned_matrix.size == 0:
                    print(f"Error: No valid data for {dn} and event {en}.")
                    continue

                num_bins = aligned_matrix.shape[1]
                time_bins = np.linspace(time_window[0], time_window[1], num_bins)
                
                # Compute mean and SEM across trials.
                mean_fluorescence = np.nanmean(aligned_matrix, axis=0)
                sem_fluorescence = np.nanstd(aligned_matrix, axis=0) / np.sqrt(aligned_matrix.shape[0])
                if smooth > 0:
                    sigma_bins = smooth / bin_size
                    mean_fluorescence = gaussian_filter1d(mean_fluorescence, sigma=sigma_bins)
                    sem_fluorescence = gaussian_filter1d(sem_fluorescence, sigma=sigma_bins)
                
                ax.plot(time_bins, mean_fluorescence, color='b', label='Mean')
                ax.fill_between(time_bins, mean_fluorescence - sem_fluorescence,
                                mean_fluorescence + sem_fluorescence, color='b', alpha=0.3, label='SEM')
                ax.axvline(0, color='red', linestyle='--', linewidth=2, label=en)
                ax.set_xlabel("Time (s) from Event")
                ax.set_ylabel("Fluorescence (dF/F)")
                ax.set_title(f"{dn} \n {en}")
                ax.legend(loc='upper left',fontsize=6)
        
        # For each row (each data channel), update the y-axis range.
        for i in range(n_rows):
            row_axes = axs[i, :] if n_cols > 1 else [axs[i, 0]]
            if y_range is None:
                global_ymin = np.inf
                global_ymax = -np.inf
                for ax in row_axes:
                    ymin, ymax = ax.get_ylim()
                    global_ymin = min(global_ymin, ymin)
                    global_ymax = max(global_ymax, ymax)
                new_y_range = [global_ymin, global_ymax]
            else:
                new_y_range = y_range
            for ax in row_axes:
                ax.set_ylim(new_y_range)
        
        plt.tight_layout()
        plt.show()


    def plot_licks_raster_and_psth(self,
                               event_names: list = None,
                               time_window: list = None,
                               bin_size: float = 0.05,
                               lick_time_window: float = 1,
                               before_go_cue_lick_time_window: list = None,
                               offset: float = 0,
                               plot_combine: bool = False,
                               align_by_latent_name: Optional[str] = None,
                               behavior_model: Optional[str] = None):
        """
        Plots a raster (top) and PSTH (bottom) of left and right licks aligned to specified events.
        Optionally reorders (aligns) the trials by ascending latent values, if `align_by_latent_name` is given
        and each event_time array matches one-to-one with the latent array.

        Parameters
        ----------
        event_names : list
            A list of event names to align to. If None, uses:
                ['reward_go_cue_start', 'no_reward_go_cue_start', 'no_response_go_cue_start', 'go_cue']
        time_window : list
            The time window [start, end] (in seconds) around the event to include. Default [-2, 2].
        bin_size : float
            The bin size (in seconds) for computing the PSTH. Default 0.05.
        lick_time_window : float
            Time window (in seconds) after the event for the user's internal logic (passed to `extract_event_timestamps`).
        before_go_cue_lick_time_window : list
            A two-element list [start_offset, end_offset] (both typically negative) for 'before_go_cue_lick...' events.
            Defaults to [-3, -2].
        offset : float
            A constant offset added to all extracted event timestamps (shifts the alignment).
        plot_combine : bool
            If True, plot the raster in separate vertical panels (one per event) on the top and overlay all PSTHs
            in a panel below. Otherwise, produce a separate figure for each event.
        align_by_latent_name : str or None
            If provided, reorder the event_times by ascending latent value for the chosen `behavior_model`.
            This only works if the event has exactly one timestamp per trial and the length matches the latent array.
            If there's a mismatch, a warning is printed and no reordering is done for that event.
        behavior_model : str or None
            Name of the fitted behavior model from which to extract latents. For instance, 'q_learning_Y1'.

        Returns
        -------
        None
            Displays the raster and PSTH plots (either combined or one-per-event).
        """
        # Default arguments
        if event_names is None:
            event_names = [
                'reward_go_cue_start',
                'no_reward_go_cue_start',
                'no_response_go_cue_start',
                'go_cue'
            ]
        if time_window is None:
            time_window = [-2, 2]
        if before_go_cue_lick_time_window is None:
            before_go_cue_lick_time_window = [-3, -2]

        # ---------------------------------------------------
        # If user requests latent-based reordering, get latents
        # ---------------------------------------------------
        latent_values = None
        if align_by_latent_name is not None:
            if behavior_model is not None:
                # Example: you can add custom logic to fit or retrieve latents
                if behavior_model == 'q_learning_Y1':
                    self.fit_q_learning_model()  # If you have such a method
                # Extract latents as a NumPy array
                latent_values = np.array(
                    self.extract_fitted_data(
                        model_name=behavior_model,
                        latent_name=align_by_latent_name
                    )
                )
            else:
                print("Warning: `align_by_latent_name` given but `behavior_model` is None. "
                    "Cannot extract latents. Will skip reordering.")
                latent_values = None

        # ------------------------------------------------------------------------------------
        # Helper function to reorder events if length matches latent_values,
        # after excluding no-response trials (animal_response == 2).
        # ------------------------------------------------------------------------------------
        def reorder_events_by_latent(event_times: np.ndarray) -> np.ndarray:
            """
            Returns a copy of event_times sorted by ascending latent_values, but only for
            trials that are not no-response (i.e., animal_response != 2). We do so by:
            1. Excluding no-response trials from event_times and latent_values.
            2. Sorting the remaining valid trials by ascending latent_values.
            3. Returning the reordered event_times for those valid trials.

            If there's any mismatch in length after excluding no-response, a warning is printed
            and we return event_times (valid subset) as-is, without sorting.

            Parameters
            ----------
            event_times : np.ndarray
                An array of event timestamps, one per trial (ideally).

            Returns
            -------
            np.ndarray
                A reordered (or partially filtered) copy of event_times.
            """
            if latent_values is None:
                # If we have no latent values, we can't reorder.
                return event_times

            # 1) Get the trial choices from NWB. Adjust if you store them differently.
            choice_history = self.nwb_ophys_data.trials['animal_response'][:]
            # Ensure shapes match at the outset.
            if len(event_times) != len(choice_history):
                print("Warning: event_times length != choice_history length. "
                    "Skipping reordering since we cannot reliably match trials.")
                return event_times

            # 2) Exclude no-response trials from event_times, latent_values, and choice_history
            valid_mask = (choice_history != 2)
            event_times_valid = event_times[valid_mask]
            latent_values_valid = latent_values[valid_mask]

            # 3) Check if we can reorder
            if len(event_times_valid) == len(latent_values_valid):
                # Sort by ascending latent values
                sort_idx = np.argsort(latent_values_valid)
                # Return only the sorted event_times (valid trials)
                return event_times_valid[sort_idx]
            else:
                print(f"Warning: after excluding no-response trials, we have "
                    f"len(event_times_valid)={len(event_times_valid)} vs. "
                    f"len(latent_values_valid)={len(latent_values_valid)}. "
                    "Skipping reordering for this event.")
                return event_times_valid


        # ---------------------------------------------------
        # Non-combined mode (one figure per event)
        # ---------------------------------------------------
        if not plot_combine:
            for event_name in event_names:
                # 1) Extract event timestamps
                event_times = np.array(
                    self.extract_event_timestamps(
                        event_name=event_name,
                        lick_time_window=lick_time_window,
                        before_go_cue_lick_time_window=before_go_cue_lick_time_window
                    )
                )

                # 2) Add offset
                event_times = event_times + offset

                # 3) If requested, reorder by latent
                event_times = reorder_events_by_latent(event_times)

                if len(event_times) == 0:
                    print(f"No event timestamps found for event '{event_name}'.")
                    continue

                # 4) Gather left/right lick times relative to each event time
                left_lick_times = self.nwb_ophys_data.acquisition['left_lick_time'].timestamps[:]
                right_lick_times = self.nwb_ophys_data.acquisition['right_lick_time'].timestamps[:]

                all_left_rasters = []
                all_right_rasters = []
                for evt in event_times:
                    l_mask = (left_lick_times >= evt + time_window[0]) & (left_lick_times <= evt + time_window[1])
                    r_mask = (right_lick_times >= evt + time_window[0]) & (right_lick_times <= evt + time_window[1])
                    l_times_relative = left_lick_times[l_mask] - evt
                    r_times_relative = right_lick_times[r_mask] - evt
                    all_left_rasters.append(l_times_relative)
                    all_right_rasters.append(r_times_relative)

                # 5) Create figure with 2 panels: top=raster, bottom=PSTH
                fig = plt.figure(figsize=(8, 8))
                gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.2)
                ax_raster = fig.add_subplot(gs[0])
                ax_psth = fig.add_subplot(gs[1], sharex=ax_raster)

                # Raster plot: each row is a trial
                for i, (l_times, r_times) in enumerate(zip(all_left_rasters, all_right_rasters)):
                    ax_raster.vlines(l_times, i + 0.5, i + 1.5, color='blue', linewidth=1)
                    ax_raster.vlines(r_times, i + 0.5, i + 1.5, color='red', linewidth=1)
                ax_raster.axvline(0, color='k', linestyle='--')
                ax_raster.set_ylabel('Trials')
                title_str = f"Raster of Licks aligned to {event_name}"
                if align_by_latent_name is not None and latent_values is not None:
                    title_str += f"\n(sorted by {align_by_latent_name})"
                if offset != 0:
                    title_str += f" (offset={offset:.2f}s)"
                ax_raster.set_title(title_str)
                ax_raster.set_xlim(time_window)
                ax_raster.set_ylim([0, len(event_times) + 1])

                # PSTH
                all_left_concat = np.concatenate(all_left_rasters) if len(all_left_rasters) else np.array([])
                all_right_concat = np.concatenate(all_right_rasters) if len(all_right_rasters) else np.array([])
                bins = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
                left_counts, _ = np.histogram(all_left_concat, bins=bins)
                right_counts, _ = np.histogram(all_right_concat, bins=bins)
                num_trials = len(event_times)
                # Avoid division by zero
                if num_trials > 0:
                    left_rate = left_counts / (num_trials * bin_size)
                    right_rate = right_counts / (num_trials * bin_size)
                else:
                    left_rate = np.zeros_like(left_counts)
                    right_rate = np.zeros_like(right_counts)
                bin_centers = (bins[:-1] + bins[1:]) / 2.0

                ax_psth.plot(bin_centers, left_rate, '-', color='blue', label='Left Lick Rate')
                ax_psth.plot(bin_centers, right_rate, '-', color='red', label='Right Lick Rate')
                ax_psth.axvline(0, color='k', linestyle='--')
                ax_psth.set_xlabel('Time (s)')
                ax_psth.set_ylabel('Lick Rate (Hz)')
                ax_psth.set_xlim(time_window)
                ax_psth.set_title('PSTH of Licks')
                ax_psth.legend()

                plt.tight_layout()
                plt.show()

        # ---------------------------------------------------
        # Combined mode (one figure, multiple raster panels + single PSTH)
        # ---------------------------------------------------
        else:
            if len(event_names) == 0:
                print("No event names provided; nothing to plot.")
                return

            n_events = len(event_names)
            # We'll have n_events raster panels + 1 PSTH panel
            fig = plt.figure(figsize=(10, 3 * n_events + 3))
            gs = gridspec.GridSpec(n_events + 1, 1, height_ratios=[1]*n_events + [1.2], hspace=0.2)

            # Prepare bins for PSTH
            bins = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            # We'll accumulate PSTH data for each event to overlay in the bottom panel
            psth_data = {}  # dict: event_name -> (left_concat, right_concat, n_trials)

            for i, event_name in enumerate(event_names):
                ax_raster = fig.add_subplot(gs[i])
                event_times = np.array(
                    self.extract_event_timestamps(
                        event_name=event_name,
                        lick_time_window=lick_time_window,
                        before_go_cue_lick_time_window=before_go_cue_lick_time_window
                    )
                )
                event_times = event_times + offset
                # Attempt reorder by latent
                event_times = reorder_events_by_latent(event_times)

                if len(event_times) == 0:
                    print(f"No event timestamps found for event '{event_name}'.")
                    continue

                left_lick_times = self.nwb_ophys_data.acquisition['left_lick_time'].timestamps[:]
                right_lick_times = self.nwb_ophys_data.acquisition['right_lick_time'].timestamps[:]

                all_left_rasters = []
                all_right_rasters = []
                for evt in event_times:
                    l_mask = (left_lick_times >= evt + time_window[0]) & (left_lick_times <= evt + time_window[1])
                    r_mask = (right_lick_times >= evt + time_window[0]) & (right_lick_times <= evt + time_window[1])
                    all_left_rasters.append(left_lick_times[l_mask] - evt)
                    all_right_rasters.append(right_lick_times[r_mask] - evt)

                n_trials = len(event_times)
                # Plot the raster for this event
                for j, (l_times, r_times) in enumerate(zip(all_left_rasters, all_right_rasters)):
                    trial_row = j + 1
                    ax_raster.vlines(l_times, trial_row + 0.5, trial_row + 1.5, color='blue', linewidth=1.5)
                    ax_raster.vlines(r_times, trial_row + 0.5, trial_row + 1.5, color='red', linewidth=1.5)

                ax_raster.axvline(0, color='k', linestyle='--')
                ax_raster.set_xlim(time_window)
                ax_raster.set_ylim([0, n_trials + 1])
                # Show only min/max trial # on y-axis
                ax_raster.set_yticks([1, n_trials])
                ax_raster.set_yticklabels([1, n_trials])
                title_str = f"Raster: {event_name}"
                if align_by_latent_name:
                    title_str += f"\n(sorted by {align_by_latent_name})"
                if offset != 0:
                    title_str += f" (offset={offset:.2f}s)"
                ax_raster.set_title(title_str, fontsize='medium')
                ax_raster.set_ylabel('Trial #', fontsize='small')

                # PSTH data
                left_snip = np.concatenate(all_left_rasters) if len(all_left_rasters) else np.array([])
                right_snip = np.concatenate(all_right_rasters) if len(all_right_rasters) else np.array([])
                psth_data[event_name] = (left_snip, right_snip, n_trials)

            # -------------------------------------------------------
            # Bottom PSTH panel: overlay all events
            # -------------------------------------------------------
            ax_psth = fig.add_subplot(gs[-1])  # last row

            for event_name, (left_concat, right_concat, n_trials) in psth_data.items():
                if n_trials < 1:
                    continue
                left_counts, _ = np.histogram(left_concat, bins=bins)
                right_counts, _ = np.histogram(right_concat, bins=bins)
                left_rate = left_counts / (n_trials * bin_size)
                right_rate = right_counts / (n_trials * bin_size)

                color = plt.cm.tab10(event_names.index(event_name) % 10)  # pick a color from tab10
                ax_psth.plot(bin_centers, left_rate, '-', color=color, linewidth=2,
                            label=f"{event_name} Left")
                ax_psth.plot(bin_centers, right_rate, '--', color=color, linewidth=2,
                            label=f"{event_name} Right")

            ax_psth.axvline(0, color='k', linestyle='--')
            ax_psth.set_xlabel('Time (s)')
            ax_psth.set_ylabel('Lick Rate (Hz)')
            ax_psth.set_xlim(time_window)
            ax_psth.set_title('Combined PSTH of Licks', fontsize='medium')
            ax_psth.legend(fontsize='small', loc='upper right')

            plt.show()


    def plot_correlation_two_channel(self,
                                    data_x_name: str='G_1_preprocessed-bright',
                                    data_y_name: str='G_2_preprocessed-bright',
                                    bin_size: float = 0.1):
        """
        Plots the correlation between two fluorescence channels by binning their activities
        in time and then computing the average within each bin.

        Parameters:
        - data_x_name (str): The acquisition name for the X-channel in self.nwb_ophys_data.
        - data_y_name (str): The acquisition name for the Y-channel in self.nwb_ophys_data.
        - bin_size (float): The size of the time bins (in seconds) used to average the activity.

        Returns:
        - (slope, intercept, r_value, p_value): The linear-fit parameters and Pearson correlation stats.
        * slope, intercept: from np.polyfit of x_means vs. y_means
        * r_value, p_value: from scipy.stats.pearsonr on x_means vs. y_means
        """

        # 1) Load X and Y data
        x_data_values = self.nwb_ophys_data.acquisition[data_x_name].data[:]
        x_data_timestamps = self.nwb_ophys_data.acquisition[data_x_name].timestamps[:]

        y_data_values = self.nwb_ophys_data.acquisition[data_y_name].data[:]
        y_data_timestamps = self.nwb_ophys_data.acquisition[data_y_name].timestamps[:]

        # 2) Determine a common time range to bin over
        start_time = max(x_data_timestamps[0], y_data_timestamps[0])
        end_time = min(x_data_timestamps[-1], y_data_timestamps[-1])
        if end_time <= start_time:
            print("No overlapping time range found between X and Y channels.")
            return None

        # Create bin edges
        bin_edges = np.arange(start_time, end_time + bin_size, bin_size)
        num_bins = len(bin_edges) - 1

        # 3) Prepare arrays to store the binned-mean activity
        x_bin_means = np.zeros(num_bins)
        y_bin_means = np.zeros(num_bins)

        # 4) For each bin, compute the mean X, mean Y
        # We'll use np.logical_and to select data within each bin range.
        for i in range(num_bins):
            t0 = bin_edges[i]
            t1 = bin_edges[i+1]

            # Indices for x_data in [t0, t1)
            x_idx = np.where((x_data_timestamps >= t0) & (x_data_timestamps < t1))[0]
            # Indices for y_data in [t0, t1)
            y_idx = np.where((y_data_timestamps >= t0) & (y_data_timestamps < t1))[0]

            # If no data points in that bin, the mean should be NaN (or 0). We'll use NaN for clarity.
            if len(x_idx) > 0:
                x_bin_means[i] = np.mean(x_data_values[x_idx])
            else:
                x_bin_means[i] = np.nan

            if len(y_idx) > 0:
                y_bin_means[i] = np.mean(y_data_values[y_idx])
            else:
                y_bin_means[i] = np.nan

        # Optionally, we might drop bins with NaN in either X or Y
        valid_mask = ~np.isnan(x_bin_means) & ~np.isnan(y_bin_means)
        x_bin_means = x_bin_means[valid_mask]
        y_bin_means = y_bin_means[valid_mask]

        if len(x_bin_means) == 0:
            print("No valid binned data points for correlation.")
            return None

        # 5) Plot a scatter of binned means
        plt.figure(figsize=(6, 6))
        plt.scatter(x_bin_means, y_bin_means, alpha=0.7, label="Binned Means")

        # 6) Fit a linear model (slope, intercept)
        slope, intercept = np.polyfit(x_bin_means, y_bin_means, 1)
        fit_line_x = np.linspace(np.min(x_bin_means), np.max(x_bin_means), 100)
        fit_line_y = slope * fit_line_x + intercept

        plt.plot(fit_line_x, fit_line_y, 'r--', label=f"Fit: y={slope:.3f}x+{intercept:.3f}")

        # 7) Compute the correlation stats
        r_value, p_value = pearsonr(x_bin_means, y_bin_means)

        plt.title(f"Correlation: r={r_value:.3f}, p={p_value:.1e}")
        plt.xlabel(data_x_name)
        plt.ylabel(data_y_name)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return slope, intercept, r_value, p_value

    def plot_example_trace_two_channel(self,
                                    data_x_name: str = 'G_1_preprocessed-bright',
                                    data_y_name: str = 'G_2_preprocessed-bright',
                                    time_window: list = None,
                                    window_length: float = 20,
                                    overlay_licks: bool = True,
                                    overlay_reward: bool = True):
        """
        Plots example traces from two data channels over either a specified or random time window,
        with optional overlays for licks and reward events.

        Parameters:
        - data_x_name (str): Acquisition name for channel X. Default 'G_1_preprocessed-bright'.
        - data_y_name (str): Acquisition name for channel Y. Default 'G_2_preprocessed-bright'.
        - time_window (list): If provided (e.g., [start, end]), use that range. If None, select a random window of length 'window_length'.
        - window_length (float): The length (in seconds) for random window selection, used only if 'time_window' is None.
        - overlay_licks (bool): Whether to overlay left/right lick timestamps as vertical dotted lines.
        - overlay_reward (bool): Whether to overlay left/right reward events as vertical dotted lines (in green).

        Returns:
        - None (displays a figure).
        """

        # 1) Load X and Y data
        x_data_values = self.nwb_ophys_data.acquisition[data_x_name].data[:]
        x_data_timestamps = self.nwb_ophys_data.acquisition[data_x_name].timestamps[:]

        y_data_values = self.nwb_ophys_data.acquisition[data_y_name].data[:]
        y_data_timestamps = self.nwb_ophys_data.acquisition[data_y_name].timestamps[:]

        # 2) Find overall overlapping time range
        start_t = max(x_data_timestamps[0], y_data_timestamps[0])
        end_t   = min(x_data_timestamps[-1], y_data_timestamps[-1])

        if end_t <= start_t:
            print("No overlapping time range found between X and Y channels.")
            return

        # 3) Determine the actual plotting window [plot_start, plot_end]
        if time_window is not None:
            # use the user-supplied window
            if len(time_window) != 2 or time_window[0] >= time_window[1]:
                print("Invalid 'time_window'. Must be [start, end] with start < end.")
                return
            plot_start = max(start_t, time_window[0])  # also clamp to available data range
            plot_end   = min(end_t,   time_window[1])
            if plot_end <= plot_start:
                print("Provided time_window does not intersect with available data range.")
                return
        else:
            # pick a random window of length 'window_length'
            max_start = end_t - window_length
            if max_start < start_t:
                overlap_dur = end_t - start_t
                print(f"Not enough overlap to show a {window_length}s window. Overlap: {overlap_dur:.2f}s.")
                return
            plot_start = random.uniform(start_t, max_start)
            plot_end   = plot_start + window_length

        # 4) Extract data for X in [plot_start, plot_end]
        x_mask = (x_data_timestamps >= plot_start) & (x_data_timestamps <= plot_end)
        x_seg_t = x_data_timestamps[x_mask]
        x_seg_v = x_data_values[x_mask]

        # 5) Extract data for Y in [plot_start, plot_end]
        y_mask = (y_data_timestamps >= plot_start) & (y_data_timestamps <= plot_end)
        y_seg_t = y_data_timestamps[y_mask]
        y_seg_v = y_data_values[y_mask]

        # Shift time so the first point is t=0 in the plot (optional)
        x_seg_t_plot = x_seg_t - plot_start
        y_seg_t_plot = y_seg_t - plot_start

        # 6) If overlaying licks, gather them
        left_snip_plot = []
        right_snip_plot = []
        if overlay_licks:
            left_lick_times  = self.nwb_ophys_data.acquisition['left_lick_time'].timestamps[:]
            right_lick_times = self.nwb_ophys_data.acquisition['right_lick_time'].timestamps[:]

            # Filter to [plot_start, plot_end]
            left_snip = left_lick_times[
                (left_lick_times >= plot_start) & (left_lick_times <= plot_end)
            ]
            right_snip = right_lick_times[
                (right_lick_times >= plot_start) & (right_lick_times <= plot_end)
            ]

            # Shift them so that plot_start is 0
            left_snip_plot = left_snip - plot_start
            right_snip_plot = right_snip - plot_start

        # 7) If overlaying reward, gather them
        reward_snip_plot = []
        if overlay_reward:
            # Typically 'left_reward_delivery_time' & 'right_reward_delivery_time' are used
            left_reward_times  = self.nwb_ophys_data.acquisition['left_reward_delivery_time'].timestamps[:]
            right_reward_times = self.nwb_ophys_data.acquisition['right_reward_delivery_time'].timestamps[:]
            all_reward_times = np.sort(np.concatenate((left_reward_times, right_reward_times)))

            reward_snip = all_reward_times[
                (all_reward_times >= plot_start) & (all_reward_times <= plot_end)
            ]
            reward_snip_plot = reward_snip - plot_start

        # 8) Create the figure
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(10, 6))

        # (A) Top panel: X data
        ax1.plot(x_seg_t_plot, x_seg_v, color='blue', label=data_x_name)
        ax1.set_ylabel(data_x_name)
        ax1.set_title(f"Example Trace: {data_x_name} vs. {data_y_name}\n"
                    f"Window ~ [{plot_start:.2f}, {plot_end:.2f}] s")

        # Overlay licks in top panel
        if overlay_licks and (len(left_snip_plot) + len(right_snip_plot) > 0):
            ymin_x, ymax_x = ax1.get_ylim()
            for L in left_snip_plot:
                ax1.vlines(L, ymin_x, ymax_x, color='blue', linestyle=':', alpha=0.3)
            for R in right_snip_plot:
                ax1.vlines(R, ymin_x, ymax_x, color='red', linestyle=':', alpha=0.3)

        # Overlay reward in top panel
        if overlay_reward and len(reward_snip_plot) > 0:
            ymin_x, ymax_x = ax1.get_ylim()
            for rew in reward_snip_plot:
                ax1.vlines(rew, ymin_x, ymax_x, color='green', linestyle='-', alpha=0.5)

        # (B) Bottom panel: Y data
        ax2.plot(y_seg_t_plot, y_seg_v, color='red', label=data_y_name)
        ax2.set_xlabel("Time (s, from window start)")
        ax2.set_ylabel(data_y_name)

        # Overlay licks in bottom panel
        if overlay_licks and (len(left_snip_plot) + len(right_snip_plot) > 0):
            ymin_y, ymax_y = ax2.get_ylim()
            for L in left_snip_plot:
                ax2.vlines(L, ymin_y, ymax_y, color='blue', linestyle=':', alpha=0.3)
            for R in right_snip_plot:
                ax2.vlines(R, ymin_y, ymax_y, color='red', linestyle=':', alpha=0.3)

        # Overlay reward in bottom panel
        if overlay_reward and len(reward_snip_plot) > 0:
            ymin_y, ymax_y = ax2.get_ylim()
            for rew in reward_snip_plot:
                ax2.vlines(rew, ymin_y, ymax_y, color='green', linestyle='-', alpha=0.5)

        # 9) Build a legend. We'll add proxy lines for licks and reward if needed
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles = handles1 + handles2
        all_labels  = labels1 + labels2

        if overlay_licks:
            left_lick_proxy = Line2D([], [], color='blue',  linestyle=':', alpha=0.3, label='Left Lick')
            right_lick_proxy= Line2D([], [], color='red',   linestyle=':', alpha=0.3, label='Right Lick')
            all_handles.extend([left_lick_proxy, right_lick_proxy])
            all_labels.extend(['Left Lick', 'Right Lick'])

        if overlay_reward:
            reward_proxy = Line2D([], [], color='green', linestyle='-', alpha=0.3, label='Reward')
            all_handles.append(reward_proxy)
            all_labels.append('Reward')

        # We'll place the legend on ax2 (bottom panel)
        ax2.legend(all_handles, all_labels, loc='upper right')

        plt.tight_layout()
        plt.show()

    def predict_fluoresence_from_licks(self,
                                    data_name: str = 'G_1_preprocessed-bright',
                                    quiet_lick_event: str = 'quiet_lick',
                                    single_lick_extraction_window: list = [-0.5, 1],
                                    lick_train_dt: float = 0.1,
                                    snippet_length: float = 20.0):
        """
        1) Finds all 'quiet_lick' timestamps (isolated licks with no neighbors in a certain window).
        2) Extracts the single-lick calcium response by averaging fluorescence in [t-2, t+2] around each quiet lick.
        3) Builds a lick train (all licks in the session) at resolution lick_train_dt.
        4) Convolves the lick train with the single-lick kernel to predict the total fluorescence signal.
        5) Compares predicted vs. observed fluorescence over the entire session (Pearson correlation).
        6) Also shows a random snippet (snippet_length seconds) of predicted vs. real data, with lick overlays.

        Parameters:
        - data_name (str): Fluorescence channel to use (e.g., 'G_1_preprocessed-bright').
        - quiet_lick_event (str): Name of the quiet lick event (e.g. 'quiet_lick', 'quiet_left_lick', or 'quiet_right_lick').
        - single_lick_extraction_window (list): e.g. [-2, 2], time around each quiet lick to gather the average response.
        - lick_train_dt (float): Time resolution (s) for building the lick train.
        - snippet_length (float): Length of the random snippet (in seconds) to show in the third panel. Default 20s.

        Returns:
        - A dictionary with:
            'kernel_time': time array for the single-lick kernel
            'single_lick_kernel': the average single-lick response
            't_array': the time array for the predicted signal
            'predicted_signal': predicted fluorescence
            'observed_signal': observed fluorescence
            'r_value', 'p_value': correlation stats from scipy.stats.pearsonr
        """

        # ----------------------------------------------------------------
        # 1) Get quiet lick timestamps
        # ----------------------------------------------------------------
        quiet_licks = self.extract_event_timestamps(quiet_lick_event)
        if len(quiet_licks) == 0:
            print(f"No '{quiet_lick_event}' timestamps found; cannot build single-lick kernel.")
            return None

        # ----------------------------------------------------------------
        # 2) Build single-lick kernel by averaging fluorescence around each quiet lick
        # ----------------------------------------------------------------
        data_values = self.nwb_ophys_data.acquisition[data_name].data[:]
        data_timestamps = self.nwb_ophys_data.acquisition[data_name].timestamps[:]

        t_before = single_lick_extraction_window[0]  # e.g. -2
        t_after  = single_lick_extraction_window[1]  # e.g.  2

        dt_kernel = 0.01  # resolution for kernel time
        kernel_time = np.arange(t_before, t_after, dt_kernel)

        all_segments = []
        for lick_t in quiet_licks:
            t0 = lick_t + t_before
            t1 = lick_t + t_after
            idx = np.where((data_timestamps >= t0) & (data_timestamps < t1))[0]
            if len(idx) == 0:
                continue

            segment_t = data_timestamps[idx] - lick_t  # shift so lick_t is 0
            segment_v = data_values[idx]

            # interpolate onto kernel_time
            seg_on_kernel = np.interp(kernel_time, segment_t, segment_v)
            all_segments.append(seg_on_kernel)

        if len(all_segments) == 0:
            print("No valid single-lick segments found. Cannot build kernel.")
            return None

        all_segments = np.array(all_segments)  # shape: (num_quiet_licks, len(kernel_time))
        single_lick_kernel = np.mean(all_segments, axis=0)

        # Optional: baseline subtract so that at t<0 we average ~0 if desired
        # baseline_idx = (kernel_time < 0)
        # baseline_val = np.mean(single_lick_kernel[baseline_idx])
        # single_lick_kernel -= baseline_val

        # ----------------------------------------------------------------
        # 3) Build lick train (ALL licks, not just quiet) at resolution lick_train_dt
        # ----------------------------------------------------------------
        # We'll combine left + right or you can pass another event_name if you want only left or right
        left_lick_times = self.nwb_ophys_data.acquisition['left_lick_time'].timestamps[:]
        right_lick_times = self.nwb_ophys_data.acquisition['right_lick_time'].timestamps[:]
        all_licks = np.sort(np.concatenate((left_lick_times, right_lick_times)))

        # We'll define a time array from the start to end of the fluorescence data
        global_start = data_timestamps[0]
        global_end   = data_timestamps[-1]
        t_array = np.arange(global_start, global_end, lick_train_dt)
        lick_train = np.zeros_like(t_array)
        # For each lick, find the bin
        for L in all_licks:
            if L >= global_start and L < global_end:
                idx = int((L - global_start) // lick_train_dt)
                lick_train[idx] += 1

        # ----------------------------------------------------------------
        # 4) Convolve lick train with single-lick kernel
        # ----------------------------------------------------------------
        # shift kernel so index 0 corresponds to t=0 in the kernel
        zero_idx = np.argmin(np.abs(kernel_time))
        shift_len = zero_idx
        shifted_kernel = np.roll(single_lick_kernel, -shift_len)
        if shift_len > 0:
            shifted_kernel[:shift_len] = 0

        predicted_signal_full = np.convolve(lick_train, shifted_kernel, mode='full') * lick_train_dt
        predicted_signal = predicted_signal_full[:len(t_array)]

        # ----------------------------------------------------------------
        # 5) Build observed signal at same time base & compare
        # ----------------------------------------------------------------
        observed_signal = np.zeros_like(t_array)
        for i in range(len(t_array)):
            t0 = t_array[i]
            t1 = t0 + lick_train_dt
            idx = np.where((data_timestamps >= t0) & (data_timestamps < t1))[0]
            if len(idx) > 0:
                observed_signal[i] = np.mean(data_values[idx])
            else:
                observed_signal[i] = np.nan

        valid_mask = ~np.isnan(observed_signal)
        observed_signal = observed_signal[valid_mask]
        predicted_signal = predicted_signal[valid_mask]
        time_valid = t_array[valid_mask]

        if len(observed_signal) == 0:
            print("No valid overlapping data for final comparison.")
            return None

        # Optionally find best linear scale factor alpha s.t. alpha * predicted_signal ~ observed_signal
        # slope, intercept = np.polyfit(predicted_signal, observed_signal, 1)
        # scaled_prediction = slope * predicted_signal + intercept
        # For now let's do a simpler approach: correlation of the raw predicted vs observed
        r_value, p_value = pearsonr(predicted_signal, observed_signal)

        # ----------------------------------------------------------------
        # 6) PLOT: 3 panels + LICK OVERLAYS in snippet
        # ----------------------------------------------------------------
        fig = plt.figure(figsize=(10, 10))
        gs = plt.GridSpec(3, 1, height_ratios=[1, 2, 1.5])

        # (A) Single lick kernel
        ax_kernel = fig.add_subplot(gs[0])
        ax_kernel.plot(kernel_time, single_lick_kernel, 'b-')
        ax_kernel.axvline(0, color='k', linestyle='--')
        ax_kernel.set_title(f"Single-Lick Kernel (avg of {len(all_segments)} quiet licks)")
        ax_kernel.set_xlabel("Time from lick (s)")
        ax_kernel.set_ylabel("Fluorescence")

        # (B) Full predicted vs observed
        ax_full = fig.add_subplot(gs[1])
        ax_full.set_title(f"Full Session Prediction vs Observed (r={r_value:.3f}, p={p_value:.1e})")
        obs_line, = ax_full.plot(time_valid, observed_signal, 'k-', label='Observed')
        pred_line, = ax_full.plot(time_valid, predicted_signal, 'r-', alpha=0.7, label='Predicted')
        ax_full.set_xlabel("Time (s)")
        ax_full.set_ylabel("Fluorescence")
        ax_full.legend()

        # (C) Random snippet
        ax_snip = fig.add_subplot(gs[2])
        session_length = time_valid[-1] - time_valid[0]
        if session_length > snippet_length:
            max_snip_start = time_valid[-1] - snippet_length
            random_snip_start = random.uniform(time_valid[0], max_snip_start)
            random_snip_end   = random_snip_start + snippet_length

            snip_mask = (time_valid >= random_snip_start) & (time_valid < random_snip_end)
            # Plot snippet for observed/predicted
            obs_snip_line, = ax_snip.plot(time_valid[snip_mask], observed_signal[snip_mask], 
                                        'k-', label='Observed')
            pred_snip_line, = ax_snip.plot(time_valid[snip_mask], predicted_signal[snip_mask], 
                                        'r-', alpha=0.7, label='Predicted')

            # Overlays: left licks & right licks in [random_snip_start, random_snip_end]
            left_snip = left_lick_times[(left_lick_times >= random_snip_start) & 
                                        (left_lick_times < random_snip_end)]
            right_snip = right_lick_times[(right_lick_times >= random_snip_start) & 
                                        (right_lick_times < random_snip_end)]

            ymin, ymax = ax_snip.get_ylim()

            # draw vertical lines
            for L in left_snip:
                ax_snip.vlines(L, ymin, ymax, color='blue', alpha=0.3, linestyle=':', linewidth=1)
            for R in right_snip:
                ax_snip.vlines(R, ymin, ymax, color='red', alpha=0.3, linestyle=':', linewidth=1)

            # Now define two "proxy" artists for the legend
            left_lick_proxy = Line2D([], [], color='blue', alpha=0.3, linestyle=':', linewidth=1, 
                                    label='Left Lick')
            right_lick_proxy = Line2D([], [], color='red', alpha=0.3, linestyle=':', linewidth=1, 
                                    label='Right Lick')

            ax_snip.set_title(f"Random {snippet_length}s snippet ~ [{random_snip_start:.1f}, {random_snip_end:.1f}]")
            ax_snip.set_xlabel("Time (s)")
            ax_snip.set_ylabel("Fluorescence")

            # Combine handles for the legend
            ax_snip.legend(handles=[obs_snip_line, pred_snip_line, left_lick_proxy, right_lick_proxy])
        else:
            ax_snip.text(0.5, 0.5, "Not enough time range for snippet",
                        ha='center', va='center', transform=ax_snip.transAxes)
            ax_snip.axis('off')

        plt.tight_layout()
        plt.show()

        results = {
            'kernel_time': kernel_time,
            'single_lick_kernel': single_lick_kernel,
            't_array': time_valid,
            'predicted_signal': predicted_signal,
            'observed_signal': observed_signal,
            'r_value': r_value,
            'p_value': p_value,
        }
        return results

    def perform_correlation(self, 
                            time_window: list = [-1, 0],
                            z_score: bool = False,
                            correlation_model: str = 'simple_LR',
                            behavior_model: str = 'QLearning_L2F1_softmax',
                            latent_name: str = 'q_value_difference',
                            data_name: str = 'G_1_preprocessed-bright') -> dict:
        """
        Performs correlation analysis between the photometry signal and a latent variable.
        
        Since OphysBehavior contains photometry data (a single channel), this method:
        - Aligns the photometry signal to go cues (using align_to_event) with the specified time_window.
        - Computes the mean photometry response for each trial.
        - Excludes trials with no animal response (animal_response == 2).
        - Extracts the trial-based latent variable.
        - Uses a regression method (dynamically retrieved from the methods package) to fit a linear model
            between the mean photometry response and the latent variable.
        - Computes Pearson correlation statistics.
        
        Parameters:
        - time_window (list): Time window (in seconds) relative to the event from which to compute the mean photometry signal (default: [-1, 0]).
        - z_score (bool): Whether to z-score the mean photometry responses across trials (default: False).
        - correlation_model (str): Regression model to use (only 'simple_LR' is supported here).
        - behavior_model (str): The behavior model used to extract latent variables.
        - latent_name (str): Name of the latent variable (e.g., 'q_value_difference').
        - data_name (str): The photometry data key (e.g., 'G_1_preprocessed-bright').
        
        Returns:
        - dict: A dictionary containing:
            {'slope': ..., 'intercept': ..., 'r_value': ..., 'p_value': ...}
        
        Raises:
        - ValueError: If there is a mismatch in the number of trials between photometry and latent data.
        """
        
        # Align photometry data to 'go_cue' using the specified time window.
        aligned_matrix = self.align_to_event(data_name=data_name, event_name='go_cue', time_window=time_window)
        # Each row corresponds to a trial; compute the mean photometry response in the specified window per trial.
        trial_mean = np.nanmean(aligned_matrix, axis=1)  # shape: (n_trials,)
        
        # Retrieve the animal responses for each trial and create a valid mask (exclude no-response trials, where animal_response == 2)
        trials = self.nwb_behavior_data.intervals['trials'][:]
        choice_history = trials['animal_response'].to_numpy()
        valid_mask = choice_history != 2

        # Filter trial_mean using the valid mask
        trial_mean = trial_mean[valid_mask]
        
        # Extract the latent variable data (assumed to be trial-based)
        latent_data = np.array(self.extract_fitted_data(model_name=behavior_model, latent_name=latent_name))
        
        if latent_data.shape[0] != trial_mean.shape[0]:
            raise ValueError("Mismatch between number of valid trials in latent data and photometry data.")

        if z_score:
            trial_mean = (trial_mean - np.mean(trial_mean)) / np.std(trial_mean)
        
        # Retrieve the regression method dynamically from the methods package.
        if not hasattr(methods, correlation_model):
            raise ValueError(f"Correlation model '{correlation_model}' is not available in 'methods'.")
        regression_method = getattr(methods, correlation_model)
        
        # Run the regression model using the photometry trial means as the independent variable and latent data as the dependent variable.
        regression_result = regression_method(trial_mean, latent_data, behavior_name=latent_name)
        
        # For a simple linear regression, extract the parameters.
        if correlation_model == 'simple_LR':
            results = {
                'slope': regression_result.params[latent_name],
                'intercept': regression_result.params['const'],
                'r_value': regression_result.rsquared_adj,
                'p_value': regression_result.pvalues[latent_name],
                'behavior_name': latent_name
            }
            print(f"Correlation results: slope={results['slope']:.3f}, intercept={results['intercept']:.3f}, "
            f"r_value={results['r_value']:.3f}, p_value={results['p_value']:.1e}")
        elif correlation_model in ['ARMA_model','ARDL_model']:
            results = regression_result  
            print(f"Correlation summary: {results.summary()}")
        elif correlation_model in ['cyclic_shift','linear_shift','phase_randomization']:
            results = regression_result
            print(f"Significance percentile: {results}")

        return results

    def extract_aligned_matrix_and_latent_data(self,
                                           time_window: list = [-3, 4], 
                                           bin_size: float = 0.05,
                                           behavior_model: str or list = 'QLearning_L2F1_softmax', 
                                           latent_name: str or list = 'q_value_difference', 
                                           data_name: str or list = 'G_1_preprocessed-bright',
                                           session_name_list: Optional[list] = None):
        """
        Extracts and combines aligned photometry data and latent variables across one or more sessions,
        while excluding no-response trials.

        For each session:
        1. For each provided data_name, the photometry data is aligned to the 'go_cue' event and 
            trials with no response (animal_response == 2) are excluded.
        2. For each combination of behavior_model and latent_name, the latent variable is extracted.
        3. If the number of valid trials in the latent extraction does not match that in the aligned data 
            (using the first data_name for comparison), that session’s latent data for that combination is skipped.
        
        When multiple session names are provided, each session is processed one by one.
        The aligned matrices from all sessions are concatenated (keyed by data_name),
        and the latent data for each (behavior_model, latent_name) pair are concatenated into a nested dictionary.

        Parameters:
        - time_window (list): Time window (in seconds) relative to the event for alignment.
        - bin_size (float): Bin size (in seconds) for binning the photometry signal.
        - behavior_model (str or list): Behavior model(s) to use for latent extraction.
        - latent_name (str or list): Latent variable name(s) to extract.
        - data_name (str or list): Photometry data key(s) to use for alignment.
        - session_name_list (Optional[list]): List of session names to process. If None or empty,
                                                the current session is used.

        Returns:
        dict: {
            "combined_aligned_matrix": { data_name: np.ndarray, ... },
            "combined_latent_data": { behavior_model: { latent_name: np.ndarray, ... }, ... },
            "params": { ... }
        }
        """

        # Convert parameters to lists if needed.
        if not isinstance(data_name, list):
            data_names = [data_name]
        else:
            data_names = data_name

        if not isinstance(behavior_model, list):
            behavior_models = [behavior_model]
        else:
            behavior_models = behavior_model

        if not isinstance(latent_name, list):
            latent_names = [latent_name]
        else:
            latent_names = latent_name

        # Initialize dictionaries to accumulate results.
        combined_aligned = {dn: [] for dn in data_names}
        combined_latent = {bm: {ln: [] for ln in latent_names} for bm in behavior_models}

        # Helper function to extract aligned matrix from a session.
        def extract_aligned_matrix_from_session(instance, dn, time_window, bin_size):
            aligned_matrix = instance.align_to_event(data_name=dn, event_name='go_cue',
                                                    time_window=time_window, bin_size=bin_size)
            trials = instance.nwb_behavior_data.intervals['trials'][:]
            choice_history = trials['animal_response'].to_numpy()
            valid_mask = choice_history != 2  # Exclude trials with no response.
            return aligned_matrix[valid_mask, :]

        # Helper function to extract latent data from a session.
        def extract_latent_data_from_session(instance, bm, ln):
            latent_data = np.array(instance.extract_fitted_data(model_name=bm, latent_name=ln))
            # Ensure latent_data is at least 1D.
            return np.atleast_1d(latent_data)

        # Decide how to iterate over sessions.
        if session_name_list is None or len(session_name_list) == 0:
            sessions_iter = [None]  # 'None' indicates the current instance.
        else:
            sessions_iter = session_name_list

        # Process sessions one by one.
        for sess in sessions_iter:
            if sess is None:
                instance = self
            else:
                instance = type(self)(sess, folder_path=self.folder_path)
            if behavior_model == 'q_learning_Y1':
                instance.fit_q_learning_model()
            # Extract and store aligned data for each data_name.
            for dn in data_names:
                aligned = extract_aligned_matrix_from_session(instance, dn, time_window, bin_size)
                combined_aligned[dn].append(aligned)
            # For latent extraction, use the aligned matrix from the first data_name as a trial count check.
            for bm, ln in itertools.product(behavior_models, latent_names):
                latent = extract_latent_data_from_session(instance, bm, ln)
                if latent.shape[0] != aligned.shape[0]:
                    print(f"Skipping session '{getattr(instance, 'session_name', 'current')}' for latent combination (behavior_model={bm}, latent_name={ln}) due to trial count mismatch.")
                    continue
                combined_latent[bm][ln].append(latent)

        # Concatenate arrays across sessions.
        for dn in data_names:
            if len(combined_aligned[dn]) > 0:
                combined_aligned[dn] = np.concatenate(combined_aligned[dn], axis=0)
            else:
                combined_aligned[dn] = None

        for bm in behavior_models:
            for ln in latent_names:
                if len(combined_latent[bm][ln]) > 0:
                    combined_latent[bm][ln] = np.concatenate(combined_latent[bm][ln], axis=0)
                else:
                    combined_latent[bm][ln] = None

        params_used = {
            "time_window": time_window,
            "bin_size": bin_size,
            "behavior_model": behavior_models,
            "latent_name": latent_names,
            "data_name": data_names,
            "session_name_list": session_name_list,
            "folder_path": self.folder_path
        }

        self.combined_aligned_matrix = combined_aligned
        self.combined_latent_data = combined_latent
        self.combined_params = params_used

        return {
            "combined_aligned_matrix": combined_aligned,
            "combined_latent_data": combined_latent,
            "params": params_used
        }


    def plot_quantiles(self, 
                        quantiles_num: int = 5, 
                        time_window: list = [-3, 4], 
                        bin_size: float = 0.05,
                        z_score: bool = True,
                        smooth: Optional[float] = None,
                        behavior_model: Optional[str] = 'QLearning_L2F1_softmax', 
                        latent_name: Optional[str] = 'q_value_difference', 
                        data_name: Optional[str] = 'G_1_preprocessed-bright',
                        title_font_size: int = 16,
                        label_font_size: int = 12,
                        colorbar_cmap: str = 'plasma',
                        baseline_window: Optional[list] = None,
                        aligned_matrix: Optional[np.ndarray] = None,
                        latent_data: Optional[np.ndarray] = None,
                        min_trials_each_quantil: int = 5  # New parameter to filter quantiles by minimum trials
                        ) -> None:
        """
        Plots the photometry signal (PSTH) grouped by quantiles of a latent variable.
        
        This method aligns the photometry signal (using align_to_event) to the go cue,
        computes the mean photometry response per trial, and then groups trials into quantiles
        based on the latent variable values. For each quantile, it computes the average PSTH and its
        standard error (SEM). Optionally, Gaussian smoothing is applied.
        
        Trials with no response (animal_response == 2) are excluded.
        
        Parameters:
        - quantiles_num (int): 
            The number of quantiles to divide the latent data into (default: 5).
            Each quantile corresponds to a subset of trials, and this parameter determines how many groups 
            the trials are split into based on the latent variable values.
            
        - time_window (list): 
            Time window (in seconds) relative to the go cue for extracting and visualizing the photometry signal 
            (default: [-3, 4]).
            Specifies the time range (from go cue) over which the photometry signal will be analyzed.

        - bin_size (float): 
            The size of each time bin (in seconds) for binning the photometry signal (default: 0.05).
            Determines the resolution of the time bins used for the PSTH.

        - z_score (bool): 
            Whether to z-score the photometry responses (default: True).
            If True, the photometry signal is normalized across all trials, ensuring zero mean and unit variance.

        - smooth (Optional[float]): 
            The standard deviation for Gaussian smoothing in seconds (default: None).
            If provided, this parameter applies Gaussian smoothing to the PSTH data to reduce noise.

        - behavior_model (Optional[str]): 
            The behavior model used to extract latent variables (default: 'QLearning_L2F1_softmax').
            This model is assumed to generate trial-based latent variable data, which is used for grouping trials.

        - latent_name (Optional[str]): 
            The latent variable name, such as 'q_value_difference' (default: 'q_value_difference').
            Specifies which latent variable will be used to group trials into quantiles.

        - data_name (Optional[str]): 
            The photometry data key, such as 'G_1_preprocessed-bright' (default: 'G_1_preprocessed-bright').
            Specifies the dataset containing photometry data to be aligned and analyzed.

        - title_font_size (int): 
            Font size for the title of the plot (default: 16).
            Controls the size of the plot title text.

        - label_font_size (int): 
            Font size for the axis labels (default: 12).
            Controls the size of the axis label text.

        - colorbar_cmap (str): 
            The colormap to use for the colorbar and PSTH curves (default: 'plasma').
            Specifies the color scheme used for visualizing the latent variable range in the plot.

        - baseline_window (Optional[list]): 
            Time window for baseline activity (default: None).
            If provided, the photometry signal will be normalized by subtracting and dividing by the baseline mean 
            in this window.

        - aligned_matrix (Optional[np.ndarray]): 
            Pre-computed aligned photometry data (default: None).
            If provided, this data is used instead of re-aligning the photometry data to the event.

        - latent_data (Optional[np.ndarray]): 
            Pre-computed latent variable data (default: None).
            If provided, this data is used for grouping trials into quantiles instead of calculating it from the behavior model.

        - min_trials_each_quantil (int): 
            Minimum number of trials required in each quantile to be included in the plot (default: 5).
            If a quantile contains fewer than this number of trials, it is excluded from the analysis and plot.
        """
        
        # 1. Align photometry data using align_to_event.
        if aligned_matrix is None:
            aligned_matrix = self.align_to_event(data_name=data_name, event_name='go_cue', time_window=time_window, bin_size=bin_size)
            
            # 2. Exclude no-response trials using animal responses from behavior data.
            trials = self.nwb_behavior_data.intervals['trials'][:]
            choice_history = trials['animal_response'].to_numpy()
            valid_mask = choice_history != 2  # Exclude trials with no response
            aligned_matrix = aligned_matrix[valid_mask, :]
        
        if latent_data is None:
            # 3. Get latent variable values (assumed trial-based)
            latent_data = np.array(self.extract_fitted_data(model_name=behavior_model, latent_name=latent_name))
        
        # Check that the number of trials in latent data matches the number of valid trials in photometry data
        if latent_data.shape[0] != aligned_matrix.shape[0]:
            raise ValueError("Mismatch between number of valid trials in latent data and photometry data.")
        
        # 4. Optionally z-score the entire aligned matrix.
        if z_score:
            aligned_matrix = (aligned_matrix - np.nanmean(aligned_matrix)) / np.nanstd(aligned_matrix)
        
        # 5. Determine quantile edges (evenly spaced between min and max of latent_data)
        quantile_edges = np.linspace(np.min(latent_data), np.max(latent_data), quantiles_num + 1)
        quantile_labels = [f'Q{i+1}' for i in range(quantiles_num)]
        
        # 6. Group trial indices by quantile
        groups = {label: [] for label in quantile_labels}
        for i, val in enumerate(latent_data):
            for j in range(quantiles_num):
                if quantile_edges[j] <= val <= quantile_edges[j+1]:
                    groups[quantile_labels[j]].append(i)
                    break
        
        # 7. Remove quantiles with fewer than min_trials_each_quantil trials
        groups = {label: indices for label, indices in groups.items() if len(indices) >= min_trials_each_quantil}
        
        # Update quantile_labels and quantile_edges after filtering
        quantile_labels = [label for label in quantile_labels if label in groups]
        quantile_edges = np.linspace(np.min(latent_data), np.max(latent_data), len(groups) + 1)
        
        # 8. Generate x-axis time bins using the same binning as in align_to_event.
        num_bins = aligned_matrix.shape[1]  # Number of time bins (based on photometry signal binning)
        time_bins = np.linspace(time_window[0], time_window[1], num_bins)  # Create time bins for x-axis
        
        # 9. For each quantile, compute average PSTH and SEM, then normalize by baseline if needed.
        quantile_psth = {}
        quantile_sem = {}
        for label, trial_indices in groups.items():
            if len(trial_indices) == 0:
                quantile_psth[label] = np.zeros(aligned_matrix.shape[1])  # If no trials, set as zeros
                quantile_sem[label] = np.zeros(aligned_matrix.shape[1])  # Same for SEM
            else:
                # Get the data for the trials in the current quantile
                group_data = aligned_matrix[trial_indices, :]  # shape: (n_trials, bins)
                quantile_psth[label] = np.nanmean(group_data, axis=0)  # Compute average PSTH
                quantile_sem[label] = np.nanstd(group_data, axis=0, ddof=1) / np.sqrt(len(trial_indices))  # Compute SEM
                
                # Normalize by baseline window (f - f0) / f0, if baseline_window is provided
                if baseline_window is not None:
                    # Mask for the baseline window based on time_bins
                    baseline_mask = (time_bins >= baseline_window[0]) & (time_bins <= baseline_window[1])
                    baseline_mean = np.nanmean(quantile_psth[label][baseline_mask])  # Compute mean baseline activity
                    quantile_psth[label] = (quantile_psth[label] - baseline_mean) / baseline_mean  # Normalize by baseline mean
                    quantile_sem[label] = quantile_sem[label] / baseline_mean  # Scale SEM by baseline mean
                
                # Apply Gaussian smoothing if requested
                if smooth is not None:
                    sigma_bins = smooth / bin_size  # Convert smoothing time to number of bins
                    quantile_psth[label] = gaussian_filter1d(quantile_psth[label], sigma=sigma_bins)  # Smooth PSTH
                    quantile_sem[label] = gaussian_filter1d(quantile_sem[label], sigma=sigma_bins)  # Smooth SEM
        
        # 10. Create a colormap for the latent data range.
        cmap_obj = cm.get_cmap(colorbar_cmap)
        norm_obj = mcolors.Normalize(vmin=np.min(latent_data), vmax=np.max(latent_data))
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj)  # Create ScalarMappable object for colorbar
        sm.set_array(latent_data)  # Set the data for the colorbar
        
        # 11. Use the same colormap for PSTH curves.
        colors = cmap_obj(np.linspace(0, 1, len(groups)))  # Generate a color for each quantile
        
        # 12. Plot PSTH for each quantile.
        fig, ax = plt.subplots(figsize=(12, 8))  # Create the plot
        for i, label in enumerate(quantile_labels):
            # Add legend label with quantile range and number of trials
            legend_label = f"{label} ({quantile_edges[i]:.2f}-{quantile_edges[i+1]:.2f}, n={len(groups[label])})"
            ax.plot(time_bins, quantile_psth[label], label=legend_label, color=colors[i])  # Plot the PSTH
            ax.fill_between(time_bins, quantile_psth[label] - quantile_sem[label], 
                            quantile_psth[label] + quantile_sem[label], color=colors[i], alpha=0.3)  # Add SEM shading
        
        # Highlight the "Go Cue" time point with a red vertical line
        ax.axvline(0, color='red', linestyle='--', label='Go Cue')
        
        # Set axis labels and title with specified font sizes
        ax.set_xlabel("Time from Go Cue (s)", fontsize=label_font_size)
        ax.set_ylabel("Fluorescence Intensity (dF/F)" if not z_score else "Z-scored Fluorescence", fontsize=label_font_size)
        ax.set_title(f"Combined PSTH Grouped by {latent_name} Quantiles", fontsize=title_font_size)
        
        # Add legend to the plot
        ax.legend()
        
        # Add a colorbar for the latent data range
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(f'Latent Variable: {latent_name}')
        cbar.set_ticks(quantile_edges)
        cbar.set_ticklabels([f'{q:.2f}' for q in quantile_edges])  # Display quantile edges in colorbar
        
        # Calculate midpoints for major ticks in the colorbar
        quantile_midpoints = [(quantile_edges[i] + quantile_edges[i+1]) / 2 for i in range(len(quantile_edges)-1)]
        cbar.ax.yaxis.set_ticks(quantile_midpoints, minor=False)
        cbar.ax.set_yticklabels([f'{mid:.2f}' for mid in quantile_midpoints])  # Set colorbar tick labels
        
        # Ensure tight layout to avoid overlapping text
        fig.tight_layout()
        plt.show()



    def plot_latent_firing_rate(self, 
                                num_bins: int = 20,
                                bin_range: Optional[list] = None,
                                time_window: list = [-1, 0],
                                smooth: Optional[float] = None,
                                z_score: bool = False,
                                behavior_model: Optional[str] = 'QLearning_L2F1_softmax',
                                latent_name: Optional[str] = 'q_value_difference',
                                data_name: Optional[str] = 'G_1_preprocessed-bright',
                                title_font_size: int = 16,
                                label_font_size: int = 12,
                                baseline_window: Optional[list] = None,
                                aligned_matrix: Optional[np.ndarray] = None,
                                latent_data: Optional[np.ndarray] = None,
                                aligned_matrix_timestampes: Optional[np.ndarray] = None,
                                bin_size: float = 0.01,
                                x_range: Optional[list] = None,
                                y_range: Optional[list] = None,
                            ) -> None:
        """
        Plots the photometry response grouped by bins of a latent variable.
        
        This function:
        1. Optionally aligns the photometry signal to go cues using `align_to_event`,
            restricted to the specified time_window.
        2. Excludes trials with no response (animal_response == 2).
        3. Extracts (or uses pre-computed) trial-based latent variable values.
        4. (Optionally) z-scores the entire aligned data across valid trials.
        5. (Optionally) applies baseline normalization per trial if baseline_window is provided:
            (f - f0) / f0.
        6. Bins the latent variable (using num_bins or a specified bin_range) and, for each bin,
            computes the average photometry response (over time_window) and its standard error (SEM).
        7. Plots the binned photometry responses and SEM.
        
        Parameters:
        - num_bins (int): Number of bins for the latent variable (default: 20).
        - bin_range (list or None): [min, max] for the latent variable. If None, range is auto-detected.
        - time_window (list): Time window (in seconds) relative to the go cue (default: [-1, 0]).
        - smooth (float or None): Standard deviation (in seconds) for Gaussian smoothing of the binned response (default: None).
        - z_score (bool): If True, z-score the entire aligned photometry data matrix across valid trials.
        - behavior_model (str): The behavior model name to extract latents from.
        - latent_name (str): The key for the latent variable (e.g., 'q_value_difference').
        - data_name (str): The key for the photometry data (default: 'G_1_preprocessed-bright').
        - title_font_size (int): Font size for the plot title.
        - label_font_size (int): Font size for the axis labels.
        - baseline_window (list or None): [start, end] time window for baseline normalization. 
        If None, no baseline normalization is done.
        - aligned_matrix (np.ndarray or None): Optionally provide pre-aligned data of shape (n_trials, n_timepoints).
        - latent_data (np.ndarray or None): Optionally provide pre-extracted latent data (n_trials, ).
        - aligned_matrix_timestampes (np.ndarray or None): Time stamps for the columns of `aligned_matrix`.
        - bin_size (float): Bin size (in seconds) if `align_to_event` needs it, default=0.01.
        
        Returns:
        - None: Displays the plot with bin centers on the x-axis and mean photometry response on the y-axis.
        """
        # -----------------------------
        # 1. Load/align photometry data
        # -----------------------------
        if aligned_matrix is None:
            # Align photometry data around 'go_cue' for the given time_window
            aligned_matrix = self.align_to_event(
                data_name=data_name,
                event_name='go_cue',
                time_window=time_window,
                bin_size=bin_size
            )
            # Exclude no-response trials
            trials = self.nwb_behavior_data.intervals['trials'][:]
            choice_history = trials['animal_response'].to_numpy()
            valid_mask = (choice_history != 2)
            aligned_matrix = aligned_matrix[valid_mask, :]

        # -----------------------------
        # 2. Load latent data
        # -----------------------------
        if latent_data is None:
            latent_data = np.array(self.extract_fitted_data(
                model_name=behavior_model,
                latent_name=latent_name
            ))

        # Check shape consistency
        if latent_data.shape[0] != aligned_matrix.shape[0]:
            raise ValueError("Mismatch between number of valid trials in latent_data and photometry data.")

        # ------------------------------------------
        # 3. (Optional) Z-score across all valid data
        # ------------------------------------------
        if z_score:
            aligned_matrix = (
                aligned_matrix - np.nanmean(aligned_matrix)
            ) / np.nanstd(aligned_matrix)

        # ------------------------------------------------------------
        # 4. Identify time indices and handle baseline normalization
        # ------------------------------------------------------------
        # If the user provided timestamps, figure out which columns fall in time_window.
        if aligned_matrix_timestampes is not None:
            # Indices for the main time window
            time_idx = np.where(
                (aligned_matrix_timestampes >= time_window[0]) &
                (aligned_matrix_timestampes <= time_window[1])
            )[0]

            if len(time_idx) == 0:
                raise ValueError("No time bins found within the specified time_window.")

            # If baseline_window is provided, get those indices separately
            if baseline_window is not None:
                baseline_idx = np.where(
                    (aligned_matrix_timestampes >= baseline_window[0]) &
                    (aligned_matrix_timestampes <= baseline_window[1])
                )[0]
            else:
                baseline_idx = None
        else:
            # If no timestamps are provided, assume the columns linearly span time_window
            n_timepoints = aligned_matrix.shape[1]
            all_time = np.linspace(time_window[0], time_window[1], n_timepoints)
            time_idx = np.where((all_time >= time_window[0]) & (all_time <= time_window[1]))[0]
            if baseline_window is not None:
                baseline_idx = np.where(
                    (all_time >= baseline_window[0]) & 
                    (all_time <= baseline_window[1])
                )[0]
            else:
                baseline_idx = None

        # For baseline normalization, we do it trial by trial.
        if baseline_window is not None and baseline_idx.size > 0:
            for t in range(aligned_matrix.shape[0]):
                f0 = np.nanmean(aligned_matrix[t, baseline_idx])
                aligned_matrix[t, :] = (aligned_matrix[t, :] - f0) / f0

        # Finally, slice the matrix to time_window columns
        aligned_matrix = aligned_matrix[:, time_idx]

        # --------------------------------------------------------------------
        # 5. Bin the latent variable and compute mean ± SEM for each bin
        # --------------------------------------------------------------------
        # Determine bin edges and centers
        if bin_range is None:
            bin_range = [np.min(latent_data), np.max(latent_data)]
        bin_edges = np.linspace(bin_range[0], bin_range[1], num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        binned_response = np.zeros(num_bins)
        binned_sem = np.zeros(num_bins)

        # Loop over bins
        for i in range(num_bins):
            in_bin = (latent_data >= bin_edges[i]) & (latent_data < bin_edges[i+1])
            count_in_bin = np.sum(in_bin)
            if count_in_bin > 0:
                # Mean over time within each trial
                trial_means = np.nanmean(aligned_matrix[in_bin, :], axis=1)
                binned_response[i] = np.mean(trial_means)
                binned_sem[i] = (np.std(trial_means, ddof=1) /
                                np.sqrt(count_in_bin))
            else:
                binned_response[i] = np.nan
                binned_sem[i] = np.nan

        # --------------------------------
        # 6. (Optional) Gaussian smoothing
        # --------------------------------
        if smooth is not None:
            # Convert 'smooth' (seconds) to # of bins in latent space
            latent_bin_width = (bin_range[1] - bin_range[0]) / num_bins
            sigma_bins = smooth / latent_bin_width
            binned_response = gaussian_filter1d(binned_response, sigma=sigma_bins)
            binned_sem = gaussian_filter1d(binned_sem, sigma=sigma_bins)

        # -------------------
        # 7. Plot the results
        # -------------------
        plt.figure(figsize=(8, 6))
        plt.plot(bin_centers, binned_response,
                label=f'Binned Response (n={aligned_matrix.shape[0]})')
        plt.fill_between(bin_centers,
                        binned_response - binned_sem,
                        binned_response + binned_sem,
                        alpha=0.5,
                        label='SEM')
        
        plt.xlabel(f"Latent Variable ({latent_name})", fontsize=label_font_size)
        plt.ylabel("Mean Photometry Response", fontsize=label_font_size)
        plt.title(f"Photometry Response vs. {latent_name}",
                fontsize=title_font_size)
        plt.legend(loc='upper right')

        if x_range is not None:
            plt.xlim(x_range)
        if y_range is not None:
            plt.ylim(y_range)
        plt.tight_layout()
        plt.show()






