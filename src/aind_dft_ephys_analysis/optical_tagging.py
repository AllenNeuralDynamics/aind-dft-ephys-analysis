import os
import warnings
import random
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches  # For custom legend patches
from matplotlib import gridspec
from matplotlib.colors import ListedColormap


from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from hdmf_zarr import NWBZarrIO
from scipy.stats import ttest_rel
from ephys_utils import find_best_electrode, cluster_estimated_x, load_ccf_channel_locations, extract_channel_info


def bh_correct(pvals):
    """Benjamini-Hochberg FDR correction. NaNs are treated as p=1.0."""
    p = np.asarray(pvals, dtype=float)
    p = np.where(np.isnan(p), 1.0, p)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p-value downward.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def select_tagged_units(metrics_df,
                        alpha=0.05,
                        effect_ratio=2.0,
                        min_abs_increase=0.0,
                        min_reliability=0.2,
                        max_latency=0.006,
                        max_jitter=0.003,
                        correction="fdr_bh",
                        return_tagged_only=False):
    """
    SELECTION step of robust opto-tagging: apply thresholds to precomputed
    metrics (from :meth:`OpticalTagging.compute_tagging_metrics`) and add a
    ``tagged`` flag.

    This is a standalone function (no ``OpticalTagging`` instance required), so
    it can be run on a metrics table combined across many sessions -- e.g. read
    several per-session CSVs, ``pd.concat`` them, and select in one call. When a
    ``session`` column is present, multiple-comparison correction is applied
    within each (``session``, ``pulse_index``) family; otherwise within each
    ``pulse_index`` family. A row is flagged ``tagged=True`` only if ALL hold:
      - t-test significant after correction (``p_ttest_corr < alpha``)
      - SALT significant after correction (``p_salt_corr < alpha``)
      - ``stim_FR_mean > baseline_FR_mean``
      - ``effect_ratio`` met AND absolute increase >= ``min_abs_increase``
      - ``reliability >= min_reliability``
      - ``median_latency <= max_latency``
      - ``jitter <= max_jitter``

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics table produced by ``compute_tagging_metrics`` (one session or
        several concatenated).
    alpha, effect_ratio, min_abs_increase, min_reliability, max_latency, max_jitter
        Selection thresholds.
    correction : {"fdr_bh", "none"}
        Multiple-comparison correction for the p-values.
    return_tagged_only : bool
        If True, return only rows where ``tagged`` is True.

    Returns
    -------
    pd.DataFrame
        The metrics table with ``p_ttest_corr``, ``p_salt_corr`` and ``tagged``
        columns added.
    """
    if metrics_df is None or len(metrics_df) == 0:
        print("Error: no metrics to select from.")
        return pd.DataFrame()

    df = metrics_df.copy()

    # Families over which p-values are corrected: keep sessions independent and
    # each pulse position independent (they are separate test families).
    family_cols = [c for c in ("session", "pulse_index") if c in df.columns]

    if correction in ("fdr_bh", "bh"):
        if family_cols:
            df["p_ttest_corr"] = df.groupby(family_cols)["p_ttest"].transform(
                lambda s: pd.Series(bh_correct(s.values), index=s.index))
            df["p_salt_corr"] = df.groupby(family_cols)["p_salt"].transform(
                lambda s: pd.Series(bh_correct(s.values), index=s.index))
        else:
            df["p_ttest_corr"] = bh_correct(df["p_ttest"].values)
            df["p_salt_corr"] = bh_correct(df["p_salt"].values)
    elif correction in (None, "none"):
        df["p_ttest_corr"] = df["p_ttest"]
        df["p_salt_corr"] = df["p_salt"]
    else:
        raise ValueError(f"Unknown correction: {correction!r}")

    effect_ok = (df["effect_ratio"] >= effect_ratio) & \
                (df["abs_increase"] >= min_abs_increase)
    latency_ok = df["median_latency"].le(max_latency)   # NaN -> False
    jitter_ok = df["jitter"].le(max_jitter)             # NaN -> False

    df["tagged"] = (
        (df["p_ttest_corr"] < alpha) &
        (df["p_salt_corr"] < alpha) &
        (df["stim_FR_mean"] > df["baseline_FR_mean"]) &
        effect_ok &
        (df["reliability"] >= min_reliability) &
        latency_ok &
        jitter_ok
    )

    if return_tagged_only:
        return df[df["tagged"]].reset_index(drop=True)
    return df


class OpticalTagging:
    def __init__(self, behavior_json_file, ephys_nwb_file):
        """
        Initializes the OpticalTagging instance for reading ephys and behavior NWB data.
        
        Parameters
        ----------
        behavior_json_file : str
            Path to the JSON file that contains behavior and optical tagging parameters.
        ephys_nwb_file : str
            Path to the NWB file that contains electrophysiology data.
        
        The constructor calls methods to load both NWB and behavior data and then filters
        out units that do not pass quality control.
        """
        self.behavior_json_file = behavior_json_file
        self.ephys_nwb_file = ephys_nwb_file
        self.nwb_ephys_data = None
        self.nwb_behavior_data = None
        self.read_ephys_nwb()
        self.read_behavior_data()
        self.get_units_passed_default_qc()
        self.append_units_locations()
        
    def append_units_locations(self):
        """
        For each unit:
          1) find best electrode by trough-to-peak,
          2) assign shank via clustering estimated_x,
          3) load each probe‐shank CCF JSON only once,
          4) extract CCF coords & region for best electrode,
          5) append a single 'ccf_location' column of dicts containing:
             best_electrode, shank, x, y, z, axial, lateral, brain_region_id, brain_region.
        """
        # safety check
        if self.nwb_ephys_data is None or not hasattr(self.nwb_ephys_data, 'units'):
            return

        units = self.nwb_ephys_data.units
        wm = units['waveform_mean'].data[:]      # (n_units, timepoints, electrodes)
        ex = units['estimated_x'].data[:]        # (n_units,)

        if wm.size == 0 or ex.size == 0:
            return

        # 1) best electrode per unit
        best_ch = find_best_electrode(wm, unit_index=None)    # shape (n_units,)

        # 2) shank assignment per unit
        shank_ids = cluster_estimated_x(ex, n_clusters=4, threshold=0.5, plot=False)

        device_names = units['device_name'][:]   # e.g. ['ProbeA','ProbeA',…]
        n_units = best_ch.shape[0]

        # 3) cache each probe–shank CCF JSON
        unique_keys = set(zip(device_names, shank_ids))
        ccf_cache: Dict[Tuple[str,int], Dict[str,Any]] = {}
        self.session_name=self.nwb_ephys_data.session_id
        for probe, shank in unique_keys:
            ccf_cache[(probe, int(shank))] = load_ccf_channel_locations(
                self.session_name, probe, int(shank)+1
            )

        # 4) gather CCF arrays
        x_arr       = np.full(n_units, np.nan)
        y_arr       = np.full(n_units, np.nan)
        z_arr       = np.full(n_units, np.nan)
        axial_arr   = np.full(n_units, np.nan)
        lateral_arr = np.full(n_units, np.nan)
        region_id   = np.full(n_units, np.nan)
        region_name = np.array(['']*n_units, dtype=object)

        for i in range(n_units):
            probe = device_names[i]
            shank = int(shank_ids[i])
            ccf   = ccf_cache.get((probe, shank), {})
            chan  = extract_channel_info(ccf, int(best_ch[i]))
            if chan:
                x_arr[i]       = chan.get('x', np.nan)
                y_arr[i]       = chan.get('y', np.nan)
                z_arr[i]       = chan.get('z', np.nan)
                axial_arr[i]   = chan.get('axial', np.nan)
                lateral_arr[i] = chan.get('lateral', np.nan)
                region_id[i]   = chan.get('brain_region_id', np.nan)
                region_name[i] = chan.get('brain_region', '')

        # 5) pack everything into one dict per unit
        ccf_location = []
        for i in range(n_units):
            entry = {
                'best_electrode': int(best_ch[i]),
                'shank':          int(shank_ids[i]),
                'probe':           str(device_names[i])
            }
            # only add CCF fields if available
            if np.isfinite(x_arr[i]):
                entry.update({
                    'x':               float(x_arr[i]),
                    'y':               float(y_arr[i]),
                    'z':               float(z_arr[i]),
                    'axial':          float(axial_arr[i]),
                    'lateral':        float(lateral_arr[i]),
                    'brain_region_id': int(region_id[i]),
                    'brain_region':    region_name[i]
                })
            ccf_location.append(entry)

        # 6) append single column
        units.add_column(
            name='ccf_location',
            description=(
                "Per-unit dict with keys: "
                "best_electrode, shank, x, y, z, axial, lateral, brain_region_id, brain_region"
            ),
            data=ccf_location
        )

    def read_behavior_data(self):
        """
        Reads the behavior JSON file and stores its contents.
        
        The behavior JSON is expected to contain optical tagging parameters and,
        optionally, a sub-dictionary of task parameters.
        """
        try:
            with open(self.behavior_json_file, 'r') as f:
                self.behavior_data = json.load(f)
            print(f"Behavior data successfully loaded from: {self.behavior_json_file}")
        except FileNotFoundError:
            print(f"Error: Behavior JSON file not found at: {self.behavior_json_file}")
            self.behavior_data = None
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON file: {self.behavior_json_file}")
            self.behavior_data = None

    def read_ephys_nwb(self):
        """
        Reads an ephys NWB file and stores its data in the instance variable.
        
        The NWB file should adhere to the NWB standard; the data is read using NWBZarrIO.
        """
        nwb_file_path = self.ephys_nwb_file
        if not nwb_file_path or not os.path.exists(nwb_file_path):
            print(f"Error: Ephys NWB file not found at: {nwb_file_path}")
            self.nwb_ephys_data = None
            return

        io = NWBZarrIO(nwb_file_path, "r")
        self.nwb_ephys_data = io.read()

    def get_units_passed_default_qc(self):
        """
        Filters and retrieves the units that pass default quality control (QC).
        
        Filtering criteria:
          - 'default_qc' must be True (as string "True" or boolean True).
          - 'decoder_label' must not be 'noise'.
        
        The filtered units are stored in self.units_passing_qc.
        """
        if self.nwb_ephys_data is None:
            print("Error: No ephys data loaded. Cannot perform QC filtering.")
            self.units_passing_qc = None
            return
        
        units = self.nwb_ephys_data.units[:]
        units_passing_qc = units[
            ((units.default_qc == "True") | (units.default_qc == True)) &
            (units.decoder_label != 'noise')
        ]
        self.units_passing_qc = units_passing_qc
        print(f"Number of units passing the default QC: {len(units_passing_qc)}")

    def get_optical_tagging_par(self):
        """
        Retrieves optical tagging parameters from the behavior JSON.
        
        Returns
        -------
        dict or None
            Dictionary of optical tagging parameters if available; otherwise, None.
        
        Checks for either a key 'optical_tagging_par' (which may include task parameters)
        or a top-level key 'laser_start_timestamp'.
        """
        if not hasattr(self, "behavior_data") or not self.behavior_data:
            print("Error: No behavior data loaded. Cannot get optical_tagging_par.")
            self.optical_tagging_par = None
            return None

        if 'optical_tagging_par' in self.behavior_data:
            self.optical_tagging_par = self.behavior_data['optical_tagging_par']
            if 'OpticalTagging_dialog' in self.behavior_data:
                self.optical_tagging_par['task_parameters'] = self.behavior_data['OpticalTagging_dialog']
            if 'task_parameters' in self.optical_tagging_par:
                lower_case_task_parameters = {
                    key.lower(): value 
                    for key, value in self.optical_tagging_par['task_parameters'].items()
                }
                self.optical_tagging_par['task_parameters'] = lower_case_task_parameters
            return self.optical_tagging_par

        elif 'laser_start_timestamp' in self.behavior_data:
            self.optical_tagging_par = self.behavior_data
            return self.optical_tagging_par

        else:
            print("Error: Neither 'optical_tagging_par' nor 'laser_start_timestamp' found in behavior JSON.")
            self.optical_tagging_par = None
            return None

    @staticmethod
    def _fill_missing(arr, placeholder):
        """
        Replaces None values in a NumPy array with a specified placeholder.
        
        Parameters
        ----------
        arr : np.array
            Input array that may contain None.
        placeholder : any
            Value to replace None entries.
        
        Returns
        -------
        np.array
            Array with None values replaced.
        """
        mask = np.array([x is None for x in arr], dtype=bool)
        arr[mask] = placeholder
        return arr

    def _build_pulse_arrays(self):
        """
        Builds arrays of pulse start times and associated condition mapping arrays.
        
        For each trial (identified by each laser start time), the function extracts trial-specific
        stimulation parameters:
            - duration_each_cycle (cycle duration),
            - frequency,
            - pulse_duration (converted from ms to seconds).
        
        It generates pulse start times using:
            np.arange(start_time, start_time + duration_each_cycle, 1.0 / frequency)
        
        Each pulse is associated with a combined condition tuple:
            (target_power, location_tag, laser_name, duration_each_cycle, frequency, pulse_duration)
        
        Returns
        -------
        dict
            Dictionary with keys:
              - "pulse_start_times": np.array of pulse onset times.
              - "pulse_power_map": np.array of target power values.
              - "pulse_location_map": np.array of location tags.
              - "pulse_lasername_map": np.array of laser names.
              - "pulse_cycle_duration_map": np.array of cycle durations.
              - "pulse_frequency_map": np.array of frequencies.
              - "pulse_pdur_map": np.array of pulse durations (in seconds).
              - "combined_conditions": list of 6-tuple conditions (one per pulse).
              - "unique_conditions": sorted list of unique 6-tuple conditions.
        """
        optical_tagging_par = self.get_optical_tagging_par()
        if optical_tagging_par is None:
            return None

        # Extract basic arrays
        laser_start_times = np.array(optical_tagging_par.get("laser_start_timestamp", []))
        target_powers = np.array(optical_tagging_par.get("target_power", []), dtype=object)
        location_tags = np.array(optical_tagging_par.get("location_tag", []), dtype=object)
        laser_names = np.array(optical_tagging_par.get("laser_name", []), dtype=object)

        # Extract trial-specific stimulation parameters (as arrays)
        duration_arr = np.array(optical_tagging_par.get("duration_each_cycle", []), dtype=object)
        frequency_arr = np.array(optical_tagging_par.get("frequency", []), dtype=object)
        pulse_duration_arr = np.array(optical_tagging_par.get("pulse_duration", []), dtype=object)

        if (len(laser_start_times) == 0 or duration_arr.size == 0 or 
            frequency_arr.size == 0 or pulse_duration_arr.size == 0):
            print("Error: Missing required optical tagging parameters.")
            return None

        # Initialize lists for pulse-level data
        pulse_start_times = []
        pulse_power_map = []
        pulse_location_map = []
        pulse_lasername_map = []
        pulse_cycle_duration_map = []
        pulse_frequency_map = []
        pulse_pdur_map = []  # pulse_duration in seconds
        pulse_index_map = []  # 0-based position of each pulse within its train

        def safe_fetch(arr, idx):
            return arr[idx] if idx < len(arr) else None

        for i, start_time in enumerate(laser_start_times):
            pwr = safe_fetch(target_powers, i)
            loc = safe_fetch(location_tags, i)
            lname = safe_fetch(laser_names, i)
            dur = float(duration_arr[i])
            fre = float(frequency_arr[i])
            pdur = float(pulse_duration_arr[i]) / 1000.0  # convert ms to sec

            pulses = np.arange(start_time, start_time + dur, 1.0 / fre)
            n_pulses_trial = len(pulses)
            pulse_start_times.extend(pulses)
            pulse_power_map.extend([pwr] * n_pulses_trial)
            pulse_location_map.extend([loc] * n_pulses_trial)
            pulse_lasername_map.extend([lname] * n_pulses_trial)
            pulse_cycle_duration_map.extend([dur] * n_pulses_trial)
            pulse_frequency_map.extend([fre] * n_pulses_trial)
            pulse_pdur_map.extend([pdur] * n_pulses_trial)
            pulse_index_map.extend(range(n_pulses_trial))

        # Convert lists to arrays
        pulse_start_times = np.array(pulse_start_times)
        pulse_index_map = np.array(pulse_index_map, dtype=int)
        pulse_power_map = np.array(pulse_power_map, dtype=object)
        pulse_location_map = np.array(pulse_location_map, dtype=object)
        pulse_lasername_map = np.array(pulse_lasername_map, dtype=object)
        pulse_cycle_duration_map = np.array(pulse_cycle_duration_map, dtype=float)
        pulse_frequency_map = np.array(pulse_frequency_map, dtype=float)
        pulse_pdur_map = np.array(pulse_pdur_map, dtype=float)

        # Replace missing entries
        pulse_power_map = self._fill_missing(pulse_power_map, "NoPower")
        pulse_location_map = self._fill_missing(pulse_location_map, "NoLocation")
        pulse_lasername_map = self._fill_missing(pulse_lasername_map, "NoLaserName")

        combined_conditions = [
            (pwr, loc, lname, dur, fre, pdur)
            for pwr, loc, lname, dur, fre, pdur in zip(
                pulse_power_map, pulse_location_map, pulse_lasername_map,
                pulse_cycle_duration_map, pulse_frequency_map, pulse_pdur_map)
        ]
        unique_conditions = sorted(set(combined_conditions))

        return {
            "pulse_start_times": pulse_start_times,
            "pulse_power_map": pulse_power_map,
            "pulse_location_map": pulse_location_map,
            "pulse_lasername_map": pulse_lasername_map,
            "pulse_cycle_duration_map": pulse_cycle_duration_map,
            "pulse_frequency_map": pulse_frequency_map,
            "pulse_pdur_map": pulse_pdur_map,
            "pulse_index_map": pulse_index_map,
            "combined_conditions": combined_conditions,
            "unique_conditions": unique_conditions
        }

    def _build_laser_start_arrays(self):
        """
        Builds arrays for laser alignment events using the laser_start_timestamp field.

        This function extracts:
          - The laser start times.
          - The condition maps for each laser event:
              * "laser_power_map": target power for each event.
              * "laser_location_map": location tag for each event.
              * "laser_lasername_map": laser name for each event.
          - The stimulation parameters for each event:
              * "laser_cycle_duration_map": cycle duration for each event.
              * "laser_frequency_map": frequency for each event.
              * "laser_pdur_map": pulse duration (converted from ms to sec) for each event.
          - It then builds the combined condition tuples (a 6-tuple) and computes the sorted unique conditions.

        Returns
        -------
        dict
            Dictionary with keys:
              - "laser_start_times": np.array of laser start times.
              - "laser_power_map": np.array of target power values.
              - "laser_location_map": np.array of location tags.
              - "laser_lasername_map": np.array of laser names.
              - "laser_cycle_duration_map": np.array of cycle durations.
              - "laser_frequency_map": np.array of frequencies.
              - "laser_pdur_map": np.array of pulse durations (in seconds).
              - "combined_conditions": list of 6-tuple conditions 
                  (target_power, location_tag, laser_name, cycle_duration, frequency, pulse_duration).
              - "unique_conditions": sorted list of unique 6-tuple conditions.
        """
        optical_tagging_par = self.get_optical_tagging_par()
        if optical_tagging_par is None:
            return None

        # Extract laser-level arrays.
        laser_start_times = np.array(optical_tagging_par.get("laser_start_timestamp", []))
        laser_power_map = np.array(optical_tagging_par.get("target_power", []), dtype=object)
        laser_location_map = np.array(optical_tagging_par.get("location_tag", []), dtype=object)
        laser_lasername_map = np.array(optical_tagging_par.get("laser_name", []), dtype=object)

        # Extract stimulation parameters for each laser event.
        cycle_duration_arr = np.array(optical_tagging_par.get("duration_each_cycle", []), dtype=object)
        frequency_arr = np.array(optical_tagging_par.get("frequency", []), dtype=object)
        pulse_duration_arr = np.array(optical_tagging_par.get("pulse_duration", []), dtype=object)

        try:
            laser_cycle_duration_map = np.array([float(x) for x in cycle_duration_arr])
            laser_frequency_map = np.array([float(x) for x in frequency_arr])
            laser_pdur_map = np.array([float(x) / 1000.0 for x in pulse_duration_arr])
        except Exception as e:
            print("Error converting stimulation parameters to float:", e)
            return None

        # Build combined condition tuples (6-tuple).
        combined_conditions = list(zip(
            laser_power_map, 
            laser_location_map, 
            laser_lasername_map, 
            laser_cycle_duration_map, 
            laser_frequency_map, 
            laser_pdur_map
        ))
        unique_conditions = sorted(set(combined_conditions))

        return {
            "laser_start_times": laser_start_times,
            "laser_power_map": laser_power_map,
            "laser_location_map": laser_location_map,
            "laser_lasername_map": laser_lasername_map,
            "laser_cycle_duration_map": laser_cycle_duration_map,
            "laser_frequency_map": laser_frequency_map,
            "laser_pdur_map": laser_pdur_map,
            "combined_conditions": combined_conditions,
            "unique_conditions": unique_conditions
        }

    def get_pulse_start_end(self):
        """
        Computes and returns pulse start and end times.
        
        The pulse end times are computed as the pulse start times plus the corresponding pulse duration.
        
        Returns
        -------
        dict
            Dictionary with keys:
              - "pulse_start_times": np.array of pulse start times.
              - "pulse_end_times": np.array of pulse end times.
        """
        pulse_data = self._build_pulse_arrays()
        if pulse_data is None:
            return None
        pulse_start_times = pulse_data["pulse_start_times"]
        pulse_pdur_map = pulse_data["pulse_pdur_map"]
        pulse_end_times = pulse_start_times + pulse_pdur_map
        return {"pulse_start_times": pulse_start_times, "pulse_end_times": pulse_end_times}

    def remove_laser_artefacts(self, unit_spike_times, pulse_start_times, pulse_end_times, removal_window=0.0005):
        """
        Removes laser artefact spikes from a unit's spike times.
        
        Laser artefacts are defined as spikes occurring within a given removal window
        (default 0.5 ms) around both the pulse onset and the pulse offset.
        
        Parameters
        ----------
        unit_spike_times : np.array
            Array of spike times (in seconds) for the unit.
        pulse_start_times : np.array
            Array of laser pulse onset times (in seconds).
        pulse_end_times : np.array
            Array of laser pulse end times (in seconds).
        removal_window : float, optional
            Time window (in seconds) around pulse start and end within which spikes are removed (default 0.0005).
        
        Returns
        -------
        np.array
            Filtered array of spike times with artefact spikes removed.
        """
        mask = np.ones(len(unit_spike_times), dtype=bool)
        for i, start in enumerate(pulse_start_times):
            end = pulse_end_times[i]
            start_interval = (start - removal_window, start + removal_window)
            end_interval = (end - removal_window, end + removal_window)
            mask &= ~((unit_spike_times >= start_interval[0]) & (unit_spike_times <= start_interval[1]))
            mask &= ~((unit_spike_times >= end_interval[0]) & (unit_spike_times <= end_interval[1]))
        return unit_spike_times[mask]

    def _get_filtered_spike_times(self, unit_index, remove_artefacts, removal_window):
        """
        Pre-calculates filtered spike times for each unit in unit_index.
        
        Artefact removal is always based on pulse start and end times (obtained via get_pulse_start_end).
        
        Parameters
        ----------
        unit_index : list
            List of unit indices.
        remove_artefacts : bool
            Flag indicating whether to remove artefacts.
        removal_window : float
            Time window (in seconds) for artefact removal.
            
        Returns
        -------
        dict
            Dictionary where each key is a unit index and the value is the filtered spike times.
            If remove_artefacts is False, returns the raw spike times.
        """
        filtered_spikes = {}
        if remove_artefacts:
            pulse_times = self.get_pulse_start_end()
            if pulse_times is None:
                return None
            pulse_start_all = pulse_times["pulse_start_times"]
            pulse_end_all = pulse_times["pulse_end_times"]
            for unit in unit_index:
                filtered_spikes[unit] = self.remove_laser_artefacts(
                    self.units_passing_qc.loc[unit]["spike_times"],
                    pulse_start_all,
                    pulse_end_all,
                    removal_window
                )
        else:
            for unit in unit_index:
                filtered_spikes[unit] = self.units_passing_qc.loc[unit]["spike_times"]
        return filtered_spikes

    def _get_event_arrays(self, align_to_event):
        """
        Retrieves event arrays and condition maps based on the alignment method.
        
        Parameters
        ----------
        align_to_event : str
            Determines which event arrays to retrieve. Valid values are "pulse" or "laser".
        
        Returns
        -------
        dict or None
            A dictionary with the following keys:
              - "event_times": np.array of event start times.
              - "unique_conditions": sorted list of unique condition tuples.
              - "power_map": np.array of target power values.
              - "location_map": np.array of location tags.
              - "lasername_map": np.array of laser name values.
              - "cycle_duration_map": np.array of cycle durations.
              - "frequency_map": np.array of frequencies.
              - "pdur_map": np.array of pulse durations (in seconds).
            Returns None if the provided alignment method is invalid or data is missing.
        """
        if align_to_event == "pulse":
            pulse_data = self._build_pulse_arrays()
            if pulse_data is None:
                return None
            return {
                "event_times": pulse_data["pulse_start_times"],
                "unique_conditions": pulse_data["unique_conditions"],
                "power_map": pulse_data["pulse_power_map"],
                "location_map": pulse_data["pulse_location_map"],
                "lasername_map": pulse_data["pulse_lasername_map"],
                "cycle_duration_map": pulse_data["pulse_cycle_duration_map"],
                "frequency_map": pulse_data["pulse_frequency_map"],
                "pdur_map": pulse_data["pulse_pdur_map"],
                "pulse_index_map": pulse_data["pulse_index_map"]
            }
        elif align_to_event == "laser":
            laser_data = self._build_laser_start_arrays()
            if laser_data is None:
                return None
            return {
                "event_times": laser_data["laser_start_times"],
                "unique_conditions": laser_data["unique_conditions"],
                "power_map": laser_data["laser_power_map"],
                "location_map": laser_data["laser_location_map"],
                "lasername_map": laser_data["laser_lasername_map"],
                "cycle_duration_map": laser_data["laser_cycle_duration_map"],
                "frequency_map": laser_data["laser_frequency_map"],
                "pdur_map": laser_data["laser_pdur_map"],
                "pulse_index_map": np.zeros(len(laser_data["laser_start_times"]), dtype=int)
            }
        else:
            print("Error: Invalid align_to_event value. Use 'pulse' or 'laser'.")
            return None

    def _compute_aligned_histogram(self, unit_spike_times, event_times, time_window, bins_arr):
        """
        Computes a histogram of spike counts aligned to a set of event times.
        
        Parameters
        ----------
        unit_spike_times : np.array
            The (filtered) spike times for a unit.
        event_times : np.array
            An array of event times (e.g. pulse or laser start times) to align spikes.
        time_window : list of two floats
            Time window (in seconds) around each event to consider.
        bins_arr : np.array
            The bin edges for histogramming.
        
        Returns
        -------
        np.array
            Histogram counts (summed over all events).
        """
        all_aligned = []
        for event_time in event_times:
            aligned = unit_spike_times[
                (unit_spike_times >= event_time + time_window[0]) &
                (unit_spike_times <= event_time + time_window[1])
            ] - event_time
            all_aligned.extend(aligned)
        hist_counts, _ = np.histogram(all_aligned, bins=bins_arr)
        return hist_counts

    @staticmethod
    def _lookup_condition_metrics(metrics_df, unit_id, cond):
        """
        Return the pooled (``pulse_index == -1``) metrics row for one
        ``unit_id`` x ``cond`` as a dict, or None if not present. ``cond`` may be
        stored as a tuple (in-memory) or a string (after a CSV round-trip).
        """
        from ast import literal_eval
        import re

        def _as_tuple(c):
            if isinstance(c, str):
                # Strip numpy scalar wrappers, e.g. "np.float64(1.0)" -> "1.0",
                # so older CSVs (saved before native-type storage) still parse.
                c = re.sub(r"np\.\w+\(([^()]*)\)", r"\1", c)
                try:
                    c = literal_eval(c)
                except (ValueError, SyntaxError):
                    return None
            return tuple(c) if isinstance(c, (tuple, list)) else None

        target = _as_tuple(cond)
        sub = metrics_df[metrics_df["unit_id"] == unit_id]
        if "pulse_index" in sub.columns:
            sub = sub[sub["pulse_index"] == -1]
        for _, row in sub.iterrows():
            if _as_tuple(row["condition"]) == target:
                return {
                    "reliability": row.get("reliability", float("nan")),
                    "median_latency": row.get("median_latency", float("nan")),
                    "jitter": row.get("jitter", float("nan")),
                    "effect_ratio": row.get("effect_ratio", float("nan")),
                }
        return None

    def plot_raster_graph(self, unit_index=None, time_window=[-0.05, 0.1], bin_size=0.005,
                          remove_artefacts=True, removal_window=0.002,
                          align_to_event="pulse", min_onset_time=0.0,
                          save_path="/root/capsule/scratch/", save_formats=['png'],
                          show_waveform=False, metrics_df=None):
        """
        Plots a raster and peri-stimulus time histogram (PSTH) for a single unit,
        sorting trials by the first spike ≥ min_onset_time, and—
        if align_to_event=="laser"—shading all individual pulses.
        All conditions for the unit are drawn in a SINGLE figure (one column per
        condition), and the figure is optionally saved to one file per unit.

        Parameters
        ----------
        unit_index : int
            Index of the QC-passed unit to plot.
        time_window : [float, float]
            Window (s) around each event for alignment.
        bin_size : float
            PSTH bin size (s).
        remove_artefacts : bool
        removal_window : float
            Artefact removal interval around each pulse onset/offset.
        align_to_event : "pulse" or "laser"
        min_onset_time : float
            Only spikes ≥ this (relative to event) count toward sorting.
        save_path : str or None
            Base path (without extension) where to save the figure. One file is
            written per unit (all conditions in the same image).
        save_formats : list of str or None
            List of formats to save in, e.g. ["png", "pdf", "eps"].
            Supported: "png", "pdf", "eps", "svg", "jpg", "jpeg".
        show_waveform : bool
            If True, draw the unit's mean waveform (peak channel) as a small inset
            in the top-right corner of the figure.
        metrics_df : pd.DataFrame, optional
            Metrics table from ``compute_tagging_metrics``. When provided, each
            condition panel is annotated with this unit's pooled (``pulse_index
            == -1``) ``reliability``, ``median_latency``, ``jitter`` and
            ``effect_ratio`` for that condition.
        """
        # Validate inputs
        optical_tagging_par = self.get_optical_tagging_par()
        if optical_tagging_par is None:
            print("Error: No optical tagging parameters.")
            return
        if not hasattr(self, "units_passing_qc"):
            print("Error: QC-passed units missing.")
            return
        if unit_index is None:
            print("Error: Must specify unit_index.")
            return
        if isinstance(unit_index, int):
            unit_index = [unit_index]
        if unit_index[0] not in self.units_passing_qc.index:
            print(f"Error: Unit {unit_index[0]} not found.")
            return

        # Retrieve event arrays
        event_dict = self._get_event_arrays(align_to_event)
        if event_dict is None:
            return
        event_times = event_dict["event_times"]
        conds = event_dict["unique_conditions"]
        pm, lm, nm, cm, fm, pdm = (
            event_dict["power_map"],
            event_dict["location_map"],
            event_dict["lasername_map"],
            event_dict["cycle_duration_map"],
            event_dict["frequency_map"],
            event_dict["pdur_map"],
        )

        # Pre-calc filtered spikes
        filtered_spikes = self._get_filtered_spike_times(unit_index, remove_artefacts, removal_window)
        if filtered_spikes is None:
            return
        spk = filtered_spikes[unit_index[0]]

        # Probe / shank info for the figure title (from ccf_location if available)
        probe_str, shank_str = "?", "?"
        try:
            ccf = self.nwb_ephys_data.units['ccf_location'][unit_index[0]]
            probe_str = ccf.get('probe', '?')
            shank_str = ccf.get('shank', '?')
        except Exception:
            pass

        # Collect only the conditions that actually have events for this unit,
        # so every condition is drawn in a SINGLE figure (one column each).
        valid = []
        for cond in conds:
            pwr, loc, lname, dur, fre, pdur = cond
            mask = (
                (pm == pwr) & (lm == loc) & (nm == lname) &
                (cm == dur) & (fm == fre) & (pdm == pdur)
            )
            these_events = event_times[mask]
            if len(these_events) == 0:
                continue
            valid.append((cond, these_events))

        if not valid:
            print(f"No events for unit {unit_index[0]} in any condition.")
            return

        n_conds = len(valid)
        bins = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
        centers = bins[:-1] + bin_size / 2

        # One figure per unit, conditions arranged in a 3-column grid
        # (3 x 3 for up to 9 conditions). Each grid cell is a sub-figure
        # holding a stacked raster (top) + PSTH (bottom).
        ncols = 3
        nrows = int(np.ceil(n_conds / ncols))
        fig = plt.figure(figsize=(6 * ncols, 4 * nrows), layout="constrained")
        fig.suptitle(
            f"Unit {unit_index[0]} | {probe_str} shank {shank_str}",
            fontsize=14,
        )
        subfigs = fig.subfigures(nrows, ncols, squeeze=False)

        last_raster_ax = None
        for idx, (cond, these_events) in enumerate(valid):
            pwr, loc, lname, dur, fre, pdur = cond
            r, c = divmod(idx, ncols)
            sf = subfigs[r][c]
            raster_ax, psth_ax = sf.subplots(
                2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}
            )
            last_raster_ax = raster_ax

            # Sort by first post-onset spike
            first_spikes = []
            for t in these_events:
                s = spk[(spk >= t + min_onset_time) & (spk <= t + time_window[1])] - t
                first_spikes.append(s.min() if s.size else np.inf)
            order = np.argsort(first_spikes)
            sorted_events = these_events[order]

            # Shade stim intervals ONCE (identical across trials) instead of per row.
            if align_to_event == "pulse":
                raster_ax.axvspan(0, pdur, color="gray", alpha=0.3)
            else:
                for p in np.arange(0, dur, 1.0 / fre):
                    raster_ax.axvspan(p, p + pdur, color="gray", alpha=0.3)

            # Raster + per-trial PSTH
            psth_counts = []
            for row, t in enumerate(sorted_events):
                aligned = spk[(spk >= t + time_window[0]) & (spk <= t + time_window[1])] - t
                raster_ax.vlines(aligned, row + 0.5, row + 1.5, color="black")
                counts, _ = np.histogram(aligned, bins=bins)
                psth_counts.append(counts)

            raster_ax.set_title(f"Cond: {cond}", fontsize=9)
            raster_ax.set_ylabel("Trial (sorted)")
            # Legend upper-left so it does not overlap the waveform inset.
            raster_ax.legend(
                [mpatches.Patch(color="gray", alpha=0.3)],
                ["Stim Window(s)"],
                loc="upper left",
                fontsize=8,
            )

            # Optional per-condition metric annotation (pooled pulse_index == -1).
            if metrics_df is not None and len(metrics_df) > 0:
                m = self._lookup_condition_metrics(metrics_df, unit_index[0], cond)
                if m is not None:
                    txt = (
                        f"reliability={m['reliability']:.2f}\n"
                        f"median_latency={m['median_latency'] * 1e3:.2f} ms\n"
                        f"jitter={m['jitter'] * 1e3:.2f} ms\n"
                        f"effect_ratio={m['effect_ratio']:.2f}"
                    )
                    raster_ax.text(
                        0.02, 0.72, txt,
                        transform=raster_ax.transAxes,
                        fontsize=7, va="top", ha="left",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7,
                                  edgecolor="none"),
                    )

            # Compute mean & SEM FR
            arr = np.array(psth_counts)
            mean_cnt = arr.mean(axis=0)
            sem_cnt = (arr.std(axis=0, ddof=1) / np.sqrt(len(arr))
                       if len(arr) > 1 else np.zeros_like(mean_cnt))
            fr = mean_cnt / bin_size
            fr_sem = sem_cnt / bin_size

            # Mask artefact bins around all pulses
            mask_bins = np.ones_like(centers, dtype=bool)
            pulse_times_rel = np.arange(0, dur, 1.0 / fre)
            for p in pulse_times_rel:
                mask_bins &= ~(
                    (np.abs(centers - p) <= removal_window) |
                    (np.abs(centers - (p + pdur)) <= removal_window)
                )

            psth_ax.plot(centers[mask_bins], fr[mask_bins], label="Mean FR")
            psth_ax.fill_between(
                centers[mask_bins],
                fr[mask_bins] - fr_sem[mask_bins],
                fr[mask_bins] + fr_sem[mask_bins],
                alpha=0.3
            )

            # Shade PSTH stim windows
            if align_to_event == "pulse":
                psth_ax.axvspan(0, pdur, color="gray", alpha=0.3)
            else:
                for p in pulse_times_rel:
                    psth_ax.axvspan(p, p + pdur, color="gray", alpha=0.3)

            psth_ax.set_ylabel("Firing Rate (Hz)")
            psth_ax.set_xlabel("Time from Event (s)")
            psth_ax.legend(fontsize=8)

        # Hide any unused grid cells (when n_conds is not a multiple of ncols).
        for idx in range(n_conds, nrows * ncols):
            r, c = divmod(idx, ncols)
            subfigs[r][c].set_visible(False)

        # Optional mean-waveform inset in the top-right corner of the last raster.
        if show_waveform and last_raster_ax is not None:
            try:
                units_table = self.nwb_ephys_data.units[:]
                wf = np.array(units_table.loc[unit_index[0], "waveform_mean"])
                # Peak channel = the one with the most negative trough
                peak_ch = int(np.argmin(np.min(wf, axis=0)))
                wf_trace = wf[:, peak_ch]
                wf_ax = last_raster_ax.inset_axes([0.80, 0.78, 0.18, 0.20])
                wf_ax.plot(wf_trace, color="black", linewidth=1)
                wf_ax.set_title(f"waveform (ch {peak_ch})", fontsize=8, pad=1)
                wf_ax.set_xticks([])
                wf_ax.set_yticks([])
                for spine in wf_ax.spines.values():
                    spine.set_visible(False)
            except Exception as e:
                print(f"Warning: could not draw waveform inset for unit {unit_index[0]}: {e}")

        # Save a single figure per unit (all conditions in one image).
        if save_path and save_formats:
            supported = ("png", "pdf", "eps", "svg", "jpg", "jpeg")
            for fmt in save_formats:
                fmt_low = fmt.lower()
                if fmt_low in supported:
                    fname = f"{save_path}.{fmt_low}"
                    fig.savefig(fname, format=fmt_low, dpi=300, bbox_inches='tight')
                else:
                    print(f"Warning: unsupported save format '{fmt}'")

        plt.show()
        plt.close(fig)







    def plot_psth(self, unit_index=None, time_window=[-2, 3], bin_size=0.05,
                  remove_artefacts=True, removal_window=0.002, align_to_event="pulse",
                  save_path='/root/capsule/scratch/', save_formats=["eps"]):
        """
        Plots the peri-stimulus time histogram (PSTH) for selected units, with each unique condition
        displayed in a separate figure. Adds artifact masking around all pulses and optional saving.

        Parameters
        ----------
        unit_index : int or list of int
            Unit index (or list) to include in the PSTH.
        time_window : list of two floats
            Time window (in seconds) around each event used for aligning spikes.
        bin_size : float
            Bin size (in seconds) for histogram binning.
        remove_artefacts : bool, optional
            If True (default), removes laser artefact spikes from the unit's spike times.
        removal_window : float, optional
            Removal window (in seconds) for artefact removal (default 0.002).
        align_to_event : str, optional
            Alignment method; "pulse" (default) aligns to pulse events,
            "laser" aligns to laser events.
        save_path : str or None
            Base path (without extension) where to save the figures, e.g. "/path/to/output".
        save_formats : list of str or None
            List of formats to save in, e.g. ["pdf", "eps"]. Supported: "pdf", "eps".
        """
        optical_tagging_par = self.get_optical_tagging_par()
        if optical_tagging_par is None:
            return

        if not hasattr(self, "units_passing_qc"):
            print("Error: QC-passing units are missing.")
            return

        # Ensure unit_index is a list of valid unit IDs
        if unit_index is None:
            print("Error: No unit index provided.")
            return
        if isinstance(unit_index, int):
            unit_index = [unit_index]
        unit_index = [u for u in unit_index if u in self.units_passing_qc.index]
        if not unit_index:
            print("Error: None of the selected units exist in QC-passing units.")
            return

        # Prepare PSTH bins
        bins_arr = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
        centers = bins_arr[:-1] + bin_size / 2

        # Retrieve event arrays
        event_dict = self._get_event_arrays(align_to_event)
        if event_dict is None:
            return
        event_times = event_dict["event_times"]
        unique_conditions = event_dict["unique_conditions"]
        pm = event_dict["power_map"]
        lm = event_dict["location_map"]
        nm = event_dict["lasername_map"]
        cm = event_dict["cycle_duration_map"]
        fm = event_dict["frequency_map"]
        pdm = event_dict["pdur_map"]

        # Pre-calculate (optionally artifact-removed) spike times
        filtered_spikes = self._get_filtered_spike_times(unit_index, remove_artefacts, removal_window)
        if filtered_spikes is None:
            return

        # Iterate over conditions
        for cond in unique_conditions:
            pwr, loc, lname, dur, fre, pdur = cond
            sel = ((pm == pwr) & (lm == loc) & (nm == lname) &
                   (cm == dur) & (fm == fre) & (pdm == pdur))
            these_events = event_times[sel]
            n_events = len(these_events)
            if n_events == 0:
                continue

            # Build per-unit PSTH counts
            psth_per_unit = []
            for unit in unit_index:
                spike_times = filtered_spikes[unit]
                total_counts = np.zeros(len(bins_arr)-1, dtype=float)
                for t in these_events:
                    aligned = spike_times[(spike_times >= t + time_window[0]) &
                                           (spike_times <= t + time_window[1])] - t
                    counts, _ = np.histogram(aligned, bins=bins_arr)
                    total_counts += counts
                # normalize by number of events and bin width → firing rate
                psth_per_unit.append(total_counts / (n_events * bin_size))

            psth_arr = np.vstack(psth_per_unit)
            mean_fr = psth_arr.mean(axis=0)
            sem_fr = psth_arr.std(axis=0, ddof=1) / np.sqrt(len(unit_index))

            # Artifact masking around all pulses
            mask = np.ones_like(centers, dtype=bool)
            pulse_rel = np.arange(0, dur, 1.0 / fre)
            for p in pulse_rel:
                mask &= ~((np.abs(centers - p) <= removal_window) |
                          (np.abs(centers - (p + pdur)) <= removal_window))

            # Plot PSTH
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(centers[mask], mean_fr[mask], label="Mean FR")
            ax.fill_between(centers[mask],
                            mean_fr[mask] - sem_fr[mask],
                            mean_fr[mask] + sem_fr[mask],
                            alpha=0.3)
            # Shade stimulation intervals
            if align_to_event == "pulse":
                ax.axvspan(0, pdur, color="gray", alpha=0.3)
            else:
                for p in pulse_rel:
                    ax.axvspan(p, p + pdur, color="gray", alpha=0.3)

            ax.set_xlabel("Time from Event (s)")
            ax.set_ylabel("Firing Rate (Hz)")
            ax.set_title(f"Condition: {cond} | n_units={len(unit_index)}")
            ax.legend()
            plt.tight_layout()

            # Save if requested
            if save_path and save_formats:
                cond_str = "_".join([
                    str(pwr).replace(" ", ""),
                    str(loc).replace(" ", ""),
                    str(lname).replace(" ", ""),
                    f"dur{dur}",
                    f"fre{fre}",
                    f"pdur{int(pdur*1000)}ms"
                ])
                base = save_path.rstrip("/") + "/" + cond_str
                for fmt in save_formats:
                    fmt_low = fmt.lower()
                    if fmt_low in ("pdf", "eps"):
                        fig.savefig(f"{base}.{fmt_low}", format=fmt_low,
                                    dpi=300, bbox_inches="tight")
                    else:
                        print(f"Warning: unsupported save format '{fmt}'")

            plt.show()


    def plot_psth_multiple_laser_power(self, unit_index=None, time_window=[-2, 3], bin_size=0.05,
                                       remove_artefacts=True, removal_window=0.002,
                                       align_to_event="pulse",
                                       sem_shade=0.6, stim_shade=0.6,
                                       save_path='/root/capsule/scratch/', save_formats=["eps"]):
        """
        Plots PSTHs for multiple laser powers on the same axes, for each unique combination
        of location, laser name, cycle duration, frequency, and pulse duration.

        Parameters
        ----------
        unit_index : int or list of int
            Unit index (or list) to include in the PSTH.
        time_window : list of two floats
            Window (s) around each event for alignment.
        bin_size : float
            Bin size (s) for histogramming.
        remove_artefacts : bool
        removal_window : float
            Artefact removal window (s) around each pulse.
        align_to_event : "pulse" or "laser"
        sem_shade : float in [0, 1]
            Lightness of the SEM shaded band. Higher = lighter / more
            transparent-looking (EPS-safe, no real alpha). Default 0.6.
        stim_shade : float in [0, 1]
            Lightness of the gray stim-window shading. Higher = lighter.
            Default 0.6.
        save_path : str or None
            Base path (without extension) to save figures.
        save_formats : list of str or None
            Formats to save: ["pdf","eps"].
        """
        # Validate units
        if unit_index is None:
            print("Error: Must specify unit_index.")
            return
        if isinstance(unit_index, int):
            unit_index = [unit_index]
        unit_index = [u for u in unit_index if u in self.units_passing_qc.index]
        if not unit_index:
            print("Error: None of the selected units exist in QC-passing units.")
            return

        # PSTH bins
        bins = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
        centers = bins[:-1] + bin_size/2

        # Get events & maps
        E = self._get_event_arrays(align_to_event)
        if E is None:
            return
        ev, pm, lm, nm, cm, fm, pdm = (
            E["event_times"], E["power_map"], E["location_map"], E["lasername_map"],
            E["cycle_duration_map"], E["frequency_map"], E["pdur_map"]
        )
        conds = E["unique_conditions"]

        # Pre-filter spikes
        spikes_dict = self._get_filtered_spike_times(unit_index, remove_artefacts, removal_window)
        if spikes_dict is None:
            return

        # Group by all cond fields except power
        groups = {}
        for (pwr, loc, lname, dur, fre, pdur) in conds:
            key = (loc, lname, dur, fre, pdur)
            groups.setdefault(key, []).append(pwr)

        # Helper: blend a color toward white so shaded regions render
        # correctly in EPS/PostScript (which does not support transparency).
        def _lighten(color, amount=0.75):
            c = np.array(mcolors.to_rgb(color))
            return tuple(c + (1.0 - c) * amount)

        # For each group, plot powers
        for (loc, lname, dur, fre, pdur), powers in groups.items():
            fig, ax = plt.subplots(figsize=(6, 6))
            for pwr in sorted(powers):
                # select events of this power
                mask = ((pm==pwr)&(lm==loc)&(nm==lname)&(cm==dur)&(fm==fre)&(pdm==pdur))
                these = ev[mask]
                if len(these)==0:
                    continue

                # compute PSTH across units & events
                all_rates = []
                for u in unit_index:
                    st = spikes_dict[u]
                    counts = np.zeros(len(bins)-1)
                    for t0 in these:
                        aligned = st[(st>=t0+time_window[0])&(st<=t0+time_window[1])] - t0
                        c,_ = np.histogram(aligned, bins=bins)
                        counts += c
                    # normalize: events * bin_size
                    all_rates.append(counts / (len(these)*bin_size))
                arr = np.vstack(all_rates)
                mean_fr = arr.mean(axis=0)
                sem_fr = arr.std(axis=0, ddof=1)/np.sqrt(len(unit_index))

                # mask artifacts around all pulses
                mask_bins = np.ones_like(centers, dtype=bool)
                rel = np.arange(0, dur, 1.0/fre)
                for t in rel:
                    mask_bins &= ~((np.abs(centers-t)<=removal_window)|
                                   (np.abs(centers-(t+pdur))<=removal_window))

                # plot (use solid lightened fill instead of alpha for EPS compatibility)
                line, = ax.plot(centers[mask_bins], mean_fr[mask_bins], label=f"power={pwr}")
                ax.fill_between(centers[mask_bins],
                                mean_fr[mask_bins]-sem_fr[mask_bins],
                                mean_fr[mask_bins]+sem_fr[mask_bins],
                                color=_lighten(line.get_color(), amount=sem_shade),
                                edgecolor="none", linewidth=0)

            # shade stim windows (solid light gray for EPS compatibility)
            if align_to_event=="pulse":
                ax.axvspan(0, pdur, color=_lighten("gray", amount=stim_shade), edgecolor="none", zorder=0)
            else:
                for t in np.arange(0, dur, 1.0/fre):
                    ax.axvspan(t, t+pdur, color=_lighten("gray", amount=stim_shade), edgecolor="none", zorder=0)

            ax.set_title(f"loc={loc}, laser={lname}, dur={dur}, fre={fre}, pdur={pdur} | n_units={len(unit_index)}")
            ax.set_xlabel("Time from Event (s)")
            ax.set_ylabel("Firing Rate (Hz)")
            ax.legend(title="Laser Power")
            plt.tight_layout()

            # save
            if save_path and save_formats:
                safe = f"{loc}_{lname}_dur{dur}_fre{fre}_pdur{int(pdur*1000)}ms"
                base = save_path.rstrip("/") + "/" + safe
                for fmt in save_formats:
                    if fmt.lower() in ("pdf","eps"):
                        fig.savefig(f"{base}.{fmt.lower()}", format=fmt.lower(), dpi=300, bbox_inches='tight')
                    else:
                        print(f"Warning: unsupported format {fmt}")

            plt.show()


    def plot_heatmap(self, 
                    unit_index=None, 
                    time_window=[-0.03, 0.03], 
                    bin_size=0.001,
                    color_range=[0, 80], 
                    save_path="/root/capsule/scratch/", 
                    save_formats=["eps"],
                    remove_artefacts=True, 
                    removal_window=0.002, 
                    align_to_event="pulse",
                    df_summary=None,               # DataFrame containing 'unit_index', 'sig_*', 'estimated_*' columns for sorting
                    sort_by=None,                  # Column name in `df_summary` to sort units by (e.g., 'estimated_y')
                    right_label_step=5,            # Show every Nth label on the right y-axis (default: 5)
                    right_label_fmt="{:.1f}",      # Format string for right y-axis labels (default: one decimal point)
                    show_p_value_line=False,       # Whether to show a line plot for p-values (default: False)
                    p_value_color="red"            # Color for the p-value line plot
                    ):
        """
        Plots a heatmap of firing rates for selected units, aligned to events.
        Optionally sorts units using `df_summary[sort_by]`, draws 1-column sidebars
        for `sig_increase`, `sig_decrease`, and shows the p-value as a line plot
        along the heatmap.

        Parameters
        ----------
        unit_index : int or list[int]
            Unit index (or list) to include in the heatmap.
        time_window : list of two floats
            Time window (in seconds) around each event for spike alignment.
        bin_size : float
            Bin size (in seconds) for computing the histogram.
        color_range : list of two floats
            Minimum and maximum firing rate for the colormap.
        save_path : str or None
            Base directory to save figures. If None, the figure is not saved.
        save_formats : list[str] or None
            Formats to save: any of ["pdf", "eps"]. If None, no files are saved.
        remove_artefacts : bool
            If True, removes laser artefact spikes from the spike times.
        removal_window : float
            Time window (in seconds) for artefact removal (default 0.002).
        align_to_event : {"pulse", "laser"}
            Alignment method for event times. "pulse" aligns to pulse events,
            "laser" aligns to laser events.
        df_summary : pd.DataFrame or None
            Summary DataFrame containing 'unit_index', 'sig_increase', 'sig_decrease', and
            'estimated_*' columns used for sorting and sidebars. If None, sorting and sidebars are not used.
        sort_by : str or None
            Column name in `df_summary` to sort units by (e.g., "estimated_y"). If None, no sorting is applied.
        right_label_step : int
            Show every Nth label on the right y-axis for sorted values. Default is 5.
        right_label_fmt : str
            Format string for the right y-axis labels (default is one decimal: "{:.1f}").
        show_p_value_line : bool
            Whether to show a line plot for p-values. Default is False.
        p_value_color : str
            Color for the p-value line plot (default is red).
        """
        # Ensure optical tagging parameters are loaded
        optical_tagging_par = self.get_optical_tagging_par()
        if optical_tagging_par is None:
            return
        if not hasattr(self, "units_passing_qc"):
            print("Error: QC-passing units missing.")
            return
        if unit_index is None:
            print("Error: No unit index provided.")
            return

        # Normalize and validate unit list
        if isinstance(unit_index, int):
            unit_index = [unit_index]
        unit_index = [u for u in unit_index if u in self.units_passing_qc.index]
        if not unit_index:
            print("Error: None of the selected units exist in QC-passing units.")
            return

        # Handle df_summary: intersect + (optional) sort + prep sidebars
        sidebars = {}
        sorted_values = None  # for right y-axis labels
        p_values = None       # for storing p-values
        if df_summary is not None:
            if "unit_index" not in df_summary.columns:
                raise ValueError("df_summary must contain a 'unit_index' column.")
            df_sub = df_summary[df_summary["unit_index"].isin(unit_index)].copy()
            if df_sub.empty:
                print("Warning: df_summary has no rows for the provided units; proceeding without it.")
            else:
                if sort_by is not None:
                    if sort_by not in df_sub.columns:
                        raise ValueError(f"sort_by='{sort_by}' not found in df_summary columns.")
                    # Sort ascending; NaNs go to the end by default
                    df_sub = df_sub.sort_values(by=sort_by, ascending=True, kind="mergesort")
                    sorted_values = df_sub[sort_by].to_numpy()
                # Adopt this row order for plotting
                unit_index = df_sub["unit_index"].tolist()
                # Prepare sidebars if present
                if "sig_increase" in df_sub.columns:
                    sidebars["sig_increase"] = df_sub["sig_increase"].astype(bool).to_numpy()
                if "sig_decrease" in df_sub.columns:
                    sidebars["sig_decrease"] = df_sub["sig_decrease"].astype(bool).to_numpy()

                # Add p-value line if requested
                if show_p_value_line and "p_value" in df_sub.columns:
                    p_values = df_sub["p_value"].to_numpy()

        # Build event arrays
        E = self._get_event_arrays(align_to_event)
        if E is None:
            return
        event_times = E["event_times"]
        conds = E["unique_conditions"]
        pm, lm, nm = E["power_map"], E["location_map"], E["lasername_map"]
        cm, fm, pdm = E["cycle_duration_map"], E["frequency_map"], E["pdur_map"]

        # Histogram bins
        bins_arr = np.arange(time_window[0], time_window[1] + bin_size, bin_size)

        # (Optionally) artifact-removed spike times
        spikes_dict = self._get_filtered_spike_times(unit_index, remove_artefacts, removal_window)
        if spikes_dict is None:
            return

        # Layout settings for optional sidebars
        n_sidebar_cols = len(sidebars)
        width_main = 20
        width_each_sidebar = 1 if n_sidebar_cols > 0 else 0
        width_ratios = [width_main] + [width_each_sidebar]*n_sidebar_cols

        # Sidebars color maps
        sidebar_cm_inc = ListedColormap(["#ffffff", "#cc0000"])   # white=False, red=True
        sidebar_cm_dec = ListedColormap(["#ffffff", "#0066cc"])   # white=False, blue=True

        for cond in conds:
            pwr, loc, lname, dur, fre, pdur = cond
            mask = (
                (pm == pwr) & (lm == loc) & (nm == lname) &
                (cm == dur) & (fm == fre) & (pdm == pdur)
            )
            these_events = event_times[mask]
            n_events = len(these_events)
            if n_events == 0:
                continue

            # Build heatmap data (units × time) in the (possibly) sorted order
            heatmap_data = []
            for u in unit_index:
                st = spikes_dict[u]
                all_aligned = []
                for t0 in these_events:
                    aligned = st[(st >= t0 + time_window[0]) & (st <= t0 + time_window[1])] - t0
                    all_aligned.extend(aligned)
                counts, _ = np.histogram(all_aligned, bins=bins_arr)
                heatmap_data.append(counts / (bin_size * n_events))
            heatmap_data = np.array(heatmap_data)

            # Figure with gridspec (main + optional sidebars)
            fig = plt.figure(figsize=(10 + 1.2*n_sidebar_cols, 6))
            gs = gridspec.GridSpec(1, 1 + n_sidebar_cols, width_ratios=width_ratios, wspace=0.05)

            # Main heatmap
            ax_main = fig.add_subplot(gs[0, 0])
            # Use origin='lower' so row 0 is at the bottom; this helps align the right y-axis
            cax = ax_main.imshow(
                heatmap_data, aspect='auto', cmap='Greys', origin='lower',
                extent=[time_window[0], time_window[1], 0, len(unit_index)],
                vmin=color_range[0], vmax=color_range[1]
            )

            # Shade stimulation windows
            if align_to_event == "pulse":
                ax_main.axvspan(0, pdur, color='blue', alpha=0.1, label='Stim')
            else:
                pulses = np.arange(0, dur, 1.0 / fre)
                for p in pulses:
                    ax_main.axvspan(p, p + pdur, color='blue', alpha=0.1)

            ax_main.set_xlabel("Time from Event Start (s)")
            left_ylabel = "Unit Index"
            if (df_summary is not None) and (sort_by is not None):
                left_ylabel += f" (sorted by {sort_by})"
            ax_main.set_ylabel(left_ylabel)
            n_inc = int(sidebars["sig_increase"].sum()) if "sig_increase" in sidebars else None
            n_dec = int(sidebars["sig_decrease"].sum()) if "sig_decrease" in sidebars else None
            title = (
                f"Heatmap – cond=(pwr={pwr}, loc={loc}, laser={lname}, "
                f"dur={dur}, fre={fre}, pdur={pdur}) | n_units={len(unit_index)}"
            )
            if n_inc is not None or n_dec is not None:
                parts = []
                if n_inc is not None:
                    parts.append(f"sig_inc={n_inc}")
                if n_dec is not None:
                    parts.append(f"sig_dec={n_dec}")
                title += " | " + ", ".join(parts)
            ax_main.set_title(title)
            fig.colorbar(cax, ax=ax_main, pad=0.1,label="Firing Rate (Hz)")


            # ---- Right y-axis: sparse, one-decimal labels aligned to rows ----
            if sorted_values is not None:
                ax_right = ax_main.twinx()
                # Match limits to main axis (rows from 0 to N)
                ax_right.set_ylim(0, len(unit_index))
                ax_right.set_xlim(ax_main.get_xlim())

                # Ticks centered on each row
                y_pos = np.arange(len(unit_index)) + 0.5
                ax_right.set_yticks(y_pos)

                # Build sparse labels
                labels = []
                for i, v in enumerate(sorted_values):
                    show = (i % max(1, int(right_label_step))) == 0
                    if show:
                        try:
                            fv = float(v)
                            label = right_label_fmt.format(fv)
                        except Exception:
                            label = str(v)
                    else:
                        label = ""  # suppress intermediate labels
                    labels.append(label)

                ax_right.set_yticklabels(labels, fontsize=8)
                ax_right.set_ylabel(sort_by, rotation=270, labelpad=15)
                ax_right.tick_params(axis='y', which='both', length=0)

            # ---- Plot the p-value as a line ----
            if show_p_value_line and p_values is not None:
                ax_p_value = ax_main.twinx()  # Create a secondary y-axis for the p-value line
                ax_p_value.plot(np.arange(len(unit_index)), p_values, color=p_value_color, label="p-value", linestyle="--", linewidth=2)
                ax_p_value.set_ylabel("p-value", color=p_value_color)
                ax_p_value.tick_params(axis="y", labelcolor=p_value_color)

            # ---- Sidebars (one narrow column per flag) ----
            sidebar_names = list(sidebars.keys())
            for i, key in enumerate(sidebars.keys(), start=1):
                ax_sb = fig.add_subplot(gs[0, i])
                data = sidebars[key].astype(int)[:, None]  # shape (n_units, 1)
                cmap = sidebar_cm_inc if key == "sig_increase" else sidebar_cm_dec
                ax_sb.imshow(
                    data, aspect='auto', cmap=cmap, origin='lower',
                    extent=[0, 1, 0, len(unit_index)], vmin=0, vmax=1
                )
                ax_sb.set_xticks([])
                ax_sb.set_yticks([])
                ax_sb.set_title(key.replace("_", "\n"), fontsize=9)

            plt.tight_layout()

            # Saving
            if save_path and save_formats:
                os.makedirs(save_path, exist_ok=True)
                cond_str = "_".join([
                    str(pwr).replace(" ", ""),
                    str(loc).replace(" ", ""),
                    str(lname).replace(" ", ""),
                    f"dur{dur}",
                    f"fre{fre}",
                    f"pdur{int(pdur*1000)}ms"
                ])
                if (df_summary is not None) and (sort_by is not None):
                    cond_str += f"_sortedby_{sort_by}"
                base = save_path.rstrip("/") + "/" + cond_str
                for fmt in save_formats:
                    fmt_low = fmt.lower()
                    if fmt_low in ("pdf", "eps"):
                        fig.savefig(f"{base}.{fmt_low}",
                                    format=fmt_low, dpi=300, bbox_inches='tight')
                    else:
                        print(f"Warning: unsupported format '{fmt}'")

            plt.show()





    def plot_psth_align_to_pulse(self, unit_index=None, time_window=[-0.03, 0.03],
                                 bin_size=0.001, y_lim=None, save_path=None,
                                 remove_artefacts=True, removal_window=0.002, align_to_event="pulse"):
        """
        Plots a PSTH (peri-stimulus time histogram) aligned to events.

        Parameters
        ----------
        unit_index : int or list of int
            Unit index (or list) to include in the PSTH.
        time_window : list of two floats
            Time window (in seconds) around each event for aligning spikes.
        bin_size : float
            Bin size (in seconds) for histogram binning.
        y_lim : tuple of two floats or None
            Optional y-axis limits (e.g., (0, 50)); if None, auto-scaled.
        save_path : str or None
            If provided, the resulting figure is saved to this path.
        remove_artefacts : bool, optional
            If True (default), removes laser artefact spikes from unit spike times.
        removal_window : float, optional
            Removal window (in seconds) for artefact removal (default 0.002).
        align_to_event : str, optional
            Alignment method; "pulse" (default) aligns to pulse events,
            "laser" aligns to laser events.
        """
        optical_tagging_par = self.get_optical_tagging_par()
        if optical_tagging_par is None:
            return

        if not hasattr(self, "units_passing_qc"):
            print("Error: QC-passing units are missing.")
            return

        if unit_index is None:
            print("Error: No unit index provided.")
            return

        if isinstance(unit_index, int):
            unit_index = [unit_index]
        unit_index = [u for u in unit_index if u in self.units_passing_qc.index]
        if len(unit_index) == 0:
            print("Error: None of the selected units exist in QC-passing units.")
            return

        bins_arr = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
        # Retrieve event arrays.
        event_dict = self._get_event_arrays(align_to_event)
        if event_dict is None:
            return
        event_times = event_dict["event_times"]
        unique_conditions = event_dict["unique_conditions"]
        cond_map = {
            "power_map": event_dict["power_map"],
            "location_map": event_dict["location_map"],
            "lasername_map": event_dict["lasername_map"],
            "cycle_duration_map": event_dict["cycle_duration_map"],
            "frequency_map": event_dict["frequency_map"],
            "pdur_map": event_dict["pdur_map"]
        }

        # For artefact removal, always use pulse start/end times.
        pulse_times = self.get_pulse_start_end()
        if pulse_times is None:
            return
        event_end_times = pulse_times["pulse_end_times"]

        # Pre-calculate filtered spike times per unit.
        filtered_spikes = self._get_filtered_spike_times(unit_index, remove_artefacts, removal_window)
        if filtered_spikes is None:
            return

        psth_counts = {cond: [] for cond in unique_conditions}
        power_map = cond_map["power_map"]
        location_map = cond_map["location_map"]
        lasername_map = cond_map["lasername_map"]
        cycle_duration_map = cond_map["cycle_duration_map"]
        frequency_map = cond_map["frequency_map"]
        pdur_map = cond_map["pdur_map"]

        for unit in unit_index:
            unit_spike_times = filtered_spikes[unit]
            for cond in unique_conditions:
                pwr_cond, loc_cond, lname_cond, dur_cond, fre_cond, pdur_cond = cond
                sel_mask = (
                    (power_map == pwr_cond) &
                    (location_map == loc_cond) &
                    (lasername_map == lname_cond) &
                    (cycle_duration_map == dur_cond) &
                    (frequency_map == fre_cond) &
                    (pdur_map == pdur_cond)
                )
                these_events = event_times[sel_mask]
                all_aligned = []
                for event_time in these_events:
                    aligned = unit_spike_times[
                        (unit_spike_times >= event_time + time_window[0]) &
                        (unit_spike_times <= event_time + time_window[1])
                    ] - event_time
                    all_aligned.extend(aligned)
                hist_counts, _ = np.histogram(all_aligned, bins=bins_arr)
                psth_counts[cond].append(hist_counts)

        avg_psth = {}
        sem_psth = {}
        for cond in unique_conditions:
            if psth_counts[cond]:
                arr = np.array(psth_counts[cond])
                avg_psth[cond] = np.mean(arr, axis=0)
                sem_psth[cond] = np.std(arr, axis=0, ddof=1) / np.sqrt(len(unit_index))
            else:
                avg_psth[cond] = np.zeros(len(bins_arr) - 1)
                sem_psth[cond] = np.zeros(len(bins_arr) - 1)

        fig, axes = plt.subplots(len(unique_conditions), 1, figsize=(8, len(unique_conditions)*3), sharex=True)
        if len(unique_conditions) == 1:
            axes = [axes]
        for ax, cond in zip(axes, unique_conditions):
            pwr_cond, loc_cond, lname_cond, dur_cond, fre_cond, pdur_cond = cond
            ax.plot(bins_arr[:-1], avg_psth[cond], color='black', linewidth=2)
            ax.fill_between(bins_arr[:-1], avg_psth[cond] - sem_psth[cond], avg_psth[cond] + sem_psth[cond],
                            color='gray', alpha=0.3)
            ax.axvspan(0, pdur_cond, color='blue', alpha=0.1, label='Laser Stimulation')
            ax.set_ylabel("Firing Rate [Hz]")
            ax.set_title(f"Condition: {cond}\nAve Firing Rate for {len(unit_index)} Units, n_events={len(these_events)}")
            ax.legend()
            if y_lim is not None:
                ax.set_ylim(y_lim)
        axes[-1].set_xlabel("Time from Event Start [s]")
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}average_PSTH_combined.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    def find_significant_units(self, unit_index=None, baseline_window=(-0.05, 0.0),
                               stim_window=(0.01, None), alpha=0.05,
                               remove_artefacts=True, removal_window=0.002, align_to_event="pulse"):
        """
        Identifies 'significant' units within each unique condition based on changes in firing rate.
        
        Parameters
        ----------
        unit_index : int or list of int
            Unit index (or list) to test. If None, all QC-passing units are used.
        baseline_window : tuple of two floats
            Time window (relative to event start) for baseline firing rate measurement (e.g., (-0.05, 0)).
        stim_window : tuple of two floats
            Time window (relative to event start) for stimulation measurement.
            If the upper bound is None, the event-specific pulse duration is used.
        alpha : float
            Significance level (e.g., 0.05) for the paired t-test.
        remove_artefacts : bool, optional
            If True (default), removes laser artefact spikes from unit spike times.
        removal_window : float, optional
            Removal window (in seconds) for artefact removal (default 0.002).
        align_to_event : str, optional
            Alignment method; "pulse" (default) aligns to pulse events,
            "laser" aligns to laser events.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
              - condition: 6-tuple (target_power, location_tag, laser_name, duration_each_cycle, frequency, pulse_duration)
              - unit_id: the unit index
              - p_value: p-value from the paired t-test
              - significant: boolean indicating if p < alpha
              - baseline_FR_mean: mean baseline firing rate
              - stim_FR_mean: mean stimulation firing rate
              - baseline_FR_std: std dev of baseline firing rates
              - stim_FR_std: std dev of stimulation firing rates
              - n_pulses: number of events for that condition
        """
        optical_tagging_par = self.get_optical_tagging_par()
        if optical_tagging_par is None:
            print("No optical tagging parameters found.")
            return pd.DataFrame()

        if not hasattr(self, "units_passing_qc"):
            print("Error: QC-passing units are missing.")
            return pd.DataFrame()

        if unit_index is None:
            unit_index = self.units_passing_qc.index.tolist()
        elif isinstance(unit_index, int):
            unit_index = [unit_index]
        else:
            unit_index = [u for u in unit_index if u in self.units_passing_qc.index]

        if len(unit_index) == 0:
            print("Error: None of the selected units exist in QC-passing units.")
            return pd.DataFrame()

        event_dict = self._get_event_arrays(align_to_event)
        if event_dict is None:
            return pd.DataFrame()
        event_times = event_dict["event_times"]
        unique_conditions = event_dict["unique_conditions"]
        cond_map = {
            "power_map": event_dict["power_map"],
            "location_map": event_dict["location_map"],
            "lasername_map": event_dict["lasername_map"],
            "cycle_duration_map": event_dict["cycle_duration_map"],
            "frequency_map": event_dict["frequency_map"],
            "pdur_map": event_dict["pdur_map"]
        }

        # For artefact removal, always use pulse start/end times.
        pulse_times = self.get_pulse_start_end()
        if pulse_times is None:
            return pd.DataFrame()
        event_end_times = pulse_times["pulse_end_times"]

        filtered_spikes = self._get_filtered_spike_times(unit_index, remove_artefacts, removal_window)
        if filtered_spikes is None:
            return pd.DataFrame()

        results_list = []
        power_map = cond_map["power_map"]
        location_map = cond_map["location_map"]
        lasername_map = cond_map["lasername_map"]
        cycle_duration_map = cond_map["cycle_duration_map"]
        frequency_map = cond_map["frequency_map"]
        pdur_map = cond_map["pdur_map"]

        for cond in unique_conditions:
            pwr_cond, loc_cond, lname_cond, cycle_cond, freq_cond, pdur_cond = cond
            sel_mask = (
                (power_map == pwr_cond) &
                (location_map == loc_cond) &
                (lasername_map == lname_cond) &
                (cycle_duration_map == cycle_cond) &
                (frequency_map == freq_cond) &
                (pdur_map == pdur_cond)
            )
            these_events = event_times[sel_mask]
            n_events = len(these_events)
            if n_events == 0:
                continue

            if stim_window[1] is None:
                effective_stim_window = (stim_window[0], pdur_cond)
            else:
                effective_stim_window = stim_window

            for unit in unit_index:
                unit_spike_times = filtered_spikes[unit]
                baseline_counts = []
                stim_counts = []
                for event_time in these_events:
                    b_start = event_time + baseline_window[0]
                    b_end = event_time + baseline_window[1]
                    s_start = event_time + effective_stim_window[0]
                    s_end = event_time + effective_stim_window[1]
                    baseline_spikes = np.sum((unit_spike_times >= b_start) & (unit_spike_times < b_end))
                    stim_spikes = np.sum((unit_spike_times >= s_start) & (unit_spike_times < s_end))
                    baseline_rate = baseline_spikes / (baseline_window[1] - baseline_window[0])
                    stim_rate = stim_spikes / (effective_stim_window[1] - effective_stim_window[0])
                    baseline_counts.append(baseline_rate)
                    stim_counts.append(stim_rate)

                baseline_counts = np.array(baseline_counts)
                stim_counts = np.array(stim_counts)
                if len(baseline_counts) < 2:
                    p_val = 1.0
                else:
                    _, p_val = ttest_rel(baseline_counts, stim_counts)

                baseline_fr_mean = np.mean(baseline_counts) if len(baseline_counts) > 0 else 0
                stim_fr_mean = np.mean(stim_counts) if len(stim_counts) > 0 else 0
                baseline_fr_std = np.std(baseline_counts, ddof=1) if len(baseline_counts) > 1 else 0
                stim_fr_std = np.std(stim_counts, ddof=1) if len(stim_counts) > 1 else 0
                sig_flag = p_val < alpha

                results_list.append({
                    "condition": cond,
                    "unit_id": unit,
                    "p_value": p_val,
                    "significant": sig_flag,
                    "baseline_FR_mean": baseline_fr_mean,
                    "stim_FR_mean": stim_fr_mean,
                    "baseline_FR_std": baseline_fr_std,
                    "stim_FR_std": stim_fr_std,
                    "n_pulses": n_events,
                    "best_electrode":self.nwb_ephys_data.units['ccf_location'][unit]['best_electrode'],
                    "shank":self.nwb_ephys_data.units['ccf_location'][unit]['shank'],
                    "probe":self.nwb_ephys_data.units['ccf_location'][unit]['probe'],
                    "estimated_x": self.nwb_ephys_data.units['estimated_x'][unit],
                    "estimated_y": self.nwb_ephys_data.units['estimated_y'][unit],
                })

        results_df = pd.DataFrame(results_list,
                                  columns=["condition", "unit_id", "p_value", "significant",
                                           "baseline_FR_mean", "stim_FR_mean",
                                           "baseline_FR_std", "stim_FR_std", "n_pulses","best_electrode","shank","probe","estimated_x","estimated_y"])
        self.significant_df = results_df
        return results_df

    # ------------------------------------------------------------------ #
    # Robust opto-tagging: SALT + latency/jitter/reliability/effect-size  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bh_correct(pvals):
        """Benjamini-Hochberg FDR correction. NaNs are treated as p=1.0."""
        p = np.asarray(pvals, dtype=float)
        p = np.where(np.isnan(p), 1.0, p)
        n = p.size
        if n == 0:
            return p
        order = np.argsort(p)
        ranked = p[order] * n / (np.arange(n) + 1)
        # Enforce monotonicity from the largest p-value downward.
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(n)
        out[order] = np.clip(ranked, 0.0, 1.0)
        return out

    @staticmethod
    def _js_divergence(P, Q):
        """Jensen-Shannon divergence (base-2) between two distributions."""
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float)
        if P.sum() > 0:
            P = P / P.sum()
        if Q.sum() > 0:
            Q = Q / Q.sum()
        M = 0.5 * (P + Q)

        def _kl(a, b):
            mask = (a > 0) & (b > 0)
            return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

        return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)

    @staticmethod
    def _latency_hist(spike_matrix, nbins):
        """
        First-spike-latency distribution for a (trials x nbins) binary matrix.

        Returns a length ``nbins + 1`` probability vector; the extra final bin
        counts trials with no spike in the window (as in Kvitsiani et al. 2013).
        """
        hist = np.zeros(nbins + 1)
        for row in spike_matrix:
            idx = np.flatnonzero(row)
            if idx.size:
                hist[idx[0]] += 1
            else:
                hist[nbins] += 1
        s = hist.sum()
        if s > 0:
            hist = hist / s
        return hist

    def _salt(self, spt_baseline, spt_test, dt, wn):
        """
        Stimulus-Associated spike Latency Test (Kvitsiani et al., 2013).

        Compares the first-spike-latency distribution in the test window against
        distributions from equally sized baseline epochs using the (square-root)
        Jensen-Shannon divergence. Returns ``(p_value, info_diff)`` where a small
        p-value means the test latency distribution is atypical vs baseline.
        """
        nmbn = int(round(wn / dt))
        if nmbn < 1:
            return 1.0, 0.0
        nbase = spt_baseline.shape[1]
        nm = nbase // nmbn  # number of baseline epochs
        if nm < 2:
            return 1.0, 0.0

        dists = [
            self._latency_hist(spt_baseline[:, i * nmbn:(i + 1) * nmbn], nmbn)
            for i in range(nm)
        ]
        test_dist = self._latency_hist(spt_test[:, :nmbn], nmbn)

        base_jsd = []
        for i in range(nm):
            for j in range(i + 1, nm):
                base_jsd.append(np.sqrt(max(self._js_divergence(dists[i], dists[j]), 0.0) * 2))
        base_jsd = np.array(base_jsd)
        if base_jsd.size == 0:
            return 1.0, 0.0

        test_jsd = np.array([
            np.sqrt(max(self._js_divergence(dists[i], test_dist), 0.0) * 2)
            for i in range(nm)
        ])
        test_stat = np.median(test_jsd)
        p = float(np.mean(base_jsd >= test_stat))
        info = float(test_stat - np.median(base_jsd))
        return p, info

    def _tag_metrics_for_events(self, st, events, b0, b1, w0, w1,
                                base_dur, win_dur, nb, nt, salt_dt):
        """
        Compute opto-tagging metrics for one unit over a set of event times.

        ``st`` MUST be sorted ascending; the baseline/test windows are located
        with ``np.searchsorted`` so only the few spikes inside each tiny window
        are touched (not the whole spike train).

        Returns a dict of metrics (rates, reliability, latency, jitter, effect
        size, and raw t-test / SALT p-values), or None if ``events`` is empty.
        """
        n_events = len(events)
        if n_events == 0:
            return None

        baseline_rates = np.empty(n_events)
        stim_rates = np.empty(n_events)
        latencies = []
        spt_baseline = np.zeros((n_events, nb))
        spt_test = np.zeros((n_events, nt))

        for i, t in enumerate(events):
            # Baseline spikes -> rate + SALT baseline matrix.
            # Window is [t+b0, t+b1); searchsorted restricts to just those spikes.
            b_lo = np.searchsorted(st, t + b0, side="left")
            b_hi = np.searchsorted(st, t + b1, side="left")
            bspk = st[b_lo:b_hi] - (t + b0)
            if bspk.size:
                bidx = np.floor(bspk / salt_dt).astype(int)
                bidx = bidx[(bidx >= 0) & (bidx < nb)]
                spt_baseline[i, bidx] = 1
            baseline_rates[i] = bspk.size / base_dur

            # Test-window spikes -> rate + SALT test matrix + first latency.
            t_lo = np.searchsorted(st, t + w0, side="left")
            t_hi = np.searchsorted(st, t + w1, side="left")
            tspk = st[t_lo:t_hi] - (t + w0)
            if tspk.size:
                tidx = np.floor(tspk / salt_dt).astype(int)
                tidx = tidx[(tidx >= 0) & (tidx < nt)]
                spt_test[i, tidx] = 1
                latencies.append(w0 + tspk.min())
            stim_rates[i] = tspk.size / win_dur

        n_hit = len(latencies)
        reliability = n_hit / n_events
        median_latency = float(np.median(latencies)) if n_hit > 0 else np.nan
        jitter = float(np.std(latencies, ddof=1)) if n_hit > 1 else np.nan

        baseline_fr_mean = float(np.mean(baseline_rates))
        stim_fr_mean = float(np.mean(stim_rates))
        abs_increase = stim_fr_mean - baseline_fr_mean
        if baseline_fr_mean > 0:
            ratio = stim_fr_mean / baseline_fr_mean
        else:
            ratio = np.inf if stim_fr_mean > 0 else 0.0

        # Paired t-test on short-latency rates
        if n_events < 2 or np.allclose(baseline_rates, stim_rates):
            p_ttest = 1.0
        else:
            _, p_ttest = ttest_rel(baseline_rates, stim_rates)
            if np.isnan(p_ttest):
                p_ttest = 1.0

        # SALT test
        p_salt, salt_info = self._salt(spt_baseline, spt_test, salt_dt, win_dur)

        return {
            "n_pulses": n_events,
            "n_hit": n_hit,
            "reliability": reliability,
            "median_latency": median_latency,
            "jitter": jitter,
            "baseline_FR_mean": baseline_fr_mean,
            "stim_FR_mean": stim_fr_mean,
            "abs_increase": abs_increase,
            "effect_ratio": ratio,
            "p_ttest": float(p_ttest),
            "p_salt": p_salt,
            "salt_info": salt_info,
        }

    def compute_tagging_metrics(self, unit_index=None,
                                baseline_window=(-0.05, 0.0),
                                latency_window=(0.0, 0.006),
                                salt_dt=0.001,
                                per_pulse=True,
                                remove_artefacts=True, removal_window=0.002,
                                align_to_event="pulse",
                                progress=True):
        """
        CALCULATION step of robust opto-tagging (no thresholding / selection).

        For every unit x condition it computes, using a SHORT ``latency_window``
        (direct-activation window, default 0-6 ms):
          - paired t-test p-value (baseline vs stim rate)  [``p_ttest``]
          - SALT test p-value (Kvitsiani et al. 2013)       [``p_salt``]
          - effect size: stim/baseline rate ratio + absolute increase
          - reliability: fraction of pulses with >=1 spike in the window
          - first-spike latency (median) and jitter (SD of first-spike latency)

        If ``per_pulse`` is True and ``align_to_event == "pulse"``, the same
        metrics are ALSO computed separately for each pulse position within the
        train. Rows are tagged by ``pulse_index``:
          - ``pulse_index == -1`` : pooled over all pulses in the train
          - ``pulse_index == 0,1,2,...`` : that pulse position only

        No p-value correction and no ``tagged`` flag are added here; use
        :meth:`select_tagged_units` for that.

        Parameters
        ----------
        unit_index : int or list of int, optional
            Units to test. Defaults to all QC-passing units.
        baseline_window : (float, float)
            Pre-event window (s) for baseline rate and SALT baseline epochs.
        latency_window : (float, float)
            Short post-event window (s) for direct activation, latency, jitter,
            reliability, effect size, and the SALT test window.
        salt_dt : float
            Time resolution (s) for the SALT latency histograms.
        per_pulse : bool
            Also compute metrics per pulse position within each train.
        remove_artefacts, removal_window, align_to_event
            As in ``find_significant_units``.

        Returns
        -------
        pd.DataFrame
            One row per unit x condition x pulse_index with the raw metrics.
            Also stored on ``self.metrics_df``.
        """
        if not hasattr(self, "units_passing_qc"):
            print("Error: QC-passing units are missing.")
            return pd.DataFrame()

        if unit_index is None:
            unit_index = self.units_passing_qc.index.tolist()
        elif isinstance(unit_index, int):
            unit_index = [unit_index]
        else:
            unit_index = [u for u in unit_index if u in self.units_passing_qc.index]
        if len(unit_index) == 0:
            print("Error: None of the selected units exist in QC-passing units.")
            return pd.DataFrame()

        event_dict = self._get_event_arrays(align_to_event)
        if event_dict is None:
            return pd.DataFrame()
        event_times = event_dict["event_times"]
        unique_conditions = event_dict["unique_conditions"]
        power_map = event_dict["power_map"]
        location_map = event_dict["location_map"]
        lasername_map = event_dict["lasername_map"]
        cycle_duration_map = event_dict["cycle_duration_map"]
        frequency_map = event_dict["frequency_map"]
        pdur_map = event_dict["pdur_map"]
        pulse_index_map = event_dict.get("pulse_index_map")
        if pulse_index_map is None:
            pulse_index_map = np.zeros(len(event_times), dtype=int)

        filtered_spikes = self._get_filtered_spike_times(unit_index, remove_artefacts, removal_window)
        if filtered_spikes is None:
            return pd.DataFrame()

        # Ensure spike arrays are sorted ascending (the inner metric loop locates
        # windows with searchsorted). NWB spike_times are already sorted, so this
        # is usually a no-op; done once per unit here to avoid re-sorting inside
        # the condition loop.
        for _u in unit_index:
            _sv = np.asarray(filtered_spikes[_u], dtype=float)
            if _sv.size and np.any(np.diff(_sv) < 0):
                _sv = np.sort(_sv)
            filtered_spikes[_u] = _sv

        b0, b1 = baseline_window
        w0, w1 = latency_window
        base_dur = b1 - b0
        win_dur = w1 - w0
        nb = max(int(round(base_dur / salt_dt)), 1)
        nt = max(int(round(win_dur / salt_dt)), 1)

        import time as _time
        _t0 = _time.perf_counter()
        _tag = getattr(self, "session_name", "session")
        n_cond = len(unique_conditions)
        n_units = len(unit_index)
        if progress:
            print(f"[{_tag}] compute_tagging_metrics: {n_units} unit(s) x "
                  f"{n_cond} condition(s)", flush=True)
        _step = max(1, n_units // 4)

        rows = []
        for ci, cond in enumerate(unique_conditions):
            pwr_cond, loc_cond, lname_cond, cycle_cond, freq_cond, pdur_cond = cond
            sel_mask = (
                (power_map == pwr_cond) &
                (location_map == loc_cond) &
                (lasername_map == lname_cond) &
                (cycle_duration_map == cycle_cond) &
                (frequency_map == freq_cond) &
                (pdur_map == pdur_cond)
            )
            idx = np.flatnonzero(sel_mask)
            if idx.size == 0:
                continue
            cond_events = event_times[idx]
            cond_pidx = pulse_index_map[idx]

            # Group 1: pooled over all pulses in the train (pulse_index = -1).
            groups = [(-1, cond_events)]
            # Groups 2..k: one per pulse position within the train.
            if per_pulse and align_to_event == "pulse":
                for p in sorted(set(int(x) for x in cond_pidx)):
                    groups.append((p, cond_events[cond_pidx == p]))

            if progress:
                print(f"[{_tag}]   cond {ci + 1}/{n_cond} "
                      f"(pwr={pwr_cond}, pdur={pdur_cond}, {lname_cond}): "
                      f"{n_units} units x {len(groups)} group(s)", flush=True)

            for ui, unit in enumerate(unit_index):
                st = filtered_spikes[unit]
                ccf = self.nwb_ephys_data.units['ccf_location'][unit]
                est_x = self.nwb_ephys_data.units['estimated_x'][unit]
                est_y = self.nwb_ephys_data.units['estimated_y'][unit]
                for pidx_label, evs in groups:
                    m = self._tag_metrics_for_events(
                        st, evs, b0, b1, w0, w1, base_dur, win_dur, nb, nt, salt_dt
                    )
                    if m is None:
                        continue
                    row = {
                        "session": self.session_name,
                        # Store the condition with native Python types so the
                        # CSV round-trips cleanly (no "np.float64(...)" wrappers).
                        "condition": (
                            float(pwr_cond), float(loc_cond), str(lname_cond),
                            float(cycle_cond), float(freq_cond), float(pdur_cond),
                        ),
                        "unit_id": unit,
                        "pulse_index": pidx_label,
                    }
                    row.update(m)
                    row.update({
                        "best_electrode": ccf['best_electrode'],
                        "shank": ccf['shank'],
                        "probe": ccf['probe'],
                        "estimated_x": est_x,
                        "estimated_y": est_y,
                    })
                    rows.append(row)

                if progress and (ui + 1) % _step == 0:
                    print(f"[{_tag}]     cond {ci + 1}/{n_cond}: "
                          f"{ui + 1}/{n_units} units | "
                          f"{_time.perf_counter() - _t0:.1f}s elapsed", flush=True)

        metrics_df = pd.DataFrame(rows)
        self.metrics_df = metrics_df
        if progress:
            print(f"[{_tag}] compute_tagging_metrics done: {len(metrics_df)} row(s) "
                  f"in {_time.perf_counter() - _t0:.1f}s", flush=True)
        return metrics_df

    def select_tagged_units(self, metrics_df=None,
                            alpha=0.05,
                            effect_ratio=2.0,
                            min_abs_increase=0.0,
                            min_reliability=0.2,
                            max_latency=0.006,
                            max_jitter=0.003,
                            correction="fdr_bh",
                            return_tagged_only=False):
        """
        Thin wrapper around the module-level :func:`select_tagged_units`.

        Defaults ``metrics_df`` to ``self.metrics_df`` and stores the result on
        ``self.tagged_df``. For combining several sessions, call the standalone
        :func:`select_tagged_units` function directly on a concatenated metrics
        table instead (it does not need an ``OpticalTagging`` instance).
        """
        if metrics_df is None:
            metrics_df = getattr(self, "metrics_df", None)
        if metrics_df is None or len(metrics_df) == 0:
            print("Error: no metrics to select from. Run compute_tagging_metrics first.")
            return pd.DataFrame()

        df = select_tagged_units(
            metrics_df,
            alpha=alpha,
            effect_ratio=effect_ratio,
            min_abs_increase=min_abs_increase,
            min_reliability=min_reliability,
            max_latency=max_latency,
            max_jitter=max_jitter,
            correction=correction,
            return_tagged_only=return_tagged_only,
        )
        self.tagged_df = df
        return df

    def find_tagged_units(self, unit_index=None,
                          baseline_window=(-0.05, 0.0),
                          latency_window=(0.0, 0.006),
                          alpha=0.05,
                          effect_ratio=2.0,
                          min_abs_increase=0.0,
                          min_reliability=0.2,
                          max_latency=0.006,
                          max_jitter=0.003,
                          salt_dt=0.001,
                          per_pulse=False,
                          remove_artefacts=True, removal_window=0.002,
                          align_to_event="pulse"):
        """
        Convenience wrapper that runs :meth:`compute_tagging_metrics` followed by
        :meth:`select_tagged_units` in one call. Prefer calling the two steps
        separately when you want to compute/save metrics once and then explore
        different selection criteria.
        """
        metrics_df = self.compute_tagging_metrics(
            unit_index=unit_index,
            baseline_window=baseline_window,
            latency_window=latency_window,
            salt_dt=salt_dt,
            per_pulse=per_pulse,
            remove_artefacts=remove_artefacts,
            removal_window=removal_window,
            align_to_event=align_to_event,
        )
        if metrics_df.empty:
            self.tagged_df = metrics_df
            return metrics_df
        return self.select_tagged_units(
            metrics_df,
            alpha=alpha,
            effect_ratio=effect_ratio,
            min_abs_increase=min_abs_increase,
            min_reliability=min_reliability,
            max_latency=max_latency,
            max_jitter=max_jitter,
        )

    def plot_waveform_mean(self, unit_id, save_path="/root/capsule/scratch/", save_formats=["eps"]):
        """
        Plots the mean waveform for a specified unit.
        
        Parameters
        ----------
        unit_id : int
            The index of the unit (from the NWB units table) for which the mean waveform is plotted.
        save_path : str or None
            Base path (without extension) where to save the figures.
        save_formats : list of str or None
            List of formats to save in, e.g. ["pdf", "eps"]. Supported: "pdf", "eps".
        """
        if self.nwb_ephys_data is None:
            print("Error: No ephys NWB data loaded.")
            return

        if not hasattr(self.nwb_ephys_data, "units"):
            print("Error: NWB file has no units table.")
            return

        units_table = self.nwb_ephys_data.units[:]
        if "waveform_mean" not in units_table.columns:
            print("Error: 'waveform_mean' column not found in units table.")
            return

        if unit_id not in units_table.index:
            print(f"Error: Unit ID {unit_id} not found in the units table.")
            return

        waveform = np.array(units_table.loc[unit_id, "waveform_mean"])
        n_time, n_chans = waveform.shape

        # Figure 1: all channels
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        for ch in range(n_chans):
            ax1.plot(waveform[:, ch], alpha=0.5)
        ax1.set_title(f"Unit {unit_id}: Mean Waveform (All {n_chans} Channels)")
        ax1.set_xlabel("Time (samples)")
        ax1.set_ylabel("Amplitude")
        plt.tight_layout()

        # Figure 2: largest‐amplitude channel
        channel_mins = np.min(waveform, axis=0)
        largest_channel = np.argmin(channel_mins)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(waveform[:, largest_channel], color="red")
        ax2.set_title(f"Unit {unit_id}: Channel with Smallest Value = {largest_channel}")
        ax2.set_xlabel("Time (samples)")
        ax2.set_ylabel("Amplitude")
        plt.tight_layout()

        # Save if requested
        if save_path and save_formats:
            base = save_path.rstrip("/")
            for fmt in save_formats:
                fmt_low = fmt.lower()
                if fmt_low in ("pdf", "eps"):
                    f1 = f"{base}_unit{unit_id}_waveforms_all.{fmt_low}"
                    fig1.savefig(f1, format=fmt_low, dpi=300, bbox_inches="tight")
                    f2 = f"{base}_unit{unit_id}_waveform_ch{largest_channel}.{fmt_low}"
                    fig2.savefig(f2, format=fmt_low, dpi=300, bbox_inches="tight")
                else:
                    print(f"Warning: unsupported save format '{fmt}'")

        # Finally show
        plt.show()

