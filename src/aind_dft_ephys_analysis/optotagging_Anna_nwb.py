"""
NWB-based optotagging analysis.

This is a rewrite of Anna's original ``optotagging_analysis.py`` /
``plotting_funcs.py`` / ``main.py`` pipeline so that it reads everything from an
AIND ephys NWB file (loaded with :class:`NWBUtils`) instead of from raw
Open Ephys folders + SpikeInterface sorting outputs + a ``*opto.csv``.

What changed vs. the original
-----------------------------
Original source of data                     -> NWB source used here
- NIDAQ event stream (laser onsets)         -> optotagging stimulus table in the NWB
- ``*opto.csv`` (trial parameters)          -> same optotagging stimulus table
- SpikeInterface ``sorting_output``         -> ``nwb_data.units['spike_times'][u]``
- waveform extractor templates              -> ``nwb_data.units['waveform_mean'][u]``
- ``extremum_channels``                     -> ``nwb_data.units['extremum_channel_index']``
- probe/stream name                         -> ``nwb_data.units['device_name']``
- ``default_qc`` / ``decoder_label``        -> ``get_units_passed_default_qc(nwb_data)``

Because the exact NWB layout of the optotagging *stimulus* (laser pulse onset
times and their parameters) varies between datasets, the extraction step
(:func:`get_optotagging_events`) is intentionally configurable and there is a
discovery helper (:func:`describe_opto_sources`) that prints the candidate
locations in a given NWB so you can map the field names once.

Metric functions (:func:`calculate_laser_response_latency`,
:func:`calculate_pulse_train_responses`) are ported faithfully from the original
code; spike alignment is done with plain NumPy (via :func:`to_events`) to match
the rest of this repository (see ``create_psth.py``).
"""

from __future__ import annotations

import os
import glob
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.multitest import multipletests

from nwb_utils import NWBUtils
from ephys_behavior import get_units_passed_default_qc
from ephys_utils import find_best_electrode
from general_utils import extract_session_name_core


# ============================================================================
# Spike alignment (NumPy replacement for aind_ephys_utils.align.to_events)
# ============================================================================
def to_events(
    spike_times: Sequence[float],
    event_times: Sequence[float],
    time_range: Sequence[float],
    bin_size: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, None]:
    """
    Align spike times to a set of event times.

    This mirrors the two calling conventions used in the original code:

    * ``bin_size is None`` -> returns ``(offsets, event_inds, None)`` where
      ``offsets`` are per-spike times relative to each event (only spikes within
      ``time_range`` are kept) and ``event_inds`` gives the trial index of each
      returned spike. Reproduces
      ``offsets, event_inds, _ = align.to_events(...)``.
    * ``bin_size`` given -> returns ``(bin_centers, counts, None)`` where
      ``counts`` has shape ``(n_bins, n_trials)``. Reproduces
      ``_, counts, _ = align.to_events(..., bin_size=...)``.

    Parameters
    ----------
    spike_times : array-like
        Spike timestamps (seconds), assumed sorted or not — both work.
    event_times : array-like
        Event (laser onset) timestamps (seconds).
    time_range : (float, float)
        ``[start, stop]`` window relative to each event (seconds).
    bin_size : float, optional
        Bin width (seconds). If None, per-spike offsets are returned.

    Returns
    -------
    (np.ndarray, np.ndarray, None)
    """
    st = np.asarray(spike_times, dtype=float)
    et = np.asarray(event_times, dtype=float)
    start, stop = float(time_range[0]), float(time_range[1])

    if bin_size is None:
        off_list: List[np.ndarray] = []
        id_list: List[np.ndarray] = []
        for i, t0 in enumerate(et):
            rel = st - t0
            mask = (rel >= start) & (rel <= stop)
            off_list.append(rel[mask])
            id_list.append(np.full(int(mask.sum()), i, dtype=int))
        offsets = np.concatenate(off_list) if off_list else np.array([], dtype=float)
        event_inds = np.concatenate(id_list) if id_list else np.array([], dtype=int)
        return offsets, event_inds, None

    edges = np.arange(start, stop + bin_size, bin_size)
    if len(edges) < 2:
        edges = np.array([start, stop], dtype=float)
    centers = edges[:-1] + np.diff(edges) / 2.0
    counts = np.zeros((len(centers), len(et)), dtype=float)
    for i, t0 in enumerate(et):
        rel = st - t0
        c, _ = np.histogram(rel, bins=edges)
        counts[:, i] = c
    return centers, counts, None


# ============================================================================
# Optotagging stimulus discovery + extraction
# ============================================================================
# Column names we try to map onto a canonical schema. Extend as needed.
_CANONICAL_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "time": ("time", "start_time", "onset", "onset_time", "laser_onset", "timestamps"),
    "type": ("type", "trial_type", "stim_type", "condition"),
    "power": ("power", "laser_power", "power_mW", "power_mw"),
    "wavelength": ("wavelength", "laser_wavelength", "color"),
    "site": ("site", "emission_site", "channel"),
    "duration": ("duration", "pulse_duration", "pulse_width"),
    "num_pulses": ("num_pulses", "n_pulses", "pulse_count", "num_pulse"),
    "pulse_interval": ("pulse_interval", "inter_pulse_interval", "ipi"),
    "interval": ("interval", "iti", "inter_trial_interval"),
    "emission_location": ("emission_location", "location", "target", "probe", "emission_probe"),
    "param_group": ("param_group", "group", "protocol"),
}


def describe_opto_sources(nwb_data: Any) -> None:
    """
    Print the candidate locations of optotagging stimulus info in an NWB file.

    Use this once to figure out which table/module holds the laser pulse onset
    times and parameters, then pass its name to :func:`get_optotagging_events`.
    """
    print("=" * 70)
    print("Scanning NWB for optotagging stimulus sources")
    print("=" * 70)

    # 1) intervals / epochs tables
    intervals = getattr(nwb_data, "intervals", None)
    if intervals:
        print("\n[intervals]")
        for name in list(intervals.keys()):
            try:
                cols = list(intervals[name].colnames)
            except Exception:
                cols = "<could not read columns>"
            print(f"  - intervals['{name}']  columns={cols}")
    else:
        print("\n[intervals] none")

    # 2) trials table (foraging opto lives here as laser_on_trial)
    trials = getattr(nwb_data, "trials", None)
    if trials is not None:
        try:
            cols = list(trials.colnames)
        except Exception:
            cols = "<could not read columns>"
        laser_cols = [c for c in cols if any(k in c.lower() for k in ("laser", "opto", "pulse"))]
        print(f"\n[trials] columns={cols}")
        if laser_cols:
            print(f"         laser/opto-related columns: {laser_cols}")

    # 3) processing modules
    processing = getattr(nwb_data, "processing", None)
    if processing:
        print("\n[processing]")
        for mod_name in list(processing.keys()):
            mod = processing[mod_name]
            data_ifs = list(getattr(mod, "data_interfaces", {}).keys())
            print(f"  - processing['{mod_name}']  data_interfaces={data_ifs}")

    # 4) stimulus (OptogeneticSeries etc.)
    for attr in ("stimulus", "stimulus_template"):
        container = getattr(nwb_data, attr, None)
        if container:
            print(f"\n[{attr}]")
            try:
                for name in list(container.keys()):
                    obj = container[name]
                    print(f"  - {attr}['{name}']  type={type(obj).__name__}")
            except Exception:
                print(f"  - {attr}: {type(container).__name__}")

    print("\n" + "=" * 70)
    print("Next: call get_optotagging_events(nwb_data, source=..., table_name=...)")
    print("and, if needed, column_map={'time': '<col>', 'power': '<col>', ...}")
    print("=" * 70)


def _map_columns(df: pd.DataFrame, column_map: Optional[Dict[str, str]]) -> pd.DataFrame:
    """
    Rename columns of ``df`` to the canonical schema.

    Explicit ``column_map`` (canonical -> actual) wins; otherwise best-effort
    matching against ``_CANONICAL_COLUMNS`` is used.
    """
    out = df.copy()
    lower = {c.lower(): c for c in out.columns}
    rename: Dict[str, str] = {}

    if column_map:
        for canon, actual in column_map.items():
            if actual in out.columns:
                rename[actual] = canon
            elif actual.lower() in lower:
                rename[lower[actual.lower()]] = canon

    for canon, candidates in _CANONICAL_COLUMNS.items():
        if canon in rename.values():
            continue
        for cand in candidates:
            if cand in out.columns:
                rename[cand] = canon
                break
            if cand.lower() in lower:
                rename[lower[cand.lower()]] = canon
                break

    return out.rename(columns=rename)


def get_optotagging_events(
    nwb_data: Any,
    source: str = "intervals",
    table_name: Optional[str] = None,
    column_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Extract the optotagging stimulus table (laser onsets + parameters) from NWB.

    Parameters
    ----------
    nwb_data : NWB file handle
    source : {'intervals', 'trials', 'processing'}
        Where to look. Use :func:`describe_opto_sources` first if unsure.
    table_name : str, optional
        Name of the intervals table or processing data-interface holding the
        opto stimulus. Required for 'intervals' (unless there is exactly one)
        and 'processing'.
    column_map : dict, optional
        Explicit mapping from canonical field -> actual column name, e.g.
        ``{'time': 'start_time', 'power': 'laser_power', 'emission_location': 'probe'}``.

    Returns
    -------
    pd.DataFrame
        One row per optotagging trial/pulse-train, indexed 0..N-1, with a
        ``time`` column (onset, seconds) and whatever parameter columns are
        available, renamed to the canonical schema.
    """
    if source == "intervals":
        intervals = getattr(nwb_data, "intervals", None)
        if not intervals:
            raise ValueError("NWB has no 'intervals'. Run describe_opto_sources(nwb_data).")
        if table_name is None:
            names = list(intervals.keys())
            if len(names) != 1:
                raise ValueError(
                    f"Multiple intervals tables {names}; pass table_name=..."
                )
            table_name = names[0]
        df = intervals[table_name].to_dataframe()

    elif source == "trials":
        trials = getattr(nwb_data, "trials", None)
        if trials is None:
            raise ValueError("NWB has no 'trials'.")
        df = trials.to_dataframe()

    elif source == "processing":
        processing = getattr(nwb_data, "processing", None)
        if not processing or table_name is None:
            raise ValueError("source='processing' requires table_name='<module>/<interface>'.")
        mod_name, _, if_name = table_name.partition("/")
        mod = processing[mod_name]
        obj = mod.data_interfaces[if_name] if if_name else mod
        df = obj.to_dataframe() if hasattr(obj, "to_dataframe") else pd.DataFrame(obj[:])

    else:
        raise ValueError(f"Unknown source '{source}'.")

    df = _map_columns(df.reset_index(drop=True), column_map)
    if "time" not in df.columns:
        raise ValueError(
            "Could not find an onset-time column. Pass column_map={'time': '<col>'}. "
            f"Available columns: {list(df.columns)}"
        )
    return df.reset_index(drop=True)


# ============================================================================
# Raw NIDAQ onsets + opto CSV parameters (Anna's original sources)
# ============================================================================
def find_recording_clipped_folder(
    session_name: str,
    data_root: str = "/root/capsule/data",
) -> Optional[str]:
    """
    Locate the raw Open Ephys ``ecephys_clipped`` folder for a session.

    Mirrors ``main.py``'s ``glob('/data/ecephys_*/**/ecephys_clipped/')`` but
    scoped to a single session core (e.g. '839480_2026-06-03_15-09-14').

    Returns the folder path, or None if not found. Warns on multiple matches.
    """
    core = extract_session_name_core(session_name)
    patterns = [
        os.path.join(data_root, f"ecephys_{core}*", "**", "ecephys_clipped"),
        os.path.join(data_root, f"*{core}*", "**", "ecephys_clipped"),
    ]
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))
        if matches:
            break
    # keep only real directories, drop duplicates and any '*_sorted*' assets
    matches = sorted({m for m in matches if os.path.isdir(m) and "sorted" not in m})
    if not matches:
        print(f"Warning: no ecephys_clipped folder found for '{core}' under {data_root}.")
        return None
    if len(matches) > 1:
        print(f"Warning: multiple ecephys_clipped folders for '{core}': {matches}. Using first.")
    return matches[0]


def get_laser_onsets_from_nidaq(
    recording_clipped_folder: str,
    event_id: str = "2",
    opto_recording: int = 0,
    flip_NIDAQ: bool = False,
    channel_id: str = "PXIe-6341Digital Input Line",
    block_index: int = 0,
    expected_n: Optional[int] = None,
    search_all_segments: bool = True,
) -> np.ndarray:
    """
    Read laser-onset times from the raw Open Ephys NIDAQ events (channel 2).

    Direct port of Anna's ``_get_laser_onset_times``: reads the NIDAQ digital
    input line via SpikeInterface and keeps events whose label equals
    ``event_id`` ('2' = channel 2), optionally applying a 0.5 s correction when
    the sync signal was flipped.

    Some sessions contain more than one recording under the same experiment
    (``.../experiment1/recording1``, ``recording2``, ...). These map to
    SpikeInterface *segments*, and the laser stimulation is not always in the
    first one. When the requested ``opto_recording`` segment has no matching
    laser events (or a count that disagrees with ``expected_n``), the other
    segments are searched and the best-matching one is used instead.

    Parameters
    ----------
    recording_clipped_folder : str
        Path to the Open Ephys ``ecephys_clipped`` folder (contains the
        ``Record Node .../experiment1/recordingN`` tree).
    event_id : str, default '2'
        NIDAQ digital-input line label to keep (channel 2).
    opto_recording : int, default 0
        Segment index of the recording to try first.
    flip_NIDAQ : bool, default False
        If True, subtract 0.5 s from every onset (flipped sync correction).
    expected_n : int, optional
        Expected number of laser onsets (e.g. the number of rows in the opto
        CSV). When given, a segment whose event count matches it is preferred.
    search_all_segments : bool, default True
        If True, fall back to the other recording segments when the requested
        one yields no / mismatched laser events.

    Returns
    -------
    np.ndarray
        Laser onset times (seconds), one per detected event.
    """
    import spikeinterface.extractors as se  # lazy import (heavy dependency)

    event = se.read_openephys_event(recording_clipped_folder, block_index=block_index)
    adjustment = 0.5 if flip_NIDAQ else 0.0

    def onsets_for_segment(seg: int) -> np.ndarray:
        events = event.get_events(channel_id=channel_id, segment_index=seg)
        laser_pulses = events[events["label"] == event_id]
        return np.asarray(laser_pulses["time"], dtype=float) - adjustment

    try:
        onsets = onsets_for_segment(opto_recording)
    except Exception:
        onsets = np.array([], dtype=float)

    # Fall back to the other recording segments when the requested one has no
    # laser events, or fewer than expected (e.g. the stim lives in recording2).
    need_search = search_all_segments and (
        onsets.size == 0 or (expected_n is not None and onsets.size != expected_n)
    )
    if need_search:
        try:
            n_seg = int(event.get_num_segments())
        except Exception:
            n_seg = 1
        best_seg, best_onsets = opto_recording, onsets
        for seg in range(n_seg):
            if seg == opto_recording:
                continue
            try:
                cand = onsets_for_segment(seg)
            except Exception:
                continue
            if expected_n is not None:
                # Prefer an exact match to the opto-CSV row count.
                if cand.size == expected_n:
                    best_seg, best_onsets = seg, cand
                    break
                if best_onsets.size != expected_n and cand.size > best_onsets.size:
                    best_seg, best_onsets = seg, cand
            elif cand.size > best_onsets.size:
                best_seg, best_onsets = seg, cand
        if best_seg != opto_recording and best_onsets.size:
            print(
                f"[laser onsets] segment {opto_recording} had {onsets.size} matching "
                f"event(s); using segment {best_seg} with {best_onsets.size} instead."
            )
        onsets = best_onsets

    return np.asarray(onsets, dtype=float)


def read_opto_trials_csv(
    recording_clipped_folder: Optional[str] = None,
    trials_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the optotagging parameter table (``*opto.csv``), as in Anna's code.

    Parameters
    ----------
    recording_clipped_folder : str, optional
        Folder to search for ``*opto.csv`` when ``trials_csv`` is not given.
    trials_csv : str, optional
        Explicit path to the opto CSV. Takes precedence.

    Returns
    -------
    pd.DataFrame
        The opto trials table (``pd.read_csv(..., index_col=0)``), preserving
        Anna's original column names (type, param_group, power, site, duration,
        num_pulses, pulse_interval, interval, emission_location, wavelength, ...).
    """
    if trials_csv is None:
        if recording_clipped_folder is None:
            raise ValueError("Provide trials_csv or recording_clipped_folder.")
        candidates = glob.glob(os.path.join(recording_clipped_folder, "*opto.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No '*opto.csv' found in {recording_clipped_folder}."
            )
        if len(candidates) > 1:
            print(f"Warning: multiple opto CSVs {candidates}. Using first.")
        trials_csv = candidates[0]
    return pd.read_csv(trials_csv, index_col=0)


# ============================================================================
# Per-unit access helpers
# ============================================================================
def get_unit_spike_times(nwb_data: Any, unit_index: int) -> np.ndarray:
    """Return the spike-time array (seconds) for a single unit index."""
    return np.asarray(nwb_data.units["spike_times"][unit_index], dtype=float)


def get_unit_peak_channels(nwb_data: Any) -> np.ndarray:
    """
    Return the peak (extremum) electrode index for every unit.

    Uses the precomputed ``extremum_channel_index`` column if present, otherwise
    derives it from ``waveform_mean`` via :func:`find_best_electrode`.
    """
    units = nwb_data.units
    if "extremum_channel_index" in units.colnames:
        return np.asarray(units["extremum_channel_index"].data[:], dtype=int)
    wm = units["waveform_mean"].data[:]
    return np.asarray(find_best_electrode(wm, unit_index=None), dtype=int)


def get_unit_probes(nwb_data: Any) -> np.ndarray:
    """Return the probe/device name for every unit (from ``device_name``)."""
    units = nwb_data.units
    if "device_name" in units.colnames:
        return np.asarray(units["device_name"][:])
    return np.array([""] * len(units), dtype=object)


def get_stream_names(nwb_data: Any) -> List[str]:
    """List the unique probe/device names present in the units table."""
    return sorted(set(map(str, get_unit_probes(nwb_data).tolist())))


# ============================================================================
# Metric functions (ported from the original optotagging_analysis.py)
# ============================================================================
def calculate_laser_response_latency(
    unit_spike_times: np.ndarray,
    this_trials_timestamps: np.ndarray,
    laser_start_times: Sequence[float],
    full_time_range: Sequence[float],
    bin_size: float = 0.001,
    sigma: float = 2.0,
    smooth_win_size: int = 3,
    ignore_onset: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-pulse response latency, median time-to-first-spike, and jitter.

    Ported from the original; alignment uses :func:`to_events`.
    """
    num_pulses = len(laser_start_times)
    all_latencies = np.full(num_pulses, np.nan)
    all_latencies_median = np.full(num_pulses, np.nan)
    all_jitter = np.full(num_pulses, np.nan)

    win = np.concatenate((np.zeros(smooth_win_size), np.ones(smooth_win_size)))
    win = win / np.sum(win)

    baseline_time_range = [full_time_range[0], laser_start_times[0]]
    _, baseline_spike_counts, _ = to_events(
        unit_spike_times, this_trials_timestamps, baseline_time_range,
        bin_size=np.diff(baseline_time_range)[0],
    )
    baseline_spike_rate = np.mean(baseline_spike_counts) / (baseline_time_range[-1] - baseline_time_range[0])
    baseline_rate_stdev = np.std(baseline_spike_counts / (baseline_time_range[-1] - baseline_time_range[0]))
    threshold_spike_rate = baseline_spike_rate + sigma * baseline_rate_stdev

    for ind_pulse, _ in enumerate(laser_start_times):
        if ind_pulse == len(laser_start_times) - 1:
            this_range = [laser_start_times[ind_pulse], full_time_range[-1]]
        else:
            this_range = [laser_start_times[ind_pulse], laser_start_times[ind_pulse + 1]]
        if ignore_onset:
            this_range[0] = this_range[0] + bin_size

        first_spikes, first_ids, _ = to_events(unit_spike_times, this_trials_timestamps, this_range)
        this_bin_edges, pulse_counts, _ = to_events(
            unit_spike_times, this_trials_timestamps, this_range, bin_size=bin_size
        )
        average_response = np.mean(pulse_counts, axis=1) / bin_size
        smooth_psth = np.convolve(average_response, win, mode="same")
        responsive_inds = np.flatnonzero(smooth_psth > threshold_spike_rate)
        if len(responsive_inds) > 0:
            i0 = responsive_inds[0]
            y_diff = smooth_psth[i0] - smooth_psth[i0 - 1]
            y_frac = (threshold_spike_rate - smooth_psth[i0 - 1]) / y_diff if y_diff != 0 else 0.0
            latency = this_bin_edges[i0] + y_frac * bin_size - this_range[0]
            if ignore_onset:
                latency = latency + bin_size
            all_latencies[ind_pulse] = latency

        if len(first_spikes) > 0:
            first_per_trial = []
            for tid in np.unique(first_ids):
                k = np.where(first_ids == tid)[0][0]
                first_per_trial.append(first_spikes[k])
            if len(first_per_trial) > 1:
                all_latencies_median[ind_pulse] = np.median(first_per_trial)
                all_jitter[ind_pulse] = np.std(first_per_trial)

    return all_latencies, all_latencies_median, all_jitter


def calculate_pulse_train_responses(
    unit_spike_times: np.ndarray,
    this_trials_timestamps: np.ndarray,
    laser_time_ranges: Sequence[Sequence[float]],
    baseline_time_range: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-pulse significance (Holm-corrected Wilcoxon vs baseline) and reliability.

    Ported from the original; alignment uses :func:`to_events`.
    """
    num_pulses = len(laser_time_ranges)
    all_pvals = np.ones(num_pulses)
    all_reliability = np.zeros(num_pulses)

    _, baseline_spike_counts, _ = to_events(
        unit_spike_times, this_trials_timestamps, baseline_time_range,
        bin_size=np.diff(baseline_time_range)[0],
    )
    base_rate = baseline_spike_counts.flatten() / np.diff(baseline_time_range)

    for ind_pulse, this_range in enumerate(laser_time_ranges):
        _, pulse_counts, _ = to_events(
            unit_spike_times, this_trials_timestamps, this_range,
            bin_size=np.diff(this_range)[0],
        )
        pulse_rate = pulse_counts.flatten() / np.diff(this_range)
        try:
            _, p = stats.wilcoxon(pulse_rate, base_rate, alternative="greater")
        except ValueError:
            p = 1.0
        all_pvals[ind_pulse] = p
        all_reliability[ind_pulse] = np.count_nonzero(pulse_counts) / pulse_counts.size

    responsive, corrected_pvals, _, _ = multipletests(all_pvals, method="holm")
    return responsive, corrected_pvals, all_reliability


# ============================================================================
# Analysis driver
# ============================================================================
class OptotaggingAnalysisNWB:
    """
    NWB-based reimplementation of the original ``OptotaggingAnalysis`` class.

    Spike times, waveforms, peak channels and QC come from the ephys NWB; laser
    onset times come from the raw Open Ephys NIDAQ events (channel 2) and the
    stimulation parameters from the ``*opto.csv`` — exactly Anna's original
    sources.

    Parameters
    ----------
    nwb_data : NWB handle, optional
        Preloaded ephys NWB. If None, it is loaded via
        ``NWBUtils.read_ephys_nwb(session_name=session_name)``.
    session_name : str, optional
        Session identifier used to load the NWB (if needed), locate the raw
        ``ecephys_clipped`` folder, and name output files.
    recording_clipped_folder : str, optional
        Path to the raw Open Ephys ``ecephys_clipped`` folder (NIDAQ events +
        ``*opto.csv``). If None, resolved from ``session_name`` under
        ``data_root``.
    trials_csv : str, optional
        Explicit path to the opto CSV. If None, found as ``*opto.csv`` inside
        ``recording_clipped_folder``.
    data_root : str
        Base folder to search for the raw asset (default '/root/capsule/data').
    laser_event_id : str, default '2'
        NIDAQ digital-input label to keep (channel 2).
    opto_recording : int, default 0
        Segment index of the recording containing the laser stimulation.
    flip_NIDAQ : bool, default False
        If True, subtract 0.5 s from every laser onset.
    alpha : float
        Significance threshold for a pulse to count as "significant".
    """

    def __init__(
        self,
        nwb_data: Any = None,
        session_name: Optional[str] = None,
        recording_clipped_folder: Optional[str] = None,
        trials_csv: Optional[str] = None,
        data_root: str = "/root/capsule/data",
        laser_event_id: str = "2",
        opto_recording: int = 0,
        flip_NIDAQ: bool = False,
        alpha: float = 0.05,
    ):
        if nwb_data is None:
            if session_name is None:
                raise ValueError("Provide either nwb_data or session_name.")
            nwb_data = NWBUtils.read_ephys_nwb(session_name=session_name)
            if nwb_data is None:
                raise RuntimeError(f"Failed to load ephys NWB for '{session_name}'.")
        self.nwb_data = nwb_data
        self.session = session_name or str(getattr(nwb_data, "session_id", "session"))
        self.alpha = alpha
        self.opto_recording = opto_recording

        # locate the raw Open Ephys asset (NIDAQ events + opto CSV)
        if recording_clipped_folder is None:
            recording_clipped_folder = find_recording_clipped_folder(
                self.session, data_root=data_root
            )
        if recording_clipped_folder is None and trials_csv is None:
            raise RuntimeError(
                "Could not locate the raw ecephys_clipped folder; pass "
                "recording_clipped_folder=... (and trials_csv=...) explicitly."
            )
        self.recording_clipped_folder = recording_clipped_folder

        # trial parameters from the opto CSV (Anna's column names preserved)
        self.trial_ids = read_opto_trials_csv(recording_clipped_folder, trials_csv)

        # laser onsets from the raw NIDAQ channel-2 events
        self.laser_onset_times = get_laser_onsets_from_nidaq(
            recording_clipped_folder,
            event_id=laser_event_id,
            opto_recording=opto_recording,
            flip_NIDAQ=flip_NIDAQ,
            expected_n=len(self.trial_ids),
        )
        if len(self.laser_onset_times) != len(self.trial_ids):
            print(
                f"Warning: {len(self.laser_onset_times)} NIDAQ onsets vs "
                f"{len(self.trial_ids)} opto-CSV rows — these should match. "
                "Check laser_event_id / opto_recording / flip_NIDAQ."
            )

        # per-unit metadata (from the NWB)
        self.qc_units = np.asarray(get_units_passed_default_qc(nwb_data), dtype=int)
        self.peak_channels = get_unit_peak_channels(nwb_data)
        self.unit_probes = get_unit_probes(nwb_data)

    # -- probe helpers -------------------------------------------------------
    def get_stream_names(self) -> List[str]:
        return get_stream_names(self.nwb_data)

    def _units_on_probe(self, probe: Optional[str]) -> np.ndarray:
        qc_units = np.asarray(self.qc_units, dtype=int)
        if probe is None:
            return qc_units
        probes = self.unit_probes
        mask = np.array(
            [str(probes[u]) == str(probe) for u in qc_units], dtype=bool
        )
        return qc_units[mask]

    # -- pulse timing helper -------------------------------------------------
    @staticmethod
    def _pulse_time_ranges(duration_ms: float, interval_ms: float, num_pulses: int,
                           params_in_ms: bool = True) -> List[List[float]]:
        scale = 1000.0 if params_in_ms else 1.0
        dur = duration_ms / scale
        gap = interval_ms / scale
        return [[k * (dur + gap), k * (dur + gap) + dur] for k in range(int(num_pulses))]

    # -- main computation ----------------------------------------------------
    def one_probe_laser_responses(
        self,
        trials_query: Dict[str, Iterable],
        probe: Optional[str] = None,
        suffixes: Optional[List[Optional[str]]] = None,
        ignore_onset_offset: bool = True,
        params_in_ms: bool = True,
        pre_opto_duration: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Compute laser-response metrics for all QC units on ``probe``.

        Returns a DataFrame with one row per unit and columns following the
        original naming convention consumed by the downstream best-power step,
        e.g. ``{type}_train_{power}mW_num_sig_pulses``, ``..._mean_latency``,
        ``..._latency_range``, ``..._mean_time_to_first_spike``,
        ``..._mean_jitter``, ``..._mean_reliability``, plus ``peak_channel``,
        ``pre_stim_isi_ratio`` and ``pre_stim_spike_rate``.
        """
        from itertools import product

        units = self._units_on_probe(probe)
        metrics = pd.DataFrame({"unit_id": units})
        param_names = list(trials_query.keys())
        combos = list(product(*[list(v) for v in trials_query.values()]))

        for row_i, unit in enumerate(units):
            spikes = get_unit_spike_times(self.nwb_data, int(unit))
            metrics.at[row_i, "peak_channel"] = int(self.peak_channels[unit])

            # pre-stim QC window
            if pre_opto_duration is not None and len(self.laser_onset_times) > 0:
                stim_start = self.laser_onset_times[0] - 1.0
                spont_start = stim_start - pre_opto_duration
                lo, hi = np.searchsorted(np.sort(spikes), [spont_start, stim_start])
                n_spont = int(hi - lo)
                metrics.at[row_i, "pre_stim_spike_rate"] = n_spont / pre_opto_duration
                metrics.at[row_i, "pre_stim_isi_ratio"] = _isi_violations_ratio(
                    np.sort(spikes)[lo:hi], pre_opto_duration
                )

            for combo in combos:
                query = " and ".join(
                    f"{name} == {_q(val)}" for name, val in zip(param_names, combo)
                )
                if probe is not None and "emission_location" in self.trial_ids.columns:
                    query += f" and emission_location == {_q(probe)}"

                try:
                    sel = self.trial_ids.query(query)
                except Exception:
                    sel = self.trial_ids.iloc[0:0]
                if len(sel) == 0:
                    continue

                col = _column_name(combo, suffixes)
                onsets = self.laser_onset_times[sel.index.to_numpy()]

                duration = float(np.unique(sel.get("duration", pd.Series([5.0])))[0])
                num_pulses = int(np.unique(sel.get("num_pulses", pd.Series([5])))[0])
                pulse_interval = float(np.unique(sel.get("pulse_interval", pd.Series([duration])))[0])
                iti = float(np.unique(sel.get("interval", pd.Series([1.0])))[0])

                pulse_ranges = self._pulse_time_ranges(
                    duration, pulse_interval, num_pulses, params_in_ms
                )
                full_range = [-max(iti, pulse_ranges[-1][1]), pulse_ranges[-1][1]]
                baseline_range = [-iti, 0.0]

                responsive, _, reliability = calculate_pulse_train_responses(
                    spikes, onsets, pulse_ranges, baseline_range
                )
                latencies, ttfs, jitter = calculate_laser_response_latency(
                    spikes, onsets,
                    [r[0] for r in pulse_ranges], full_range,
                    ignore_onset=ignore_onset_offset,
                )

                metrics.at[row_i, f"{col}_num_sig_pulses"] = int(np.nansum(responsive))
                metrics.at[row_i, f"{col}_mean_latency"] = _safe_nanmean(latencies)
                lat_valid = latencies[~np.isnan(latencies)]
                metrics.at[row_i, f"{col}_latency_range"] = (
                    float(np.nanmax(lat_valid) - np.nanmin(lat_valid)) if lat_valid.size else np.nan
                )
                metrics.at[row_i, f"{col}_mean_time_to_first_spike"] = _safe_nanmean(ttfs)
                metrics.at[row_i, f"{col}_mean_jitter"] = _safe_nanmean(jitter)
                metrics.at[row_i, f"{col}_mean_reliability"] = _safe_nanmean(reliability)

        return metrics

    # -- best-power collapse (ported from main.py) ---------------------------
    @staticmethod
    def add_best_power_columns(metrics: pd.DataFrame, trial_types: Iterable[str]) -> pd.DataFrame:
        """
        For each trial type, pick each unit's best power (most significant pulses)
        and copy its metrics into ``{type}_train_best_*`` columns.
        """
        cols = list(metrics.columns)
        suffix = "_num_sig_pulses"
        for trial_type in trial_types:
            sig_cols = [c for c in cols if trial_type in c and c.endswith(suffix)]
            for i in metrics.index:
                best_val, best_col = 0, None
                for c in sig_cols:
                    v = metrics.at[i, c]
                    if pd.notna(v) and v > best_val:
                        best_val, best_col = v, c
                if best_col is None:
                    continue
                # Derive the exact column base from the winning column so we do
                # not assume a particular parameter naming (e.g. a 'train' token).
                base = best_col[: -len(suffix)]
                power_tok = next((p for p in base.split("_") if p.endswith("mW")), None)
                if power_tok is not None:
                    metrics.at[i, f"{trial_type}_train_best_power"] = float(power_tok[:-2])
                for metric in ("num_sig_pulses", "mean_latency", "latency_range",
                               "mean_time_to_first_spike", "mean_jitter", "mean_reliability"):
                    src = f"{base}_{metric}"
                    if src in metrics.columns:
                        dst = f"{trial_type}_train_max_num_sig_pulses" if metric == "num_sig_pulses" \
                            else f"{trial_type}_train_best_{metric}"
                        metrics.at[i, dst] = metrics.at[i, src]
        return metrics


# ============================================================================
# Small utilities
# ============================================================================
def _q(val: Any) -> str:
    """Quote a value for use in a pandas ``DataFrame.query`` expression."""
    if isinstance(val, str):
        return f"'{val}'"
    return str(val)


def _safe_nanmean(arr: Any) -> float:
    """Mean over non-NaN entries; NaN for empty / all-NaN input (no warning)."""
    a = np.asarray(arr, dtype=float)
    if a.size == 0 or np.all(np.isnan(a)):
        return float("nan")
    return float(np.nanmean(a))


def _column_name(param_values: Sequence[Any], suffixes: Optional[Sequence[Optional[str]]]) -> str:
    """Reproduce the original ``_construct_one_column_name`` behavior."""
    parts = []
    for i, v in enumerate(param_values):
        chunk = f"{v}" if i == 0 else f"_{v}"
        if suffixes is not None and i < len(suffixes) and suffixes[i] is not None:
            chunk = chunk + suffixes[i]
        parts.append(chunk)
    return "".join(parts)


def _isi_violations_ratio(spike_times: np.ndarray, duration: float,
                          isi_threshold: float = 0.0015,
                          min_isi: float = 0.0) -> float:
    """
    Refractory-period ISI-violation ratio (Hill et al.), NumPy implementation.

    Mirrors the quantity used by the original ``isi_violations`` call so a
    per-unit ``pre_stim_isi_ratio`` can be computed from spike times alone.
    """
    st = np.sort(np.asarray(spike_times, dtype=float))
    n = st.size
    if n < 2 or duration <= 0:
        return np.nan
    isis = np.diff(st)
    if min_isi > 0:
        isis = isis[isis > min_isi]
    n_viol = int(np.sum(isis < isi_threshold))
    viol_time = 2.0 * n * (isi_threshold - min_isi)
    total_rate = n / duration
    if total_rate == 0:
        return np.nan
    viol_rate = n_viol / viol_time if viol_time > 0 else np.nan
    return float(viol_rate / total_rate)
