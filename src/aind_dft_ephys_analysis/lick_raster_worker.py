# lick_raster_worker.py
from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless in workers
import matplotlib.pyplot as plt

from nwb_utils import NWBUtils
from behavior_utils import extract_event_timestamps


# ---- Shared config (edit if needed) ----
OUTDIR = Path("/root/capsule/scratch/lick_raster_plot")

TIME_WINDOW = (-1.0, 1.0)      # seconds around each lick
BIN_SIZE = 0.02               # seconds
SAVE_FORMATS = ("png",)       # e.g. ("png", "eps")
MAX_RASTER_EVENTS = 500        # cap raster rows (PSTH always uses all events)
UNITS: Optional[Sequence[int]] = None  # None = all QC-passed units


def session_core_from_folder(sorted_folder: str) -> str:
    """'ecephys_839480_2026-06-02_16-20-58_sorted_...' -> '839480_2026-06-02_16-20-58'."""
    core = sorted_folder.split("_sorted")[0]
    if core.startswith("ecephys_"):
        core = core[len("ecephys_"):]
    return core


def get_units_passing_qc(nwb):
    """Return the QC-passed units table from a loaded ephys NWB.

    Mirrors OpticalTagging.get_units_passed_default_qc: keep units whose
    ``default_qc`` is True and whose ``decoder_label`` is not 'noise'.
    """
    units = nwb.units[:]
    return units[
        ((units.default_qc == "True") | (units.default_qc == True)) &
        (units.decoder_label != "noise")
    ]


def compute_aligned_raster_psth(spike_times, event_times, time_window, bin_size):
    """Align spikes to event_times; return per-event offsets and PSTH mean/SEM."""
    spike_times = np.asarray(spike_times, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    event_times = event_times[~np.isnan(event_times)]
    event_times = np.sort(event_times)

    bins = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
    centers = bins[:-1] + bin_size / 2.0

    per_event = []
    counts_mat = np.zeros((len(event_times), len(centers)), dtype=float)
    for i, t0 in enumerate(event_times):
        rel = spike_times[(spike_times >= t0 + time_window[0]) &
                          (spike_times <= t0 + time_window[1])] - t0
        per_event.append(rel)
        counts_mat[i], _ = np.histogram(rel, bins=bins)

    if len(event_times) > 0:
        fr = counts_mat.mean(axis=0) / bin_size
    else:
        fr = np.zeros_like(centers)
    if len(event_times) > 1:
        sem = counts_mat.std(axis=0, ddof=1) / np.sqrt(len(event_times)) / bin_size
    else:
        sem = np.zeros_like(centers)
    return per_event, centers, fr, sem


def get_lick_times(nwb, side: str) -> np.ndarray:
    """Return sorted left/right lick times (s) in the spike-time clock."""
    event_name = "left_lick" if side.lower() == "left" else "right_lick"
    lick_times = np.asarray(
        extract_event_timestamps(nwb, event_name), dtype=float
    )
    lick_times = lick_times[~np.isnan(lick_times)]
    return np.sort(lick_times)


def plot_lick_aligned(units_df, unit_index, side, lick_times,
                      time_window, bin_size, session_label,
                      save_path, save_formats, max_raster_events):
    """Save a raster (top) + PSTH (bottom) figure for one unit and one lick side."""
    side = side.lower()
    n_licks = len(lick_times)

    spikes = units_df.loc[unit_index]["spike_times"]
    per_event, centers, fr, sem = compute_aligned_raster_psth(
        spikes, lick_times, time_window, bin_size
    )

    display_events = per_event
    if max_raster_events is not None and n_licks > max_raster_events:
        idx = np.linspace(0, n_licks - 1, max_raster_events).astype(int)
        display_events = [per_event[i] for i in idx]

    color = "tab:blue" if side == "left" else "tab:red"
    fig, (raster_ax, psth_ax) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(f"{session_label} | Unit {unit_index} | {side} lick ({n_licks} events)", fontsize=12)

    for row, rel in enumerate(display_events):
        raster_ax.vlines(rel, row + 0.5, row + 1.5, color=color, linewidth=0.5)
    raster_ax.axvline(0, color="k", linestyle="--", linewidth=1)
    raster_ax.set_ylabel(f"{side.capitalize()} lick event")
    raster_ax.set_ylim(0.5, max(len(display_events), 1) + 0.5)
    if max_raster_events is not None and n_licks > max_raster_events:
        raster_ax.set_title(f"(showing {max_raster_events} of {n_licks} events)", fontsize=9)

    psth_ax.plot(centers, fr, color=color, label="Mean FR")
    psth_ax.fill_between(centers, fr - sem, fr + sem, color=color, alpha=0.3)
    psth_ax.axvline(0, color="k", linestyle="--", linewidth=1)
    psth_ax.set_xlabel("Time from lick (s)")
    psth_ax.set_ylabel("FR (Hz)")
    psth_ax.set_xlim(time_window)
    psth_ax.legend(loc="upper right")
    fig.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    base = os.path.join(save_path, f"{session_label}_unit_{unit_index}_{side}_lick")
    for fmt in save_formats:
        fig.savefig(f"{base}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_session(session: str) -> str:
    """
    Generate left & right lick-aligned raster+PSTH figures for every QC-passed unit
    of one session. Returns a short status string.

    This function must be top-level in a module so it is importable by 'spawn'.
    Spikes are read from the ephys NWB (``NWBUtils.read_ephys_nwb``) and lick times
    from the behavior NWB (``NWBUtils.read_behavior_nwb``); both are synchronized to
    the same clock by the upstream pipeline, so no extra alignment is needed.
    """
    nwb: Optional[Any] = None
    beh: Optional[Any] = None
    try:
        label = session_core_from_folder(session)

        nwb = NWBUtils.read_ephys_nwb(session_name=session)
        if nwb is None or not hasattr(nwb, "units"):
            return f"[{label}] skip: ephys NWB not found or has no units"

        beh = NWBUtils.read_behavior_nwb(session_name=session)
        if beh is None:
            return f"[{label}] skip: behavior NWB not found (needed for lick times)"

        units_passing_qc = get_units_passing_qc(nwb)

        left_licks = get_lick_times(beh, "left")
        right_licks = get_lick_times(beh, "right")

        units = list(units_passing_qc.index) if UNITS is None else list(UNITS)
        save_dir = str(OUTDIR / label)
        os.makedirs(save_dir, exist_ok=True)

        print(f"[{label}] units={len(units)} | left_licks={len(left_licks)} | right_licks={len(right_licks)}")

        for i, unit_index in enumerate(units, start=1):
            if unit_index not in units_passing_qc.index:
                continue
            for side, licks in (("left", left_licks), ("right", right_licks)):
                try:
                    plot_lick_aligned(
                        units_passing_qc, unit_index, side, licks,
                        TIME_WINDOW, BIN_SIZE, label,
                        save_dir, SAVE_FORMATS, MAX_RASTER_EVENTS,
                    )
                except Exception as e:
                    print(f"[{label}] error plotting unit {unit_index} {side}: {e}")
                finally:
                    plt.close("all")
            if (i % 25) == 0:
                gc.collect()

        return f"[{label}] done ({len(units)} units)"

    except Exception as e:
        return f"[{session}] failed: {e}"

    finally:
        try:
            plt.close("all")
        except Exception:
            pass
        for _obj in (nwb, beh):
            try:
                if _obj is not None and hasattr(_obj, "io"):
                    _obj.io.close()
            except Exception:
                pass
        try:
            del nwb
            del beh
        except Exception:
            pass
        gc.collect()
