"""
Plotting for NWB-based optotagging analysis.

Port of the original ``plotting_funcs.py`` (``multi_unit_raster_plot`` and
``multi_unit_pulse_plot``) that sources spike times and waveforms from an NWB
units table instead of SpikeInterface objects. Works together with
:mod:`optotagging_Anna_nwb`.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

from optotagging_Anna_nwb import get_unit_spike_times, to_events, _q


def raster_plot(
    event_locked_spike_times: Sequence[np.ndarray],
    time_range: Sequence[float],
    ax: Optional[plt.Axes] = None,
    color: str = "k",
    ms: float = 2.5,
    **kwargs,
) -> plt.Axes:
    """Simple per-trial raster. ``event_locked_spike_times`` is a ragged list."""
    ax = ax or plt.gca()
    for trial, spikes in enumerate(event_locked_spike_times):
        if len(spikes):
            ax.plot(spikes, np.full(len(spikes), trial + 1), ".",
                    color=color, ms=ms, **kwargs)
    ax.set_xlim(time_range)
    ax.set_ylim(0, len(event_locked_spike_times) + 2)
    return ax


def _ragged_align(spikes: np.ndarray, onsets: np.ndarray, time_range: Sequence[float]) -> List[np.ndarray]:
    """Align spikes to onsets, returning a ragged list of per-trial offsets."""
    offsets, event_inds, _ = to_events(spikes, onsets, time_range)
    return [offsets[event_inds == t] for t in range(len(onsets))]


def _select_trials(analysis: Any, trial_type: str, probe: Optional[str]):
    """Return the trial-table subset for a given trial type / probe."""
    df = analysis.trial_ids
    query_parts = []
    if "type" in df.columns:
        query_parts.append(f"type == {_q(trial_type)}")
    if probe is not None and "emission_location" in df.columns:
        query_parts.append(f"emission_location == {_q(probe)}")
    if not query_parts:
        return df
    try:
        return df.query(" and ".join(query_parts))
    except Exception:
        return df.iloc[0:0]


def multi_unit_raster_plot(
    analysis: Any,
    unit_ids: Iterable[int],
    trial_types: Sequence[str],
    probe: Optional[str],
    fig_title: str,
    save_folder: str = "/root/capsule/results",
    time_range: Sequence[float] = (-0.05, 0.15),
) -> Optional[str]:
    """
    For each unit: one raster per trial type (aligned to laser onset) plus the
    unit's mean-waveform heatmap and peak-channel trace.
    """
    unit_ids = list(unit_ids)
    if not unit_ids:
        print("No units to plot.")
        return None

    waveform_mean = analysis.nwb_data.units["waveform_mean"]
    n_types = len(trial_types)
    width = int(np.ceil(np.sqrt(len(unit_ids))))
    height = int(np.ceil(len(unit_ids) / width))

    fig = plt.figure(figsize=(width * (n_types + 1) * 3, height * 3), constrained_layout=True)
    gs = gridspec.GridSpec(height, width, figure=fig)

    for idx, unit in enumerate(unit_ids):
        spikes = get_unit_spike_times(analysis.nwb_data, int(unit))
        sub = gs[idx // width, idx % width].subgridspec(1, n_types + 1, wspace=0.5)

        for it, trial_type in enumerate(trial_types):
            sel = _select_trials(analysis, trial_type, probe)
            ax = fig.add_subplot(sub[0, it])
            if len(sel):
                onsets = analysis.laser_onset_times[sel.index.to_numpy()]
                ragged = _ragged_align(spikes, onsets, time_range)
                raster_plot(ragged, time_range, ax=ax)
                # shade laser pulses if parameters available
                _shade_pulses(ax, sel, trial_type)
            ax.set_title(f"{trial_type}", fontsize=8)
            ax.set_xlabel("Time from laser (s)", fontsize=7)
            if it == 0:
                ax.set_ylabel("Trial", fontsize=7)

        # waveform heatmap
        ax_w = fig.add_subplot(sub[0, n_types])
        wm = np.asarray(waveform_mean[int(unit)])  # (timepoints, electrodes)
        im = ax_w.imshow(wm.T, aspect="auto", cmap="PRGn",
                         vmin=-np.nanmax(np.abs(wm)), vmax=np.nanmax(np.abs(wm)))
        ax_w.set_title(f"unit {unit}", fontsize=8, fontweight="bold")
        ax_w.set_xlabel("Sample", fontsize=7)
        ax_w.set_ylabel("Channel", fontsize=7)
        fig.colorbar(im, ax=ax_w, fraction=0.046)

    os.makedirs(save_folder, exist_ok=True)
    out = os.path.join(save_folder, f"{fig_title}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"{out} saved")
    return out


def multi_unit_pulse_plot(
    analysis: Any,
    unit_ids: Iterable[int],
    metrics: Any,
    trial_types: Sequence[str],
    probe: Optional[str],
    fig_title: str,
    save_folder: str = "/root/capsule/results",
) -> Optional[str]:
    """
    Per-unit, per-trial-type pulse rasters at each unit's best power, with the
    laser pulse shaded and the estimated latency marked.
    """
    unit_ids = list(unit_ids)
    if not unit_ids:
        print("No units to plot.")
        return None

    n_types = len(trial_types)
    width = int(np.ceil(np.sqrt(len(unit_ids))))
    height = int(np.ceil(len(unit_ids) / width))

    fig = plt.figure(figsize=(width * n_types * 4, height * 3.5), constrained_layout=True)
    gs = gridspec.GridSpec(height, width, figure=fig)

    for idx, unit in enumerate(unit_ids):
        spikes = get_unit_spike_times(analysis.nwb_data, int(unit))
        row = metrics.loc[metrics["unit_id"] == unit]
        sub = gs[idx // width, idx % width].subgridspec(1, n_types, wspace=0.5)

        for it, trial_type in enumerate(trial_types):
            ax = fig.add_subplot(sub[0, it])
            best_power = _best_power(row, trial_type, analysis, trial_type_col="type")
            sel = _select_trials(analysis, trial_type, probe)
            if best_power is not None and "power" in sel.columns:
                sel = sel[sel["power"] == best_power]
            if len(sel):
                onsets = analysis.laser_onset_times[sel.index.to_numpy()]
                duration = float(np.unique(sel.get("duration", [5.0]))[0]) / 1000.0
                trange = [-duration / 2, duration * 1.5]
                ragged = _ragged_align(spikes, onsets, trange)
                raster_plot(ragged, trange, ax=ax)
                ax.axvspan(0, duration, color="skyblue", alpha=0.3)
                lat = _metric_value(row, f"{trial_type}_train_best_mean_latency")
                if lat is not None and np.isfinite(lat):
                    ax.axvline(lat, color="blue", ls="--", lw=1)
            title = f"{best_power} mW" if best_power is not None else trial_type
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("Time from laser (s)", fontsize=7)
            if it == 0:
                ax.set_ylabel("Trial", fontsize=7)

    os.makedirs(save_folder, exist_ok=True)
    out = os.path.join(save_folder, f"{fig_title}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"{out} saved")
    return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _shade_pulses(ax: plt.Axes, sel: Any, trial_type: str) -> None:
    """Overlay laser-pulse rectangles if timing parameters are present."""
    if not {"duration", "num_pulses", "pulse_interval"}.issubset(sel.columns):
        return
    duration = float(np.unique(sel["duration"])[0]) / 1000.0
    num_pulses = int(np.unique(sel["num_pulses"])[0])
    gap = float(np.unique(sel["pulse_interval"])[0]) / 1000.0
    color = "tomato" if "red" in str(trial_type).lower() else "skyblue"
    y0, y1 = ax.get_ylim()
    for k in range(num_pulses):
        ax.add_patch(patches.Rectangle(
            (k * (duration + gap), y0), duration, y1 - y0,
            edgecolor=color, facecolor=color, alpha=0.3, linewidth=0,
        ))


def _best_power(row: Any, trial_type: str, analysis: Any, trial_type_col: str = "type"):
    val = _metric_value(row, f"{trial_type}_train_best_power")
    if val is not None and np.isfinite(val):
        return val
    df = analysis.trial_ids
    if "type" in df.columns and "power" in df.columns:
        powers = np.unique(df[df["type"] == trial_type]["power"])
        if len(powers):
            return float(np.max(powers))
    return None


def _metric_value(row: Any, col: str):
    if col in row.columns and len(row):
        try:
            return float(row[col].values[0])
        except Exception:
            return None
    return None
