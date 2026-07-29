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
    cond_each_trial: Optional[Sequence] = None,
    cond_colors: Optional[Sequence[str]] = None,
    **kwargs,
) -> plt.Axes:
    """Per-trial raster. If ``cond_each_trial`` is given, trials are grouped by
    condition value (e.g. power) with separating lines, left-edge condition bars
    and condition labels on the y-axis (matching Anna's raster_plot)."""
    ax = ax or plt.gca()

    if cond_each_trial is not None:
        cond_each_trial = np.asarray(cond_each_trial)
        conds = np.unique(cond_each_trial)
        if cond_colors is None:
            cond_colors = np.tile(["0.5", "0.75"], int(np.ceil(len(conds) / 2)))
        xspan = time_range[1] - time_range[0]
        total = 0
        centers = []
        for i, cond in enumerate(conds):
            idxs = np.flatnonzero(cond_each_trial == cond)
            start = total
            for j in idxs:
                total += 1
                spikes = event_locked_spike_times[j]
                if len(spikes):
                    ax.plot(spikes, np.full(len(spikes), total), ".",
                            color=color, ms=ms, **kwargs)
            centers.append((start + total) / 2.0)
            ax.axhline(total, color="0.7", lw=0.5, zorder=-100)
            xpos = [time_range[0] - 0.03 * xspan, time_range[0]]
            ax.fill_between(xpos, [start, start], [total, total], ec="none",
                            fc=cond_colors[i % len(cond_colors)], clip_on=False)
        ax.set_yticks(centers)
        ax.set_yticklabels([f"{c}" for c in conds])
        ax.tick_params("y", length=0, pad=8)
        ax.set_ylim(0, max(total, 1))
    else:
        for trial, spikes in enumerate(event_locked_spike_times):
            if len(spikes):
                ax.plot(spikes, np.full(len(spikes), trial + 1), ".",
                        color=color, ms=ms, **kwargs)
        ax.set_ylim(0, len(event_locked_spike_times) + 2)

    ax.set_xlim(time_range)
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

    fig = plt.figure(figsize=(width * (n_types + 1) * 3, height * 2), constrained_layout=True)
    gs = gridspec.GridSpec(height, width, hspace=0.9, wspace=0.4, figure=fig)

    for idx, unit in enumerate(unit_ids):
        spikes = get_unit_spike_times(analysis.nwb_data, int(unit))
        sub = gs[idx // width, idx % width].subgridspec(
            2, n_types + 1, wspace=0.8, hspace=0.6, height_ratios=[0.005, 1]
        )

        for it, trial_type in enumerate(trial_types):
            ax = fig.add_subplot(sub[1, it])
            sel = _select_trials(analysis, trial_type, probe)
            # Anna's filter: train pulses on a single external emission site
            if "param_group" in sel.columns:
                sel = sel[sel["param_group"] == "train"]
            if "site" in sel.columns:
                sel = sel[sel["site"] == 0]

            if len(sel):
                # Pulse-train geometry -> full-train time range (Anna's convention)
                duration = float(np.unique(sel["duration"])[0]) if "duration" in sel.columns else 5.0
                num_pulses = int(np.unique(sel["num_pulses"])[0]) if "num_pulses" in sel.columns else 5
                pulse_interval = float(np.unique(sel["pulse_interval"])[0]) if "pulse_interval" in sel.columns else duration
                total_duration = (duration * num_pulses) + (pulse_interval * num_pulses)  # ms
                trange = [-(total_duration / 2) / 1000.0, (1.5 * total_duration) / 1000.0]

                onsets = analysis.laser_onset_times[sel.index.to_numpy()]
                ragged = _ragged_align(spikes, onsets, trange)
                cond = sel["power"].tolist() if "power" in sel.columns else None
                raster_plot(ragged, trange, ax=ax, cond_each_trial=cond, ms=2.5,
                            markeredgecolor="none")

                # Shade every laser pulse across the train
                laser_color = "tomato" if "red" in str(trial_type).lower() else "skyblue"
                y0, y1 = ax.get_ylim()
                for pulse in range(num_pulses):
                    ax.add_patch(patches.Rectangle(
                        (pulse * (duration + pulse_interval) / 1000.0, y0),
                        duration / 1000.0, y1 - y0,
                        edgecolor=laser_color, facecolor=laser_color,
                        alpha=0.35, linewidth=0, clip_on=False,
                    ))

                # Title = wavelength (473 nm / 638 nm), like Anna
                if "wavelength" in sel.columns:
                    wl = np.unique(sel["wavelength"])[0]
                    ax.set_title(f"{wl} nm", fontsize=8)
                else:
                    ax.set_title(f"{trial_type}", fontsize=8)
            else:
                ax.set_title(f"{trial_type}", fontsize=8)

            ax.tick_params("both", labelsize=8)
            ax.set_xlabel("Time from laser onset (s)", fontsize=7)
            if it == 0:
                ax.set_ylabel("Power (mW)", fontsize=7)

        # waveform heatmap (cropped like Anna) with peak-channel inset
        ax_w = fig.add_subplot(sub[1, n_types])
        wm = np.asarray(waveform_mean[int(unit)])  # (timepoints, electrodes)
        t0, t1, c1 = 40, 160, 150
        wm_slice = wm[t0:t1, :c1] if wm.shape[0] >= t1 else wm[:, :c1]
        vmax = float(np.nanmax(np.abs(wm_slice))) or 1.0
        im = ax_w.imshow(wm_slice.T, aspect="auto", cmap="PRGn", vmin=-vmax, vmax=vmax)
        ax_w.invert_yaxis()
        ax_w.set_xlabel("Sample number", fontsize=7)
        ax_w.set_ylabel("Channel", fontsize=7)
        cbar = fig.colorbar(im, ax=ax_w, fraction=0.046)
        cbar.set_label("Voltage (uV)", fontsize=7)

        # inset: peak-channel waveform trace
        peak_ch = int(np.unravel_index(np.nanargmin(wm), wm.shape)[1])
        peak_trace = wm[t0:t1, peak_ch] if wm.shape[0] >= t1 else wm[:, peak_ch]
        ax_in = ax_w.inset_axes([0.62, 0.68, 0.34, 0.28])
        ax_in.plot(peak_trace, lw=1.5, c="k")
        ax_in.set_xticks([])
        ax_in.set_yticks([])

        # Title row with unit ID
        ax_title = fig.add_subplot(sub[0, :])
        ax_title.axis("off")
        ax_title.set_title(f"cluster {unit}", fontweight="heavy")

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
    laser pulse shaded and the estimated latency marked.  All pulses are stacked
    vertically (y = pulse number) matching Anna's pulse_plot style.
    """
    unit_ids = list(unit_ids)
    if not unit_ids:
        print("No units to plot.")
        return None

    n_types = len(trial_types)
    width = int(np.ceil(np.sqrt(len(unit_ids))))
    height = int(np.ceil(len(unit_ids) / width))

    fig = plt.figure(figsize=(width * 2 * n_types, height * 3), constrained_layout=True)
    gs = gridspec.GridSpec(height, width, hspace=0.9, wspace=0.4, figure=fig)

    for idx, unit in enumerate(unit_ids):
        spikes = get_unit_spike_times(analysis.nwb_data, int(unit))
        row = metrics.loc[metrics["unit_id"] == unit]
        sub = gs[idx // width, idx % width].subgridspec(
            2, n_types, wspace=0.6, height_ratios=[0.005, 1]
        )

        for it, trial_type in enumerate(trial_types):
            ax = fig.add_subplot(sub[1, it])
            best_power = _best_power(row, trial_type, analysis, trial_type_col="type")
            sel = _select_trials(analysis, trial_type, probe)
            # Filter to train param_group only (matching Anna's main.py)
            if "param_group" in sel.columns:
                sel = sel[sel["param_group"] == "train"]
            if "site" in sel.columns:
                sel = sel[sel["site"] == 0]
            if best_power is not None and "power" in sel.columns:
                sel = sel[sel["power"] == best_power]
            if len(sel):
                onsets = analysis.laser_onset_times[sel.index.to_numpy()]
                duration = float(np.unique(sel.get("duration", [5.0]))[0]) / 1000.0
                num_pulses = int(np.unique(sel.get("num_pulses", [5]))[0])
                pulse_interval = float(np.unique(sel.get("pulse_interval", [duration * 1000]))[0]) / 1000.0
                trange = [-duration / 2, duration * 1.5]
                color = "tomato" if "red" in str(trial_type).lower() else "skyblue"

                # Stack all pulses vertically (Anna's pulse_plot style)
                n_trials = len(onsets)
                for pulse in range(num_pulses):
                    pulse_onsets = onsets + pulse * (duration + pulse_interval)
                    ragged = _ragged_align(spikes, pulse_onsets, trange)
                    for trial_i, trial_spikes in enumerate(ragged):
                        y_pos = trial_i + (pulse * n_trials)
                        if len(trial_spikes) > 0:
                            ax.plot(trial_spikes, np.full(len(trial_spikes), y_pos + 1),
                                    "k.", ms=2, markeredgecolor="none")
                    ax.axhline(n_trials * pulse, color="0.7", lw=0.5, zorder=-100)

                ax.axvspan(0, duration, color=color, alpha=0.3)
                lat = _metric_value(row, f"{trial_type}_train_best_mean_latency")
                if lat is not None and np.isfinite(lat):
                    ax.axvline(lat, color="blue", ls="--", lw=1)
                ax.set_xlim(trange)
                ax.set_ylim(0, n_trials * num_pulses)
                ax.set_yticks(np.arange(n_trials / 2, n_trials * num_pulses, n_trials).astype(int))
                ax.set_yticklabels(range(1, num_pulses + 1))

            title = f"{best_power} mW" if best_power is not None else trial_type
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("Time from laser onset (s)", fontsize=7)
            if it == 0:
                ax.set_ylabel("Pulse", fontsize=7)

        # Title row with cluster ID
        ax_title = fig.add_subplot(sub[0, :])
        ax_title.axis("off")
        ax_title.set_title(f"cluster {unit}", fontweight="heavy")

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
