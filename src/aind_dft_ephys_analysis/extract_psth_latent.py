


from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Optional, Union

import numpy as np
import pandas as pd
import zarr
import matplotlib.pyplot as plt

from nwb_utils import NWBUtils
from behavior_utils import find_trials
from create_psth import load_psth_raster_subset
from general_utils import smart_read_csv


# -------------------------------------------------------------------
# 0) Locate PSTH zarr for a given session
# -------------------------------------------------------------------
def _find_psth_zarr_for_session(
    session_id: str,
    psth_root: Union[str, Path] = "/root/capsule/scratch/psth",
    bin_size_label: str = "0.2s",
) -> Path:
    """
    Locate the PSTH Zarr folder for a given ephys session.

    It searches for files like:
        ecephys_{session_id}_*_{bin_size_label}.zarr
    """
    psth_root = Path(psth_root)
    pattern = f"ecephys_{session_id}_*_{bin_size_label}.zarr"
    matches = sorted(psth_root.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No PSTH Zarr found for session {session_id!r} with pattern {pattern!r}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple PSTH Zarr folders found for {session_id!r}: {matches}"
        )
    return matches[0]


# -------------------------------------------------------------------
# 1) Locate behavior_summary CSV for a session (with variable suffix)
# -------------------------------------------------------------------
def _find_behavior_summary_csv_for_session(
    session_id: str,
    behavior_summary_root: Union[str, Path] = "/root/capsule/scratch",
    behavior_summary_prefix: str = "behavior_summary-",
    behavior_summary_suffix: str = ".csv",
) -> Path:
    """
    Find behavior summary CSV for a session.

    You said your files look like:
        /root/capsule/scratch/
            behavior_summary-ecephys_{session_id}_sorted_2025-01-24_14-20-13.csv

    So we match:
        behavior_summary-ecephys_{session_id}_sorted_*{behavior_summary_suffix}
    """
    behavior_dir = Path(behavior_summary_root)
    pattern = (
        f"{behavior_summary_prefix}ecephys_{session_id}_sorted_*{behavior_summary_suffix}"
    )
    matches = sorted(behavior_dir.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No behavior summary CSV found for session {session_id!r} "
            f"with pattern {pattern!r} in {behavior_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple behavior summary CSVs found for session {session_id!r}: {matches}"
        )

    return matches[0]


# -------------------------------------------------------------------
# 2) Single-session helper: extract response PSTH for all units *once*
# -------------------------------------------------------------------
def _extract_response_trials_for_one_session(
    *,
    session_id: str,
    unit_indices: Sequence[int],
    latent_cols: Sequence[str],
    psth_root: Union[str, Path],
    behavior_summary_root: Union[str, Path],
    behavior_summary_prefix: str,
    behavior_summary_suffix: str,
    align_to_event: Optional[str],
    time_window: Optional[Tuple[float, float]],
    bin_size_label: str,
    consolidated: bool,
) -> Dict[str, Any]:
    """
    For ONE session, extract single-trial PSTH on response trials for
    MULTIPLE units, loading NWB and behavior CSV only once.

    Assumes each latent column in df_beh is a single row whose value is
    an array/list over trials, e.g.:

        df_beh["QLearning_L2F1_softmax-reward"].iloc[0]  ->  (n_trials,)

    Returns
    -------
    result : dict with keys
        "time"       : (T,)
        "unit_index" : (N_units_s,) int array of units actually found
        "psth"       : (N_units_s, N_trials_s, T)
        "latent"     : (N_trials_s, L)  per-trial latent values
        "latent_cols": list[str]
    """
    # --------------------------
    # Deduplicate units
    # --------------------------
    seen_units: set[int] = set()
    unit_indices_dedup: list[int] = []
    for uid in unit_indices:
        uid_int = int(uid)
        if uid_int not in seen_units:
            seen_units.add(uid_int)
            unit_indices_dedup.append(uid_int)

    if not unit_indices_dedup:
        raise ValueError(f"No unit indices for session {session_id!r}.")

    # --------------------------
    # Load PSTH zarr & NWB
    # --------------------------
    psth_path = _find_psth_zarr_for_session(
        session_id=session_id,
        psth_root=psth_root,
        bin_size_label=bin_size_label,
    )

    nwb_data = NWBUtils.read_behavior_nwb(session_name=session_id)

    # --------------------------
    # Load behavior summary CSV
    # --------------------------
    behavior_csv_path = _find_behavior_summary_csv_for_session(
        session_id=session_id,
        behavior_summary_root=behavior_summary_root,
        behavior_summary_prefix=behavior_summary_prefix,
        behavior_summary_suffix=behavior_summary_suffix,
    )
    df_beh = smart_read_csv(str(behavior_csv_path))

    for col in latent_cols:
        if col not in df_beh.columns:
            raise ValueError(
                f"Latent column {col!r} not in behavior summary CSV: {behavior_csv_path}"
            )

    if len(df_beh) == 0:
        return {
            "time": np.zeros((0,), dtype=float),
            "unit_index": np.zeros((0,), dtype=int),
            "psth": np.zeros((0, 0, 0), dtype=float),
            "latent": np.zeros((0, len(latent_cols)), dtype=float),
            "latent_cols": list(latent_cols),
        }

    # --------------------------
    # Response trial indices
    # --------------------------
    response_ids = np.asarray(find_trials(nwb_data, "response"), dtype=int)
    if response_ids.size == 0:
        return {
            "time": np.zeros((0,), dtype=float),
            "unit_index": np.zeros((0,), dtype=int),
            "psth": np.zeros((0, 0, 0), dtype=float),
            "latent": np.zeros((0, len(latent_cols)), dtype=float),
            "latent_cols": list(latent_cols),
        }

    # --------------------------
    # Build latent matrix from ONE row of df_beh
    #   df_beh[col].iloc[0] -> 1D array over trials
    # --------------------------
    latent_arrays = []
    for col in latent_cols:
        arr_full = np.asarray(df_beh[col].iloc[0], dtype=float)  # (n_trials_all,)
        latent_arrays.append(arr_full)

    # shape: (n_trials_all, L)
    latent_all = np.stack(latent_arrays, axis=1)

    # Safety: only keep response_ids within array bounds
    n_trials_all = latent_all.shape[0]
    valid_mask = (response_ids >= 0) & (response_ids < n_trials_all)
    if not np.any(valid_mask):
        return {
            "time": np.zeros((0,), dtype=float),
            "unit_index": np.zeros((0,), dtype=int),
            "psth": np.zeros((0, 0, 0), dtype=float),
            "latent": np.zeros((0, len(latent_cols)), dtype=float),
            "latent_cols": list(latent_cols),
        }

    response_ids = response_ids[valid_mask]
    latent_vals = latent_all[response_ids, :]  # (N_trials_s, L)

    # --------------------------
    # Load PSTH subset ONCE for all units in this session
    # --------------------------
    psth_da, _ = load_psth_raster_subset(
        psth_path,
        trial_ids=response_ids,
        unit_ids=unit_indices_dedup,
        align_to_event=align_to_event,
        time_window=time_window,
        consolidated=consolidated,
    )

    time = psth_da.coords["time"].values
    unit_ids_in_ds = psth_da.coords["unit_index"].values.astype(int)

    psth_list: list[np.ndarray] = []
    kept_units: list[int] = []

    for uid in unit_indices_dedup:
        where_unit = np.where(unit_ids_in_ds == int(uid))[0]
        if where_unit.size == 0:
            # Unit not present; skip
            continue

        kept_units.append(int(uid))
        u_pos = int(where_unit[0])
        psth_unit = psth_da.isel(unit=u_pos).values  # (N_trials_s, T)

        # Make sure lengths match (just in case)
        n_keep2 = min(psth_unit.shape[0], latent_vals.shape[0])
        psth_unit = psth_unit[:n_keep2, :]
        psth_list.append(psth_unit)

    if not kept_units:
        return {
            "time": time,
            "unit_index": np.zeros((0,), dtype=int),
            "psth": np.zeros((0, 0, time.shape[0]), dtype=float),
            "latent": np.zeros((0, len(latent_cols)), dtype=float),
            "latent_cols": list(latent_cols),
        }

    # Stack units: (N_units_s, N_trials_s, T)
    psth_arr = np.stack(psth_list, axis=0)
    N_trials_s = psth_arr.shape[1]
    latent_vals = latent_vals[:N_trials_s, :]

    return {
        "time": time,
        "unit_index": np.asarray(kept_units, int),
        "psth": psth_arr,
        "latent": latent_vals,
        "latent_cols": list(latent_cols),
    }



# -------------------------------------------------------------------
# 3) Multi-session extractor: concatenate trials across all units
# -------------------------------------------------------------------
def extract_response_psth_and_latent(
    unit_specs: Sequence[Dict[str, Any]],
    *,
    latent_cols: Union[str, Sequence[str]],
    psth_root: Union[str, Path] = "/root/capsule/scratch/psth",
    behavior_summary_root: Union[str, Path] = "/root/capsule/scratch",
    behavior_summary_prefix: str = "behavior_summary-",
    behavior_summary_suffix: str = ".csv",
    align_to_event: Optional[str] = "go_cue",
    time_window: Optional[Tuple[float, float]] = None,
    bin_size_label: str = "0.2s",
    consolidated: bool = True,
    save_zarr_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Extract single-trial PSTHs for RESPONSE trials and latents
    across multiple sessions and units.

    For each session, NWB + behavior CSV + PSTH zarr are loaded ONCE,
    and PSTHs for all requested units in that session are extracted.

    The final result concatenates all (unit, trial) combinations:

        psth           : (N_trials_total, T)
        latent_values  : (N_trials_total, L)

    Only psth, time, latent_values are saved into Zarr (plus attrs).
    """
    # Normalize latent_cols to list
    if isinstance(latent_cols, str):
        latent_cols = [latent_cols]
    latent_cols = list(latent_cols)
    if len(latent_cols) == 0:
        raise ValueError("latent_cols is empty.")

    # Deduplicate (session, unit) while preserving order
    seen: set[Tuple[str, int]] = set()
    unit_keys: list[Tuple[str, int]] = []
    units_by_session: Dict[str, list[int]] = {}
    for spec in unit_specs:
        sid = spec["session_id"]
        uid = int(spec["unit_index"])
        key = (sid, uid)
        if key not in seen:
            seen.add(key)
            unit_keys.append(key)
            units_by_session.setdefault(sid, []).append(uid)

    psth_blocks: list[np.ndarray] = []
    latent_blocks: list[np.ndarray] = []
    unit_info: list[Tuple[str, int]] = []
    common_time: Optional[np.ndarray] = None

    # --------------------------------------------------
    # Per-session extraction (avoids repeated loading)
    # --------------------------------------------------
    for session_id, unit_indices in units_by_session.items():
        sess_res = _extract_response_trials_for_one_session(
            session_id=session_id,
            unit_indices=unit_indices,
            latent_cols=latent_cols,
            psth_root=psth_root,
            behavior_summary_root=behavior_summary_root,
            behavior_summary_prefix=behavior_summary_prefix,
            behavior_summary_suffix=behavior_summary_suffix,
            align_to_event=align_to_event,
            time_window=time_window,
            bin_size_label=bin_size_label,
            consolidated=consolidated,
        )

        time = sess_res["time"]
        if time.size == 0:
            continue

        if common_time is None:
            common_time = time.copy()
        else:
            if len(time) != len(common_time) or not np.allclose(time, common_time):
                raise ValueError(
                    f"Time axis mismatch when merging session {session_id!r}."
                )

        unit_idx_sess: np.ndarray = sess_res["unit_index"]  # (N_units_s,)
        psth_sess: np.ndarray = sess_res["psth"]            # (N_units_s, N_trials_s, T)
        latent_sess: np.ndarray = sess_res["latent"]        # (N_trials_s, L)

        N_units_s, N_trials_s, _ = psth_sess.shape

        # For each unit in this session, append its trials
        for i_local in range(N_units_s):
            uid = int(unit_idx_sess[i_local])
            psth_u = psth_sess[i_local, :, :]    # (N_trials_s, T)

            psth_blocks.append(psth_u)
            latent_blocks.append(latent_sess)    # same latent per trial, replicated per unit
            unit_info.extend([(session_id, uid)] * N_trials_s)

    if common_time is None or not psth_blocks:
        raise RuntimeError("No PSTH data extracted for the given unit_specs.")

    psth_global = np.vstack(psth_blocks)         # (N_trials_total, T)
    latent_global = np.vstack(latent_blocks)     # (N_trials_total, L)

    res: Dict[str, Any] = {
        "time": common_time,
        "psth": psth_global,
        "latent_values": latent_global,
        "latent_cols": latent_cols,
        "unit_list": unit_info,   # each row ↔ (session_id, unit_index)
    }

    # Save to Zarr (only time, psth, latent_values + attrs)
    if save_zarr_path is not None:
        save_zarr_path = Path(save_zarr_path)
        root = zarr.open_group(str(save_zarr_path), mode="w")
        root.create_dataset("time", data=common_time)
        root.create_dataset("psth", data=psth_global)
        root.create_dataset("latent_values", data=latent_global)
        root.attrs["latent_cols"] = latent_cols
        print(f"Saved combined psth + latent_values to Zarr: {save_zarr_path}")

    return res


# -------------------------------------------------------------------
# 4) Loader: only time, psth, latent_values + latent_cols
# -------------------------------------------------------------------
def load_response_psth_and_latent_zarr(
    zarr_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Load combined response PSTH + latent data saved by
    extract_response_psth_and_latent(..., save_zarr_path=...).

    Zarr layout:
      time           (T,)
      psth           (N_trials_total, T)
      latent_values  (N_trials_total, L)
      attrs["latent_cols"]
    """
    zarr_path = Path(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="r")

    time = root["time"][:]
    psth = root["psth"][:].astype(float)
    latent_values = root["latent_values"][:].astype(float)

    latent_cols = root.attrs.get("latent_cols", [])
    if isinstance(latent_cols, np.ndarray):
        latent_cols = latent_cols.astype(str).tolist()

    res: Dict[str, Any] = {
        "time": time,
        "psth": psth,
        "latent_values": latent_values,
        "latent_cols": latent_cols,
    }
    return res

def plot_psth_quantiles_from_response_res(
    res: Dict[str, Any],
    *,
    latent_name: str,
    n_quantiles: int = 4,
    sort_ascending: bool = True,
    figsize: Tuple[float, float] = (8.0, 5.0),
    cmap: str = "viridis",
    linewidth: float = 2.0,
    sem_alpha: float = 0.25,
) -> None:
    """
    Plot PSTH averaged in equal-width latent bins (called "quantiles" here)
    with SEM shading, and show the mean latent in each bin on a colorbar.

    Parameters
    ----------
    res : dict
        Returned by extract_response_psth_and_latent or
        load_response_psth_and_latent_zarr. Must contain:
            res["psth"]          -> (N_samples, T)
            res["latent_values"] -> (N_samples, L)
            res["time"]          -> (T,)
            res["latent_cols"]   -> list of column names

        Each row in psth / latent_values corresponds to one (unit, trial) sample.

    latent_name : str
        Name of the latent variable to use (must exist in res["latent_cols"]).

    n_quantiles : int, default 4
        Number of bins in latent space. Bins are equal-distance in latent value:
        edges are linspace(min_lat, max_lat, n_quantiles + 1).

    sort_ascending : bool, default True
        If True, plot curves from lowest-valued bin to highest.
        If False, reverse the plotting order.

    figsize : (float, float), default (8.0, 5.0)
        Figure size in inches.

    cmap : str, default "viridis"
        Name of a matplotlib colormap used to draw the bins.

    linewidth : float, default 2.0
        Line width for the mean PSTH curves.

    sem_alpha : float, default 0.25
        Alpha transparency for the SEM shading bands.
    """
    psth = res["psth"]                 # (N_samples, T)
    latent_values = res["latent_values"]  # (N_samples, L)
    time = res["time"]
    latent_cols = res.get("latent_cols", [])

    if latent_name not in latent_cols:
        raise ValueError(
            f"latent_name {latent_name!r} not found in res['latent_cols']: {latent_cols}"
        )

    latent_index = latent_cols.index(latent_name)
    lat = latent_values[:, latent_index]  # (N_samples,)

    # -------------------------------------------------------
    # Keep only finite latent values
    # -------------------------------------------------------
    finite_mask = np.isfinite(lat)
    if not np.any(finite_mask):
        raise ValueError("All latent values are NaN/inf; cannot compute bins.")

    lat_valid = lat[finite_mask]
    psth_valid = psth[finite_mask, :]   # (N_valid, T)

    # -------------------------------------------------------
    # Equal-distance bins in latent value space
    # -------------------------------------------------------
    if n_quantiles < 1:
        raise ValueError("n_quantiles must be >= 1.")

    lat_min = float(np.nanmin(lat_valid))
    lat_max = float(np.nanmax(lat_valid))
    if lat_min == lat_max:
        raise ValueError(
            f"All latent values are the same ({lat_min}); cannot make multiple bins."
        )

    edges = np.linspace(lat_min, lat_max, n_quantiles + 1)  # equal-width bins
    n_groups = n_quantiles

    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=lat_min, vmax=lat_max)

    # Decide plotting order of bins
    bin_indices = list(range(n_groups))
    if not sort_ascending:
        bin_indices = bin_indices[::-1]

    plt.figure(figsize=figsize)

    # For colorbar tick labels
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    # -------------------------------------------------------
    # Plot each bin's mean PSTH + SEM
    # -------------------------------------------------------
    for i_plot, i in enumerate(bin_indices):
        lo = edges[i]
        hi = edges[i + 1]

        # Non-overlapping bins: [lo, hi) except last: [lo, hi]
        if i < n_groups - 1:
            mask_bin = (lat_valid >= lo) & (lat_valid < hi)
        else:
            mask_bin = (lat_valid >= lo) & (lat_valid <= hi)

        if not np.any(mask_bin):
            continue

        lat_bin = lat_valid[mask_bin]
        group_psth = psth_valid[mask_bin, :]  # (N_bin, T)

        mean_trace = group_psth.mean(axis=0)
        if group_psth.shape[0] > 1:
            sem_trace = group_psth.std(axis=0, ddof=1) / np.sqrt(group_psth.shape[0])
        else:
            sem_trace = np.zeros_like(mean_trace)

        # Mean latent in this bin (for label + colorbar position)
        mean_lat = float(np.nanmean(lat_bin))
        if not np.isfinite(mean_lat):
            # Fallback: mid-point of bin if something weird happens
            mean_lat_for_color = 0.5 * (lo + hi)
        else:
            mean_lat_for_color = mean_lat

        color = cmap_obj(norm(mean_lat_for_color))

        label = (
            f"[{lo:.3f}, {hi:.3f}] "
            f"(mean={mean_lat:.3f}, n={group_psth.shape[0]})"
        )

        plt.plot(time, mean_trace, label=label, color=color, linewidth=linewidth)
        plt.fill_between(
            time,
            mean_trace - sem_trace,
            mean_trace + sem_trace,
            color=color,
            alpha=sem_alpha,
            linewidth=0,
        )

        tick_positions.append(mean_lat_for_color)
        tick_labels.append(f"{mean_lat:.3f}")

    plt.xlabel("Time (s)")
    plt.ylabel("Firing rate (spk/s)")
    plt.title(f"PSTH in equal-width {latent_name} bins")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # -------------------------------------------------------
    # Colorbar: full colormap, ticks at mean latent per bin
    # -------------------------------------------------------
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=plt.gca())
    if tick_positions:
        # sort ticks by latent value
        order = np.argsort(tick_positions)
        ticks_sorted = np.array(tick_positions)[order]
        labels_sorted = np.array(tick_labels)[order]

        cbar.set_ticks(ticks_sorted)
        cbar.set_ticklabels(labels_sorted)

    cbar.set_label(f"Mean {latent_name} in bin")

    plt.legend(title=f"{latent_name} bin\n[value range, mean, n]")
    plt.show()

