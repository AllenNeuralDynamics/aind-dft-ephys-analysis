"""
Utilities for analyzing ephys activity while optogenetic stimulation is applied.

This module bundles helpers that were previously duplicated across the
``ephys_while_stimulation.ipynb`` notebook so the notebook itself only needs
configuration + a few function calls.

Public API
----------
- :func:`summarize_laser_conditions`         — group laser-on trials by their
  stimulation parameters and return a tidy summary DataFrame.
- :func:`filter_psth_units_by_probe`         — keep only PSTH units whose
  ``device_name`` is in a target list.
- :func:`build_response_nonlaser_trials`     — compute the
  ``response ∩ non-laser`` absolute trial set plus the response-only mask used
  to align response-only latent vectors.
- :func:`load_tdr_axis`                      — load ``axis_w`` (+ aligned
  ``unit_id``) from a saved TDR zarr.
- :func:`project_psth_with_tdr_axis`         — project a (probe-filtered) PSTH
  onto a saved TDR axis for a chosen set of absolute trial IDs.
- :func:`plot_psth_condition_scatter`        — per-neuron mean-activity scatter
  (cond A vs cond B) with Welch/FDR significance.
- :func:`plot_multi_group_timecourses`       — population-average firing rate
  over time for multiple trial groups.
- :func:`run_session_tdr_projection_pipeline` — end-to-end: from a session
  name, ensure PSTH + behavior summary + TDR axis exist (build them on demand)
  and return projection results under any number of laser conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection


# ---------------------------------------------------------------------------
# Small, generic helpers
# ---------------------------------------------------------------------------

_ALIGN_TO_VAR = {
    "go_cue": ("psth_go_cue", "trial_go_cue", "trial_index_go_cue"),
    "reward_go_cue_start": (
        "psth_reward_go_cue_start",
        "trial_reward_go_cue_start",
        "trial_index_reward_go_cue_start",
    ),
}


def _align_keys(align: str) -> Tuple[str, str, str]:
    """Return ``(data_var, trial_dim, trial_coord)`` for a PSTH alignment.

    Known alignments are taken from :data:`_ALIGN_TO_VAR`. For any other
    string, the names are derived by convention (``psth_{align}``,
    ``trial_{align}``, ``trial_index_{align}``) so newly-supported alignments
    (e.g. ``"ITI_start"``) work without editing the map.
    """
    if align in _ALIGN_TO_VAR:
        return _ALIGN_TO_VAR[align]
    if not isinstance(align, str) or not align:
        raise ValueError(f"align must be a non-empty string, got {align!r}")
    return (f"psth_{align}", f"trial_{align}", f"trial_index_{align}")


def _device_names(nwb_data: Any) -> np.ndarray:
    """Return ``nwb_data.units['device_name']`` as a string array."""
    names = np.asarray(nwb_data.units["device_name"][:])
    if names.dtype.kind in {"S", "O"}:
        names = names.astype(str)
    return names


def _map_trials_to_positions(
    abs_ids: np.ndarray, trial_labels: np.ndarray
) -> np.ndarray:
    """Map absolute trial IDs to positions along a trial coord (intersect)."""
    abs_ids = np.asarray(abs_ids, dtype=int).ravel()
    present = np.intersect1d(abs_ids, trial_labels, assume_unique=False)
    if present.size == 0:
        return np.array([], dtype=int)
    pos_by_id = {int(tid): i for i, tid in enumerate(trial_labels)}
    return np.array([pos_by_id[int(tid)] for tid in present], dtype=int)


# ---------------------------------------------------------------------------
# 1. Laser-condition summary
# ---------------------------------------------------------------------------

def summarize_laser_conditions(
    nwb_data: Any,
    *,
    group_cols: Sequence[str] = (
        "laser_start",
        "laser_start_offset",
        "laser_end",
        "laser_end_offset",
        "laser_duration",
    ),
) -> pd.DataFrame:
    """
    Build a tidy summary of laser-on trials grouped by stimulation parameters.

    The returned DataFrame has one row per unique combination of
    ``group_cols``, plus ``n_trials`` and ``trial_ids`` (list of absolute trial
    indices, 0-based — same convention as the PSTH/behavior CSV pipeline).

    Sorted by ``n_trials`` descending.
    """
    trials = nwb_data.trials
    laser_on = np.asarray(trials["laser_on_trial"][:])
    n = len(laser_on)

    cols = {"trial_id": np.arange(n), "laser_on_trial": laser_on}
    for c in group_cols:
        cols[c] = np.asarray(trials[c][:])
    df = pd.DataFrame(cols)

    df_laser = df[df["laser_on_trial"] == 1].copy()
    summary = (
        df_laser.groupby(list(group_cols))
        .agg(
            n_trials=("trial_id", "size"),
            trial_ids=("trial_id", lambda x: list(x)),
        )
        .reset_index()
        .sort_values("n_trials", ascending=False)
        .reset_index(drop=True)
    )
    return summary


# ---------------------------------------------------------------------------
# 2. Probe / unit filtering
# ---------------------------------------------------------------------------

def filter_psth_units_by_probe(
    psth: xr.Dataset | xr.DataArray,
    nwb_data: Any,
    target_probes: Sequence[str],
) -> Tuple[xr.Dataset | xr.DataArray, np.ndarray]:
    """
    Restrict a PSTH dataset/array to units recorded on ``target_probes``.

    Returns
    -------
    psth_filtered : xr.Dataset | xr.DataArray
        Same type as the input, sliced along the ``unit`` dim.
    unit_ids : np.ndarray
        Absolute NWB unit IDs that survived the filter, in the same order as
        the returned PSTH along the ``unit`` dim.
    """
    if "unit_index" not in psth.coords:
        raise KeyError(
            "PSTH must have a 'unit_index' coordinate (absolute NWB unit IDs)."
        )
    device_names = _device_names(nwb_data)
    keep_rows = np.where(
        np.isin(device_names, np.asarray(target_probes).astype(str))
    )[0]
    avail = psth["unit_index"].values.astype(int)
    pos = np.nonzero(np.isin(avail, keep_rows))[0]
    if pos.size == 0:
        raise RuntimeError(
            f"No units remain after filtering to probes {list(target_probes)}."
        )
    return psth.isel(unit=pos), avail[pos]


# ---------------------------------------------------------------------------
# 3. Trial-set helpers (response ∩ non-laser)
# ---------------------------------------------------------------------------

def build_response_nonlaser_trials(
    nwb_data: Any, response_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the ``response ∩ non-laser`` absolute trial IDs.

    Parameters
    ----------
    nwb_data
        NWB-like object exposing ``trials['laser_on_trial']``.
    response_ids
        1-D absolute trial IDs of response trials (defines the row order of
        response-only latent vectors).

    Returns
    -------
    include_trials : np.ndarray
        Absolute trial IDs that are response AND non-laser, in response-only
        order.
    resp_keep_mask : np.ndarray
        Boolean mask over ``response_ids`` selecting those non-laser trials.
        Use this to subset response-only latent arrays so they align with
        ``include_trials``.
    """
    response_ids = np.asarray(response_ids, dtype=int).ravel()
    laser_flags = np.asarray(nwb_data.trials["laser_on_trial"][:], dtype=int)
    nonlaser_ids = np.where(laser_flags == 0)[0]
    resp_keep_mask = np.isin(response_ids, nonlaser_ids)
    include_trials = response_ids[resp_keep_mask]
    return include_trials, resp_keep_mask


# ---------------------------------------------------------------------------
# 4. TDR axis loading + projection
# ---------------------------------------------------------------------------

def load_tdr_axis(tdr_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ``(axis_w, unit_id)`` from a saved TDR zarr.

    Falls back to a handful of alternative key names if ``axis_w`` is absent.
    """
    ds = xr.open_zarr(str(tdr_path), consolidated=False)
    if "axis_w" in ds:
        w = np.asarray(ds["axis_w"].values).astype(float).ravel()
        if "unit_id" not in ds.coords:
            raise KeyError("Saved TDR zarr is missing the 'unit_id' coord.")
        unit_ids = np.asarray(ds["unit_id"].values).astype(int)
        if unit_ids.shape[0] != w.shape[0]:
            raise ValueError("axis_w length != unit_id length in TDR dataset.")
        return w, unit_ids

    for k in ("projection_axis", "axis_weights", "weights", "w", "u", "tdr_axis"):
        if k in ds:
            w = np.asarray(ds[k].values).astype(float).ravel()
            if "unit_id" in ds.coords:
                return w, np.asarray(ds["unit_id"].values).astype(int)
            if "unit_index" in ds.coords:
                return w, np.asarray(ds["unit_index"].values).astype(int)
            return w, np.arange(w.shape[0], dtype=int)
    raise KeyError("No axis weights found in TDR dataset.")


def _pick_psth_array(
    psth: xr.Dataset | xr.DataArray, align: Optional[str] = None
) -> xr.DataArray:
    """Return a DataArray with dims that include ``unit`` and ``time``.

    If ``align`` is provided, the corresponding variable from
    :data:`_ALIGN_TO_VAR` is selected explicitly; otherwise the first known
    PSTH variable is used.
    """
    if isinstance(psth, xr.DataArray):
        return psth
    if align is not None:
        var, _, _ = _align_keys(align)
        if var not in psth.data_vars:
            raise KeyError(
                f"PSTH dataset has no variable '{var}' for align='{align}'. "
                f"Available: {sorted(psth.data_vars)}"
            )
        v = psth[var]
        if "unit" not in v.dims or "time" not in v.dims:
            raise RuntimeError(
                f"Variable '{var}' is missing 'unit' or 'time' dim."
            )
        return v
    for name in ("psth_go_cue", "psth_reward_go_cue_start"):
        if name in psth.data_vars:
            v = psth[name]
            if "unit" in v.dims and "time" in v.dims:
                return v
    for name, var in psth.data_vars.items():
        if "unit" in var.dims and "time" in var.dims:
            return psth[name]
    raise RuntimeError("No DataArray with dims ('unit','time') found in PSTH.")


def _find_trial_dim_and_coord(
    arr: xr.DataArray, align: Optional[str] = None
) -> Tuple[str, str]:
    """Locate the trial dim and an integer-labeled trial coord on ``arr``.

    If ``align`` is given, prefer the trial dim/coord from :data:`_ALIGN_TO_VAR`.
    """
    if align is not None:
        _, td, tc = _align_keys(align)
        if td in arr.dims and tc in arr.coords:
            return td, tc
    candidates = (
        "trial_go_cue",
        "trial_reward_go_cue_start",
        "trial",
        "trials",
    )
    for td in candidates:
        if td in arr.dims:
            if td in arr.coords and np.issubdtype(arr[td].dtype, np.integer):
                return td, td
            for c in arr.coords:
                if td in arr[c].dims and np.issubdtype(arr[c].dtype, np.integer):
                    return td, c
    for d in arr.dims:
        if "trial" in d:
            for c in arr.coords:
                if d in arr[c].dims and np.issubdtype(arr[c].dtype, np.integer):
                    return d, c
    raise RuntimeError(
        "Could not infer trial dimension/coordinate (need int-labeled trial coord)."
    )


def project_psth_with_tdr_axis(
    psth: xr.Dataset | xr.DataArray,
    nwb_data: Any,
    tdr_path: str | Path,
    include_trials: np.ndarray,
    target_probes: Sequence[str],
    *,
    align: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project a PSTH onto a stored TDR axis for a chosen set of absolute trials.

    Workflow
    --------
    1. Pick a ``(unit, trial, time)`` array from the PSTH dataset.
    2. Filter units to ``target_probes``.
    3. Load ``axis_w`` from the TDR zarr and align it to the surviving unit IDs.
    4. Slice the PSTH to ``include_trials`` (preserving their order).
    5. Project: ``Y = w @ X`` → ``(trials, time)``.

    Returns
    -------
    Y : np.ndarray, shape (n_trials, n_time)
    t : np.ndarray, shape (n_time,)
    unit_ids : np.ndarray
        Unit IDs that were actually used in the projection (after probe ∩ TDR
        alignment), in the order matching ``w``.
    """
    arr = _pick_psth_array(psth, align=align)
    trial_dim, trial_coord = _find_trial_dim_and_coord(arr, align=align)

    arr_filt, p_units = filter_psth_units_by_probe(arr, nwb_data, target_probes)

    tdr_w, tdr_units = load_tdr_axis(tdr_path)

    index_in_tdr = {u: i for i, u in enumerate(tdr_units)}
    order = np.array([index_in_tdr.get(u, -1) for u in p_units])
    valid = order >= 0
    if not np.any(valid):
        raise RuntimeError(
            "No overlapping units between probe-filtered PSTH and TDR axis."
        )
    arr_filt = arr_filt.isel(unit=np.where(valid)[0])
    p_units = p_units[valid]
    w_aligned = tdr_w[order[valid]]

    trial_labels = arr_filt[trial_coord].values.astype(int)
    include_trials = np.asarray(include_trials, dtype=int)
    pos_by_id = {tid: i for i, tid in enumerate(trial_labels)}
    missing = [tid for tid in include_trials if tid not in pos_by_id]
    if missing:
        raise RuntimeError(
            f"{len(missing)} requested trials not found in PSTH "
            f"(e.g. {missing[:10]})."
        )
    trial_pos = np.array([pos_by_id[tid] for tid in include_trials])
    sel = arr_filt.isel({trial_dim: trial_pos})

    dims = list(sel.dims)
    u_ax = dims.index("unit")
    t_ax = dims.index("time")
    tr_ax = [i for i, d in enumerate(dims) if d == trial_dim][0]
    X = np.moveaxis(sel.values, [u_ax, t_ax, tr_ax], [0, 1, 2])  # (U, T, K)

    Y = np.einsum("u,utk->tk", w_aligned, X).T  # (K, T)
    t = sel["time"].values
    return Y, t, p_units


# ---------------------------------------------------------------------------
# 5. Per-neuron condition scatter (cond A vs cond B)
# ---------------------------------------------------------------------------

def plot_psth_condition_scatter(
    psth_ds: xr.Dataset,
    nwb_data: Any,
    trials_a: np.ndarray,
    trials_b: np.ndarray,
    target_probes: Sequence[str],
    *,
    align: str = "go_cue",
    time_window: Tuple[float, float] = (0.0, 0.5),
    alpha: float = 0.05,
    use_fdr: bool = True,
    equal_var: bool = False,
    marker_size: float = 30.0,
    title: Optional[str] = None,
    xy_lim: Optional[Tuple[float, float]] = None,
    save_path: str | Path | None = None,
    save_format: str | Sequence[str] = ("png",),
    dpi: int = 300,
    overwrite: bool = True,
    show: bool = True,
) -> dict:
    """
    Scatter mean PSTH activity (cond A vs cond B) per neuron, with significance.

    Parameters mirror the previous in-notebook implementation. See the module
    docstring for the high-level pipeline.
    """
    save_formats = (
        [save_format.lower()]
        if isinstance(save_format, str)
        else [str(f).lower() for f in save_format]
    )
    valid_formats = {"png", "eps", "pdf"}
    for fmt in save_formats:
        if fmt not in valid_formats:
            raise ValueError(
                f"Unsupported save_format '{fmt}'. Must be one of {valid_formats}."
            )

    var, trial_dim, trial_coord = _align_keys(align)
    if var not in psth_ds.data_vars:
        raise KeyError(f"{var} not found in psth dataset data_vars.")

    da, unit_ids = filter_psth_units_by_probe(psth_ds[var], nwb_data, target_probes)

    time = psth_ds["time"].values
    t0, t1 = float(time_window[0]), float(time_window[1])
    if t1 <= t0:
        raise ValueError("time_window must be (tmin, tmax) with tmax > tmin.")
    tmask = (time >= t0) & (time < t1)
    if not np.any(tmask):
        raise ValueError(f"time_window {time_window} selects no samples in 'time'.")

    if trial_coord not in psth_ds.coords:
        raise KeyError(f"Missing '{trial_coord}' coordinate in dataset.")
    trial_labels = psth_ds[trial_coord].values.astype(int)
    idx_a = _map_trials_to_positions(trials_a, trial_labels)
    idx_b = _map_trials_to_positions(trials_b, trial_labels)
    if idx_a.size == 0 or idx_b.size == 0:
        raise RuntimeError("No overlapping trials for A or B with the dataset's trial IDs.")

    A = da.isel({trial_dim: idx_a}).sel(time=tmask).mean(dim="time").values
    B = da.isel({trial_dim: idx_b}).sel(time=tmask).mean(dim="time").values

    U = int(A.shape[0])
    pvals = np.empty(U, dtype=float)
    mean_a = A.mean(axis=1)
    mean_b = B.mean(axis=1)

    for u in range(U):
        a_u = A[u, np.isfinite(A[u, :])]
        b_u = B[u, np.isfinite(B[u, :])]
        if a_u.size < 2 or b_u.size < 2:
            pvals[u] = np.nan
            continue
        _, p = ttest_ind(a_u, b_u, equal_var=equal_var)
        pvals[u] = float(p)

    if use_fdr:
        valid = np.isfinite(pvals)
        sig_mask = np.zeros_like(pvals, dtype=bool)
        qvals = np.full_like(pvals, np.nan, dtype=float)
        if valid.any():
            rej, q = fdrcorrection(pvals[valid], alpha=alpha, method="indep")
            sig_mask[valid] = rej
            qvals[valid] = q
    else:
        sig_mask = pvals < alpha
        qvals = np.full_like(pvals, np.nan, dtype=float)

    valid_means = np.isfinite(mean_a) & np.isfinite(mean_b)
    valid_units = valid_means & np.isfinite(pvals)
    sig = sig_mask & valid_units
    inc_mask = sig & (mean_b > mean_a)
    dec_mask = sig & (mean_b < mean_a)

    n_valid = int(valid_units.sum())
    n_sig = int(sig.sum())
    n_sig_inc = int(inc_mask.sum())
    n_sig_dec = int(dec_mask.sum())
    frac_inc_all = (n_sig_inc / n_valid) if n_valid else np.nan
    frac_dec_all = (n_sig_dec / n_valid) if n_valid else np.nan
    frac_inc_sig = (n_sig_inc / n_sig) if n_sig else np.nan
    frac_dec_sig = (n_sig_dec / n_sig) if n_sig else np.nan

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ns_mask = (~sig) & valid_units
    ax.scatter(
        mean_a[ns_mask], mean_b[ns_mask], s=marker_size,
        facecolors="none", edgecolors="k", linewidths=1.0, label="NS",
    )
    if n_sig_inc:
        ax.scatter(
            mean_a[inc_mask], mean_b[inc_mask], s=marker_size,
            facecolors="C1", edgecolors="k", linewidths=0.6, label="Sig ↑ (B>A)",
        )
    if n_sig_dec:
        ax.scatter(
            mean_a[dec_mask], mean_b[dec_mask], s=marker_size,
            facecolors="C0", edgecolors="k", linewidths=0.6, label="Sig ↓ (B<A)",
        )

    lo = float(np.nanmin([np.nanmin(mean_a), np.nanmin(mean_b)]))
    hi = float(np.nanmax([np.nanmax(mean_a), np.nanmax(mean_b)]))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.0, alpha=0.7)

    if xy_lim is not None:
        ax.set_xlim(*xy_lim)
        ax.set_ylim(*xy_lim)

    ax.set_xlabel(f"Mean firing in window {time_window} — Condition A")
    ax.set_ylabel(f"Mean firing in window {time_window} — Condition B")
    ttl = title or f"PSTH mean: condition A vs B ({align})"
    subtitle = (
        f"Sig (alpha={alpha}, {'FDR' if use_fdr else 'uncorr'}) = {n_sig}/{n_valid} | "
        f"↑ {n_sig_inc} ({frac_inc_all:.2%} all, {frac_inc_sig:.2%} of sig), "
        f"↓ {n_sig_dec} ({frac_dec_all:.2%} all, {frac_dec_sig:.2%} of sig)"
    )
    ax.set_title(f"{ttl}\n{subtitle}")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == "":
            save_path.mkdir(parents=True, exist_ok=True)
            base = "psth_condition_scatter"
            for fmt in save_formats:
                fp = save_path / f"{base}.{fmt}"
                if fp.exists() and not overwrite:
                    print(f"Skipping save (exists, overwrite=False): {fp}")
                    continue
                fig.savefig(fp, dpi=dpi, bbox_inches="tight")
                print(f"Saved figure to {fp}")
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            for fmt in save_formats:
                fp = save_path.with_suffix(f".{fmt}")
                if fp.exists() and not overwrite:
                    print(f"Skipping save (exists, overwrite=False): {fp}")
                    continue
                fig.savefig(fp, dpi=dpi, bbox_inches="tight")
                print(f"Saved figure to {fp}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return dict(
        mean_a=mean_a, mean_b=mean_b, pvals=pvals, qvals=qvals,
        sig_mask=sig_mask, unit_ids=unit_ids, fig=fig, ax=ax,
        n_valid=n_valid, n_sig=n_sig, n_sig_inc=n_sig_inc, n_sig_dec=n_sig_dec,
        frac_inc_all=frac_inc_all, frac_dec_all=frac_dec_all,
        frac_inc_sig=frac_inc_sig, frac_dec_sig=frac_dec_sig,
        inc_mask=inc_mask, dec_mask=dec_mask,
    )


# ---------------------------------------------------------------------------
# 6. Population timecourses for multiple trial groups
# ---------------------------------------------------------------------------

@dataclass
class MultiGroupTimecourseResult:
    time: np.ndarray
    group_labels: list
    mean_over_units: np.ndarray
    spread_over_units: np.ndarray
    n_units: int
    n_trials_per_group: list
    per_unit_timecourses: np.ndarray
    unit_ids: np.ndarray
    align: str
    fig: plt.Figure
    ax: plt.Axes


def _gaussian_smooth_rows(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma is None or sigma <= 0:
        return arr
    k = int(6 * sigma) | 1
    x = np.arange(k) - k // 2
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    ker = ker / ker.sum()
    pad = k // 2
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        padrow = np.pad(arr[i], pad, mode="reflect")
        out[i] = np.convolve(padrow, ker, mode="valid")
    return out


def plot_multi_group_timecourses(
    psth_ds: xr.Dataset,
    nwb_data: Any,
    trial_groups: Sequence[Sequence[int] | np.ndarray],
    group_labels: Optional[Sequence[str]] = None,
    *,
    target_probes: Sequence[str],
    align: Literal["go_cue", "reward_go_cue_start"] = "go_cue",
    time_window: Optional[Tuple[float, float]] = None,
    statistic_across_trials: Literal["mean", "median"] = "mean",
    ci: Literal["sem", "95"] = "sem",
    smooth_sigma: Optional[float] = None,
    colors: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> MultiGroupTimecourseResult:
    """
    Plot average firing rate (across units) over time for multiple trial groups.

    For each group:
      ``(unit, trial, time) → average over trials per unit → (unit, time)``
      then average across units → ``(time)``. Spread = SEM or 95% CI across
      units.
    """
    var, trial_dim, trial_coord = _align_keys(align)
    if var not in psth_ds.data_vars:
        raise KeyError(f"{var} not found in psth dataset data_vars.")

    da, unit_ids = filter_psth_units_by_probe(psth_ds[var], nwb_data, target_probes)
    U = da.sizes["unit"]

    time = psth_ds["time"].values
    if time_window is not None:
        t0, t1 = time_window
        tmask = (time >= t0) & (time <= t1)
        if not np.any(tmask):
            raise ValueError(f"time_window {time_window} selects no samples in 'time'.")
        da = da.sel(time=tmask)
        time = time[tmask]
    T = time.shape[0]

    if trial_coord not in psth_ds.coords:
        raise KeyError(f"Missing '{trial_coord}' coordinate in dataset.")
    trial_labels = psth_ds[trial_coord].values.astype(int)

    groups = [np.asarray(g, dtype=int).ravel() for g in trial_groups]
    G = len(groups)
    if group_labels is None:
        group_labels = [f"G{i + 1}" for i in range(G)]
    else:
        group_labels = list(group_labels)
        if len(group_labels) != G:
            raise ValueError("Length of group_labels must match number of trial_groups.")
    if colors is not None and len(colors) != G:
        raise ValueError("If provided, 'colors' length must equal number of groups.")

    per_unit_timecourses = np.full((G, U, T), np.nan, dtype=float)
    n_trials_per_group: list[int] = []
    for g, trials in enumerate(groups):
        idx = _map_trials_to_positions(trials, trial_labels)
        n_trials_per_group.append(int(idx.size))
        if idx.size == 0:
            continue
        vals = da.isel({trial_dim: idx}).values  # (U, Tg, T)
        if statistic_across_trials == "mean":
            per_unit = np.nanmean(vals, axis=1)
        elif statistic_across_trials == "median":
            per_unit = np.nanmedian(vals, axis=1)
        else:
            raise ValueError("statistic_across_trials must be 'mean' or 'median'")

        if smooth_sigma is not None and smooth_sigma > 0:
            per_unit = _gaussian_smooth_rows(per_unit, smooth_sigma)
        per_unit_timecourses[g] = per_unit

    valid_units = np.any(np.isfinite(per_unit_timecourses.reshape(G, U, -1)), axis=(0, 2))
    if not np.any(valid_units):
        raise RuntimeError("No valid units after computing per-unit timecourses (all NaN).")
    X = per_unit_timecourses[:, valid_units, :]

    if ci == "sem":
        mean_over_units = np.nanmean(X, axis=1)
        denom = np.sqrt(np.sum(np.isfinite(X), axis=1))
        spread = np.nanstd(X, axis=1, ddof=1) / np.maximum(denom, 1.0)
    elif ci == "95":
        mean_over_units = np.nanmean(X, axis=1)
        denom = np.sqrt(np.sum(np.isfinite(X), axis=1))
        sem = np.nanstd(X, axis=1, ddof=1) / np.maximum(denom, 1.0)
        spread = 1.96 * sem
    else:
        raise ValueError("ci must be 'sem' or '95'")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    cols = colors if colors is not None else [None] * G
    for g in range(G):
        c = cols[g]
        ax.plot(time, mean_over_units[g], label=group_labels[g], color=c, linewidth=2.0)
        ax.fill_between(
            time,
            mean_over_units[g] - spread[g],
            mean_over_units[g] + spread[g],
            alpha=0.25, edgecolor="none", facecolor=c,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Average firing rate across units (spikes/s)")
    ax.set_title(title or f"Average firing over time across units — {align}")
    ax.legend(frameon=False, ncols=min(G, 3))
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    fig.tight_layout()

    return MultiGroupTimecourseResult(
        time=time,
        group_labels=group_labels,
        mean_over_units=mean_over_units,
        spread_over_units=spread,
        n_units=int(valid_units.sum()),
        n_trials_per_group=n_trials_per_group,
        per_unit_timecourses=X,
        unit_ids=unit_ids[valid_units],
        align=align,
        fig=fig,
        ax=ax,
    )


# ---------------------------------------------------------------------------
# 7. High-level pipeline: session -> projections under multiple conditions
# ---------------------------------------------------------------------------

# Type aliases for the `conditions` argument of
# :func:`run_session_tdr_projection_pipeline`.
#
# Each condition value can be either:
#   - a sequence/array of absolute trial IDs (used as-is, then intersected
#     with response trials), or
#   - a mapping with one of the following keys:
#       * "laser_index": int  -> use the laser-condition summary row at this
#         positional index (after sorting by n_trials desc, the default of
#         :func:`summarize_laser_conditions`).
#       * "laser_off": True  -> use non-laser trials (intersected with response).
#       * "shift": int       -> shift the trial IDs by this many trials before
#         intersecting with response (useful for "stim-1" / "stim+1" controls).
ConditionSpec = Union[Sequence[int], np.ndarray, Mapping[str, Any]]


@dataclass
class TDRProjectionResult:
    """Container for one condition's TDR projection."""

    label: str
    Y: np.ndarray             # (n_trials, n_time)
    t: np.ndarray             # (n_time,)
    trial_ids: np.ndarray     # absolute trial IDs, matches Y rows
    latent: np.ndarray        # latent values for these trials
    unit_ids: np.ndarray      # absolute NWB unit IDs used in the projection


def _ensure_psth_zarr(
    session_name: str,
    binsize: float,
    save_folder: str | Path,
    align_to_event: Sequence[str],
    time_window: Tuple[float, float],
) -> Tuple[Path, Any]:
    """Return ``(zarr_path, nwb_data)``, generating the PSTH zarr if missing."""
    from nwb_utils import NWBUtils
    from create_psth import extract_neuron_psth_to_zarr

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    zarr_path = save_folder / f"{session_name}_{binsize}s.zarr"

    nwb_data, _ = NWBUtils.combine_nwb(session_name=session_name)
    if nwb_data is None:
        raise RuntimeError(f"Could not open NWB for session: {session_name}")

    if not zarr_path.exists():
        print(f"[INFO] PSTH missing, generating: {zarr_path}")
        extract_neuron_psth_to_zarr(
            nwb_data=nwb_data,
            align_to_event=list(align_to_event),
            time_window=tuple(time_window),
            bin_size=float(binsize),
            save_folder=str(save_folder),
            save_name=f"{session_name}_{binsize}s",
        )
    return zarr_path, nwb_data


def _ensure_behavior_summary(
    session_name: str, save_folder: str | Path
) -> Path:
    """Return path to the behavior summary CSV, generating it if missing."""
    from behavior_utils import generate_behavior_summary_combined

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    csv_path = save_folder / f"behavior_summary-{session_name}.csv"
    if not csv_path.exists():
        print(f"[INFO] Behavior summary missing, generating: {csv_path}")
        generate_behavior_summary_combined(
            session_names=[session_name],
            save_result=True,
            save_folder=str(save_folder),
            save_name=csv_path.name,
        )
    return csv_path


def _ensure_tdr_axis(
    *,
    session_name: str,
    latent_var: str,
    time_window: Tuple[float, float],
    align: str,
    target_probes: Sequence[str],
    psth_da: xr.Dataset | xr.DataArray,
    behavior_df: pd.DataFrame,
    nwb_data: Any,
    tdr_folder: str | Path,
) -> Path:
    """Return path to the TDR zarr for ``(session, latent, window)``; build if absent."""
    from ephys_dimension_reduction_tdr import tdr_from_psth

    tdr_folder = Path(tdr_folder)
    tdr_folder.mkdir(parents=True, exist_ok=True)
    tw0, tw1 = time_window
    tdr_path = tdr_folder / f"tdr_{session_name}_{latent_var}_timewindow_{tw0}_{tw1}.zarr"
    if tdr_path.exists():
        return tdr_path

    print(f"[INFO] TDR axis missing, building: {tdr_path}")
    if latent_var not in behavior_df.columns:
        raise KeyError(f"'{latent_var}' not in behavior CSV columns.")
    if "response_trials" not in behavior_df.columns:
        raise KeyError("Column 'response_trials' missing in behavior summary CSV.")
    response_ids = np.asarray(behavior_df["response_trials"][0], dtype=int)

    include_trials, resp_keep_mask = build_response_nonlaser_trials(
        nwb_data, response_ids
    )
    if include_trials.size == 0:
        raise RuntimeError(f"No response ∩ non-laser trials for {session_name}")

    latent_response_only = np.asarray(behavior_df[latent_var][0], dtype=float)
    if latent_response_only.shape[0] != response_ids.shape[0]:
        raise ValueError(
            f"Length mismatch for {latent_var}: latent={latent_response_only.shape[0]} "
            f"vs response_ids={response_ids.shape[0]}"
        )
    latent_filtered = latent_response_only[resp_keep_mask]

    psth_da_filtered, _ = filter_psth_units_by_probe(
        psth_da, nwb_data, target_probes
    )
    tdr_from_psth(
        psth_da_filtered,
        latent=latent_filtered,
        align=align,
        time_window=[tw0, tw1],
        include_trials=include_trials,
        require_all_ids=True,
        save_path=str(tdr_path),
        save_format="zarr",
    )
    return tdr_path


def _resolve_condition_trials(
    spec: ConditionSpec,
    *,
    summary: pd.DataFrame,
    response_ids: np.ndarray,
    nwb_data: Any,
) -> np.ndarray:
    """Resolve a condition spec to absolute trial IDs (intersected with response)."""
    if isinstance(spec, Mapping):
        if spec.get("laser_off"):
            laser_flags = np.asarray(
                nwb_data.trials["laser_on_trial"][:], dtype=int
            )
            trials = np.where(laser_flags == 0)[0]
        elif "laser_index" in spec:
            idx = int(spec["laser_index"])
            if idx < 0 or idx >= len(summary):
                raise IndexError(
                    f"laser_index={idx} out of range for summary with "
                    f"{len(summary)} rows."
                )
            trials = np.asarray(summary["trial_ids"].iloc[idx], dtype=int)
        else:
            raise ValueError(
                "Condition mapping must contain 'laser_index' or 'laser_off'."
            )
        shift = int(spec.get("shift", 0))
        if shift:
            trials = trials + shift
    else:
        trials = np.asarray(spec, dtype=int).ravel()

    return np.intersect1d(trials, response_ids)


def run_session_tdr_projection_pipeline(
    session_name: Union[str, Sequence[str]],
    *,
    latent_var: str,
    time_window: Tuple[float, float],
    conditions: Mapping[str, ConditionSpec],
    target_probes: Union[Sequence[str], Mapping[str, Sequence[str]]],
    binsize: float = 0.1,
    align: str = "go_cue",
    psth_align_events: Sequence[str] = ("go_cue", "reward_go_cue_start"),
    psth_time_window: Tuple[float, float] = (-6.0, 6.0),
    psth_folder: str | Path = "/root/capsule/scratch",
    behavior_folder: str | Path = "/root/capsule/scratch",
    tdr_folder: str | Path = "/root/capsule/scratch",
) -> Union[Dict[str, TDRProjectionResult], Dict[str, Dict[str, TDRProjectionResult]]]:
    """
    End-to-end pipeline: from one or more session names, return TDR projections
    under one or more laser conditions.

    Steps (per session)
    -------------------
    1. Ensure the PSTH zarr exists at
       ``{psth_folder}/{session_name}_{binsize}s.zarr`` (build with
       ``extract_neuron_psth_to_zarr`` if missing).
    2. Ensure the behavior summary CSV exists at
       ``{behavior_folder}/behavior_summary-{session_name}.csv`` (build with
       ``generate_behavior_summary_combined`` if missing).
    3. Ensure the TDR axis zarr exists for ``(latent_var, time_window)`` at
       ``{tdr_folder}/tdr_{session_name}_{latent_var}_timewindow_{tw0}_{tw1}.zarr``
       (build with ``tdr_from_psth`` on response ∩ non-laser trials if missing).
    4. For each condition in ``conditions``, project the PSTH onto the stored
       TDR axis and return the result.

    Parameters
    ----------
    session_name
        Either a single session id (str) or a list of session ids. Sessions are
        processed independently; failures are reported and skipped.
    latent_var
        Behavior-summary column to fit the TDR axis on.
    time_window
        ``(t0, t1)`` (seconds) window used when the TDR axis was/is fit.
    conditions
        Mapping ``label -> spec``. Spec can be:

        - a list/array of absolute trial IDs, or
        - a dict with keys:

          * ``"laser_off": True`` — use non-laser trials.
          * ``"laser_index": int`` — use row ``int`` of
            :func:`summarize_laser_conditions`.
          * ``"shift": int`` (optional) — shift the resolved trial IDs by this
            many trials before intersecting with response (e.g. ``-1`` /
            ``+1``).
    target_probes
        Either a single list of probe names applied to every session, or a
        mapping ``session_name -> probes`` to use different probes per session.
        A ``KeyError`` is raised if a session is missing from the mapping.
    binsize, align, psth_align_events, psth_time_window
        PSTH parameters; only used when the PSTH zarr has to be built.
    psth_folder, behavior_folder, tdr_folder
        Where to read/write the corresponding artifacts.

    Returns
    -------
    dict
        - If ``session_name`` is a single string: ``{condition_label: result}``
          (backwards compatible).
        - If ``session_name`` is a list/tuple: ``{session_name: {condition_label: result}}``.

        Each result is a :class:`TDRProjectionResult` containing ``Y``
        (trials × time), ``t``, the absolute ``trial_ids`` used, the matching
        ``latent`` values (NaN if the trial is not in the response set), and
        the absolute ``unit_ids`` used in the projection.
    """
    single_session = isinstance(session_name, str)
    sessions = [session_name] if single_session else list(session_name)

    # Resolve target_probes into a per-session mapping.
    if isinstance(target_probes, Mapping):
        missing = [s for s in sessions if s not in target_probes]
        if missing:
            raise KeyError(
                f"target_probes mapping is missing entries for sessions: {missing}"
            )
        probes_for: Dict[str, Sequence[str]] = {s: target_probes[s] for s in sessions}
    else:
        probes_for = {s: target_probes for s in sessions}

    all_results: Dict[str, Dict[str, TDRProjectionResult]] = {}
    for sess in sessions:
        try:
            all_results[sess] = _run_single_session_tdr_projection(
                session_name=sess,
                latent_var=latent_var,
                time_window=time_window,
                conditions=conditions,
                target_probes=probes_for[sess],
                binsize=binsize,
                align=align,
                psth_align_events=psth_align_events,
                psth_time_window=psth_time_window,
                psth_folder=psth_folder,
                behavior_folder=behavior_folder,
                tdr_folder=tdr_folder,
            )
        except Exception as e:  # noqa: BLE001
            print(f"❌ Error processing session {sess}: {type(e).__name__}: {e}")
            if single_session:
                raise
            all_results[sess] = {}

    return all_results[sessions[0]] if single_session else all_results


def _run_single_session_tdr_projection(
    *,
    session_name: str,
    latent_var: str,
    time_window: Tuple[float, float],
    conditions: Mapping[str, ConditionSpec],
    target_probes: Sequence[str],
    binsize: float,
    align: str,
    psth_align_events: Sequence[str],
    psth_time_window: Tuple[float, float],
    psth_folder: str | Path,
    behavior_folder: str | Path,
    tdr_folder: str | Path,
) -> Dict[str, TDRProjectionResult]:
    """Run the full pipeline for a single session. See public wrapper for docs."""
    from create_psth import load_zarr
    from general_utils import smart_read_csv

    # 1) PSTH + NWB
    psth_path, nwb_data = _ensure_psth_zarr(
        session_name=session_name,
        binsize=binsize,
        save_folder=psth_folder,
        align_to_event=psth_align_events,
        time_window=psth_time_window,
    )
    psth = load_zarr(str(psth_path))

    # 2) Behavior summary
    csv_path = _ensure_behavior_summary(session_name, behavior_folder)
    behavior_df = smart_read_csv(str(csv_path))
    if "response_trials" not in behavior_df.columns:
        raise KeyError("Column 'response_trials' missing in behavior summary CSV.")
    response_ids = np.asarray(behavior_df["response_trials"][0], dtype=int)

    # 3) TDR axis
    tdr_path = _ensure_tdr_axis(
        session_name=session_name,
        latent_var=latent_var,
        time_window=time_window,
        align=align,
        target_probes=target_probes,
        psth_da=psth,
        behavior_df=behavior_df,
        nwb_data=nwb_data,
        tdr_folder=tdr_folder,
    )

    # Latent on the response-only axis (used to label projected trials).
    if latent_var not in behavior_df.columns:
        raise KeyError(f"'{latent_var}' not in behavior CSV columns.")
    latent_response_only = np.asarray(behavior_df[latent_var][0], dtype=float)
    if latent_response_only.shape[0] != response_ids.shape[0]:
        raise ValueError(
            f"Latent length mismatch for {latent_var}: "
            f"{latent_response_only.shape[0]} vs {response_ids.shape[0]}"
        )
    response_id_to_latent = dict(zip(response_ids.tolist(), latent_response_only))

    # 4) Resolve conditions and project
    summary = summarize_laser_conditions(nwb_data)
    results: Dict[str, TDRProjectionResult] = {}
    for label, spec in conditions.items():
        include_trials = _resolve_condition_trials(
            spec, summary=summary, response_ids=response_ids, nwb_data=nwb_data
        )
        if include_trials.size == 0:
            print(f"[WARN] [{session_name}] Condition '{label}' resolved to 0 trials; skipping.")
            continue

        Y, t, used_unit_ids = project_psth_with_tdr_axis(
            psth=psth,
            nwb_data=nwb_data,
            tdr_path=tdr_path,
            include_trials=include_trials,
            target_probes=target_probes,
            align=align,
        )
        latent_vals = np.array(
            [response_id_to_latent.get(int(tid), np.nan) for tid in include_trials],
            dtype=float,
        )
        results[label] = TDRProjectionResult(
            label=label,
            Y=Y,
            t=t,
            trial_ids=include_trials.astype(int),
            latent=latent_vals,
            unit_ids=used_unit_ids,
        )
        print(
            f"[OK] [{session_name}] {label}: Y={Y.shape}, "
            f"n_trials={include_trials.size}, n_units={used_unit_ids.size}"
        )

    return results
