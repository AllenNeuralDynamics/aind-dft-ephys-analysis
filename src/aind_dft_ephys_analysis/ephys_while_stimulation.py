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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Tuple

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
    """Return ``(data_var, trial_dim, trial_coord)`` for a PSTH alignment."""
    if align not in _ALIGN_TO_VAR:
        raise ValueError(
            f"Unknown align='{align}'. Valid: {sorted(_ALIGN_TO_VAR)}"
        )
    return _ALIGN_TO_VAR[align]


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


def _pick_psth_array(psth: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """Return a DataArray with dims that include ``unit`` and ``time``."""
    if isinstance(psth, xr.DataArray):
        return psth
    for name in ("psth_go_cue", "psth_reward_go_cue_start"):
        if name in psth.data_vars:
            v = psth[name]
            if "unit" in v.dims and "time" in v.dims:
                return v
    for name, var in psth.data_vars.items():
        if "unit" in var.dims and "time" in var.dims:
            return psth[name]
    raise RuntimeError("No DataArray with dims ('unit','time') found in PSTH.")


def _find_trial_dim_and_coord(arr: xr.DataArray) -> Tuple[str, str]:
    """Locate the trial dim and an integer-labeled trial coord on ``arr``."""
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
    arr = _pick_psth_array(psth)
    trial_dim, trial_coord = _find_trial_dim_and_coord(arr)

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
