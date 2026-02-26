# -*- coding: utf-8 -*-
"""
Self-contained Coding Direction (CD) helper for xarray PSTH datasets.

Assumptions about `psth_da`:
- Data variables (choose via `align`):
    psth_go_cue(unit, trial_go_cue, time)
    psth_reward_go_cue_start(unit, trial_reward_go_cue_start, time)
- Coordinates:
    time(time), unit_index(unit), trial_index_<align>(trial_<align>)
- Attributes (optional):
    bin_size

Main entry point:
    coding_direction_from_psth(...)

Key returns:
    - 'axis_w'                       : (N_units,) unit weights (unit-norm) from the first train split
    - 'projection_test'              : (T_test_total,) held-out scalar projections (normalized)
    - 'projection_trace_test'        : (T_test_total, Tt) held-out time-resolved projections (normalized)
    - 'labels_test'                  : (T_test_total,) +1 for typeA, -1 for typeB
    - 'trial_ids_test'               : (T_test_total,)
    - 'projection_train'             : (T_train_total,) training scalar projections (normalized)
    - 'projection_trace_train'       : (T_train_total, Tt) training time-resolved projections (normalized)
    - 'labels_train'                 : (T_train_total,)
    - 'trial_ids_train'              : (T_train_total,)
    - 'time_for_projection'          : (Tt,)
    - 'metrics'                      : {'folds': [...], 'overall': {'dprime', 'auc'}}
    - 'final_all'                    : {'axis_w', 'mu', 'sigma', 'idx_all', 'labels_all_pm1'}
    - Bookkeeping                    : 'unit_ids','align','time_window_fit','projection_time_window',
                                       'norm_mode','norm_factor','y_fit_mean','y_fit_std'
    - Optional saving                : 'saved_to','saved_format'
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize a vector with numerical safety.

    Parameters
    ----------
    v : np.ndarray
        Input vector of shape (N,).
    eps : float, default 1e-12
        Small constant added to the norm for numerical stability.

    Returns
    -------
    np.ndarray
        Unit-norm vector of shape (N,).
    """
    n = np.linalg.norm(v) + eps
    return v / n


# ---------------------------------------------------------------------------
# PSTH extractors (xarray)
# ---------------------------------------------------------------------------

def extract_trial_unit_rates(
    psth_da: xr.Dataset,
    align: str = "go_cue",
    time_window: Tuple[float, float] = (0.0, 0.5),
    zscore_units: bool = False,
    unit_ids: Optional[Union[np.ndarray, List[int]]] = None,
) -> Dict:
    """
    Average firing rates within a time window to form a (trials × units) matrix.

    Parameters
    ----------
    psth_da : xr.Dataset
        PSTH dataset with data variables for both alignments:
        - 'psth_go_cue' (unit, trial_go_cue, time)
        - 'psth_reward_go_cue_start' (unit, trial_reward_go_cue_start, time)
        Coordinates must include 'time', 'unit_index', and 'trial_index_<align>'.
    align : {'go_cue', 'reward_go_cue_start'}, default 'go_cue'
        Which alignment/event to use. Chooses the data variable and trial index/axis.
    time_window : (float, float), default (0.0, 0.5)
        Time range [t0, t1) in seconds relative to `align` over which to average.
    zscore_units : bool, default False
        If True, z-score each unit (column-wise) across trials after averaging.
    unit_ids : array-like of int or None, optional
        If provided, restrict the computation to units whose `unit_index` is in
        this list. If None, all units are used.

    Returns
    -------
    dict
        'R'         : np.ndarray, shape (T, N)
            Trial × unit mean firing rates (optionally z-scored).
        'trial_ids' : np.ndarray, shape (T,)
            Trial IDs from 'trial_index_<align>'.
        'unit_ids'  : np.ndarray, shape (N,)
            Unit IDs from 'unit_index' (possibly restricted to `unit_ids`).
        'time_mask' : np.ndarray (bool), shape (T_time,)
            Boolean mask over `psth_da['time']` selecting the averaging window.
        'bin_size'  : float or None
            Bin size from dataset attrs if present.
    """
    if align == "go_cue":
        var, trial_dim, trial_coord = "psth_go_cue", "trial_go_cue", "trial_index_go_cue"
    elif align == "reward_go_cue_start":
        var, trial_dim, trial_coord = (
            "psth_reward_go_cue_start",
            "trial_reward_go_cue_start",
            "trial_index_reward_go_cue_start",
        )
    else:
        raise ValueError(f"Unknown align='{align}'")

    if var not in psth_da.data_vars:
        raise KeyError(f"Data variable '{var}' not found in dataset.")

    da = psth_da[var]  # (unit, trial, time)

    # Optional unit subsetting
    if unit_ids is not None:
        if "unit_index" not in psth_da.coords:
            raise KeyError("Cannot subset by unit_ids: 'unit_index' coordinate not found in dataset.")
        unit_ids_arr = np.asarray(unit_ids)
        full_unit_index = psth_da["unit_index"].values
        mask_units = np.isin(full_unit_index, unit_ids_arr)
        if not np.any(mask_units):
            raise ValueError("None of the requested unit_ids were found in 'unit_index'.")
        da = da.isel(unit=mask_units)

    time = psth_da["time"].values
    t0, t1 = time_window
    tmask = (time >= t0) & (time < t1)
    if not np.any(tmask):
        raise ValueError("time_window selects no samples. Check 'time' values.")

    mean_rates = da.sel(time=tmask).mean(dim="time").transpose(trial_dim, "unit")
    R = mean_rates.values  # (T × N)

    trial_ids = (
        psth_da[trial_coord].values
        if trial_coord in psth_da.coords
        else np.arange(R.shape[0])
    )

    # Use the (possibly subset) unit_index from the data array if available
    if "unit_index" in da.coords:
        unit_ids_out = da["unit_index"].values
    elif "unit_index" in psth_da.coords:
        unit_ids_out = psth_da["unit_index"].values
    else:
        unit_ids_out = np.arange(R.shape[1])

    if zscore_units:
        m = R.mean(axis=0, keepdims=True)
        s = R.std(axis=0, keepdims=True) + 1e-9
        R = (R - m) / s

    bin_size = psth_da.attrs.get("bin_size", None)
    return dict(
        R=R,
        trial_ids=trial_ids,
        unit_ids=unit_ids_out,
        time_mask=tmask,
        bin_size=bin_size,
    )

def extract_trial_unit_timecube(
    psth_da: xr.Dataset,
    align: str = "go_cue",
    time_window: Optional[Tuple[float, float]] = None,
    unit_ids: Optional[Union[np.ndarray, List[int]]] = None,
) -> Dict:
    """
    Extract a 3D tensor (trials × units × timepoints) for time-resolved projection.

    Parameters
    ----------
    psth_da : xr.Dataset
        PSTH dataset as above.
    align : {'go_cue', 'reward_go_cue_start'}, default 'go_cue'
        Which alignment/event to use. Chooses the data variable and trial index/axis.
    time_window : (float, float) or None, default None
        If provided, restrict to this time range [t0, t1) from `psth_da['time']`.
        If None, the full time axis is used.
    unit_ids : array-like of int or None, optional
        If provided, restrict the tensor to units whose `unit_index` is in
        this list. If None, all units are used.

    Returns
    -------
    dict
        'cube'      : np.ndarray, shape (T, N, Tt)
            Trial × unit × time tensor of firing rates.
        'trial_ids' : np.ndarray, shape (T,)
            Trial IDs corresponding to the first dimension of 'cube'.
        'time'      : np.ndarray, shape (Tt,)
            Time vector used for the third dimension of 'cube'.
    """
    if align == "go_cue":
        var, trial_dim, trial_coord = "psth_go_cue", "trial_go_cue", "trial_index_go_cue"
    elif align == "reward_go_cue_start":
        var, trial_dim, trial_coord = (
            "psth_reward_go_cue_start",
            "trial_reward_go_cue_start",
            "trial_index_reward_go_cue_start",
        )
    else:
        raise ValueError(f"Unknown align='{align}'")

    if var not in psth_da.data_vars:
        raise KeyError(f"Data variable '{var}' not found in dataset.")

    da = psth_da[var]

    # Optional unit subsetting
    if unit_ids is not None:
        if "unit_index" not in psth_da.coords:
            raise KeyError("Cannot subset by unit_ids: 'unit_index' coordinate not found in dataset.")
        unit_ids_arr = np.asarray(unit_ids)
        full_unit_index = psth_da["unit_index"].values
        mask_units = np.isin(full_unit_index, unit_ids_arr)
        if not np.any(mask_units):
            raise ValueError("None of the requested unit_ids were found in 'unit_index'.")
        da = da.isel(unit=mask_units)

    time = psth_da["time"].values
    if time_window is None:
        tmask = np.ones_like(time, dtype=bool)
    else:
        t0, t1 = time_window
        tmask = (time >= t0) & (time < t1)
    if not np.any(tmask):
        raise ValueError("projection_time_window selects no samples.")

    cube = da.sel(time=tmask).transpose(trial_dim, "unit", "time").values
    trial_ids = (
        psth_da[trial_coord].values
        if trial_coord in psth_da.coords
        else np.arange(cube.shape[0])
    )
    return dict(cube=cube, trial_ids=trial_ids, time=time[tmask])



# ---------------------------------------------------------------------------
# Splitting & indexing
# ---------------------------------------------------------------------------

@dataclass
class CDSplit:
    """
    Container for per-class half-split indices.

    Attributes
    ----------
    idx_train_a : np.ndarray
        Training indices (row indices into full trial list) for class A.
    idx_test_a : np.ndarray
        Held-out/test indices for class A.
    idx_train_b : np.ndarray
        Training indices for class B.
    idx_test_b : np.ndarray
        Held-out/test indices for class B.
    """
    idx_train_a: np.ndarray
    idx_test_a: np.ndarray
    idx_train_b: np.ndarray
    idx_test_b: np.ndarray


def _ids_to_indices(trial_ids_full: np.ndarray, ids: np.ndarray, *, require_all: bool=True) -> np.ndarray:
    """
    Map a list/array of trial IDs to row indices in `trial_ids_full` (preserves dataset order).

    Parameters
    ----------
    trial_ids_full : np.ndarray, shape (T_full,)
        Trial ID vector from the dataset for the chosen alignment.
    ids : array-like of int
        Trial IDs to map to row indices.
    require_all : bool, default True
        If True, raise if any provided ID is not present in `trial_ids_full`.
        If False, silently ignore IDs not present.

    Returns
    -------
    np.ndarray
        Sorted indices into `trial_ids_full` corresponding to `ids` that were found.
    """
    ids = np.asarray(ids, dtype=int).ravel()
    present_mask = np.isin(ids, trial_ids_full)
    if require_all and not np.all(present_mask):
        missing = ids[~present_mask]
        raise ValueError(f"Some provided trial IDs are not present in the dataset: {missing.tolist()}")
    keep_ids = ids[present_mask]
    positions = {tid: i for i, tid in enumerate(trial_ids_full)}
    idx = np.array([positions[tid] for tid in keep_ids], dtype=int)
    return np.sort(idx)


def _half_split(idx: np.ndarray, *, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic 50/50 split of indices.

    Parameters
    ----------
    idx : np.ndarray
        1D array of indices to split.
    random_state : int, default 0
        Seed for the permutation used before halving.

    Returns
    -------
    (np.ndarray, np.ndarray)
        (train_indices, test_indices) sorted within each array.
        If len(idx) is odd, floor(len(idx)/2) go to train; rest to test.
    """
    rng = np.random.RandomState(int(random_state))
    perm = idx.copy()
    rng.shuffle(perm)
    k = len(perm) // 2
    return np.sort(perm[:k]), np.sort(perm[k:])


def _balanced_half_split(idx_a: np.ndarray, idx_b: np.ndarray, *, random_state: int = 0) -> CDSplit:
    """
    Make 50/50 splits within each class (A and B) independently.

    Parameters
    ----------
    idx_a : np.ndarray
        Indices for class A (rows into the dataset trial list).
    idx_b : np.ndarray
        Indices for class B.
    random_state : int, default 0
        Seed for shuffling before halving.

    Returns
    -------
    CDSplit
        Structured container with train/test indices for each class.
    """
    tr_a, te_a = _half_split(idx_a, random_state=random_state)
    tr_b, te_b = _half_split(idx_b, random_state=random_state)
    return CDSplit(tr_a, te_a, tr_b, te_b)


# ---------------------------------------------------------------------------
# Coding direction & metrics
# ---------------------------------------------------------------------------

def _compute_cd_axis(R_z: np.ndarray, labels_pm1: np.ndarray) -> np.ndarray:
    """
    Compute the coding-direction (CD) axis as mean(+1) - mean(-1) on TRAIN data.

    Parameters
    ----------
    R_z : np.ndarray, shape (T_train, N_units)
        Z-scored (per unit) firing rates in the fit window for training trials.
    labels_pm1 : np.ndarray, shape (T_train,)
        Labels for training trials, must be +1 for class A and -1 for class B.

    Returns
    -------
    np.ndarray
        Unit-norm CD axis of shape (N_units,).
    """
    pos = R_z[labels_pm1 == +1]
    neg = R_z[labels_pm1 == -1]
    mu_pos = pos.mean(axis=0) if len(pos) > 0 else np.zeros(R_z.shape[1], float)
    mu_neg = neg.mean(axis=0) if len(neg) > 0 else np.zeros(R_z.shape[1], float)
    return _unit_norm(mu_pos - mu_neg)

def _roc_auc_binary(y_true_pm1: np.ndarray, y_score: np.ndarray) -> float:
    """
    Pure-NumPy ROC-AUC for binary labels in {+1, -1}.

    Parameters
    ----------
    y_true_pm1 : np.ndarray, shape (T,)
        Binary labels in {+1, -1}.
    y_score : np.ndarray, shape (T,)
        Continuous prediction scores.

    Returns
    -------
    float
        ROC-AUC in [0, 1]; NaN if either class is missing.
        Computation uses Mann–Whitney U normalized by (n_pos * n_neg) with tie-aware average ranks.
    """
    y_true_pm1 = np.asarray(y_true_pm1, dtype=float).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    mask = np.isfinite(y_score) & np.isfinite(y_true_pm1)
    y_true_pm1 = y_true_pm1[mask]
    y_score = y_score[mask]
    pos = (y_true_pm1 > 0).astype(bool)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = np.argsort(y_score, kind="mergesort")  # 1-based avg ranks
    sum_pos_ranks = ranks[pos].sum()
    U = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    auc = U / (n_pos * n_neg)
    return float(auc)


def _apply_norm(y_fit: np.ndarray, Y_trace: np.ndarray, mode: str, N_units: int):
    """
    Post-hoc normalization (matches TDR-style semantics).

    Parameters
    ----------
    y_fit : np.ndarray, shape (T,)
        Scalar projections (e.g., held-out per-trial values in the fit window).
    Y_trace : np.ndarray, shape (T, Tt)
        Time-resolved projections per trial.
    mode : {'none','divide_sqrtN','unit_variance_fit','zscore_fit'}
        Normalization strategy:
        - 'none'             : no change.
        - 'divide_sqrtN'     : divide by sqrt(#units).
        - 'unit_variance_fit': divide by std(y_fit).
        - 'zscore_fit'       : (y - mean(y_fit)) / std(y_fit).
    N_units : int
        Number of units; used for 'divide_sqrtN'.

    Returns
    -------
    (np.ndarray, np.ndarray, str, float, float, float)
        (y_norm, Y_norm, norm_mode_used, norm_factor, y_fit_mean, y_fit_std)
        - y_norm, Y_norm are the normalized scalar/time-resolved projections
        - norm_mode_used is the resolved mode string
        - norm_factor is the scaling factor (meaning depends on mode)
        - y_fit_mean, y_fit_std are computed from `y_fit`
    """
    mode_in = (mode or "none").lower()
    if mode_in in {"divide_sqrtn", "divide_sqrtn "}:
        mode_in = "divide_sqrtN"
    if mode_in not in {"none", "divide_sqrtN", "unit_variance_fit", "zscore_fit"}:
        raise ValueError("norm_mode must be one of {'none','divide_sqrtN','unit_variance_fit','zscore_fit'}")

    y_mean = float(np.nanmean(y_fit))
    y_std  = float(np.nanstd(y_fit) + 1e-12)

    if mode == "divide_sqrtN":
        factor = float(np.sqrt(N_units))
        return y_fit / factor, Y_trace / factor, mode, factor, y_mean, y_std
    elif mode == "unit_variance_fit":
        factor = y_std
        return y_fit / factor, Y_trace / factor, mode, factor, y_mean, y_std
    elif mode == "zscore_fit":
        factor = y_std
        return (y_fit - y_mean) / factor, (Y_trace - y_mean) / factor, mode, factor, y_mean, y_std
    else:
        return y_fit.copy(), Y_trace.copy(), mode, 1.0, y_mean, y_std


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def coding_direction_from_psth(
    psth_da: xr.Dataset,
    trial_ids_typeA: Union[np.ndarray, List[int]],
    trial_ids_typeB: Union[np.ndarray, List[int]],
    *,
    align: str = "go_cue",
    time_window: Tuple[float, float] = (0.0, 0.5),
    projection_time_window: Optional[Tuple[float, float]] = None,
    random_state: int = 0,
    two_fold_cv: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    save_format: str = "zarr",
    overwrite: bool = True,
    norm_mode: str = "divide_sqrtN",
    unit_ids: Optional[Union[np.ndarray, List[int]]] = None,
) -> Dict:
    """
    Compute a Coding Direction (typeA - typeB) using half of trials for training,
    project both held-out (test) and training halves (train), and optionally save.

    NEW: also returns and saves A/B **separate** projections and traces for both train and test.

    Parameters
    ----------
    psth_da : xr.Dataset
        PSTH dataset (see module docstring for structure).
    trial_ids_typeA : array-like of int
        Trial IDs for class A (mapped via trial_index_<align>).
    trial_ids_typeB : array-like of int
        Trial IDs for class B (mapped via trial_index_<align>).
    align : {'go_cue', 'reward_go_cue_start'}, default 'go_cue'
        Alignment/event used.
    time_window : (float, float), default (0.0, 0.5)
        Fit window for CD axis.
    projection_time_window : (float, float) or None, default None
        Window for time-resolved projections (None → full time axis).
    random_state : int, default 0
        Seed for half-split permutation.
    two_fold_cv : bool, default True
        If True, also swap train/test and concatenate outputs.
    save_path : str | Path | None, default None
        Optional output path.
    save_format : {'npz','nc','zarr'}, default 'zarr'
        Persist format when saving.
    overwrite : bool, default True
        Overwrite zarr store if it exists.
    norm_mode : {'none','divide_sqrtN','unit_variance_fit','zscore_fit'}, default 'divide_sqrtN'
        Post-hoc normalization strategy.
    unit_ids : array-like of int or None, optional
        If provided, compute the coding direction using only units whose
        `unit_index` matches these IDs. If None, all units are used.

    Returns
    -------
    dict
        Includes everything from the previous version, plus **per-class** splits:
        - Train (normalized):
            'projection_train_A', 'projection_trace_train_A', 'trial_ids_train_A'
            'projection_train_B', 'projection_trace_train_B', 'trial_ids_train_B'
        - Test (normalized):
            'projection_test_A', 'projection_trace_test_A', 'trial_ids_test_A'
            'projection_test_B', 'projection_trace_test_B', 'trial_ids_test_B'
        The combined arrays ('projection_train', 'projection_test', etc.) are still returned.
    """
    # 1) Fit-window trial × unit matrix (no z-score yet); collect IDs
    fit_ext = extract_trial_unit_rates(
        psth_da,
        align=align,
        time_window=time_window,
        zscore_units=False,
        unit_ids=unit_ids,
    )
    R_fit_full = fit_ext["R"]             # (T_full × N_units)
    trial_ids_full = fit_ext["trial_ids"] # (T_full,)
    unit_ids_selected = fit_ext["unit_ids"]

    # Map trial IDs → row indices
    idx_a_all = _ids_to_indices(trial_ids_full, np.asarray(trial_ids_typeA), require_all=True)
    idx_b_all = _ids_to_indices(trial_ids_full, np.asarray(trial_ids_typeB), require_all=True)

    # 2) Balanced half split
    split = _balanced_half_split(idx_a_all, idx_b_all, random_state=random_state)
    folds = [
        dict(
            train=np.concatenate([split.idx_train_a, split.idx_train_b]),
            test=np.concatenate([split.idx_test_a, split.idx_test_b]),
            train_labels=np.concatenate([np.ones(len(split.idx_train_a)), -np.ones(len(split.idx_train_b))]),
            test_labels=np.concatenate([np.ones(len(split.idx_test_a)), -np.ones(len(split.idx_test_b))]),
        ),
    ]
    if two_fold_cv:
        folds.append(
            dict(
                train=np.concatenate([split.idx_test_a, split.idx_test_b]),
                test=np.concatenate([split.idx_train_a, split.idx_train_b]),
                train_labels=np.concatenate([np.ones(len(split.idx_test_a)), -np.ones(len(split.idx_test_b))]),
                test_labels=np.concatenate([np.ones(len(split.idx_train_a)), -np.ones(len(split.idx_train_b))]),
            )
        )

    # 3) Time-cube for time-resolved projection (we z-score with TRAIN mu/sigma)
    cube_ext = extract_trial_unit_timecube(
        psth_da,
        align=align,
        time_window=projection_time_window,
        unit_ids=unit_ids,
    )
    cube_full = cube_ext["cube"]  # (T_full, N_units, Tt)
    time_vec = cube_ext["time"]

    # Accumulators
    y_test_all, lab_test_all, id_test_all, Yt_test_all = [], [], [], []
    y_train_all, lab_train_all, id_train_all, Yt_train_all = [], [], [], []
    per_fold_stats = []
    w_first_fold = None

    # 4) Per-fold train/test computation
    for fd in folds:
        train_order = np.argsort(fd["train"])
        test_order = np.argsort(fd["test"])

        tr_idx = fd["train"][train_order]  # Sorted training indices
        te_idx = fd["test"][test_order]    # Sorted testing indices

        y_tr_labels = fd["train_labels"].astype(float)[train_order]
        y_te_labels = fd["test_labels"].astype(float)[test_order]

        # Z-score using TRAIN trials only (fit window)
        R_tr = R_fit_full[tr_idx]
        mu = R_tr.mean(axis=0, keepdims=True)
        sigma = R_tr.std(axis=0, keepdims=True) + 1e-9
        R_tr_z = (R_tr - mu) / sigma

        # CD axis on training
        w = _compute_cd_axis(R_tr_z, y_tr_labels)
        if w_first_fold is None:
            w_first_fold = w.copy()

        # ---- Training projections (fit-window & time-resolved) ----
        y_tr = R_tr_z @ w  # (T_tr,)

        cube_tr = cube_full[tr_idx]  # (T_tr, N, Tt)
        cube_tr_z = (cube_tr - mu[:, :, None]) / sigma[:, :, None]
        Yt_tr = np.tensordot(cube_tr_z, w, axes=([1], [0]))  # (T_tr, Tt)

        # ---- Held-out projections (fit-window & time-resolved) ----
        R_te = R_fit_full[te_idx]
        R_te_z = (R_te - mu) / sigma
        y_te = R_te_z @ w  # (T_te,)

        cube_te = cube_full[te_idx]
        cube_te_z = (cube_te - mu[:, :, None]) / sigma[:, :, None]
        Yt_te = np.tensordot(cube_te_z, w, axes=([1], [0]))  # (T_te, Tt)

        # Metrics on held-out
        y_pos = y_te[y_te_labels == +1]
        y_neg = y_te[y_te_labels == -1]
        dprime = float(
            (np.nanmean(y_pos) - np.nanmean(y_neg))
            / (np.sqrt(0.5 * (np.nanvar(y_pos) + np.nanvar(y_neg))) + 1e-12)
        )
        auc = _roc_auc_binary(y_te_labels, y_te)
        per_fold_stats.append(
            {
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "dprime": dprime,
                "auc": auc,
            }
        )

        # Accumulate
        y_train_all.append(y_tr)
        lab_train_all.append(y_tr_labels)
        id_train_all.append(trial_ids_full[tr_idx])
        Yt_train_all.append(Yt_tr)

        y_test_all.append(y_te)
        lab_test_all.append(y_te_labels)
        id_test_all.append(trial_ids_full[te_idx])
        Yt_test_all.append(Yt_te)

    # 5) Concatenate across folds
    y_train_all = np.concatenate(y_train_all, axis=0)
    lab_train_all = np.concatenate(lab_train_all, axis=0)
    id_train_all = np.concatenate(id_train_all, axis=0)
    Yt_train_all = np.concatenate(Yt_train_all, axis=0)

    y_test_all = np.concatenate(y_test_all, axis=0)
    lab_test_all = np.concatenate(lab_test_all, axis=0)
    id_test_all = np.concatenate(id_test_all, axis=0)
    Yt_test_all = np.concatenate(Yt_test_all, axis=0)

    # ---- Normalize using TEST stats (keeps train/test directly comparable) ----
    y_test_norm, Yt_test_norm, norm_mode_used, norm_factor, y_mean, y_std = _apply_norm(
        y_test_all, Yt_test_all, norm_mode, N_units=len(unit_ids_selected)
    )
    if norm_mode_used == "divide_sqrtN":
        y_tr_norm = y_train_all / norm_factor
        Yt_tr_norm = Yt_train_all / norm_factor
    elif norm_mode_used == "unit_variance_fit":
        y_tr_norm = y_train_all / norm_factor
        Yt_tr_norm = Yt_train_all / norm_factor
    elif norm_mode_used == "zscore_fit":
        y_tr_norm = (y_train_all - y_mean) / norm_factor
        Yt_tr_norm = (Yt_train_all - y_mean) / norm_factor
    else:
        y_tr_norm = y_train_all.copy()
        Yt_tr_norm = Yt_train_all.copy()

    # ---- NEW: split train/test by class (A=+1, B=-1), keep normalized outputs ----
    trA_mask = lab_train_all == +1
    trB_mask = lab_train_all == -1
    teA_mask = lab_test_all == +1
    teB_mask = lab_test_all == -1

    projection_train_A = y_tr_norm[trA_mask]
    projection_trace_train_A = Yt_tr_norm[trA_mask]
    trial_ids_train_A = id_train_all[trA_mask].astype(int)

    projection_train_B = y_tr_norm[trB_mask]
    projection_trace_train_B = Yt_tr_norm[trB_mask]
    trial_ids_train_B = id_train_all[trB_mask].astype(int)

    projection_test_A = y_test_norm[teA_mask]
    projection_trace_test_A = Yt_test_norm[teA_mask]
    trial_ids_test_A = id_test_all[teA_mask].astype(int)

    projection_test_B = y_test_norm[teB_mask]
    projection_trace_test_B = Yt_test_norm[teB_mask]
    trial_ids_test_B = id_test_all[teB_mask].astype(int)

    # 7) Provide a "final" CD using ALL provided A & B trials (z-scored on all A∪B)
    idx_all_inc = np.sort(np.concatenate([idx_a_all, idx_b_all]))
    labels_all_pm1 = np.concatenate(
        [np.ones(len(idx_a_all), float), -np.ones(len(idx_b_all), float)]
    )
    perm = np.argsort(np.concatenate([idx_a_all, idx_b_all]))
    labels_all_pm1 = labels_all_pm1[perm]

    R_all = R_fit_full[idx_all_inc]
    mu_all = R_all.mean(axis=0, keepdims=True)
    sigma_all = R_all.std(axis=0, keepdims=True) + 1e-9
    R_all_z = (R_all - mu_all) / sigma_all
    w_final_all = _compute_cd_axis(R_all_z, labels_all_pm1)

    out = {
        # Axes & combined (normalized) outputs
        "axis_w": w_first_fold,
        "projection_test": y_test_norm,
        "projection_trace_test": Yt_test_norm,
        "labels_test": lab_test_all,
        "trial_ids_test": id_test_all.astype(int),
        "projection_train": y_tr_norm,
        "projection_trace_train": Yt_tr_norm,
        "labels_train": lab_train_all,
        "trial_ids_train": id_train_all.astype(int),
        # ---- NEW: per-class (normalized) outputs ----
        "projection_train_A": projection_train_A,
        "projection_trace_train_A": projection_trace_train_A,
        "trial_ids_train_A": trial_ids_train_A,
        "projection_train_B": projection_train_B,
        "projection_trace_train_B": projection_trace_train_B,
        "trial_ids_train_B": trial_ids_train_B,
        "projection_test_A": projection_test_A,
        "projection_trace_test_A": projection_trace_test_A,
        "trial_ids_test_A": trial_ids_test_A,
        "projection_test_B": projection_test_B,
        "projection_trace_test_B": projection_trace_test_B,
        "trial_ids_test_B": trial_ids_test_B,
        # Time & metrics
        "time_for_projection": time_vec,
        "metrics": {
            "folds": per_fold_stats,
            "overall": {
                "dprime": float(
                    (
                        np.nanmean(y_test_norm[teA_mask])
                        - np.nanmean(y_test_norm[teB_mask])
                    )
                    / (
                        np.sqrt(
                            0.5
                            * (
                                np.nanvar(y_test_norm[teA_mask])
                                + np.nanvar(y_test_norm[teB_mask])
                            )
                        )
                        + 1e-12
                    )
                ),
                "auc": _roc_auc_binary(lab_test_all, y_test_norm),
            },
        },
        # Bookkeeping
        "train_indices_folds": [np.sort(fd["train"]) for fd in folds],
        "test_indices_folds": [np.sort(fd["test"]) for fd in folds],
        "final_all": {
            "axis_w": w_final_all,
            "mu": mu_all,
            "sigma": sigma_all,
            "idx_all": idx_all_inc,
            "labels_all_pm1": labels_all_pm1,
        },
        "norm_mode": norm_mode_used,
        "norm_factor": float(norm_factor),
        "y_fit_mean": float(y_mean),
        "y_fit_std": float(y_std),
        "unit_ids": unit_ids_selected,
        "align": align,
        "time_window_fit": time_window,
        "projection_time_window": projection_time_window,
        "n_typeA": int(len(idx_a_all)),
        "n_typeB": int(len(idx_b_all)),
    }

    # 8) Optional save (unchanged except for using unit_ids_selected)
    if save_path is not None:
        path = Path(save_path)
        fmt = str(save_format).lower()
        if fmt not in {"npz", "nc", "zarr"}:
            raise ValueError("save_format must be one of {'npz','nc','zarr'}")

        attrs_payload = {
            "align": align,
            "time_window_fit": time_window,
            "projection_time_window": projection_time_window,
            "two_fold_cv": bool(two_fold_cv),
            "fold_stats": per_fold_stats,
            "norm_mode": norm_mode_used,
            "norm_factor": float(norm_factor),
            "y_fit_mean": float(y_mean),
            "y_fit_std": float(y_std),
            "n_typeA": int(len(idx_a_all)),
            "n_typeB": int(len(idx_b_all)),
        }

        if fmt == "npz":
            np.savez_compressed(
                path,
                # ---- raw (pre-normalization) ----
                projection_test=y_test_all,
                projection_trace_test=Yt_test_all,
                projection_train=y_train_all,
                projection_trace_train=Yt_train_all,
                # ---- normalized combined ----
                projection_test_norm=y_test_norm,
                projection_trace_test_norm=Yt_test_norm,
                projection_train_norm=y_tr_norm,
                projection_trace_train_norm=Yt_tr_norm,
                # ---- NEW: normalized per-class ----
                projection_train_A=projection_train_A,
                projection_trace_train_A=projection_trace_train_A,
                trial_ids_train_A=trial_ids_train_A,
                projection_train_B=projection_train_B,
                projection_trace_train_B=projection_trace_train_B,
                trial_ids_train_B=trial_ids_train_B,
                projection_test_A=projection_test_A,
                projection_trace_test_A=projection_trace_test_A,
                trial_ids_test_A=trial_ids_test_A,
                projection_test_B=projection_test_B,
                projection_trace_test_B=projection_trace_test_B,
                trial_ids_test_B=trial_ids_test_B,
                # meta
                labels_test=lab_test_all,
                trial_ids_test=id_test_all,
                labels_train=lab_train_all,
                trial_ids_train=id_train_all,
                time=time_vec,
                axis_w_first_fold=w_first_fold,
                axis_w_all=w_final_all,
                unit_ids=unit_ids_selected,
                attrs_str=json.dumps(attrs_payload),
            )
        else:
            ds = xr.Dataset(
                data_vars={
                    # combined
                    "projection_test": (("trial_test",), y_test_norm),
                    "projection_trace_test": (("trial_test", "time"), Yt_test_norm),
                    "labels_test": (("trial_test",), lab_test_all),
                    "projection_train": (("trial_train",), y_tr_norm),
                    "projection_trace_train": (("trial_train", "time"), Yt_tr_norm),
                    "labels_train": (("trial_train",), lab_train_all),
                    # per-class
                    "projection_test_A": (("trial_test_A",), projection_test_A),
                    "projection_trace_test_A": (("trial_test_A", "time"), projection_trace_test_A),
                    "projection_test_B": (("trial_test_B",), projection_test_B),
                    "projection_trace_test_B": (("trial_test_B", "time"), projection_trace_test_B),
                    "projection_train_A": (("trial_train_A",), projection_train_A),
                    "projection_trace_train_A": (("trial_train_A", "time"), projection_trace_train_A),
                    "projection_train_B": (("trial_train_B",), projection_train_B),
                    "projection_trace_train_B": (("trial_train_B", "time"), projection_trace_train_B),
                    # axes
                    "axis_w": (("unit",), w_first_fold),
                    "axis_w_all": (("unit",), w_final_all),
                },
                coords={
                    "trial_id_test": ("trial_test", id_test_all.astype(int)),
                    "trial_id_train": ("trial_train", id_train_all.astype(int)),
                    "trial_id_test_A": ("trial_test_A", trial_ids_test_A),
                    "trial_id_test_B": ("trial_test_B", trial_ids_test_B),
                    "trial_id_train_A": ("trial_train_A", trial_ids_train_A),
                    "trial_id_train_B": ("trial_train_B", trial_ids_train_B),
                    "time": ("time", np.asarray(time_vec, dtype=float)),
                    "unit_id": ("unit", np.asarray(unit_ids_selected, dtype=int)),
                },
                attrs=attrs_payload,
            )
            if fmt == "nc":
                ds.to_netcdf(path)
            elif fmt == "zarr":
                import shutil

                if path.exists() and overwrite:
                    shutil.rmtree(path)
                ds.to_zarr(path, mode="w")

        out["saved_to"] = str(path)
        out["saved_format"] = fmt

    return out

