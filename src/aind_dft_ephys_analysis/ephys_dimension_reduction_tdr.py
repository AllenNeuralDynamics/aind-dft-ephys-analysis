# -*- coding: utf-8 -*-
"""
TDR (demixed / targeted dimensionality reduction) for xarray PSTH datasets,
with per-trial time-resolved projections.

Assumptions about psth_da:
- Data variables:
    psth_go_cue(unit, trial_go_cue, time)
    psth_reward_go_cue_start(unit, trial_reward_go_cue_start, time)
- Coordinates:
    time(time), unit_index(unit), trial_index_<align>(trial_<align>)
- Attributes:
    bin_size (optional)

Main entry point:
    tdr_from_psth(...)

Returns dictionary with:
    - 'axis_w'                : (N_units,) neuronal dimension (unit loadings)
    - 'projection'            : (T_used,) scalar per-trial projection in the fit window
    - 'projection_trace'      : (T_used, Tt_proj) time-resolved per-trial projection
    - 'time_for_projection'   : (Tt_proj,) time vector used for projection_trace
    - 'trial_ids'             : (T_used,) trial IDs used (dataset order)
    - 'unit_ids'              : (N_units,) unit IDs
    - 'cv_corr', 'cv_r2', 'y_cv', 'z_cv', 'final' (from CV wrapper)
    - scaling + bookkeeping: 'zscore_mu', 'zscore_sigma', 'time_window_fit', etc.
"""
import json
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import xarray as xr
from scipy import stats
from sklearn.model_selection import KFold
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable



# ----------------------------- utilities -----------------------------

def _one_hot(vec: np.ndarray) -> np.ndarray:
    """One-hot encode a 1D categorical vector (strings/ints); drop base level."""
    cats = np.unique(vec.astype(object))
    oh = np.zeros((len(vec), len(cats)), dtype=float)
    for j, c in enumerate(cats):
        oh[:, j] = (vec == c).astype(float)
    if oh.shape[1] > 1:
        oh = oh[:, 1:]
    return oh

def _zscore_col(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Column-wise z-score for a 2D array or 1D vector."""
    x = np.asarray(x)
    if x.ndim == 1:
        m = x.mean(); s = x.std() + eps
        return (x - m) / s
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True) + eps
    return (x - m) / s

def _unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a vector to unit L2 norm (with a tiny epsilon for stability)."""
    n = np.linalg.norm(v) + eps
    return v / n

def _orthogonalize_axis(axis: np.ndarray, A: Optional[np.ndarray]) -> np.ndarray:
    """Make 'axis' orthogonal to span(A) using projection subtraction."""
    if A is None or A.size == 0:
        return axis
    P = A @ np.linalg.pinv(A)
    return axis - P @ axis

# --------------------- xarray extractors ------------------------------

def extract_trial_unit_rates(
    psth_da: xr.Dataset,
    align: str = "go_cue",
    time_window: Tuple[float, float] = (0.0, 0.5),
    zscore_units: bool = True
) -> Dict:
    """
    Average firing rates in a time window to get a 2D matrix (trials × units).

    Parameters
    ----------
    psth_da : xr.Dataset
        Dataset containing peri-stimulus time histograms with data variables:
        'psth_go_cue' or 'psth_reward_go_cue_start' of shape (unit, trial, time),
        and coordinates 'time', 'unit_index', and 'trial_index_<align>'.
    align : {"go_cue", "reward_go_cue_start"}
        Which alignment/event to use (chooses the data variable and trial dims).
    time_window : (float, float)
        Time window (in seconds) relative to the chosen alignment over which to
        average firing rate, inclusive of start and exclusive of end (t0 <= t < t1).
    zscore_units : bool
        If True, z-score each unit (column-wise) across trials after averaging.

    Returns
    -------
    dict
        R : np.ndarray
            (T × N) matrix of mean firing rates (trials × units).
        trial_ids : np.ndarray
            (T,) array of trial IDs from 'trial_index_<align>'.
        unit_ids : np.ndarray
            (N,) array of unit IDs from 'unit_index'.
        time_mask : np.ndarray (bool)
            Boolean mask over psth_da['time'] selecting the averaging window.
        bin_size : float or None
            Bin size from dataset attrs if present.
    """
    if align == "go_cue":
        var = "psth_go_cue"
        trial_dim = "trial_go_cue"
        trial_coord = "trial_index_go_cue"
    elif align == "reward_go_cue_start":
        var = "psth_reward_go_cue_start"
        trial_dim = "trial_reward_go_cue_start"
        trial_coord = "trial_index_reward_go_cue_start"
    else:
        raise ValueError(f"Unknown align='{align}'")

    assert var in psth_da.data_vars, f"{var} not in dataset data_vars."
    da = psth_da[var]  # (unit, trial, time)

    time = psth_da["time"].values
    t0, t1 = time_window
    tmask = (time >= t0) & (time < t1)
    if not np.any(tmask):
        raise ValueError("time_window selects no samples. Check 'time' values.")

    mean_rates = da.sel(time=tmask).mean(dim="time").transpose(trial_dim, "unit")
    R = mean_rates.values  # (T × N)
    trial_ids = psth_da[trial_coord].values if trial_coord in psth_da.coords else np.arange(R.shape[0])
    unit_ids = psth_da["unit_index"].values if "unit_index" in psth_da.coords else np.arange(R.shape[1])

    if zscore_units:
        R = _zscore_col(R)

    bin_size = psth_da.attrs.get("bin_size", None)
    return dict(R=R, trial_ids=trial_ids, unit_ids=unit_ids, time_mask=tmask, bin_size=bin_size)

def extract_trial_unit_timecube(
    psth_da: xr.Dataset,
    align: str = "go_cue",
    time_window: Optional[Tuple[float, float]] = None
) -> Dict:
    """
    Extract a 3D tensor (trials × units × timepoints) for time-resolved projection.

    Parameters
    ----------
    psth_da : xr.Dataset
        Dataset as described in `extract_trial_unit_rates`.
    align : {"go_cue", "reward_go_cue_start"}
        Which alignment/event to use.
    time_window : (float, float) or None
        If provided, select only this time range from 'time' (t0 <= t < t1).
        If None, use the full available time axis.

    Returns
    -------
    dict
        cube : np.ndarray
            (T × N × Tt) tensor of firing rates (trials × units × timepoints).
        trial_ids : np.ndarray
            (T,) trial IDs aligned with the first dimension of `cube`.
        time : np.ndarray
            (Tt,) time vector used for the third dimension of `cube`.
    """
    if align == "go_cue":
        var = "psth_go_cue"
        trial_dim = "trial_go_cue"
        trial_coord = "trial_index_go_cue"
    elif align == "reward_go_cue_start":
        var = "psth_reward_go_cue_start"
        trial_dim = "trial_reward_go_cue_start"
        trial_coord = "trial_index_reward_go_cue_start"
    else:
        raise ValueError(f"Unknown align='{align}'")
    da = psth_da[var]
    time = psth_da["time"].values
    if time_window is None:
        tmask = np.ones_like(time, dtype=bool)
    else:
        t0, t1 = time_window
        tmask = (time >= t0) & (time < t1)
    if not np.any(tmask):
        raise ValueError("projection_time_window selects no samples.")
    cube = da.sel(time=tmask).transpose(trial_dim, "unit", "time").values
    trial_ids = psth_da[trial_coord].values
    time_sel = time[tmask]
    return dict(cube=cube, trial_ids=trial_ids, time=time_sel)

# --------------------- design matrix construction ---------------------

def build_design_matrix(
    T: int,
    latent: Union[np.ndarray, List[float]],
    continuous: Optional[Dict[str, np.ndarray]] = None,
    categorical: Optional[Dict[str, np.ndarray]] = None,
    zscore_continuous: bool = True
) -> Tuple[np.ndarray, Dict[str, slice], int]:
    """
    Construct the predictor matrix for demixing: X = [ latent | continuous | one-hot(categorical) ].
    (Intercept is added later during fitting.)

    Parameters
    ----------
    T : int
        Number of trials (rows) to build.
    latent : array-like, shape (T,)
        The primary latent regressor to target with TDR. Internally z-scored.
    continuous : dict[str, array-like], optional
        Mapping from name → (T,) array for additional continuous covariates
        (e.g., pupil, running speed). Each is optionally z-scored (see below).
    categorical : dict[str, array-like], optional
        Mapping from name → (T,) array of categories (ints/strings). Each is
        one-hot encoded with the first level dropped to avoid collinearity.
    zscore_continuous : bool
        If True, z-score each continuous covariate column.

    Returns
    -------
    X_wo_intercept : np.ndarray
        Design matrix of shape (T × Q) *without* the intercept.
    col_slices : dict[str, slice]
        Map from each regressor name (e.g., "latent", "choice[1]") to its
        column slice in `X_wo_intercept`.
    latent_col : int
        Starting column index of the 'latent' regressor within `X_wo_intercept`.
    """
    latent = np.asarray(latent).reshape(T,)
    cols = []
    names = []

    cols.append(_zscore_col(latent)); names.append("latent")

    if continuous:
        for k, v in continuous.items():
            v = np.asarray(v).reshape(T,)
            cols.append(_zscore_col(v) if zscore_continuous else v)
            names.append(k)

    if categorical:
        for k, v in categorical.items():
            v = np.asarray(v).reshape(T,)
            oh = _one_hot(v)
            if oh.shape[1] == 0:
                continue
            cols.append(oh)
            for j in range(oh.shape[1]):
                names.append(f"{k}[{j+1}]")

    X_wo_intercept = np.column_stack(cols) if len(cols) > 0 else np.zeros((T, 0))

    col_slices: Dict[str, slice] = {}
    start = 0
    for nm, arr in zip(names, cols):
        width = arr.shape[1] if arr.ndim == 2 else 1
        col_slices[nm] = slice(start, start + width)
        start += width

    latent_col = col_slices["latent"].start
    return X_wo_intercept, col_slices, latent_col

# ----------------------------- core TDR ------------------------------

def tdr_fit(
    R: np.ndarray,
    X_wo_intercept: np.ndarray,
    latent_col: int,
    orth_names: Optional[List[str]] = None,
    col_slices: Optional[Dict[str, slice]] = None
) -> Dict:
    """
    Fit neuron-wise OLS to obtain a targeted projection axis for the latent.

    Parameters
    ----------
    R : np.ndarray, shape (T, N)
        Z-scored firing rates (trials × units) in the fit window.
    X_wo_intercept : np.ndarray, shape (T, Q)
        Design matrix without the intercept. The latent column must be present.
    latent_col : int
        Column index of the 'latent' regressor within `X_wo_intercept`.
    orth_names : list[str], optional
        Names of other regressors (keys in `col_slices`) to orthogonalize
        the latent axis against. If a block has multiple columns (e.g. a
        categorical one-hot), the leading singular vector is used as that
        block’s competing axis.
    col_slices : dict[str, slice], optional
        Mapping from regressor names to their column ranges within `X_wo_intercept`.
        Required if `orth_names` is used.

    Returns
    -------
    dict
        w : np.ndarray
            (N,) unit weights defining the neuronal dimension (unit-norm).
        betas : np.ndarray
            (N × (Q+1)) OLS coefficients per neuron (including intercept at col 0).
        y : np.ndarray
            (T,) projection of `R` onto the final axis w (same as R @ w).
        z : np.ndarray
            (T,) standardized latent regressor used to define the axis.
        corr : float
            Pearson correlation between y and z.
        r2 : float
            Squared correlation between y and z.
    """
    T, N = R.shape
    X = np.column_stack([np.ones((T, 1)), X_wo_intercept])  # add intercept

    XtX_pinv = np.linalg.pinv(X.T @ X)
    B = XtX_pinv @ (X.T @ R)  # (Q+1 × N)
    betas = B.T

    latent_idx = 1 + latent_col
    axis = betas[:, latent_idx].copy()

    if orth_names and col_slices is not None:
        A_cols = []
        for nm in orth_names:
            if nm not in col_slices: continue
            sl = col_slices[nm]
            block = betas[:, 1 + sl]
            if block.ndim == 1:
                A_cols.append(block)
            else:
                u, s, vh = np.linalg.svd(block, full_matrices=False)
                A_cols.append(u[:, 0] * s[0])
        if len(A_cols) > 0:
            A = np.column_stack(A_cols)
            P = A @ np.linalg.pinv(A)
            axis = axis - P @ axis

    w = _unit_norm(axis)

    y = R @ w
    z_std = X_wo_intercept[:, latent_col]
    sgn = np.sign(np.corrcoef(y, z_std)[0, 1] + 1e-12)
    w *= sgn
    y = R @ w

    corr = float(np.corrcoef(y, z_std)[0, 1])
    r2 = float(stats.pearsonr(y, z_std)[0] ** 2)

    return dict(w=w, betas=betas, y=y, z=z_std, corr=corr, r2=r2)

def tdr_cv(
    R: np.ndarray,
    X_wo_intercept: np.ndarray,
    latent_col: int,
    orth_names: Optional[List[str]] = None,
    col_slices: Optional[Dict[str, slice]] = None,
    n_splits: int = 5,
    random_state: Optional[int] = 0
) -> Dict:
    """
    Cross-validated evaluation of the TDR axis.

    Parameters
    ----------
    R : np.ndarray, shape (T, N)
        Z-scored firing rates (trials × units) for the fit window.
    X_wo_intercept : np.ndarray, shape (T, Q)
        Design matrix without intercept.
    latent_col : int
        Column index of the latent regressor in `X_wo_intercept`.
    orth_names : list[str], optional
        Regressor names to orthogonalize against (see `tdr_fit`).
    col_slices : dict[str, slice], optional
        Mapping of names to column slices (required if using `orth_names`).
    n_splits : int
        Number of KFold splits.
    random_state : int or None
        Random seed for shuffle.

    Returns
    -------
    dict
        cv_corr : np.ndarray
            (K,) correlations on held-out folds.
        cv_r2 : np.ndarray
            (K,) squared correlations.
        y_cv : np.ndarray
            (T,) concatenated held-out projections across folds.
        z_cv : np.ndarray
            (T,) concatenated held-out latents across folds.
        final : dict
            Result of refitting on all trials (same structure as `tdr_fit`).
    """
    T = R.shape[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    corrs = []; r2s = []
    y_all = np.zeros(T); z_all = np.zeros(T)
    for tr, te in kf.split(np.arange(T)):
        fit = tdr_fit(R[tr], X_wo_intercept[tr], latent_col, orth_names, col_slices)
        w = fit["w"]
        y_te = R[te] @ w
        z_te = X_wo_intercept[te, latent_col]
        y_all[te] = y_te
        z_all[te] = z_te
        c = float(np.corrcoef(y_te, z_te)[0,1])
        corrs.append(c); r2s.append(c**2)
    final = tdr_fit(R, X_wo_intercept, latent_col, orth_names, col_slices)
    return dict(cv_corr=np.array(corrs), cv_r2=np.array(r2s), y_cv=y_all, z_cv=z_all, final=final)

# ---------------------------- trial-ID mask ---------------------------

def _mask_from_trial_ids(
    trial_ids_full: np.ndarray,
    include_ids: Optional[Union[np.ndarray, List[int]]],
    require_all: bool = True,
) -> np.ndarray:
    """
    Build a boolean mask over trials using *trial IDs*.

    Parameters
    ----------
    trial_ids_full : np.ndarray, shape (T,)
        Trial IDs from psth_da['trial_index_<align>'] for the chosen alignment.
    include_ids : array-like of int or None
        Trial IDs to keep. If None, all trials are kept.
        Duplicates are ignored (masking does not depend on order).
    require_all : bool
        If True, raise an error if any requested ID is not present in `trial_ids_full`.

    Returns
    -------
    mask : np.ndarray (bool), shape (T,)
        True for trials to keep (preserves dataset order).
    """
    T = len(trial_ids_full)
    if include_ids is None:
        return np.ones(T, dtype=bool)

    inc = np.asarray(include_ids, dtype=int).ravel()
    _, uniq_idx = np.unique(inc, return_index=True)
    inc = inc[np.sort(uniq_idx)]

    present = np.isin(inc, trial_ids_full)
    if require_all and not np.all(present):
        missing = inc[~present]
        raise ValueError(
            f"The following include_trials IDs are not present in psth_da: {missing.tolist()}"
        )

    mask = np.isin(trial_ids_full, inc)
    if mask.sum() == 0:
        raise ValueError("include_trials matched 0 trials in psth_da.")
    return mask

# ---------------------------- High-level API -------------------------


def _package_tdr_to_xr(
    projection_trace: np.ndarray,
    projection_fit: np.ndarray,
    w_final: np.ndarray,
    t: np.ndarray,
    trial_ids_used: np.ndarray,
    unit_ids: np.ndarray,
    meta: Dict,
    *,
    projection_trace_norm: Optional[np.ndarray] = None,   # NEW
    projection_norm: Optional[np.ndarray] = None,         # NEW
    latent_raw: Optional[np.ndarray] = None,
    latent_z: Optional[np.ndarray] = None
) -> xr.Dataset:
    """
    Build an xarray.Dataset container for saving TDR outputs.

    Data variables in the result:
      - projection_trace(trial, time)
      - projection(trial)
      - projection_trace_norm(trial, time)
      - projection_norm(trial)
      - axis_w(unit)
      - latent(trial)      [optional: raw latent values]
      - latent_z(trial)    [optional: standardized latent values]
    Coordinates:
      - trial_id(trial)
      - time(time)
      - unit_id(unit)
    """
    data_vars = {
        "projection_trace": (("trial", "time"), projection_trace),
        "projection": (("trial",), projection_fit),
        "axis_w": (("unit",), w_final),
    }

    # --- include normalized data if present ---
    if projection_trace_norm is not None:
        data_vars["projection_trace_norm"] = (("trial", "time"), np.asarray(projection_trace_norm))
    if projection_norm is not None:
        data_vars["projection_norm"] = (("trial",), np.asarray(projection_norm))

    # --- include latent variables if provided ---
    if latent_raw is not None:
        data_vars["latent"] = (("trial",), np.asarray(latent_raw))
    if latent_z is not None:
        data_vars["latent_z"] = (("trial",), np.asarray(latent_z))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "trial_id": ("trial", np.asarray(trial_ids_used, dtype=int)),
            "time": ("time", np.asarray(t, dtype=float)),
            "unit_id": ("unit", np.asarray(unit_ids, dtype=int)),
        },
        attrs={k: v for k, v in meta.items()},
    )
    return ds




def tdr_from_psth(
    psth_da: xr.Dataset,
    latent: np.ndarray,
    align: str = "go_cue",
    time_window: Tuple[float, float] = (0.0, 0.5),
    continuous_covs: Optional[Dict[str, np.ndarray]] = None,
    categorical_covs: Optional[Dict[str, np.ndarray]] = None,
    orth_names: Optional[List[str]] = None,
    n_splits: int = 5,
    include_trials: Optional[Union[np.ndarray, List[int]]] = None,
    require_all_ids: bool = True,
    projection_time_window: Optional[Tuple[float, float]] = None,
    *,
    save_path: Optional[Union[str, Path]] = None,
    save_format: str = "zarr",         # {"npz", "nc", "zarr"}
    overwrite: bool = True,
    # -------- NEW --------
    norm_mode: str = "divide_sqrtN",           # {"none","divide_sqrtN","unit_variance_fit","zscore_fit"}
) -> Dict:
    """
    Perform Targeted Dimensionality Reduction (TDR) / demixed PCA on neural firing data
    (xarray PSTH dataset) against a behavioral/model latent, return a single neuronal
    axis and per-trial projections, and optionally SAVE the results. Supports optional
    **post-hoc normalization** of projections for cross-session comparability.

    -------------------------------------------------------------------------
    Parameters
    -------------------------------------------------------------------------
    psth_da : xr.Dataset
        PSTH dataset with data variables:
          - psth_go_cue(unit, trial_go_cue, time)
          - psth_reward_go_cue_start(unit, trial_reward_go_cue_start, time)
        Coordinates:
          - time(time)
          - unit_index(unit)
          - trial_index_<align>(trial_<align>)
        Attrs (optional): "bin_size".
        Produced by your PSTH extraction pipeline.

    latent : np.ndarray
        Behavioral/model latent you want the neural axis to encode.
        Accepts:
          • full length = #trials for the chosen `align` (T_full), or
          • subset length = #trials kept by `include_trials` (T_used).
        If full length is given, it is *auto-subset* to the included trials.

    align : {"go_cue","reward_go_cue_start"}, default "go_cue"
        Which alignment/event to use for extracting PSTHs and trials.

    time_window : (float, float), default (0.0, 0.5)
        Analysis window (in seconds, relative to `align`) used to *fit* the TDR axis.
        Rates are averaged in this window, z-scored per unit, and used for regression.

    continuous_covs : dict[str, np.ndarray], optional
        Additional continuous regressors (e.g., pupil, speed). Each vector may be
        length T_full (auto-subset) or T_used (already subset). Each column is z-scored.

    categorical_covs : dict[str, np.ndarray], optional
        Additional categorical regressors (strings/ints). Internally one-hot encodes
        each, dropping a base level to avoid collinearity.

    orth_names : list[str], optional
        Names of regressors (keys of `continuous_covs` / `categorical_covs`) to
        orthogonalize out of the latent axis after fitting. If a regressor expands
        into multiple columns (one-hot), its *leading singular vector* across neurons
        is used as the “competing” axis for orthogonalization.

    n_splits : int, default 5
        Number of KFold splits for cross-validation.

    include_trials : array-like of int, **REQUIRED**
        Trial IDs to keep from this alignment. Dataset order is preserved.

    require_all_ids : bool, default True
        If True, raise an error if any of `include_trials` is not present.

    projection_time_window : (float, float) or None, default None
        Time window to compute the time-resolved **projection traces** Y(trial, t).
        If None, projects across the *full* peri-event time axis in `psth_da`.

    save_path : str | Path | None, default None
        If given, save outputs:
          • "npz": compressed NumPy archive
          • "nc":  NetCDF via xarray
          • "zarr": Zarr directory (best for large arrays)
        The saved data will contain either **raw** or **normalized** projections
        depending on `norm_mode` (see Notes under "Saving behavior").

    save_format : {"npz","nc","zarr"}, default "zarr"
        File format used when saving.

    overwrite : bool, default True
        If True, overwrite existing file (or delete and rewrite zarr store).

    norm_mode : {"none","divide_sqrtN","unit_variance_fit","zscore_fit"}, default "none"
        Post-hoc normalization applied to the projections to improve cross-session
        comparability when neuron counts or data statistics differ:
          • "none": no extra scaling (original behavior).
          • "divide_sqrtN": divide y and Y(t) by sqrt(#units), compensating for
            different neuron counts across sessions.
          • "unit_variance_fit": scale so Var(y_fit) == 1 while preserving mean.
          • "zscore_fit": mean-center and scale y_fit to unit variance; apply
            the same mean/std to the entire projection trace Y(t).
        Normalized results are returned in `projection_norm` and `projection_trace_norm`,
        along with bookkeeping (`norm_factor`, `y_fit_mean`, `y_fit_std`, `norm_mode`).

    -------------------------------------------------------------------------
    Returns
    -------------------------------------------------------------------------
    dict with keys:
      Axis & projections
        - "axis_w" : (N_units,) unit weights (unit-norm) defining the neuronal axis.
        - "projection" : (T_used,) per-trial scalar in the *fit window* (raw).
        - "projection_trace" : (T_used, Tt) per-trial time-resolved projection (raw).
        - "projection_norm" : (T_used,) per-trial scalar after `norm_mode` scaling.
        - "projection_trace_norm" : (T_used, Tt) time-resolved projection after scaling.
      Normalization bookkeeping
        - "norm_mode" : str, the chosen normalization mode.
        - "norm_factor" : float, scale factor used (meaning depends on mode).
        - "y_fit_mean" : float, mean of raw `projection` (for zscore mode).
        - "y_fit_std" : float, std of raw `projection` (for unit_variance/zscore).
      Time & identity
        - "time_for_projection" : (Tt,), time vector used for projection traces.
        - "trial_ids" : (T_used,), trial IDs kept.
        - "unit_ids" : (N_units,), unit indices from dataset.
      Latent & scaling info
        - "latent" : (T_used,), raw latent values for included trials.
        - "latent_z" : (T_used,), standardized latent used in regression.
        - "zscore_mu" : (1, N_units), mean used to z-score units (fit window).
        - "zscore_sigma" : (1, N_units), std used to z-score units (fit window).
      CV metrics & meta
        - "cv_corr" : (K,), correlation on held-out folds.
        - "cv_r2" : (K,), squared correlation on held-out folds.
        - "final" : dict, fit on all included trials (as returned by `tdr_fit`).
        - "align" : str, "go_cue" or "reward_go_cue_start".
        - "time_window_fit" : (float, float), fit window used for axis.
        - "projection_time_window" : (float, float) | None, projection window.
        - "n_trials_used" : int, number of trials kept.
        - "n_trials_total" : int, total trials available before filtering.
      Saving (if used)
        - "saved_to" : str, path written.
        - "saved_format" : {"npz","nc","zarr"}.

    -------------------------------------------------------------------------
    Notes on saving behavior
    -------------------------------------------------------------------------
    • When save_format = "npz", both raw and normalized projections are saved:
        - projection, projection_trace, projection_norm, projection_trace_norm
      (plus time, ids, axis, latent, zscore stats, and attrs JSON).
    • When save_format in {"nc","zarr"}:
        - We save **the normalized projection** if `norm_mode != "none"`,
          otherwise we save the raw projection. (The normalization settings
          are recorded in dataset attributes: norm_mode, norm_factor, etc.)
    """
    # ---------- 1) Extract averaged rates in the fit window (no z-score yet) ----------
    fit_ext = extract_trial_unit_rates(
        psth_da, align=align, time_window=time_window, zscore_units=False
    )
    R_fit_full = fit_ext["R"]
    trial_ids_full = fit_ext["trial_ids"]
    T_full, N_units = R_fit_full.shape

    if include_trials is None:
        raise ValueError("include_trials (trial IDs) must be provided.")

    # ---------- 2) Subset trials by ID ----------
    mask = _mask_from_trial_ids(trial_ids_full, include_trials, require_all=require_all_ids)
    if mask.sum() == 0:
        raise ValueError("include_trials matched 0 trials in psth_da for this alignment.")
    R_fit = R_fit_full[mask]
    trial_ids_used = trial_ids_full[mask]
    T_used = R_fit.shape[0]

    # ---------- 3) Handle latent length (full vs subset) ----------
    latent = np.asarray(latent).reshape(-1)
    if len(latent) == T_full:
        latent_inc = latent[mask]
    elif len(latent) == T_used:
        latent_inc = latent
    else:
        raise ValueError(f"'latent' length must be either T_full={T_full} or #included={T_used}.")

    # ---------- 4) Z-score units (fit window, included trials only) ----------
    mu = R_fit.mean(axis=0, keepdims=True)
    sigma = R_fit.std(axis=0, keepdims=True) + 1e-9
    R_fit_z = (R_fit - mu) / sigma

    # ---------- 5) Build design matrix (auto-subset any covariates) ----------
    def _auto_subset(vec):
        arr = np.asarray(vec)
        if len(arr) == T_full:
            return arr[mask]
        elif len(arr) == T_used:
            return arr
        raise ValueError(f"Covariate length {len(arr)} incompatible with T_full={T_full} or T_used={T_used}")

    if continuous_covs:
        continuous_covs = {k: _auto_subset(v) for k, v in continuous_covs.items()}
    if categorical_covs:
        categorical_covs = {k: _auto_subset(v) for k, v in categorical_covs.items()}

    X_wo, col_slices, latent_col = build_design_matrix(
        T=T_used,
        latent=latent_inc,
        continuous=continuous_covs,
        categorical=categorical_covs,
        zscore_continuous=True,
    )
    latent_z = X_wo[:, latent_col].copy()

    # ---------- 6) Fit TDR with cross-validation ----------
    cv = tdr_cv(
        R=R_fit_z,
        X_wo_intercept=X_wo,
        latent_col=latent_col,
        orth_names=orth_names,
        col_slices=col_slices,
        n_splits=n_splits,
    )
    w_final = cv["final"]["w"]                      # (N_units,)
    projection_fit = R_fit_z @ w_final              # (T_used,)

    # ---------- 7) Time-resolved projection across the chosen window ----------
    cube_ext = extract_trial_unit_timecube(
        psth_da, align=align, time_window=projection_time_window
    )
    cube_full = cube_ext["cube"]                    # (T_full, N_units, Tt)
    time_proj = cube_ext["time"]                    # (Tt,)
    cube_used = cube_full[mask]
    cube_z = (cube_used - mu[:, :, None]) / sigma[:, :, None]
    projection_trace = np.tensordot(cube_z, w_final, axes=([1],[0]))  # (T_used, Tt)

    # ---------- 8) Post-hoc normalization (for across-session comparability) ----------
    norm_mode = (norm_mode or "none").lower()
    if norm_mode not in {"none","divide_sqrtn","unit_variance_fit","zscore_fit"}:
        raise ValueError("norm_mode must be one of {'none','divide_sqrtN','unit_variance_fit','zscore_fit'}")

    y_fit_mean = float(np.nanmean(projection_fit))
    y_fit_std  = float(np.nanstd(projection_fit) + 1e-12)
    norm_factor = 1.0

    if norm_mode == "divide_sqrtN":
        norm_factor = float(np.sqrt(N_units))
        projection_norm = projection_fit / norm_factor
        projection_trace_norm = projection_trace / norm_factor

    elif norm_mode == "unit_variance_fit":
        norm_factor = y_fit_std
        projection_norm = projection_fit / norm_factor
        projection_trace_norm = projection_trace / norm_factor

    elif norm_mode == "zscore_fit":
        norm_factor = y_fit_std
        projection_norm = (projection_fit - y_fit_mean) / norm_factor
        projection_trace_norm = (projection_trace - y_fit_mean) / norm_factor

    else:  # "none"
        projection_norm = projection_fit.copy()
        projection_trace_norm = projection_trace.copy()

    # ---------- 9) Package outputs ----------
    cv.update({
        "axis_w": w_final,
        "projection": projection_fit,                      # raw
        "projection_trace": projection_trace,              # raw
        "projection_norm": projection_norm,                # normalized
        "projection_trace_norm": projection_trace_norm,    # normalized
        "norm_mode": norm_mode,
        "norm_factor": float(norm_factor),
        "y_fit_mean": y_fit_mean,
        "y_fit_std": y_fit_std,
        "time_for_projection": time_proj,
        "trial_ids": trial_ids_used,
        "unit_ids": fit_ext["unit_ids"],
        "latent": latent_inc,
        "latent_z": latent_z,
        "zscore_mu": mu,
        "zscore_sigma": sigma,
        "align": align,
        "time_window_fit": time_window,
        "projection_time_window": projection_time_window,
        "n_trials_used": int(T_used),
        "n_trials_total": int(T_full),
    })

    # ---------- 10) Optional save ----------
    if save_path is not None:
        path = Path(save_path)
        fmt = save_format.lower()
        if fmt not in {"npz", "nc", "zarr"}:
            raise ValueError("save_format must be one of {'npz','nc','zarr'}")

        if fmt == "npz":
            # Save BOTH raw and normalized arrays
            np.savez_compressed(
                path,
                projection_trace=projection_trace,
                projection=projection_fit,
                projection_trace_norm=projection_trace_norm,
                projection_norm=projection_norm,
                axis_w=w_final,
                time=time_proj,
                trial_ids=trial_ids_used,
                unit_ids=fit_ext["unit_ids"],
                latent=latent_inc,
                latent_z=latent_z,
                zscore_mu=mu,
                zscore_sigma=sigma,
                attrs_str=json.dumps({
                    "align": align,
                    "time_window_fit": time_window,
                    "projection_time_window": projection_time_window,
                    "n_trials_used": int(T_used),
                    "n_trials_total": int(T_full),
                    "cv_corr": np.asarray(cv["cv_corr"]).tolist(),
                    "cv_r2": np.asarray(cv["cv_r2"]).tolist(),
                    "norm_mode": norm_mode,
                    "norm_factor": float(norm_factor),
                    "y_fit_mean": y_fit_mean,
                    "y_fit_std": y_fit_std,
                })
            )
        else:
            # For xarray (nc/zarr) we persist the normalized arrays if norm_mode != "none",
            # otherwise we persist the raw arrays. Normalization metadata is stored in attrs.
            meta = {
                "align": align,
                "time_window_fit": time_window,
                "projection_time_window": projection_time_window,
                "n_trials_used": int(T_used),
                "n_trials_total": int(T_full),
                "cv_corr": np.asarray(cv["cv_corr"]),
                "cv_r2": np.asarray(cv["cv_r2"]),
                "norm_mode": norm_mode,
                "norm_factor": float(norm_factor),
                "y_fit_mean": y_fit_mean,
                "y_fit_std": y_fit_std,
            }
            ds = _package_tdr_to_xr(
                projection_trace=projection_trace,
                projection_fit=projection_fit,
                projection_trace_norm=projection_trace_norm,
                projection_norm=projection_norm,
                w_final=w_final,
                t=time_proj,
                trial_ids_used=trial_ids_used,
                unit_ids=fit_ext["unit_ids"],
                meta=meta,
                latent_raw=latent_inc,
                latent_z=latent_z,
            )

            if fmt == "nc":
                ds.to_netcdf(path)
            elif fmt == "zarr":
                import shutil
                if path.exists() and overwrite:
                    shutil.rmtree(path)
                ds.to_zarr(path, mode="w")

        cv["saved_to"] = str(path)
        cv["saved_format"] = fmt

    return cv




# ------------------------------- Example -----------------------------
# Usage:
#
# align = "go_cue"
# keep_ids = np.asarray(df_combined_behavior_summary['response_trials'][0], dtype=int)
# latent_full = np.asarray(df_combined_behavior_summary['ForagingCompareThreshold-value-1'][0], dtype=float)
#
# out = tdr_from_psth(
#     psth_da,
#     latent=latent_full,               # full length OK; auto-subsets via include_trials
#     align=align,
#     time_window=(0.0, 0.5),           # fit TDR axis here
#     include_trials=keep_ids,          # trial IDs to keep (REQUIRED)
#     projection_time_window=None,      # None → project across full peri-event time
# )
#
# # Per-trial scalar in fit window:
# y_fit = out["projection"]                        # (n_trials_used,)
#
# # Per-trial time-resolved projection:
# Y = out["projection_trace"]                      # (n_trials_used, n_timepoints)
# t = out["time_for_projection"]                   # (n_timepoints,)
# ids = out["trial_ids"]                           # (n_trials_used,)




def plot_tdr_trace_by_quantile(
    Y: np.ndarray,
    t: np.ndarray,
    latent: np.ndarray,
    n_quantiles: int = 5,
    *,
    y_fit: np.ndarray = None,
    group_by: str = "latent",    # "latent" or "y_fit"
    ci: str = "sem",             # "sem" or "none"
    show_trials: bool = False,
    cmap: str = "viridis",
    title: str = "TDR projection by bins",
    alpha_trials: float = 0.08,
    lw_mean: float = 2.0,
    smooth: dict | None = None,  # e.g., {"method":"gaussian","sigma":0.05,"unit":"s"}
    legend_ci_patch: bool = False,  # add a faint patch in legend to indicate CI color

    # --- binning controls ---
    binning: str = "equal_width",        # "equal_width" or "quantile"
    bin_edges: np.ndarray | None = None, # explicit edges (overrides binning)
    quantile_method: str = "linear",     # np.nanquantile(method=...) when binning="quantile"

    # --- filtering ---
    min_traces_per_bin: int = 1,         # exclude bins with < this many trials
):
    """
    Plot TDR projection traces grouped by bins of a variable, with optional smoothing.

    Binning modes:
      - binning="equal_width": evenly spaced edges between min and max (value-based bins)
      - binning="quantile": equal-count bins via np.nanquantile(..., method=quantile_method)
      - bin_edges: provide explicit edges (length n_quantiles+1); overrides binning

    Exclusion:
      - Bins with < min_traces_per_bin trials are excluded from plot and legend.
    """

    # ---- helpers ----
    def _build_gaussian_kernel(sigma_pts: float, truncate: float = 3.0) -> np.ndarray:
        sigma_pts = float(max(sigma_pts, 1e-6))
        half = int(np.ceil(truncate * sigma_pts))
        xk = np.arange(-half, half + 1, dtype=float)
        k = np.exp(-0.5 * (xk / sigma_pts) ** 2)
        return k / k.sum()

    def _build_moving_kernel(window_pts: int) -> np.ndarray:
        window_pts = int(max(1, window_pts))
        if window_pts % 2 == 0:
            window_pts += 1
        k = np.ones(window_pts, dtype=float)
        return k / k.sum()

    def _nanaware_convolve_same(y: np.ndarray, k: np.ndarray) -> np.ndarray:
        out = np.empty_like(y, dtype=float)
        for i in range(y.shape[0]):
            row = y[i]
            valid = np.isfinite(row).astype(float)
            data = np.nan_to_num(row, nan=0.0)
            num = np.convolve(data, k, mode="same")
            den = np.convolve(valid, k, mode="same")
            sm = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-12)
            out[i] = sm
        return out

    def _maybe_smooth(Yin: np.ndarray, tvec: np.ndarray, spec: dict | None) -> np.ndarray:
        if not spec:
            return Yin
        method = spec.get("method", "gaussian").lower()
        unit = spec.get("unit", "s").lower()
        truncate = float(spec.get("truncate", 3.0))
        if unit == "s":
            dt = float(np.median(np.diff(tvec)))
            if method == "gaussian":
                sigma = float(spec.get("sigma", 0.05))  # seconds
                k = _build_gaussian_kernel(max(sigma, 1e-6) / max(dt, 1e-9), truncate)
            else:
                window = float(spec.get("window", 0.05))  # seconds
                k = _build_moving_kernel(int(round(window / max(dt, 1e-9))))
        else:  # "samples"
            if method == "gaussian":
                k = _build_gaussian_kernel(float(spec.get("sigma", 5)), truncate)
            else:
                k = _build_moving_kernel(int(spec.get("window", 5)))
        return _nanaware_convolve_same(Yin, k)

    # ---- validations ----
    Y = np.asarray(Y)
    t = np.asarray(t).reshape(-1)
    if Y.ndim != 2 or Y.shape[1] != t.size:
        raise ValueError("Y must be (n_trials, n_timepoints) and match t length")

    if group_by == "latent":
        x = np.asarray(latent).reshape(-1)
    elif group_by == "y_fit":
        if y_fit is None:
            raise ValueError("y_fit must be provided when group_by='y_fit'")
        x = np.asarray(y_fit).reshape(-1)
    else:
        raise ValueError("group_by must be 'latent' or 'y_fit'")

    if x.size != Y.shape[0]:
        raise ValueError(f"{group_by} length must match number of trials in Y")

    # drop invalid trials
    good = np.isfinite(x) & np.isfinite(Y).any(axis=1)
    Y = Y[good]
    x = x[good]
    if Y.shape[0] < n_quantiles:
        raise ValueError("Not enough valid trials to form the requested bins.")

    # optional smoothing
    Y_sm = _maybe_smooth(Y, t, smooth)

    # ---- choose bin edges ----
    if bin_edges is not None:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or edges.size != (n_quantiles + 1):
            raise ValueError("bin_edges must be 1D with length n_quantiles+1")
    else:
        if binning not in {"equal_width", "quantile"}:
            raise ValueError("binning must be 'equal_width' or 'quantile'")
        if binning == "quantile":
            edges = np.nanquantile(
                x, np.linspace(0, 1, n_quantiles + 1), method=quantile_method
            )
        else:  # equal_width
            xmin = float(np.nanmin(x))
            xmax = float(np.nanmax(x))
            if not np.isfinite(xmin) or not np.isfinite(xmax):
                raise ValueError("Non-finite values in grouping variable after filtering.")
            if xmax == xmin:
                xmin -= 0.5
                xmax += 0.5
            edges = np.linspace(xmin, xmax, n_quantiles + 1)

    # Ensure strictly increasing edges (guard against ties)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + np.finfo(float).eps

    # Digitize; include max by slightly expanding last edge
    span = edges[-1] - edges[0]
    edges_expanded = edges.copy()
    edges_expanded[-1] = edges[-1] + (1e-12 * (span if span > 0 else 1.0))
    bin_idx = np.digitize(x, edges_expanded[1:-1], right=True)

    # ---- decide which bins to keep based on min_traces_per_bin ----
    kept_bins = []
    bin_members = []
    for b in range(n_quantiles):
        idx = np.where(bin_idx == b)[0]
        if idx.size >= max(1, int(min_traces_per_bin)):
            kept_bins.append(b)
            bin_members.append(idx)

    if len(kept_bins) == 0:
        raise ValueError("All bins were excluded by min_traces_per_bin.")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(20, 8))

    if show_trials:
        ax.plot(t, Y_sm.T, color="0.6", alpha=alpha_trials,
                linewidth=0.6, zorder=1)

    # continuous colormap for the colorbar
    cmap_cont = plt.get_cmap(cmap)
    # discrete colors for the traces (one per kept bin)
    trace_colors = cmap_cont(np.linspace(0.0, 1.0, len(kept_bins)))

    legend_handles, legend_labels = [], []

    # representative bin values for colorbar = bin centers
    bin_centers = np.array([(edges[b] + edges[b + 1]) / 2.0 for b in kept_bins])

    for i_plot, (b, idx) in enumerate(zip(kept_bins, bin_members)):
        color = trace_colors[i_plot]
        Yb = Y_sm[idx]

        mean_b = np.nanmean(Yb, axis=0)
        ax.plot(t, mean_b, color=color, linewidth=lw_mean, zorder=3)

        if ci == "sem":
            sem_b = np.nanstd(Yb, axis=0) / max(1, np.sqrt(idx.size))
            ax.fill_between(
                t, mean_b - sem_b, mean_b + sem_b,
                color=color, alpha=0.25, zorder=2
            )

        line_handle = Line2D([0], [0], color=color, lw=lw_mean)
        if legend_ci_patch and ci == "sem":
            patch_handle = Patch(facecolor=color, alpha=0.25, edgecolor="none")
            legend_handles.append((line_handle, patch_handle))
        else:
            legend_handles.append(line_handle)

        legend_labels.append(
            f"B{b+1}/{n_quantiles} [{edges[b]:.3g}, {edges[b+1]:.3g}] n={idx.size}"
        )

    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Projection (a.u.)")

    by = "latent" if group_by == "latent" else "y_fit"
    subtitle = f"{binning}"
    if binning == "quantile":
        subtitle += f", method={quantile_method}"
    if min_traces_per_bin > 1:
        subtitle += f", min_n={min_traces_per_bin}"
    ax.set_title(title + f" (by {by}, {subtitle})")

    # legend outside on the right
    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            fontsize=9,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            handlelength=2.8,
            handletextpad=0.8,
        )

    # -------------------------
    # Continuous colorbar for bins
    # -------------------------
    cax = fig.add_axes([0.86, 0.2, 0.02, 0.6])
    norm = plt.Normalize(vmin=bin_centers.min(), vmax=bin_centers.max())
    sm = ScalarMappable(norm=norm, cmap=cmap_cont)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"{by} (bin center)", rotation=90)

    # leave room on the right for legend (up to 0.82)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()

