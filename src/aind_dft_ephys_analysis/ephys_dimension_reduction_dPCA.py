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

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import xarray as xr
from scipy import stats
from sklearn.model_selection import KFold

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
    n = np.linalg.norm(v) + eps
    return v / n

def _orthogonalize_axis(axis: np.ndarray, A: Optional[np.ndarray]) -> np.ndarray:
    """Make 'axis' orthogonal to the span of columns in A (N × K)."""
    if A is None or A.size == 0:
        return axis
    P = A @ np.linalg.pinv(A)
    return axis - P @ axis

# --------------------- xarray extractors ------------------------------

def extract_trial_unit_rates(
    psth_da: xr.Dataset,
    align: str = "go_cue",                 # "go_cue" or "reward_go_cue_start"
    time_window: Tuple[float, float] = (0.0, 0.5),
    zscore_units: bool = True
) -> Dict:
    """
    Average firing rates in a time window to get a (T × N) matrix.

    Returns:
        R: array (T trials × N units)
        trial_ids: vector (T,) trial indices from the xarray coordinate
        unit_ids: vector (N,) unit indices from the xarray coordinate
        time_mask: boolean mask of time bins used
        bin_size: float if available in attrs else None
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
    da = psth_da[var]  # (unit, trial_dim, time)

    time = psth_da["time"].values
    t0, t1 = time_window
    tmask = (time >= t0) & (time < t1)
    if not np.any(tmask):
        raise ValueError("time_window selects no samples. Check 'time' values.")

    # Mean over time window → (trial, unit)
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
    Return a 3D tensor (T × N × Tt) of rates for all trials (we will subset later).
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
    da = psth_da[var]  # (unit, trial, time)
    time = psth_da["time"].values
    if time_window is None:
        tmask = np.ones_like(time, dtype=bool)
    else:
        t0, t1 = time_window
        tmask = (time >= t0) & (time < t1)
    if not np.any(tmask):
        raise ValueError("projection_time_window selects no samples.")
    cube = da.sel(time=tmask).transpose(trial_dim, "unit", "time").values  # (T × N × Tt)
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
    Build a demixing design matrix:
      X = [1 | latent | continuous... | one-hot(categorical)...]
    Returns:
      X (T × Q), column_slices dict mapping name -> slice in X (wo/ intercept),
      latent_col index (wo/ intercept)
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
    R: np.ndarray,                 # (T × N) z-scored unit rates
    X_wo_intercept: np.ndarray,    # (T × Q) design (without intercept)
    latent_col: int,               # index of 'latent' within X_wo_intercept
    orth_names: Optional[List[str]] = None,
    col_slices: Optional[Dict[str, slice]] = None
) -> Dict:
    """
    Fit OLS per neuron: r_n = [1, X] * beta_n.
    Latent axis = vector of beta_n for latent across neurons, optionally
    orthogonalized against other task axes.
    """
    T, N = R.shape
    X = np.column_stack([np.ones((T, 1)), X_wo_intercept])  # add intercept

    XtX_pinv = np.linalg.pinv(X.T @ X)
    B = XtX_pinv @ (X.T @ R)  # (Q × N)
    betas = B.T               # (N × Q)

    latent_idx = 1 + latent_col
    axis = betas[:, latent_idx].copy()   # length N

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
            A = np.column_stack(A_cols)  # N × K
            P = A @ np.linalg.pinv(A)
            axis = axis - P @ axis

    w = _unit_norm(axis)

    # Sign: make projection positively correlated with latent
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
    """K-fold CV for the TDR axis; returns fold stats and the refit on all data."""
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
    """Build a boolean mask over trials using *trial IDs* only."""
    T = len(trial_ids_full)
    if include_ids is None:
        return np.ones(T, dtype=bool)

    inc = np.asarray(include_ids, dtype=int).ravel()
    # De-duplicate (order not used for masking)
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

def tdr_from_psth(
    psth_da: xr.Dataset,
    latent: np.ndarray,                       # latent for included trials OR full length (see below)
    align: str = "go_cue",
    time_window: Tuple[float, float] = (0.0, 0.5),   # window used to FIT the axis
    continuous_covs: Optional[Dict[str, np.ndarray]] = None,
    categorical_covs: Optional[Dict[str, np.ndarray]] = None,
    orth_names: Optional[List[str]] = None,
    n_splits: int = 5,
    include_trials: Optional[Union[np.ndarray, List[int]]] = None,  # trial IDs to keep (REQUIRED)
    require_all_ids: bool = True,
    projection_time_window: Optional[Tuple[float, float]] = None    # window to PROJECT traces (None → full)
) -> Dict:
    """
    Fit TDR on an analysis window, then project a time-resolved trace y(t)
    for each included trial onto the same neuronal dimension.

    'latent' can be either:
      - Full-length vector for ALL trials in this alignment (T_full, matches psth_da),
        in which case it is auto-subset using include_trials; OR
      - Already subset to included trials (length == #included trials).

    Returns (in addition to CV fields):
      'axis_w'               : (N_units,) final neuronal axis
      'projection'           : (T_used,) scalar per-trial projection (fit window)
      'projection_trace'     : (T_used, Tt_proj) time-resolved projection per trial
      'time_for_projection'  : (Tt_proj,) time vector for projection_trace
      'zscore_mu','zscore_sigma' : (1, N_units) scaling used to standardize units
    """
    # 1) Extract fit-window trial × unit (no z-score here; we’ll standardize once)
    fit_ext = extract_trial_unit_rates(
        psth_da, align=align, time_window=time_window, zscore_units=False
    )
    R_fit_full = fit_ext["R"]              # (T_full × N_units)
    trial_ids_full = fit_ext["trial_ids"]  # (T_full,)
    T_full, N_units = R_fit_full.shape

    if include_trials is None:
        raise ValueError("include_trials (trial IDs) must be provided.")

    # 2) Build mask from trial IDs and subset
    mask = _mask_from_trial_ids(trial_ids_full, include_trials, require_all=require_all_ids)
    if mask.sum() == 0:
        raise ValueError("include_trials matched 0 trials in psth_da for this alignment.")

    R_fit = R_fit_full[mask]               # (T_used × N_units)
    trial_ids_used = trial_ids_full[mask]
    T_used = R_fit.shape[0]

    # 3) Handle latent size: accept either full length (T_full) or included length (T_used)
    latent = np.asarray(latent).reshape(-1)
    if len(latent) == T_full:
        latent = latent[mask]  # auto-subset to included trials
    elif len(latent) != T_used:
        raise ValueError(
            f"'latent' length must be either T_full={T_full} or #included={T_used}, got {len(latent)}"
        )

    # 4) Standardize units using mean/std from the fit window (included trials only)
    mu = R_fit.mean(axis=0, keepdims=True)               # (1 × N_units)
    sigma = R_fit.std(axis=0, keepdims=True) + 1e-9
    R_fit_z = (R_fit - mu) / sigma

    # 5) Build design for these SAME trials (auto-subset covariates if full-length)
    def _auto_subset(vec):
        arr = np.asarray(vec)
        if len(arr) == T_full:
            return arr[mask]
        elif len(arr) == T_used:
            return arr
        else:
            raise ValueError(
                f"Covariate length must be T_full={T_full} or #included={T_used}, got {len(arr)}"
            )

    if continuous_covs:
        continuous_covs = {k: _auto_subset(v) for k, v in continuous_covs.items()}
    if categorical_covs:
        categorical_covs = {k: _auto_subset(v) for k, v in categorical_covs.items()}

    X_wo, col_slices, latent_col = build_design_matrix(
        T=T_used,
        latent=latent,
        continuous=continuous_covs,
        categorical=categorical_covs,
        zscore_continuous=True,
    )

    # 6) Fit TDR (CV + final) on standardized fit-window data
    cv = tdr_cv(
        R=R_fit_z,
        X_wo_intercept=X_wo,
        latent_col=latent_col,
        orth_names=orth_names,
        col_slices=col_slices,
        n_splits=n_splits,
    )

    w_final = cv["final"]["w"]                # (N_units,)
    projection_fit = R_fit_z @ w_final        # (T_used,)

    # 7) Build time-resolved projection for included trials (using SAME scaling)
    cube_ext = extract_trial_unit_timecube(
        psth_da, align=align, time_window=projection_time_window
    )
    cube_full = cube_ext["cube"]              # (T_full × N_units × Tt_proj)
    time_proj = cube_ext["time"]              # (Tt_proj,)
    cube_used = cube_full[mask]               # (T_used × N_units × Tt_proj)

    # Standardize the whole time-cube using fit-window mu/sigma
    cube_z = (cube_used - mu[:, :, None]) / sigma[:, :, None]          # (T_used × N_units × Tt_proj)
    projection_trace = np.tensordot(cube_z, w_final, axes=([1],[0]))   # (T_used × Tt_proj)

    # 8) Package results
    cv.update({
        "axis_w": w_final,                        # (N_units,)
        "projection": projection_fit,             # (T_used,)
        "projection_trace": projection_trace,     # (T_used × Tt_proj)
        "time_for_projection": time_proj,         # (Tt_proj,)
        "trial_ids": trial_ids_used,              # (T_used,)
        "unit_ids": fit_ext["unit_ids"],
        "time_mask_fit": fit_ext["time_mask"],    # boolean mask used to fit
        "align": align,
        "time_window_fit": time_window,
        "projection_time_window": projection_time_window,  # None → full time
        "n_trials_used": int(T_used),
        "n_trials_total": int(T_full),
        "zscore_mu": mu,                          # (1 × N_units)
        "zscore_sigma": sigma,                    # (1 × N_units)
    })
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
