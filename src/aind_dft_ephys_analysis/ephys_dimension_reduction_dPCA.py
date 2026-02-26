# -*- coding: utf-8 -*-
"""
Demixed PCA (dPCA) for xarray PSTH datasets with arbitrary categorical factors.

Expected PSTH (your format):
- Data variables (choose one via `align`):
    psth_go_cue(unit, trial_go_cue, time)
    psth_reward_go_cue_start(unit, trial_reward_go_cue_start, time)
- Coordinates:
    time(time), unit_index(unit), trial_index_<align>(trial_<align>)
- Attributes: bin_size (optional)

Core entry point:
    dpca_from_psth(psth_da, factors, ...)

Returns:
    dict with decoders/loadings per marginalization, component timecourses per
    condition, explained variance breakdown, (optional) single-trial projections,
    and convenient xarray packaging for saving.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Iterable
import itertools
import numpy as np
import xarray as xr
from pathlib import Path
import json

# ----------------------------- utilities -----------------------------

def _as_int_levels(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map a 1D categorical vector (ints/strings) to compact integer levels 0..L-1.
    Returns (levels_int, unique_sorted_levels).
    """
    arr = np.asarray(vec, dtype=object).ravel()
    cats, inv = np.unique(arr, return_inverse=True)
    return inv.astype(int), cats

def _zscore_units_over_trials(R: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score units (columns) across trials. R: (T × N) or (T × N × Tt) via reshaping.
    Returns (Z, mu, sigma) where mu, sigma are (1 × N).
    """
    mu = np.nanmean(R, axis=0, keepdims=True)
    sigma = np.nanstd(R, axis=0, keepdims=True) + eps
    return (R - mu) / sigma, mu, sigma

def _safe_mean(a: np.ndarray, axis=None):
    return np.nanmean(a, axis=axis)

# ------------------------- PSTH extraction ---------------------------

def _extract_trial_unit_timecube(
    psth_da: xr.Dataset,
    align: str = "go_cue",
    time_window: Optional[Tuple[float, float]] = None
) -> Dict:
    """
    Return cube: (T × N × Tt), trial_ids (T,), time (Tt,)
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
    if time_window is None:
        tmask = np.ones_like(time, dtype=bool)
    else:
        t0, t1 = time_window
        tmask = (time >= t0) & (time < t1)
    if not np.any(tmask):
        raise ValueError("time_window selects no samples.")

    cube = da.sel(time=tmask).transpose(trial_dim, "unit", "time").values  # (T × N × Tt)
    trial_ids = psth_da[trial_coord].values
    return dict(cube=cube, trial_ids=trial_ids, time=time[tmask])

def _mask_from_trial_ids(
    trial_ids_full: np.ndarray,
    include_ids: Optional[Union[np.ndarray, List[int]]],
    require_all: bool = True,
) -> np.ndarray:
    """Boolean mask selecting only included trial IDs (dataset order preserved)."""
    T = len(trial_ids_full)
    if include_ids is None:
        return np.ones(T, dtype=bool)
    inc = np.asarray(include_ids, dtype=int).ravel()
    inc = inc[np.unique(inc, return_index=True)[1]]  # de-dup, preserve order
    present = np.isin(inc, trial_ids_full)
    if require_all and not np.all(present):
        missing = inc[~present]
        raise ValueError(f"Some include_trials IDs not found: {missing.tolist()}")
    mask = np.isin(trial_ids_full, inc)
    if mask.sum() == 0:
        raise ValueError("include_trials matched 0 trials.")
    return mask

# ---------------------- Condition tensor builder ---------------------

def _build_condition_tensor(
    cube_used: np.ndarray,          # (T_used × N × Tt)
    factors_used: Dict[str, np.ndarray],  # each length = T_used (categorical)
    min_count: int = 1,
    drop_sparse: bool = False
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, int], np.ndarray]:
    """
    Average trials into condition means.

    Returns
    -------
    C : np.ndarray
        Condition tensor with shape (N, Tt, L1, L2, ..., Lk)
    levels: dict[str, np.ndarray]
        Mapping factor → unique sorted original levels (strings/ints)
    n_levels: dict[str, int]
        Number of levels per factor
    counts : np.ndarray
        Trial counts per condition, shape (L1, L2, ..., Lk)
    """
    T_used, N, Tt = cube_used.shape
    # Map factors to integer codes
    names = list(factors_used.keys())
    codes = []
    levels = {}
    n_levels = {}
    for k in names:
        code, lev = _as_int_levels(factors_used[k])
        codes.append(code)
        levels[k] = lev
        n_levels[k] = len(lev)

    # Build a linear index for each trial over the cartesian product of levels
    strides = np.cumprod([1] + [n_levels[nm] for nm in names[:-1]])
    lin = np.zeros(T_used, dtype=int)
    for i, nm in enumerate(names):
        lin += codes[i] * strides[i]
    total_cells = np.prod([n_levels[nm] for nm in names]) if names else 1

    # Accumulate sums and counts per cell
    sums = np.zeros((total_cells, N, Tt), dtype=float)
    counts = np.zeros(total_cells, dtype=int)
    for ti in range(T_used):
        idx = lin[ti]
        sums[idx] += cube_used[ti]
        counts[idx] += 1

    # Handle sparsity
    if min_count > 1:
        mask_cells = counts >= min_count
    else:
        mask_cells = counts > 0

    if drop_sparse and not np.all(mask_cells):
        # optional: drop underfilled cells (rarely needed)
        sums = sums[mask_cells]
        counts = counts[mask_cells]
        # but then shape stops being a proper tensor; default is to keep all

    # Means
    means = np.divide(
        sums, counts[:, None, None],
        out=np.zeros_like(sums), where=counts[:, None, None] > 0
    )

    # Reshape to (N, Tt, L1,...,Lk)
    shape = (N, Tt) + tuple(n_levels[nm] for nm in names)
    C = means.reshape(shape, order="F")

    counts_tensor = counts.reshape(tuple(n_levels[nm] for nm in names), order="F") if names else np.array([counts.sum()])
    return C, levels, n_levels, counts_tensor

# --------------------------- Marginalizations ------------------------

def _powerset_nonempty(items: List[str]) -> List[Tuple[str,...]]:
    """All non-empty subsets of a list, as tuples, sorted by size then lexicographically."""
    out = []
    for r in range(1, len(items)+1):
        out.extend(itertools.combinations(items, r))
    return out

def _marginalize(C: np.ndarray, factor_names: List[str]) -> Dict[Tuple[str,...], np.ndarray]:
    """
    Inclusion–exclusion style marginalization.

    C has shape (N, Tt, L1, L2, ..., Lk) where factors order = factor_names.
    Returns a dict mapping subset S (tuple of factor names) -> C_S of same shape,
    such that sum over all non-empty S equals C - grand_mean (per time).
    """
    # grand mean over all factors (condition average)
    axes_factors = tuple(range(2, C.ndim))
    grand = _safe_mean(C, axis=axes_factors,)

    # Helper: mean over all factors not in S (keep dims for those in S)
    def mean_keep(S_idx: List[int]) -> np.ndarray:
        # Average over all factor axes NOT in S_idx
        keep = set(S_idx)
        red_axes = [ax for ax in range(2, C.ndim) if (ax-2) not in keep]
        X = C
        for ax in sorted(red_axes, reverse=True):
            X = _safe_mean(X, axis=ax, )
        # Now X has shape (N, Tt, *levels for S)
        # Broadcast back to full shape by repeating along missing factors
        for ax in range(2, C.ndim):
            if (ax-2) not in keep:
                X = np.expand_dims(X, axis=ax)
                X = np.repeat(X, C.shape[ax], axis=ax)
        return X

    names = list(factor_names)
    subsets = _powerset_nonempty(names)
    # Precompute means for each subset (E_S)
    E = {}
    for S in subsets:
        idxs = [names.index(s) for s in S]
        E[S] = mean_keep(idxs)

    # Inclusion–exclusion to get unique marginal contributions
    M = {}
    for S in subsets:
        # M_S = E_S - sum_{R subset S, R≠S} M_R
        contrib = E[S].copy()
        for r in range(1, len(S)):
            for R in itertools.combinations(S, r):
                contrib -= M[R]
        M[S] = contrib

    # Time-only marginalization (optional but common): remove grand mean to isolate time
    # time_marg = C - grand expanded to full shape minus sum of all factor marginals
    # Expand grand to full shape
    G = grand
    for ax in range(2, C.ndim):
        G = np.expand_dims(G, axis=ax)
        G = np.repeat(G, C.shape[ax], axis=ax)
    sum_all = np.zeros_like(C)
    for v in M.values():
        sum_all += v
    M_time = (C - G) - sum_all
    return dict(time=M_time, **M)

# --------------------------- dPCA fitting ----------------------------

def _svd_components(block: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD on a 2D block (N × K) -> decoders (N × d), singvals (d,), scores (d × K).
    """
    if n_components <= 0:
        return np.zeros((block.shape[0], 0)), np.zeros((0,)), np.zeros((0, block.shape[1]))
    U, s, Vt = np.linalg.svd(block, full_matrices=False)
    d = min(n_components, U.shape[1])
    return U[:, :d], s[:d], (np.diag(s[:d]) @ Vt[:d, :])

def _reshape_for_svd(MS: np.ndarray) -> Tuple[np.ndarray, Tuple[int,...]]:
    """
    Flatten (N, Tt, L1,...,Lr) into (N, K) for SVD; return (X2D, tail_shape).
    """
    N = MS.shape[0]
    tail = int(np.prod(MS.shape[1:]))
    return MS.reshape(N, tail), MS.shape[1:]

# --------------------------- Public API ------------------------------

def dpca_from_psth(
    psth_da: xr.Dataset,
    factors: Dict[str, Union[np.ndarray, List]],     # categorical per trial (strings/ints)
    *,
    align: str = "go_cue",
    time_window: Optional[Tuple[float, float]] = None,
    include_trials: Optional[Union[np.ndarray, List[int]]] = None,
    require_all_ids: bool = True,
    zscore_units: bool = False,                      # usually False for dPCA (work on means)
    n_components_per_marg: Union[int, Dict[str, int]] = 3,  # int or dict per marginalization name
    min_count: int = 1,                              # min trials per condition cell
    single_trial_projection: bool = False,           # project each trial onto each decoder
    save_path: Optional[Union[str, Path]] = None,
    save_format: str = "zarr",                       # {"zarr","nc","npz"}
    overwrite: bool = True,
) -> Dict:
    """
    dPCA on your PSTH using categorical task factors.

    Parameters
    ----------
    psth_da : xr.Dataset
        Your PSTH dataset (see header).
    factors : dict[str, array-like]
        Mapping from factor name (e.g., "stim", "choice") to per-trial categorical labels
        (strings/ints), length = #trials for the chosen alignment.
        You can pass any number of factors (>=1).
    align : {"go_cue","reward_go_cue_start"}
        Which data variable / trial axis to use.
    time_window : (float, float) or None
        Time selection relative to align for dPCA. None = full axis.
    include_trials : array-like of int or None
        Trial IDs to keep. If None, all trials are used.
    require_all_ids : bool
        Error out if any include_trials are absent.
    zscore_units : bool
        If True, z-score units across trials before condition-averaging (rare for dPCA;
        most analyses work on means without z-scoring here).
    n_components_per_marg : int or dict
        If int: number of components per marginalization to keep.
        If dict: per-marginalization (keys are names like "time", "stim", "choice",
        "stim,choice" for interactions).
    min_count : int
        Minimum number of trials required for a condition cell to contribute.
    single_trial_projection : bool
        If True, project each single trial's timecourse onto each marginalization's
        decoders (returns `trial_projection[marg]` as dict of arrays).
    save_path : str | Path | None
        Optional path to save results (zarr/nc/npz).
    save_format : {"zarr","nc","npz"}
        File format if saving.
    overwrite : bool
        Overwrite existing output (for zarr: delete dir first).

    Returns
    -------
    out : dict
        Keys include:
          - "decoders"[marg]  : (N_units × d_m) decoder/loadings per marginalization
          - "scores"[marg]    : (d_m × Tt × L1 × ... × Lr) component timecourses
          - "singvals"[marg]  : (d_m,) singular values per marginalization
          - "explained_var"   : dict with per-marg and total variance fractions
          - "levels"          : dict[factor] -> original level labels
          - "counts"          : condition trial counts (L1 × ... × Lk)
          - "time"            : (Tt,) time vector used
          - "unit_ids"        : (N_units,)
          - "marginalizations": list of marg names like "time", "stim", "choice", "stim,choice", ...
          - Optionally "trial_projection"[marg]: (T_used × d_m × Tt) single-trial projections

          - If saved: "saved_to", "saved_format"

    Notes
    -----
    - Marginalization names are:
        "time" and every non-empty subset of provided factor names, e.g., "stim",
        "choice", "stim,choice", "stim,choice,context", etc.
    - Explained variance is computed as sum of squared block norms over total squared norm.
    """
    # 1) Extract trial × unit × time
    ext = _extract_trial_unit_timecube(psth_da, align=align, time_window=time_window)
    cube_full, trial_ids_full, time = ext["cube"], ext["trial_ids"], ext["time"]
    T_full, N_units, Tt = cube_full.shape

    # 2) Subset trials by IDs
    mask = _mask_from_trial_ids(trial_ids_full, include_trials, require_all=require_all_ids)
    cube_used = cube_full[mask]                                  # (T_used × N × Tt)
    trial_ids_used = trial_ids_full[mask]
    T_used = cube_used.shape[0]

    # 3) Slice/validate factors
    factor_names = list(factors.keys())
    factors_used = {}
    for k in factor_names:
        vec = np.asarray(factors[k])
        if vec.size == T_full:
            factors_used[k] = vec[mask]
        elif vec.size == T_used:
            factors_used[k] = vec
        else:
            raise ValueError(f"Factor '{k}' length {vec.size} must match T_full={T_full} or T_used={T_used}.")

    # 4) Optional z-scoring across trials per unit (before averaging)
    if zscore_units:
        # Flatten time into features, z-score per unit per time, then reshape
        X = cube_used.reshape(T_used, N_units*Tt)
        Xz, _, _ = _zscore_units_over_trials(X)
        cube_used = Xz.reshape(T_used, N_units, Tt)

    # 5) Condition means tensor (N, Tt, L1, ..., Lk)
    C, levels, n_levels, counts = _build_condition_tensor(
        cube_used, factors_used, min_count=min_count, drop_sparse=False
    )

    # 6) Marginalizations (inclusion–exclusion)
    marg_blocks = _marginalize(C, factor_names)   # dict with "time" and tuples
    # Convert tuple keys to comma-joined strings for readability
    blocks = {"time": marg_blocks["time"]}
    for k, v in marg_blocks.items():
        if k == "time":
            continue
        name = ",".join(k)  # e.g., "stim,choice"
        blocks[name] = v

    # 7) For each block, SVD on (N × K) matrix
    def _ncomp_for(name: str) -> int:
        if isinstance(n_components_per_marg, dict):
            return int(n_components_per_marg.get(name, 3))
        return int(n_components_per_marg)

    decoders = {}
    scores = {}
    singvals = {}

    total_power = 0.0
    block_powers = {}

    for name, B in blocks.items():   # B: (N, Tt, L1,...)
        X2D, tail_shape = _reshape_for_svd(B)  # (N × K)
        # Power (sum of squares) for explained var
        pow_block = float(np.nansum(X2D**2))
        block_powers[name] = pow_block
        total_power += pow_block

        d = _ncomp_for(name)
        U, s, S = _svd_components(X2D, d)
        decoders[name] = U                                      # (N × d)
        singvals[name] = s                                      # (d,)
        # Reshape scores back to (d × Tt × levels...)
        scores[name] = S.reshape((S.shape[0],) + tail_shape)

    # 8) Explained variance fractions
    explained_var = {
        "per_marginalization": {name: (block_powers[name] / total_power if total_power > 0 else np.nan)
                                for name in blocks.keys()},
        "total_power": total_power
    }

    out: Dict[str, object] = dict(
        decoders=decoders,
        scores=scores,
        singvals=singvals,
        explained_var=explained_var,
        levels=levels,
        counts=counts,
        time=time,
        unit_ids=(psth_da["unit_index"].values if "unit_index" in psth_da.coords else np.arange(N_units)),
        marginalizations=list(blocks.keys()),
        align=align,
        time_window=time_window,
        trial_ids_used=trial_ids_used,
        n_units=N_units,
        n_trials_used=T_used,
    )

    # 9) Optional: single-trial projection (project raw single trials on each decoder)
    if single_trial_projection:
        trial_proj = {}
        # For each trial, projection = (N × Tt) dot (N × d) → (d × Tt)
        # First arrange trial data as (T × N × Tt)
        # No extra z-scoring here; it projects in the same space where means lived.
        for name, U in decoders.items():
            d = U.shape[1]
            # proj[t, d, time] = U^T (N×d)' × trial[N×time]
            proj = np.empty((T_used, d, Tt), dtype=float)
            for ti in range(T_used):
                proj[ti] = U.T @ cube_used[ti].T     # (d × Tt)
            trial_proj[name] = proj
        out["trial_projection"] = trial_proj

    # 10) Optional save
    if save_path is not None:
        path = Path(save_path)
        fmt = save_format.lower()
        if fmt not in {"zarr", "nc", "npz"}:
            raise ValueError("save_format must be one of {'zarr','nc','npz'}")

        # Build an xarray Dataset for convenient persistence
        # We store component timecourses per marginalization in a multi-index-like layout.
        ds_vars = {}
        for name in blocks.keys():
            # decoders: unit × comp
            U = decoders[name]
            ds_vars[f"decoder__{name}"] = (("unit", "comp"), U)
            # scores: comp × time × (levels...)
            S = scores[name]
            base_coords = {"comp": np.arange(S.shape[0], dtype=int),
                           "time": time}
            # add per-factor coords for factors present in this marginalization
            if name == "time":
                ds_vars[f"scores__{name}"] = (("comp", "time"), S)
            else:
                facs = name.split(",")
                lvl_coords = {f: levels[f] for f in facs}
                dims = ("comp", "time") + tuple(f"level__{f}" for f in facs)
                coords = {**base_coords, **{"level__"+f: levels[f] for f in facs}}
                ds_vars[f"scores__{name}"] = (dims, S, )

        # trial projections (optional)
        if single_trial_projection:
            for name, P in out["trial_projection"].items():
                ds_vars[f"trial_projection__{name}"] = (("trial", "comp", "time"), P)

        # Pack counts and factor level coords
        coords = dict(
            unit=("unit", out["unit_ids"]),
            time=("time", time),
            trial=("trial", trial_ids_used),
        )
        # Add factor levels as coords
        for f, lev in levels.items():
            coords[f"level__{f}"] = ("level__"+f, lev)

        attrs = dict(
            align=align,
            time_window=time_window,
            n_units=int(N_units),
            n_trials_used=int(T_used),
            marginalizations=",".join(out["marginalizations"]),
            explained_var_json=json.dumps(explained_var),
        )

        ds = xr.Dataset(ds_vars, coords=coords, attrs=attrs)

        if fmt == "zarr":
            import shutil
            if path.exists() and overwrite:
                shutil.rmtree(path)
            ds.to_zarr(path, mode="w")
        elif fmt == "nc":
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} exists and overwrite=False")
            ds.to_netcdf(path)
        elif fmt == "npz":
            # Minimal NPZ: store decoders/scores per marginalization and metadata JSON
            np.savez_compressed(
                path,
                **{f"decoder__{k}": v for k, v in decoders.items()},
                **{f"scores__{k}": v for k, v in scores.items()},
                unit_ids=out["unit_ids"],
                time=time,
                trial_ids=trial_ids_used,
                levels_json=json.dumps({k: [str(x) for x in v] for k, v in levels.items()}),
                explained_var_json=json.dumps(explained_var),
                marginalizations_json=json.dumps(out["marginalizations"]),
                meta_json=json.dumps(dict(align=align, time_window=time_window,
                                          n_units=int(N_units), n_trials_used=int(T_used))),
            )
        out["saved_to"] = str(path)
        out["saved_format"] = fmt

    return out

