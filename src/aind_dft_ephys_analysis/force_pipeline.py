# -*- coding: utf-8 -*-
"""
Unified Full-FORCE (stable RLS) Framework — PCA-ready
-----------------------------------------------------
Supports:
  • Single full-session training (block_len_s <= 0)
  • Block-wise training with slow-drift context (block_len_s > 0)
  • Optional PCA compression before training (set pca_n or pca_var)
  • Stable per-unit RLS learning, robust tanh scaling
  • Automatic visualization and optional saving
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 0) Small PCA helper (no sklearn)
# =============================================================================

@dataclass
class PCAModel:
    mean: np.ndarray          # (U,)
    components: np.ndarray    # (K, U) rows are PC directions in unit-space
    explained_var: np.ndarray # (K,) eigenvalues / (T-1)
    explained_ratio: np.ndarray # (K,) fraction of variance per PC

def pca_fit(Y: np.ndarray, n_components: Optional[int]=None,
            var_thresh: Optional[float]=None) -> PCAModel:
    """
    Fit PCA across the UNIT dimension on (U,T) data (spikes/s).
    Returns a basis with rows in unit-space.
    """
    U, T = Y.shape
    mu = Y.mean(axis=1)                # (U,)
    Yc = Y - mu[:, None]               # (U,T)

    # economy SVD: Yc = Umat @ diag(S) @ Vt
    Umat, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    eigvals = (S**2) / max(T-1, 1)     # variance per PC
    total = eigvals.sum()

    # choose K
    if var_thresh is not None:
        assert 0 < var_thresh <= 1.0
        csum = np.cumsum(eigvals) / (total + 1e-12)
        K = int(np.searchsorted(csum, var_thresh) + 1)
    elif n_components is not None:
        K = int(min(n_components, Umat.shape[1]))
    else:
        K = Umat.shape[1]  # no reduction

    C = Umat[:, :K].T                  # (K,U)
    expl = eigvals[:K]
    ratio = expl / (total + 1e-12)
    return PCAModel(mean=mu, components=C, explained_var=expl, explained_ratio=ratio)

def pca_transform(model: PCAModel, Y: np.ndarray) -> np.ndarray:
    """Project (U,T) to PC scores (K,T)."""
    Yc = Y - model.mean[:, None]
    return model.components @ Yc  # (K,U) @ (U,T) -> (K,T)

def pca_inverse(model: PCAModel, Z: np.ndarray) -> np.ndarray:
    """Map PC scores (K,T) back to unit space (U,T)."""
    return model.mean[:, None] + (model.components.T @ Z)


# =============================================================================
# 1) Event → input builder  (UPDATED: optional smoothing for stability)
# =============================================================================

@dataclass
class InputDesign:
    U: np.ndarray
    t_aligned: np.ndarray
    t_abs: np.ndarray
    channel_names: List[str]
    t0_abs: float

def _smooth_rows(
    X: np.ndarray, dt: float,
    *, kind: Optional[str] = "exp",
    tau: float = 0.2,
    win_fwhm_s: float = 0.25,
    renorm: bool = True,
    clip01: bool = True
) -> np.ndarray:
    """
    Row-wise smoothing for (C,T) inputs.
    - kind == "exp": causal exponential IIR: y[t] = y[t-1] + alpha*(x[t]-y[t-1])
                     with alpha = dt / tau, capped at 1.
    - kind == "gauss": zero-phase FIR via symmetric Gaussian kernel.
      FWHM (seconds) is win_fwhm_s, sigma = FWHM / 2.355.
    - None: returns X unchanged.

    Renormalization keeps per-row max ≤ 1 (useful when smoothing blurs pulses).
    """
    if kind is None:
        Y = X.copy()
        if clip01:
            Y = np.clip(Y, 0.0, 1.0)
        return Y

    C, T = X.shape
    Y = np.empty_like(X, dtype=float)

    if kind == "exp":
        # causal IIR smoothing
        alpha = dt / max(tau, 1e-6)
        alpha = min(alpha, 1.0)
        for c in range(C):
            y = 0.0
            for t in range(T):
                x = float(X[c, t])
                y = y + alpha * (x - y)
                Y[c, t] = y

    elif kind == "gauss":
        # zero-phase FIR using symmetric Gaussian window (convolve with 'same')
        sigma = max(win_fwhm_s, 1e-6) / 2.355
        half_w = int(np.ceil(4.0 * sigma / dt))  # ~±4σ support
        kx = np.arange(-half_w, half_w + 1) * dt
        g = np.exp(-0.5 * (kx / sigma) ** 2)
        g /= g.sum() + 1e-12
        # pad with zeros; inputs are 0/1 pulses so zero-padding is natural
        for c in range(C):
            Y[c] = np.convolve(X[c], g, mode="same")
    else:
        raise ValueError("smooth kind must be 'exp', 'gauss', or None")

    if renorm:
        # keep each channel at most 1.0 after smoothing
        mx = np.max(Y, axis=1, keepdims=True)
        mx = np.maximum(mx, 1e-12)
        Y = Y / mx
    if clip01:
        Y = np.clip(Y, 0.0, 1.0)
    return Y

def build_event_inputs_from_trials(
    nwb_data: Any,
    dt: float,
    pre_first_go: float = 3.0,
    post_last_go: float = 6.0,
    go_dur: float = 0.2,
    resp_delay: float = 0.2,
    resp_dur: float = 0.4,
    rew_delay: float = 0.2,
    rew_dur: float = 1.0,
    *,
    # ---- NEW: smoothing controls ----
    smooth_kind: Optional[str] = "exp",  # "exp", "gauss", or None
    smooth_tau: float = 0.2,             # for "exp" (seconds)
    smooth_win_s: float = 0.25,          # FWHM for "gauss" (seconds)
    smooth_renorm: bool = True,
    smooth_clip: bool = True,
) -> InputDesign:
    """
    Build binary event channels then optionally smooth them for stability.

    Channels:
      0 go
      1 resp_L (animal_response==0)
      2 resp_R (animal_response==1)
      3 resp_NoResp (animal_response otherwise)
      4 rew_L (rewarded_historyL True)
      5 rew_R (rewarded_historyR True)
    """
    go = np.asarray(nwb_data.trials["goCue_start_time"], dtype=float)
    resp = np.asarray(nwb_data.trials["animal_response"], dtype=int)
    rewL = np.asarray(nwb_data.trials["rewarded_historyL"], dtype=bool)
    rewR = np.asarray(nwb_data.trials["rewarded_historyR"], dtype=bool)

    t0_abs, t1_abs = float(go[0]), float(go[-1])
    start_abs, end_abs = t0_abs - pre_first_go, t1_abs + post_last_go
    T = int(np.floor((end_abs - start_abs) / dt)) + 1
    t_abs = start_abs + np.arange(T) * dt
    t_aligned = t_abs - t0_abs

    n_inputs = 6
    U = np.zeros((n_inputs, T), dtype=float)
    ch_names = ["go", "resp_L", "resp_R", "resp_NoResp", "rew_L", "rew_R"]
    CH_GO, CH_RL, CH_RR, CH_RN, CH_WL, CH_WR = range(6)

    def set_high(ch: int, a_abs: float, b_abs: float):
        """Set channel ch high on [a_abs, b_abs)."""
        i0 = max(0, int(np.floor((a_abs - start_abs) / dt)))
        i1 = min(T, int(np.ceil((b_abs - start_abs) / dt)))
        if i1 > i0:
            U[ch, i0:i1] = 1.0

    # Fill binary pulses
    for i in range(go.size):
        t_go = float(go[i])
        # go cue
        set_high(CH_GO, t_go, t_go + go_dur)

        # response window
        t_r0, t_r1 = t_go + resp_delay, t_go + resp_delay + resp_dur
        if resp[i] == 0:
            set_high(CH_RL, t_r0, t_r1)
        elif resp[i] == 1:
            set_high(CH_RR, t_r0, t_r1)
        else:
            set_high(CH_RN, t_r0, t_r1)

        # reward window
        t_w0, t_w1 = t_go + rew_delay, t_go + rew_delay + rew_dur
        if rewL[i]:
            set_high(CH_WL, t_w0, t_w1)
        if rewR[i]:
            set_high(CH_WR, t_w0, t_w1)

    # ---- Smooth for stability (keeps channels in [0,1]) ----
    U_sm = _smooth_rows(
        U, dt,
        kind=smooth_kind,
        tau=smooth_tau,
        win_fwhm_s=smooth_win_s,
        renorm=smooth_renorm,
        clip01=smooth_clip
    )

    return InputDesign(U=U_sm, t_aligned=t_aligned, t_abs=t_abs,
                       channel_names=ch_names, t0_abs=t0_abs)


# =============================================================================
# 2) Add slow context channels
# =============================================================================

def add_context_channels(U_in, t, *, n_ctx=4, use_block_onehot=False, block_edges_idx=None):
    T = t.size
    ctx_list, ctx_names = [], []
    x = np.linspace(0, 1, T)
    for k in range(1, n_ctx + 1):
        ctx_list.append(np.cos(np.pi * k * x))
        ctx_names.append(f"ctx_cos{k}")
    if use_block_onehot and block_edges_idx:
        for b, (i0, i1) in enumerate(block_edges_idx):
            v = np.zeros(T)
            v[i0:i1] = 1.0
            ctx_list.append(v)
            ctx_names.append(f"ctx_block{b+1}")
    if ctx_list:
        U_aug = np.vstack([U_in, np.vstack(ctx_list)])
    else:
        U_aug = U_in
    return U_aug, ctx_names


# =============================================================================
# 3) Robust scaling
# =============================================================================

def scale_psth_to_tanh_percentile(Y_raw, lo=1.0, hi=99.0, clip=0.9):
    Y = np.asarray(Y_raw, float)
    base = np.percentile(Y, lo, axis=1, keepdims=True)
    Y0 = np.maximum(Y - base, 0.0)
    scale = np.maximum(np.percentile(Y0, hi, axis=1, keepdims=True), 1e-6)
    Y01 = np.clip(Y0 / scale, 0, 1)
    Y_sc = (2 * clip) * (Y01 - 0.5)
    return np.clip(Y_sc, -clip, clip), {"baseline": base, "scale": scale, "clip": clip}

def invert_tanh_scaling(Y_sc, scaler):
    clip = scaler["clip"]
    base, scale = scaler["baseline"], scaler["scale"]
    Y01 = (Y_sc / (2 * clip)) + 0.5
    return Y01 * scale + base


# =============================================================================
# 4) FORCE learning
# =============================================================================

def fullforce_fit_W_normalized_stable(
    Y_raw: np.ndarray,           # (U,T)
    U_inputs: np.ndarray,        # (C,T)
    tau: float,
    dt: float,
    *,
    alpha: float = 10.0,
    lam: float = 0.995,
    clip: float = 0.90,
    err_clip: float = 50.0,
    k_clip: float = 5.0,
    p_cond_cap: float = 1e6,
    recond_every: int = 5000,
    seed: int = 42,
    log_every: int = 1000,
    use_tqdm: bool = True,
    update_every: int = 1,    # update weights every N steps (speed knob)
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    if U_inputs.shape[1] != Y_raw.shape[1]:
        raise ValueError("U_inputs and Y_raw must have the SAME time length.")
    rng = np.random.default_rng(seed)
    U_units, T = Y_raw.shape
    C = U_inputs.shape[0]

    print(f"[fullFORCE] weight updates every {update_every} steps")

    # scaling & teacher
    Y_sc, scaler = scale_psth_to_tanh_percentile(Y_raw, lo=1, hi=99, clip=clip)
    Y_sc = np.clip(Y_sc, -0.999, 0.999)
    x_star = np.arctanh(Y_sc)
    r_star = np.tanh(x_star)  # == Y_sc

    # input weights
    Win = rng.standard_normal((U_units, C)) / np.sqrt(max(C, 1))

    # target recurrent drive
    factor = tau / dt
    one_minus = 1.0 - dt / tau
    h_star = np.empty((U_units, T - 1), dtype=float)
    for t in range(T - 1):
        h_star[:, t] = factor * (x_star[:, t + 1] - one_minus * x_star[:, t]) - (Win @ U_inputs[:, t])

    # RLS init
    W = np.zeros((U_units, U_units), dtype=float)
    P = np.array([np.eye(U_units) / alpha for _ in range(U_units)])

    mse_ema, beta = 0.0, 0.98
    lam = float(lam)
    inv_lam = 1.0 / lam
    eps = 1e-8

    iterator = range(T - 1)
    if use_tqdm:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, total=T - 1, desc="fullFORCE (stable)", ncols=96)
        except Exception:
            pass

    for ti in iterator:
        if (ti % update_every) != 0:
            continue

        r_t = r_star[:, ti]
        r_t_col = r_t[:, None]
        for i in range(U_units):
            Pi = P[i]
            Pi_r = Pi @ r_t_col
            denom = lam + float(r_t_col.T @ Pi_r)
            denom = max(denom, eps)

            k = (Pi_r / denom).squeeze()
            k_norm = np.linalg.norm(k)
            if k_norm > k_clip:
                k *= (k_clip / (k_norm + 1e-12))

            err = float(W[i] @ r_t - h_star[i, ti])
            if not np.isfinite(err):
                err = 0.0
            err = np.clip(err, -err_clip, err_clip)

            W[i] -= err * k
            P[i] = (Pi - np.outer(k, r_t) @ Pi) * inv_lam

        # Logging
        if ((ti + 1) % log_every == 0) or (ti == T - 2):
            fit_t = W @ r_t
            mse = float(np.mean((fit_t - h_star[:, ti]) ** 2))
            mse_ema = beta * mse_ema + (1 - beta) * mse
            if use_tqdm and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"step": ti + 1, "mse": f"{mse:.3g}", "ema": f"{mse_ema:.3g}"})
            else:
                print(f"[{ti+1}/{T-1}] mse={mse:.3g}  ema={mse_ema:.3g}")

        # Occasional reconditioning
        if (ti + 1) % recond_every == 0:
            for i in range(U_units):
                nrm = np.linalg.norm(P[i], ord=2)
                if not np.isfinite(nrm) or nrm > p_cond_cap:
                    P[i] = np.eye(U_units) / alpha

    return W, Win, scaler, r_star


# =============================================================================
# 5) Forward simulation
# =============================================================================

def fullforce_run(W, Win, U_inputs, tau, dt, x0=None):
    U_units = Win.shape[0]
    T = U_inputs.shape[1]
    x = np.zeros(U_units) if x0 is None else np.array(x0, float)
    r = np.tanh(x)
    R = np.zeros((U_units, T))
    for t in range(T):
        x += (dt / tau) * (-x + (W @ r) + (Win @ U_inputs[:, t]))
        r = np.tanh(x)
        R[:, t] = r
    return R


# =============================================================================
# 6) Visualization utilities
# =============================================================================

def _rowwise_r2(y_true, y_pred):
    num = np.sum((y_true - y_pred) ** 2, 1)
    den = np.sum((y_true - y_true.mean(1, keepdims=True)) ** 2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1 - num / den
    r2[~np.isfinite(r2)] = np.nan
    return r2

def plot_fullforce_dashboard(Y_raw, Y_hat, t, U_in, channel_names,
                             heat_vmin=0, heat_vmax=20, n_examples=6):
    plt.figure(figsize=(9, 3.4))
    plt.imshow(U_in, aspect="auto", extent=[t[0], t[-1], -0.5, U_in.shape[0]-0.5], origin="lower")
    plt.yticks(range(len(channel_names)), channel_names)
    plt.xlabel("Time (s)")
    plt.title("Inputs (Events + Context)")
    plt.colorbar(label="Amplitude")
    plt.tight_layout(); plt.show()

    order = np.argsort(np.nanargmax(Y_raw, 1))
    Yt, Yh = Y_raw[order], Y_hat[order]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, Y, ttl in zip(axes, [Yt, Yh], ["Target PSTH (spk/s)", "Reconstruction (spk/s)"]):
        im = ax.imshow(Y, aspect="auto", origin="lower", extent=[t[0], t[-1], 0, Y.shape[0]],
                       vmin=heat_vmin, vmax=heat_vmax)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax)
    axes[1].set_xlabel("Time (s)")
    plt.tight_layout(); plt.show()

    r2 = _rowwise_r2(Y_raw, Y_hat)
    plt.figure(figsize=(6,3))
    plt.hist(r2[~np.isnan(r2)], bins=30)
    plt.title(f"R² mean={np.nanmean(r2):.3f}")
    plt.tight_layout(); plt.show()


# =============================================================================
# 7) Block utilities
# =============================================================================

@dataclass
class BlockResult:
    start: int
    stop: int
    W: np.ndarray
    Win: np.ndarray
    scaler: Dict[str, np.ndarray]

@dataclass
class BlockwiseFullForceResult:
    blocks: List[BlockResult]
    Y_hat: np.ndarray
    Y_raw: np.ndarray
    t: np.ndarray
    U_in: np.ndarray
    channel_names: List[str]
    # PCA info (if used)
    pca_used: bool
    pca_mean: Optional[np.ndarray] = None
    pca_components: Optional[np.ndarray] = None
    pca_explained_ratio: Optional[np.ndarray] = None
    latent_dim: Optional[int] = None   # K
    original_units: Optional[int] = None  # U

def _make_blocks(T, dt, block_len_s):
    blen = int(round(block_len_s / dt))
    blen = max(blen, 2)
    return [(s, min(T, s + blen)) for s in range(0, T, blen)]

def train_blocks_and_stitch(Y_units: np.ndarray, U_in_events: np.ndarray, t: np.ndarray, *,
                            tau, alpha, lam, clip,
                            log_every, use_tqdm, block_len_s,
                            n_ctx, use_block_onehot,
                            warm_blend=0.25, seed=42,
                            # PCA model (optional)
                            pca_model: Optional[PCAModel] = None,
                            update_every: int = 1):
    """
    If pca_model is not None, training is done in PC space (K x T).
    """
    U_orig, T = Y_units.shape
    dt = float(np.median(np.diff(t)))
    blocks_idx = _make_blocks(T, dt, block_len_s)

    # Optional PCA transform (fit already done outside)
    if pca_model is not None:
        Y_train = pca_transform(pca_model, Y_units)  # (K,T)
        latent_dim = Y_train.shape[0]
        pca_used = True
    else:
        Y_train = Y_units
        latent_dim = Y_units.shape[0]
        pca_used = False

    # Build context channels once (same for all training variants)
    U_aug, ctx_names = add_context_channels(
        U_in_events, t, n_ctx=n_ctx,
        use_block_onehot=use_block_onehot,
        block_edges_idx=blocks_idx if use_block_onehot else None
    )
    names = ["go", "resp_L", "resp_R", "resp_NoResp", "rew_L", "rew_R"] + ctx_names

    # Storage
    blocks: List[BlockResult] = []
    Y_hat_units = np.zeros_like(Y_units)  # final reconstruction in ORIGINAL unit space
    W_prev = None

    for b, (i0, i1) in enumerate(blocks_idx, 1):
        Y_blk = Y_train[:, i0:i1]
        U_blk = U_aug[:, i0:i1]

        # Train in current space (units or PCs)
        W, Win, scaler, _ = fullforce_fit_W_normalized_stable(
            Y_blk, U_blk, tau=tau, dt=dt,
            alpha=alpha, lam=lam, clip=clip,
            err_clip=50.0, k_clip=5.0,
            p_cond_cap=1e6, recond_every=5000,
            seed=seed + b, log_every=log_every, use_tqdm=use_tqdm,
            update_every=update_every
        )

        # Warm blending
        if W_prev is not None and warm_blend > 0:
            W = warm_blend * W_prev + (1.0 - warm_blend) * W

        # Simulate this block in current space
        R_blk = fullforce_run(W, Win, U_blk, tau=tau, dt=dt)
        # Back to spikes/s of current space
        Y_blk_hat_current = invert_tanh_scaling(R_blk, scaler)

        # If trained in PC space, map back to original units
        if pca_used:
            Y_blk_hat_units = pca_inverse(pca_model, Y_blk_hat_current)
        else:
            Y_blk_hat_units = Y_blk_hat_current

        # Stitch
        Y_hat_units[:, i0:i1] = Y_blk_hat_units

        blocks.append(BlockResult(i0, i1, W, Win, scaler))
        W_prev = W.copy()

    return BlockwiseFullForceResult(
        blocks=blocks, Y_hat=Y_hat_units, Y_raw=Y_units, t=t,
        U_in=U_aug, channel_names=names,
        pca_used=pca_used,
        pca_mean=(pca_model.mean if pca_used else None),
        pca_components=(pca_model.components if pca_used else None),
        pca_explained_ratio=(pca_model.explained_ratio if pca_used else None),
        latent_dim=latent_dim,
        original_units=U_orig
    )


# =============================================================================
# 8) Unified entry point with optional PCA + saving
# =============================================================================

def run_fullforce_pipeline_auto(
    psth_da, nwb_data,
    *,
    block_len_s=0.0, tau=0.05, alpha=10.0, lam=0.995, clip=0.9,
    n_ctx=4, use_block_onehot=True, warm_blend=0.25,
    # --- PCA controls ---
    pca_n: Optional[int] = None,          # exact # PCs
    pca_var: Optional[float] = None,      # variance threshold (e.g., 0.95)
    # ---------------------
    log_every=1000, use_tqdm=True, heat_vmax=20.0,
    save_path=None,
    update_every: int = 1,                # pass to trainer
):
    # Pull data
    Y_units = np.asarray(psth_da.values, float)  # (U,T)
    t = np.asarray(psth_da.coords["time"].values, float)
    dt_psth = float(np.median(np.diff(t)))

    # Build event inputs on SAME grid (now smoothed by default)
    inputs = build_event_inputs_from_trials(
        nwb_data, dt_psth,
        # You may override defaults here; example:
        # smooth_kind="exp", smooth_tau=0.2,
        # or smooth_kind="gauss", smooth_win_s=0.3
    )
    U_events = inputs.U

    # ---- PCA fit (if requested). We ALWAYS fit on the entire session so the basis
    # is consistent across blocks. Training may be single-session or block-wise.
    pca_model = None
    if (pca_n is not None) or (pca_var is not None):
        pca_model = pca_fit(Y_units, n_components=pca_n, var_thresh=pca_var)
        K = pca_model.components.shape[0]
        print(f"[PCA] reduction: U={Y_units.shape[0]} -> K={K} "
              f"(explained={pca_model.explained_ratio.sum():.3f})")

    # -------------------- SINGLE SESSION --------------------
    if block_len_s <= 0:
        print(f"[fullFORCE: single] U={Y_units.shape[0]}, T={Y_units.shape[1]}, dt={dt_psth:.6g}s")

        # Choose training target (units or PCs)
        if pca_model is not None:
            Y_train = pca_transform(pca_model, Y_units)  # (K,T)
        else:
            Y_train = Y_units

        # Train
        W, Win, scaler, _ = fullforce_fit_W_normalized_stable(
            Y_train, U_events, tau, dt_psth, alpha=alpha, lam=lam, clip=clip,
            log_every=log_every, use_tqdm=use_tqdm, update_every=update_every
        )

        # Forward & invert scaling (in training space)
        R = fullforce_run(W, Win, U_events, tau, dt_psth)
        Y_hat_current = invert_tanh_scaling(R, scaler)

        # If PCA used, map back to original units
        if pca_model is not None:
            Y_hat = pca_inverse(pca_model, Y_hat_current)
        else:
            Y_hat = Y_hat_current

        # Plot
        plot_fullforce_dashboard(Y_units, Y_hat, t, U_events, inputs.channel_names,
                                 heat_vmin=0, heat_vmax=heat_vmax)

        # Save
        if save_path:
            pack = {
                "W": W.astype(np.float32),
                "Win": Win.astype(np.float32),
                "baseline": scaler["baseline"].astype(np.float32),
                "scale": scaler["scale"].astype(np.float32),
                "clip": np.array([scaler["clip"]], np.float32),
                "tau": np.array([tau], np.float32),
                "dt": np.array([dt_psth], np.float32),
                "Y_hat": Y_hat.astype(np.float32),
                "time": t.astype(np.float32),
                "channel_names": np.array(inputs.channel_names, dtype=object),
                "pca_used": np.array([pca_model is not None], np.int8),
            }
            if pca_model is not None:
                pack.update({
                    "pca_mean": pca_model.mean.astype(np.float32),
                    "pca_components": pca_model.components.astype(np.float32),
                    "pca_explained_ratio": pca_model.explained_ratio.astype(np.float32),
                })
            np.savez_compressed(save_path, **pack)
            print(f"[fullFORCE: single] Saved model to: {save_path}")

        return {
            "W": W, "Win": Win, "Y_hat": Y_hat, "scaler": scaler,
            "U_in": U_events, "t": t, "channel_names": inputs.channel_names,
            "pca_used": (pca_model is not None),
            "latent_dim": (Y_train.shape[0]),
            "original_units": Y_units.shape[0],
            "pca_model": pca_model,
        }

    # -------------------- BLOCK-WISE --------------------
    print(f"[fullFORCE: block-wise] U={Y_units.shape[0]}, T={Y_units.shape[1]}, dt={dt_psth:.6g}s")
    res = train_blocks_and_stitch(
        Y_units, U_events, t,
        tau=tau, alpha=alpha, lam=lam, clip=clip,
        log_every=log_every, use_tqdm=use_tqdm,
        block_len_s=block_len_s,
        n_ctx=n_ctx, use_block_onehot=use_block_onehot,
        warm_blend=warm_blend, seed=42,
        pca_model=pca_model, update_every=update_every
    )

    # Plot
    plot_fullforce_dashboard(res.Y_raw, res.Y_hat, res.t, res.U_in, res.channel_names,
                             heat_vmin=0, heat_vmax=heat_vmax)

    # Save
    if save_path:
        packed = {
            "Y_hat": res.Y_hat.astype(np.float32),
            "time": res.t.astype(np.float32),
            "channel_names": np.array(res.channel_names, dtype=object),
            "pca_used": np.array([res.pca_used], np.int8),
        }
        if res.pca_used:
            packed.update({
                "pca_mean": res.pca_mean.astype(np.float32),
                "pca_components": res.pca_components.astype(np.float32),
                "pca_explained_ratio": res.pca_explained_ratio.astype(np.float32),
            })
        for bi, blk in enumerate(res.blocks):
            packed[f"W_{bi}"] = blk.W.astype(np.float32)
            packed[f"Win_{bi}"] = blk.Win.astype(np.float32)
            packed[f"base_{bi}"] = blk.scaler["baseline"].astype(np.float32)
            packed[f"scale_{bi}"] = blk.scaler["scale"].astype(np.float32)
            packed[f"clip_{bi}"] = np.array([blk.scaler["clip"]], np.float32)
            packed[f"span_{bi}"] = np.array([blk.start, blk.stop], np.int32)
        np.savez_compressed(save_path, **packed)
        print(f"[fullFORCE: block-wise] Saved model to: {save_path}")

    return {
        "blocks": res.blocks,
        "Y_hat": res.Y_hat, "Y_raw": res.Y_raw, "t": res.t,
        "U_in": res.U_in, "channel_names": res.channel_names,
        "pca_used": res.pca_used,
        "pca_mean": res.pca_mean, "pca_components": res.pca_components,
        "pca_explained_ratio": res.pca_explained_ratio,
        "latent_dim": res.latent_dim, "original_units": res.original_units,
    }
