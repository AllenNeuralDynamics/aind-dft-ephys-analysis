from typing import Any, Dict, Optional, Sequence, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt

from behavior_utils import get_fitted_latent


# ============================================================
# Reward-rate metrics for 2-lickspout dynamic foraging (NWB)
#
# Standardized NWB trial fields (assumed fixed)
# ---------------------------------------------
#   - nwb_data.trials['rewarded_historyL'][:] : bool
#   - nwb_data.trials['rewarded_historyR'][:] : bool
#   - nwb_data.trials['animal_response'][:]   : int
#       0 = left, 1 = right, 2 = noresponse
#
# Global convention used throughout
# ---------------------------------
# Every function has include_noresponse:
#   - include_noresponse=False:
#       Exclude noresponse (choice==2) trials from the metric.
#       Means: those trials are treated as NaN / skipped; EWMA holds value.
#   - include_noresponse=True:
#       Include noresponse trials as 0-reward contributions.
#       Means: omissions reduce reward rate.
# ============================================================


# ------------------------------
# Basic extractors (standard keys)
# ------------------------------

def _as_bool_trials(nwb_data, key: str) -> np.ndarray:
    """Load a boolean trial column from nwb_data.trials."""
    return np.asarray(nwb_data.trials[key][:], dtype=bool)


def _as_int_trials(nwb_data, key: str) -> np.ndarray:
    """Load an integer trial column from nwb_data.trials."""
    return np.asarray(nwb_data.trials[key][:], dtype=int)


def get_choices(nwb_data) -> np.ndarray:
    """
    Get per-trial choices from the standardized column 'animal_response'.

    Returns
    -------
    choices : (n_trials,) int ndarray
      0 = left, 1 = right, 2 = noresponse
    """
    choices = _as_int_trials(nwb_data, "animal_response")
    if choices.ndim != 1:
        raise ValueError(f"Expected 1D choices vector, got shape={choices.shape}")
    return choices


def responded_mask(choices: np.ndarray) -> np.ndarray:
    """Boolean mask for trials with an actual response (left/right)."""
    choices = np.asarray(choices, dtype=int)
    return (choices == 0) | (choices == 1)


# ------------------------------
# Reward vectors with omission handling
# ------------------------------

def get_reward_vectors(nwb_data, *, include_noresponse: bool = False) -> Dict[str, np.ndarray]:
    """
    Get canonical reward vectors with consistent omission handling.

    Definitions (before omission handling)
    --------------------------------------
    rL[t]   = 1 if left reward delivered on trial t else 0
    rR[t]   = 1 if right reward delivered on trial t else 0
    rAny[t] = 1 if reward delivered on either side on trial t else 0

    Omission handling
    -----------------
    If include_noresponse=False:
      reward values at choice==2 trials are set to NaN (so means ignore them).
    If include_noresponse=True:
      reward values at choice==2 trials are set to 0 (so omissions reduce rates).

    Returns
    -------
    dict:
      'left'  -> (n_trials,) float ndarray
      'right' -> (n_trials,) float ndarray
      'any'   -> (n_trials,) float ndarray
    """
    rL = _as_bool_trials(nwb_data, "rewarded_historyL").astype(float)
    rR = _as_bool_trials(nwb_data, "rewarded_historyR").astype(float)
    if rL.shape != rR.shape:
        raise ValueError(f"rewarded_historyL shape {rL.shape} != rewarded_historyR shape {rR.shape}")

    rAny = ((rL > 0) | (rR > 0)).astype(float)

    choices = get_choices(nwb_data)
    if choices.shape[0] != rAny.shape[0]:
        raise ValueError("choices length must match reward vectors length")

    noresp = (choices == 2)
    if include_noresponse:
        rL[noresp] = 0.0
        rR[noresp] = 0.0
        rAny[noresp] = 0.0
    else:
        rL[noresp] = np.nan
        rR[noresp] = np.nan
        rAny[noresp] = np.nan

    return {"left": rL, "right": rR, "any": rAny}


def get_experienced_reward(nwb_data, *, include_noresponse: bool = False) -> np.ndarray:
    """
    Experienced reward per trial = reward on the chosen side.

    Definition
    ----------
    experienced[t] =
      rL[t] if choice[t] == 0
      rR[t] if choice[t] == 1
      NaN   if choice[t] == 2 and include_noresponse=False
      0     if choice[t] == 2 and include_noresponse=True

    Returns
    -------
    experienced : (n_trials,) float ndarray
    """
    # Use base reward vectors with omissions included as 0, then apply experienced mapping
    r_base = get_reward_vectors(nwb_data, include_noresponse=True)
    choices = get_choices(nwb_data)

    exp = np.full_like(r_base["any"], np.nan, dtype=float)
    exp[choices == 0] = r_base["left"][choices == 0]
    exp[choices == 1] = r_base["right"][choices == 1]

    if include_noresponse:
        exp[choices == 2] = 0.0

    return exp


# ------------------------------
# Metric 1: Global reward rate
# ------------------------------

def reward_rate_global(
    reward: np.ndarray,
    *,
    include_noresponse: bool,
    choices: Optional[np.ndarray] = None,
) -> float:
    """
    Global reward rate = mean(reward) over selected trials.

    Noresponse handling
    -------------------
    - include_noresponse=True:
        Include all trials as they appear in `reward` (typically 0 on omissions).
    - include_noresponse=False:
        Exclude omission trials. Requires `choices` (or pass reward with NaNs on omissions).

    Notes
    -----
    If you generated `reward` using get_reward_vectors(..., include_noresponse=False),
    omission trials are NaN and np.nanmean will already exclude them. In that case,
    passing choices is optional, but supported for clarity.

    Returns
    -------
    float
    """
    reward = np.asarray(reward, dtype=float)

    if not include_noresponse:
        if choices is not None:
            m = responded_mask(choices)
            reward = reward[m]
        # If choices is None, we rely on NaNs (if present) and nanmean below.

    if reward.size == 0:
        return float("nan")
    return float(np.nanmean(reward))


# ------------------------------
# Metric 2: Running reward rate
# ------------------------------

def reward_rate_running(
    reward: np.ndarray,
    *,
    window: int = 20,
    causal: bool = True,
    include_noresponse: bool,
    choices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sliding-window reward rate.

    Definition (causal=True)
    ------------------------
    rate[t] = mean(reward[max(0, t-window):t])  (past-only; excludes current trial)

    Noresponse handling
    -------------------
    - include_noresponse=True:
        Window mean includes omissions as they appear in `reward` (often 0).
    - include_noresponse=False:
        Exclude omissions from each window mean.
        You can do this either by:
          (a) passing choices (recommended), OR
          (b) using NaNs for omission trials in `reward` and relying on nanmean.

    Returns
    -------
    rate : (n_trials,) float ndarray
    """
    reward = np.asarray(reward, dtype=float)
    n = reward.size
    rate = np.full(n, np.nan, dtype=float)

    if (not include_noresponse) and (choices is not None):
        choices = np.asarray(choices, dtype=int)

    for t in range(n):
        if causal:
            lo = max(0, t - window)
            hi = t
        else:
            lo = max(0, t - window // 2)
            hi = min(n, t + window // 2 + 1)

        if hi <= lo:
            continue

        x = reward[lo:hi]

        if include_noresponse:
            rate[t] = float(np.nanmean(x)) if x.size > 0 else np.nan
        else:
            if choices is None:
                # Rely on NaNs to exclude omissions
                rate[t] = float(np.nanmean(x)) if x.size > 0 else np.nan
            else:
                m = responded_mask(choices[lo:hi])
                vals = x[m]
                rate[t] = float(np.nanmean(vals)) if vals.size > 0 else np.nan

    return rate


# ------------------------------
# Metric 3: EWMA reward rate
# ------------------------------

def reward_rate_ewma(
    reward: np.ndarray,
    *,
    alpha: float = 0.05,
    init: float = 0.0,
    include_noresponse: bool,
    choices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    EWMA reward rate (past-only).

    Definition
    ----------
    rr[0] = init
    rr[t] = (1 - alpha) * rr[t-1] + alpha * reward[t-1]   if update trial is included
    rr[t] = rr[t-1]                                       if update trial is excluded

    Noresponse handling
    -------------------
    - include_noresponse=True:
        EWMA updates every trial using reward[t-1] (omissions typically contribute 0).
    - include_noresponse=False:
        EWMA does NOT update on omission trials (holds value constant).
        Requires choices, unless omission trials are encoded as NaN in reward, in which
        case NaNs also trigger "hold constant".

    Returns
    -------
    rr : (n_trials,) float ndarray
    """
    reward = np.asarray(reward, dtype=float)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")

    if (not include_noresponse) and (choices is not None):
        choices = np.asarray(choices, dtype=int)

    n = reward.size
    rr = np.zeros(n, dtype=float)
    rr[0] = float(init)

    for t in range(1, n):
        prev = rr[t - 1]
        x = reward[t - 1]  # past-only update source

        if include_noresponse:
            rr[t] = (1.0 - alpha) * prev + alpha * (0.0 if np.isnan(x) else x)
        else:
            if choices is None:
                # If x is NaN (e.g., omission), hold rr constant
                rr[t] = prev if np.isnan(x) else (1.0 - alpha) * prev + alpha * x
            else:
                # Only update if previous trial had a response
                if choices[t - 1] in (0, 1) and np.isfinite(x):
                    rr[t] = (1.0 - alpha) * prev + alpha * x
                else:
                    rr[t] = prev

    return rr


def alpha_from_half_life(half_life_trials: float) -> float:
    """
    Convert a desired half-life (trials) to EWMA alpha:
      alpha = 1 - 2^(-1/H)
    """
    if half_life_trials <= 0:
        raise ValueError("half_life_trials must be > 0")
    return 1.0 - 2.0 ** (-1.0 / half_life_trials)


# ------------------------------
# One-call wrapper
# ------------------------------

def compute_all_reward_rates(
    nwb_data,
    *,
    window: Optional[int] = None,
    alpha: Optional[float] = None,
    include_noresponse: bool = True,
    metrics: Union[str, Sequence[str]] = "all",
    include_raw: bool = True,
) -> Dict[str, Any]:
    """
    Compute reward-rate metrics with configurable output families.

    Parameters
    ----------
    window : int or None
        Window length for running reward rate (required only if "running" is requested).
    alpha : float or None
        EWMA smoothing parameter (required only if "ewma" is requested).
    include_noresponse : bool
        Omission handling (consistent with get_reward_vectors / get_experienced_reward).
    metrics : str or sequence of str
        Which metric families to compute/return. Supported:
          - "global"
          - "running"
          - "ewma"
          - "all" (default) = global + running + ewma
    include_raw : bool
        If True, return raw vectors (reward_any/left/right, experienced_reward, choices, responded_mask).

    Returns
    -------
    dict
        Dictionary containing requested metrics.
    """

    # ------------------------------------------------------------
    # Normalize metrics selection
    # ------------------------------------------------------------
    if isinstance(metrics, str):
        metrics_set = {metrics.lower()}
    else:
        metrics_set = {str(m).lower() for m in metrics}

    if "all" in metrics_set:
        metrics_set = {"global", "running", "ewma"}

    allowed = {"global", "running", "ewma"}
    unknown = sorted(metrics_set - allowed)
    if unknown:
        raise ValueError(f"Unknown metrics={unknown}. Allowed: {sorted(allowed)} or 'all'.")

    # ------------------------------------------------------------
    # Validate required parameters *only if needed*
    # ------------------------------------------------------------
    if "running" in metrics_set:
        if window is None or window <= 0:
            raise ValueError("`window` must be a positive int when 'running' metrics are requested.")

    if "ewma" in metrics_set:
        if alpha is None or not (0.0 < alpha <= 1.0):
            raise ValueError("`alpha` must be in (0, 1] when 'ewma' metrics are requested.")

    # ------------------------------------------------------------
    # Shared base vectors
    # ------------------------------------------------------------
    choices = get_choices(nwb_data)
    rewards = get_reward_vectors(nwb_data, include_noresponse=include_noresponse)
    exp = get_experienced_reward(nwb_data, include_noresponse=include_noresponse)

    out: Dict[str, Any] = {
        "n_trials": int(rewards["any"].size),
        "include_noresponse": bool(include_noresponse),
    }

    # ------------------------------------------------------------
    # Global metrics
    # ------------------------------------------------------------
    if "global" in metrics_set:
        out["global_any"] = reward_rate_global(
            rewards["any"], include_noresponse=include_noresponse, choices=choices
        )
        out["global_left_reward"] = reward_rate_global(
            rewards["left"], include_noresponse=include_noresponse, choices=choices
        )
        out["global_right_reward"] = reward_rate_global(
            rewards["right"], include_noresponse=include_noresponse, choices=choices
        )
        out["global_experienced"] = reward_rate_global(
            exp, include_noresponse=include_noresponse, choices=choices
        )

    # ------------------------------------------------------------
    # Running metrics (past-only)
    # ------------------------------------------------------------
    if "running" in metrics_set:
        out["running_any"] = reward_rate_running(
            rewards["any"],
            window=window,
            causal=True,
            include_noresponse=include_noresponse,
            choices=choices,
        )
        out["running_left_reward"] = reward_rate_running(
            rewards["left"],
            window=window,
            causal=True,
            include_noresponse=include_noresponse,
            choices=choices,
        )
        out["running_right_reward"] = reward_rate_running(
            rewards["right"],
            window=window,
            causal=True,
            include_noresponse=include_noresponse,
            choices=choices,
        )
        out["running_experienced"] = reward_rate_running(
            exp,
            window=window,
            causal=True,
            include_noresponse=include_noresponse,
            choices=choices,
        )

    # ------------------------------------------------------------
    # EWMA metrics (past-only)
    # ------------------------------------------------------------
    if "ewma" in metrics_set:
        out["ewma_any"] = reward_rate_ewma(
            rewards["any"],
            alpha=alpha,
            init=0.0,
            include_noresponse=include_noresponse,
            choices=choices,
        )
        out["ewma_left_reward"] = reward_rate_ewma(
            rewards["left"],
            alpha=alpha,
            init=0.0,
            include_noresponse=include_noresponse,
            choices=choices,
        )
        out["ewma_right_reward"] = reward_rate_ewma(
            rewards["right"],
            alpha=alpha,
            init=0.0,
            include_noresponse=include_noresponse,
            choices=choices,
        )
        out["ewma_experienced"] = reward_rate_ewma(
            exp,
            alpha=alpha,
            init=0.0,
            include_noresponse=include_noresponse,
            choices=choices,
        )

    # ------------------------------------------------------------
    # Raw vectors (optional)
    # ------------------------------------------------------------
    if include_raw:
        out["reward_any"] = rewards["any"]
        out["reward_left"] = rewards["left"]
        out["reward_right"] = rewards["right"]
        out["experienced_reward"] = exp
        out["choices"] = choices
        out["responded_mask"] = responded_mask(choices)

    return out




# ------------------------------
# Example usage
# ------------------------------
# alpha = alpha_from_half_life(half_life_trials=20)
# rates = compute_all_reward_rates(nwb_data, window=20, alpha=alpha, include_noresponse=False)
# print("Global any-reward rate:", rates["global_any"])



import math
from typing import List, Dict, Any, Optional


def alpha_from_half_life(H: float) -> float:
    """
    Convert a desired half-life (in trials) into EWMA alpha.

    What “half-life” means
    ----------------------
    Half-life H means: the influence (weight) of an observation decays by 50%
    after H trials.

    In EWMA, the relative influence of an observation k trials ago (compared to the most recent)
    decays geometrically as (1 - alpha)^k. So the half-life condition is:

        (1 - alpha)^H = 1/2

    Solve for alpha:

        1 - alpha = 2^(-1/H)
        alpha = 1 - 2^(-1/H)

    Interpretation
    --------------
    - Smaller alpha -> longer half-life (slower forgetting; smoother estimate).
    - Larger alpha  -> shorter half-life (faster adaptation; noisier estimate).

    Parameters
    ----------
    H : float
        Half-life in trials. Must be > 0.

    Returns
    -------
    alpha : float
        EWMA smoothing parameter in (0, 1).
    """
    if H <= 0:
        raise ValueError("H must be > 0")
    return 1.0 - 2.0 ** (-1.0 / H)


def alpha_from_mean_lag(L: float) -> float:
    """
    Convert a desired mean lag (expected age of evidence, in trials) into EWMA alpha.

    What “mean lag” means
    ---------------------
    Consider EWMA weights over past trials:

        w_k = alpha * (1 - alpha)^k,   k = 0, 1, 2, ...

    where k=0 is the most recent past trial, k=1 is two trials back, etc.

    The expected lag (mean age) of the evidence under this distribution is:

        E[k] = (1 - alpha) / alpha

    Setting E[k] = L and solving:

        (1 - alpha) / alpha = L
        1 - alpha = L * alpha
        1 = (L + 1) * alpha
        alpha = 1 / (L + 1)

    How to use this for “uses last N trials”
    ----------------------------------------
    A hard window of N trials has an average lag roughly at the midpoint:

        L ≈ (N - 1) / 2

    So if you believe the animal’s “center of mass” of evidence is around the middle
    of the last N trials, this mapping is appropriate.

    Interpretation
    --------------
    - This does NOT force “most weight inside last N trials”.
    - Instead, it places the *average age* of evidence near your chosen value.

    Parameters
    ----------
    L : float
        Mean lag in trials. Must be >= 0.

    Returns
    -------
    alpha : float
        EWMA smoothing parameter in (0, 1].
    """
    if L < 0:
        raise ValueError("L must be >= 0")
    return 1.0 / (L + 1.0)


def alpha_from_mass_within_last_N(N: int, p: float) -> float:
    """
    Choose alpha so that a fraction p of total EWMA weight lies within the last N trials.

    What “mass within last N” means
    -------------------------------
    The EWMA weights form a geometric distribution over lags:

        w_k = alpha * (1 - alpha)^k

    The cumulative weight in the most recent N trials is:

        sum_{k=0..N-1} w_k = 1 - (1 - alpha)^N

    We choose alpha so that:

        1 - (1 - alpha)^N = p

    Solve:

        (1 - alpha)^N = 1 - p
        1 - alpha = (1 - p)^(1/N)
        alpha = 1 - (1 - p)^(1/N)

    Interpretation
    --------------
    This is the most literal translation of:
      “the animal mainly uses the last N trials”
    because it lets you state exactly how dominant the last N trials are.

    Examples (N=5)
    --------------
    - p = 0.90 means 90% of the total EWMA weight is within last 5 trials.
    - p = 0.95 means EWMA is very close to a 5-trial hard window.

    Parameters
    ----------
    N : int
        Number of recent trials to capture.
    p : float
        Desired cumulative weight in (0, 1). Typical: 0.80, 0.90, 0.95.

    Returns
    -------
    alpha : float
        EWMA smoothing parameter in (0, 1).
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    return 1.0 - (1.0 - p) ** (1.0 / N)


def effective_window(alpha: float) -> float:
    """
    Rule-of-thumb “effective window length” for EWMA.

    What it means
    -------------
    EWMA uses infinitely many past trials, but the contribution of older trials becomes tiny.
    A useful heuristic is:

        effective_window ≈ 1 / alpha

    Interpretation
    --------------
    - alpha = 0.05 -> ~20-trial effective window
    - alpha = 0.10 -> ~10-trial effective window
    - alpha = 0.33 -> ~3-trial effective window

    This is not an exact identity; it is a widely used approximation.

    Returns
    -------
    float
    """
    if alpha <= 0:
        return float("inf")
    return 1.0 / alpha


def alpha_table_for_trial_memory(
    N: int,
    *,
    mass_levels: List[float] = (0.80, 0.90, 0.95),
    print_table: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build a table of alpha choices when you believe the animal uses ~N recent trials.

    What the table contains
    -----------------------
    Three families of mappings from “uses N trials” to EWMA alpha:

    1) Half-life = N trials
       - Meaning: influence halves after N trials.
       - Use when: you want a characteristic decay timescale around N, not a hard window.

    2) Mean lag = (N - 1) / 2
       - Meaning: the *average age* of evidence is the midpoint of the last N trials.
       - Use when: you think the animal’s evidence is centered around the middle of last N,
         but older trials may still have non-trivial influence.

    3) p% mass in last N trials (for each p in mass_levels)
       - Meaning: exactly p fraction of total EWMA weight lies within last N trials.
       - Use when: you literally mean “mostly last N trials” and want a transparent,
         quantifiable statement.

    Recommended default when you say “uses last N trials”
    -----------------------------------------------------
    If your statement implies older-than-N trials are largely negligible,
    pick p=0.90 or p=0.95 in the mass-within-last-N mapping.

    Parameters
    ----------
    N : int
        “Memory length” in trials (e.g., N=5).
    mass_levels : list of float
        Fractions p for mass-within-last-N mapping.
    print_table : bool
        Whether to print a readable summary.

    Returns
    -------
    table : list of dict
        Each row includes:
          - interpretation (string)
          - meaning (string)
          - alpha (float)
          - effective_window (float)
    """
    if N <= 0:
        raise ValueError("N must be > 0")

    table: List[Dict[str, Any]] = []

    # 1) Half-life mapping
    a_half = alpha_from_half_life(float(N))
    table.append(
        {
            "interpretation": f"Half-life = {N} trials",
            "meaning": (
                f"Influence decays by 50% after {N} trials. "
                "Good when you want a characteristic decay timescale ~N."
            ),
            "alpha": a_half,
            "effective_window": effective_window(a_half),
        }
    )

    # 2) Mean-lag mapping (midpoint of N-trial window)
    L = (N - 1) / 2.0
    a_lag = alpha_from_mean_lag(L)
    table.append(
        {
            "interpretation": f"Mean lag = {L:.1f} trials",
            "meaning": (
                f"Sets the expected age of evidence E[k] to {L:.1f} trials "
                f"(midpoint of a {N}-trial window). Older trials still contribute."
            ),
            "alpha": a_lag,
            "effective_window": effective_window(a_lag),
        }
    )

    # 3) Mass-within-last-N mappings
    for p in mass_levels:
        a_mass = alpha_from_mass_within_last_N(N, float(p))
        table.append(
            {
                "interpretation": f"{int(p * 100)}% mass in last {N} trials",
                "meaning": (
                    f"Ensures {int(p * 100)}% of EWMA total weight lies within the most recent "
                    f"{N} trials. Most literal interpretation of 'uses last {N} trials'."
                ),
                "alpha": a_mass,
                "effective_window": effective_window(a_mass),
            }
        )

    if print_table:
        print(f"\nAlpha choices when the animal uses the previous {N} trials:\n")
        for row in table:
            a = row["alpha"]
            print(
                f"- {row['interpretation']}\n"
                f"  meaning: {row['meaning']}\n"
                f"  alpha = {a:.3f}   "
                f"(effective window ~ {row['effective_window']:.1f} trials)\n"
            )

        # Provide a transparent "recommended" entry (90% mass if present, else max mass)
        if len(mass_levels) > 0:
            p_rec = 0.90 if (0.90 in mass_levels) else max(mass_levels)
            a_rec = alpha_from_mass_within_last_N(N, p_rec)
            print(
                f"Recommended alpha when you mean 'mostly last {N} trials' "
                f"({int(p_rec * 100)}% mass): {a_rec:.3f}\n"
            )

    return table


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
# table = alpha_table_for_trial_memory(N=5, mass_levels=[0.80, 0.90, 0.95], print_table=True)
# alpha_for_model = alpha_from_mass_within_last_N(5, 0.90)






def plot_reward_rates_vs_value_all_models(
    *,
    nwb_data: Any,
    window: Union[int, Sequence[int]] = 8,
    alpha: Union[float, Sequence[float]] = 0.2,
    include_noresponse: bool = False,
    model_aliases: Optional[Sequence[str]] = None,
    drop_last_latent: bool = True,
    point_size: float = 6.0,
    alpha_points: float = 0.5,
    max_points: Optional[int] = None,
    random_state: int = 0,
    figsize_per_panel: Tuple[float, float] = (4.0, 3.2),
    warn_on_length_mismatch: bool = True,
) -> None:
    """
    Scatter plots of selected reward-rate metrics vs model value signals.

    Reward-rate metrics plotted (fixed set)
    --------------------------------------
    - ewma_experienced
    - ewma_left_reward
    - ewma_right_reward
    - running_experienced
    - running_left_reward
    - running_right_reward

    Model-specific y-axis
    ---------------------
    - QLearning* models: sumQ = q_value[0] + q_value[1]
    - ForagingCompareThreshold: fitted_latent_variables['value']

    Window/alpha sweeps
    -------------------
    - window can be an int or a list of ints
    - alpha can be a float or a list of floats
    The function plots all combinations of (window, alpha), one figure per combination.

    Alignment behavior (practical + transparent)
    --------------------------------------------
    - Reward-rate arrays have length = n_trials.
    - Fitted latents (q_value/value) are assumed to have noresponse trials removed upstream,
      so we drop noresponse trials from reward-rate arrays using:
          responded = (animal_response != 2)

    - If lengths still mismatch between reward-rate (after masking) and latent:
        * We DO NOT error.
        * We trim both to min_len (to ensure plotting works).
        * If warn_on_length_mismatch=True, we print a warning that tells you
          exactly which model/metric/combination mismatched and what min_len was used.

    This preserves your “it works” behavior while still surfacing hidden alignment issues.
    """
    # ------------------------------------------------------------
    # Normalize window and alpha to lists
    # ------------------------------------------------------------
    if isinstance(window, (list, tuple, np.ndarray)):
        windows = [int(w) for w in window]
    else:
        windows = [int(window)]

    if isinstance(alpha, (list, tuple, np.ndarray)):
        alphas = [float(a) for a in alpha]
    else:
        alphas = [float(alpha)]

    if any(w <= 0 for w in windows):
        raise ValueError(f"All window values must be positive integers. Got: {windows}")
    if any((a <= 0.0 or a > 1.0) for a in alphas):
        raise ValueError(f"All alpha values must be in (0, 1]. Got: {alphas}")

    # ------------------------------------------------------------
    # Default models
    # ------------------------------------------------------------
    if model_aliases is None:
        model_aliases = [
            "QLearning_L1F1_CK1_softmax",
            "QLearning_L2F1_softmax",
            "QLearning_L2F1_CKfull_softmax",
            "ForagingCompareThreshold",
            "QLearning_L2F1_CK1_softmax",
        ]

    # ------------------------------------------------------------
    # Reward-rate keys to plot (fixed)
    # ------------------------------------------------------------
    rr_keys = [
        "ewma_experienced",
        "ewma_left_reward",
        "ewma_right_reward",
        "running_experienced",
        "running_left_reward",
        "running_right_reward",
    ]

    # ------------------------------------------------------------
    # Drop noresponse trials to align to fitted latents
    # ------------------------------------------------------------
    choices = np.asarray(nwb_data.trials["animal_response"][:], dtype=int)
    responded = choices != 2

    # ------------------------------------------------------------
    # RNG for optional subsampling
    # ------------------------------------------------------------
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------
    # Cache fitted latents per model (avoid refetch per combo)
    # ------------------------------------------------------------
    model_latents: Dict[str, Dict[str, Any]] = {}
    for model_alias in model_aliases:
        try:
            fit = get_fitted_latent(session_name=nwb_data.session_id, model_alias=model_alias)
            latents = fit.get("fitted_latent_variables", {})

            if model_alias == "ForagingCompareThreshold":
                if "value" in latents:
                    y = np.asarray(latents["value"], dtype=float)
                    y_name = "value"
                else:
                    y = None
                    y_name = "value"
            else:
                if "q_value" in latents and len(latents["q_value"]) >= 2:
                    qL = np.asarray(latents["q_value"][0], dtype=float)
                    qR = np.asarray(latents["q_value"][1], dtype=float)
                    y = qL + qR
                    y_name = "sumQ"
                else:
                    y = None
                    y_name = "q_value"

            if y is not None and drop_last_latent and y.size > 0:
                y = y[:-1]

            model_latents[model_alias] = {"y": y, "y_name": y_name}
        except Exception as e:
            model_latents[model_alias] = {"y": None, "y_name": "latent", "error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------
    # Loop over all (window, alpha) combinations
    # ------------------------------------------------------------
    for w in windows:
        for a in alphas:
            reward_rates = compute_all_reward_rates(
                nwb_data=nwb_data,
                window=w,
                alpha=a,
                include_noresponse=include_noresponse,
            )

            # Ensure requested keys exist (at least one)
            present_keys = [k for k in rr_keys if k in reward_rates]
            if len(present_keys) == 0:
                raise ValueError(
                    f"None of the requested reward-rate metrics were found. "
                    f"Requested: {rr_keys}. Available: {sorted(list(reward_rates.keys()))}"
                )

            use_keys = present_keys
            n_rows = len(model_aliases)
            n_cols = len(use_keys)

            fig_w = max(6.0, figsize_per_panel[0] * n_cols)
            fig_h = max(3.5, figsize_per_panel[1] * n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
            fig.suptitle(
                f"Reward-rate vs latent: window={w}, alpha={a}, include_noresponse={include_noresponse}",
                y=1.02,
            )

            for i, model_alias in enumerate(model_aliases):
                y = model_latents[model_alias].get("y", None)
                y_name = model_latents[model_alias].get("y_name", "latent")

                if y is None:
                    for j in range(n_cols):
                        ax = axes[i, j]
                        ax.axis("off")
                        if j == 0:
                            err = model_latents[model_alias].get("error", f"missing {y_name}")
                            ax.text(0.02, 0.5, f"{model_alias}\n{err}", va="center")
                    continue

                for j, rr_key in enumerate(use_keys):
                    ax = axes[i, j]

                    rr_full = np.asarray(reward_rates[rr_key], dtype=float)  # length == n_trials
                    x = rr_full[responded]  # drop noresponse trials to match latent convention

                    # If lengths mismatch, warn and trim to make plotting work (your desired behavior).
                    if x.size != y.size:
                        min_len = min(x.size, y.size)
                        if warn_on_length_mismatch:
                            print(
                                f"[WARN] Length mismatch: model='{model_alias}', metric='{rr_key}', "
                                f"window={w}, alpha={a}: x={x.size}, y={y.size}. "
                                f"Trimming to min_len={min_len}."
                            )
                        x2 = x[:min_len]
                        y2 = y[:min_len]
                    else:
                        x2 = x
                        y2 = y

                    # Remove NaN/inf values (e.g., early running window can be NaN)
                    finite = np.isfinite(x2) & np.isfinite(y2)
                    x2 = x2[finite]
                    y2 = y2[finite]

                    # Optional subsampling for readability
                    if max_points is not None and x2.size > max_points:
                        idx = rng.choice(x2.size, size=max_points, replace=False)
                        x2 = x2[idx]
                        y2 = y2[idx]

                    ax.scatter(x2, y2, s=point_size, alpha=alpha_points)

                    if i == 0:
                        ax.set_title(rr_key, fontsize=10)
                    if j == 0:
                        ax.set_ylabel(f"{model_alias}\n{y_name}", fontsize=10)
                    else:
                        ax.set_ylabel("")

                    if i == n_rows - 1:
                        ax.set_xlabel(rr_key)

            plt.tight_layout()
            plt.show()
