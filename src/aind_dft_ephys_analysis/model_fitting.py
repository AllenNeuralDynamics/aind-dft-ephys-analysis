from __future__ import annotations

import numpy as np
from typing import Any, Optional, Tuple, Dict, List, Union, Sequence
from scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import BinaryResults 
from nwb_utils import NWBUtils


def fit_q_learning_model(
    nwb_behavior_data,
    model_name: str = "q_learning_Y1"
) -> Optional[Dict]:
    """
    Fit an asymmetric-learning-rate Q-learning model (5 free parameters) to a
    two-choice dynamic-foraging task stored in a behavior-only NWB file.

    -------------------------------------------------------------------------
    REQUIRED TRIAL COLUMNS IN THE NWB
    -------------------------------------------------------------------------
      • trials['animal_response']   0 = left, 1 = right, 2 = no response
      • trials['rewarded_historyL'] 0/1  reward delivered on left
      • trials['rewarded_historyR'] 0/1  reward delivered on right

    -------------------------------------------------------------------------
    MODEL SUMMARY
    -------------------------------------------------------------------------
      Q-updates (Left chosen shown; Right analogous):

          δ(t) = R(t) − Q_left(t)
          Q_left(t+1) = Q_left(t) + α_pos⋅δ   if δ ≥ 0
                      = Q_left(t) + α_neg⋅δ   if δ < 0
          Q_right(t+1) = ζ ⋅ Q_right(t)               (decay)

      Soft-max choice rule:

          P(Right) = 1 / (1 + exp(−[β⋅(Q_right−Q_left) + bias]))

      Free parameters = [α_pos, α_neg, ζ, β, bias]

    -------------------------------------------------------------------------
    RETURNS
    -------------------------------------------------------------------------
      On success: dict with
        • 'model_name'                – as given in *model_name*
        • 'fitted_params'             – {α_pos, α_neg, ζ, β, bias}
        • 'neg_log_likelihood'        – scalar NLL at optimum
        • 'fitted_latent_variables'   – {'q_value':[Ql,Qr],
                                          'choice_prob':[P(L),P(R)]}
        • 'ql_values', 'qr_values'    – arrays (len = n_trials+1; leading 0)
        • 'success'                   – True
      On failure: None
    """

    # ──────────────────────────────────────────────────────────────────
    # 1. EXTRACT AND CLEAN TRIAL-LEVEL VECTORS FROM NWB
    #    -------------------------------------------------------------
    #    • Strip “no response” trials (code 2) before fitting.
    # ──────────────────────────────────────────────────────────────────
    animal_response = np.asarray(
        nwb_behavior_data.trials['animal_response'][:]
    )  # 0,1,2
    reward_left  = np.asarray(
        nwb_behavior_data.trials['rewarded_historyL'][:], dtype=float
    )
    reward_right = np.asarray(
        nwb_behavior_data.trials['rewarded_historyR'][:], dtype=float
    )

    valid_mask   = animal_response != 2          # drop no-response trials
    animal_response = animal_response[valid_mask]
    reward_left     = reward_left[valid_mask]
    reward_right    = reward_right[valid_mask]

    n_trials = len(animal_response)
    if n_trials == 0:
        print("No valid trials after removing 'no response'.")
        return None

    # ──────────────────────────────────────────────────────────────────
    # 2. DEFINE NEGATIVE LOG-LIKELIHOOD FUNCTION (inner helper)
    # ──────────────────────────────────────────────────────────────────
    def _neg_log_lik(params: np.ndarray) -> float:
        """
        Compute the total negative log-likelihood of the behavioural sequence
        under the current parameter vector.

        Parameters
        ----------
        params : np.ndarray
            [α_pos, α_neg, ζ, β, bias]

        Returns
        -------
        float
            Accumulated negative log-likelihood across all *valid* trials.
        """
        # Unpack parameters for readability
        alpha_pos, alpha_neg, zeta, beta, bias = params

        # Initialise Q-values
        Ql, Qr = 0.0, 0.0

        nll = 0.0          # running sum of −log P(data)
        eps = 1e-10        # protects against log(0)

        # Loop over trials
        for t in range(n_trials):
            # 2.1 Compute choice probability before observing current choice
            p_right = 1.0 / (1.0 + np.exp(-(beta * (Qr - Ql) + bias)))

            # 2.2 Add log-likelihood contribution of the observed choice
            if animal_response[t] == 1:            # chose Right
                nll -= np.log(p_right + eps)
            else:                                  # chose Left
                nll -= np.log(1.0 - p_right + eps)

            # 2.3 Update Q-values based on outcome
            if animal_response[t] == 0:            # Left chosen
                delta = reward_left[t] - Ql
                Ql   += (alpha_pos if delta >= 0 else alpha_neg) * delta
                Qr   *= zeta                       # decay unchosen value
            else:                                  # Right chosen
                delta = reward_right[t] - Qr
                Qr   += (alpha_pos if delta >= 0 else alpha_neg) * delta
                Ql   *= zeta

        return nll

    # ──────────────────────────────────────────────────────────────────
    # 3. OPTIMISE PARAMETERS BY MINIMISING NLL
    # ──────────────────────────────────────────────────────────────────
    init_params = np.array([0.1, 0.1, 0.9, 5.0, 0.0])          # heuristic
    bounds      = [(0,1), (0,1), (0,1), (1e-3, 100), (-10,10)] # sensible
    result      = minimize(_neg_log_lik, init_params,
                           bounds=bounds, method='L-BFGS-B')

    if not result.success:
        print("Q-learning optimisation failed:", result.message)
        return None

    # Map parameter names → fitted values
    fitted_params = dict(
        zip(['alpha_pos', 'alpha_neg', 'zeta', 'beta', 'bias'], result.x)
    )

    # ──────────────────────────────────────────────────────────────────
    # 4. FORWARD SIMULATION WITH FITTED PARAMETERS → LATENT TIME SERIES
    # ──────────────────────────────────────────────────────────────────
    Ql, Qr = 0.0, 0.0
    ql_vals = np.zeros(n_trials + 1)      # include t=0 initial 0
    qr_vals = np.zeros(n_trials + 1)
    p_right = np.zeros(n_trials)          # P(Right) at each trial start

    for t in range(n_trials):
        # 4.1 Choice probability given current Qs
        p_r = 1.0 / (1.0 + np.exp(
            -(fitted_params['beta'] * (Qr - Ql) + fitted_params['bias'])
        ))
        p_right[t] = p_r

        # 4.2 Update Qs with actual choice & reward
        if animal_response[t] == 0:        # Left chosen
            delta = reward_left[t] - Ql
            Ql   += (fitted_params['alpha_pos']
                     if delta >= 0 else fitted_params['alpha_neg']) * delta
            Qr   *= fitted_params['zeta']
        else:                              # Right chosen
            delta = reward_right[t] - Qr
            Qr   += (fitted_params['alpha_pos']
                     if delta >= 0 else fitted_params['alpha_neg']) * delta
            Ql   *= fitted_params['zeta']

        # 4.3 Store updated Q-values *after* trial t
        ql_vals[t+1], qr_vals[t+1] = Ql, Qr

    # 4.4 Convert P(Right) → P(Left) for completeness
    p_left = 1.0 - p_right

    # ──────────────────────────────────────────────────────────────────
    # 5. PACKAGE AND RETURN RESULTS
    # ──────────────────────────────────────────────────────────────────
    return {
        'model_name'              : model_name,
        'fitted_params'           : fitted_params,
        'neg_log_likelihood'      : result.fun,
        'fitted_latent_variables' : {
            'q_value'    : [ql_vals, qr_vals],     # index 0 = Left, 1 = Right
            'choice_prob': [p_left , p_right]      # index 0 = Left, 1 = Right
        },
        'ql_values'               : ql_vals,
        'qr_values'               : qr_vals,
        'success'                 : True
    }


def fit_choice_logistic_regression_from_nwb(
    nwb_data: Any,
    lag: int = 10,
    session_id: Optional[str] = None,
    model_name: str = "logistic_regression",
    verbose: bool = False, 
) -> Dict[str, Any]:
    """
    Fit a logistic regression model:
        logit(P(cr(t)=1)) = β0
                            + Σ_i β^R_i (Rr(t-i)-Rl(t-i))
                            + Σ_i β^NR_i (Nr(t-i)-Nl(t-i))

    Only the fields REQUIRED for fitting are used:
        - animal_response (0 left, 1 right, 2 no response)
        - rewarded_historyL, rewarded_historyR
        - laser_on_trial (OPTIONALLY saved, not used for fitting)

    The function returns only:
        - result (statsmodels LogitResults)
        - predicted probabilities
        - used_trial_indices
        - lag
    """

    if session_id is None:
        session_id = "session_0"

    trials = nwb_data.trials

    # Required fields
    animal_response = np.array(trials["animal_response"][:])  # 0,1,2
    reward_left = np.array(trials["rewarded_historyL"][:]).astype(float)
    reward_right = np.array(trials["rewarded_historyR"][:]).astype(float)

    # Optional but kept for potential grouping (not used in fitting)
    laser_on_trial = np.array(trials["laser_on_trial"][:]).astype(int)

    n_trials = len(animal_response)
    if n_trials <= lag:
        raise ValueError(f"Not enough trials ({n_trials}) for lag={lag}.")

    # Build Rr, Rl, Nr, Nl
    Rr = np.zeros(n_trials, dtype=int)
    Rl = np.zeros(n_trials, dtype=int)
    Nr = np.zeros(n_trials, dtype=int)
    Nl = np.zeros(n_trials, dtype=int)

    for t in range(n_trials):
        if animal_response[t] == 0:  # left chosen
            Rl[t] = reward_left[t]
            Nl[t] = 1 - reward_left[t]
        elif animal_response[t] == 1:  # right chosen
            Rr[t] = reward_right[t]
            Nr[t] = 1 - reward_right[t]

    X_rows, y_list, used_indices = [], [], []

    for t in range(lag, n_trials):
        if animal_response[t] not in (0, 1):
            continue

        row = []
        for i in range(1, lag+1):
            row.append(Rr[t-i] - Rl[t-i])   # Rewarded diff
        for i in range(1, lag+1):
            row.append(Nr[t-i] - Nl[t-i])   # Unrewarded diff

        X_rows.append(row)
        y_list.append(animal_response[t])
        used_indices.append(t)

    if len(X_rows) == 0:
        raise ValueError("No valid trials for logistic regression.")

    X = np.array(X_rows, float)
    y = np.array(y_list, int)
    used_indices = np.array(used_indices, int)

    # Add intercept and fit
    X_const = sm.add_constant(X)
    model = sm.Logit(y, X_const)
    result = model.fit(disp=False)
    if verbose:
        print(result.summary()) 

    pred_p_right = result.predict(X_const)
    pred_p_left = 1 - pred_p_right

    return {
        "model_name": model_name,
        "fit_result": result,
        "used_trial_indices": used_indices,
        "fitted_latent_variables": {
            "choice_prob": [pred_p_left, pred_p_right]
        },
        "metadata": {
            "lag": lag,
            "column_names": ["const"]
                           + [f"Rdiff_lag{i}" for i in range(1, lag+1)]
                           + [f"Ndiff_lag{i}" for i in range(1, lag+1)]
        }
    }

def visualize_choice_logistic_regression(
    fit_output: Dict[str, Any],
    plot_coefficients: bool = True,
    plot_predictions: bool = True,
    title_font_size: int = 16,
    label_font_size: int = 12,
    figsize_coef: Tuple[int, int] = (13, 4),
    figsize_pred: Tuple[int, int] = (13, 4),
    legend_font_size: Optional[int] = 14,
) -> Dict[str, Any]:
    """
    Visualize logistic regression results with improved external legends
    and goodness-of-fit (GOF) matrix block.

    New parameters:
    --------------
    figsize_coef : tuple
        Size of the coefficient figure.

    figsize_pred : tuple
        Size of the prediction figure.

    legend_font_size : int or None
        Font size for external legends. If None, auto = label_font_size - 4.
    """

    result = fit_output["fit_result"]
    used_indices = fit_output["used_trial_indices"]
    metadata = fit_output["metadata"]
    lag = metadata["lag"]

    # ----------------------------------------
    # Extract bias / intercept coefficient
    # ----------------------------------------
    params = result.params

    # Try to be robust to different param naming
    if hasattr(params, "index"):
        # If there is an explicit 'const' term, use it
        if "const" in params.index:
            bias_val = float(params.loc["const"])
        else:
            # Fall back to the first parameter as bias
            bias_val = float(params.iloc[0])
    else:
        # params is a plain ndarray
        bias_val = float(params[0])

    # ------------------------------
    # Extract goodness-of-fit stats
    # ------------------------------
    gof = {
        "nobs": int(result.nobs),
        "pseudo_r2": float(result.prsquared),
        "llf": float(result.llf),
        "llnull": float(result.llnull),
        "llr_pvalue": float(result.llr_pvalue),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "converged": bool(getattr(result, "converged", True)),
    }

    params = result.params
    conf_int = result.conf_int()
    ci_half = (conf_int[:, 1] - conf_int[:, 0]) / 2.0

    # Extract coefficients
    intercept = params[0]
    intercept_ci = ci_half[0]

    rewarded_coeffs = params[1:lag+1]
    rewarded_ci = ci_half[1:lag+1]

    unrewarded_coeffs = params[lag+1:2*lag+1]
    unrewarded_ci = ci_half[lag+1:2*lag+1]

    x_vals = np.arange(1, lag + 1)

    # Legend font size (user overrides default)
    if legend_font_size is None:
        legend_font_size = max(6, label_font_size - 4)

    gof_font_size = label_font_size

    out: Dict[str, Any] = {
        "coefficients": None,
        "predictions": None,
        "goodness_of_fit": gof,
    }

    # --------------------------------------------------
    # 1. Coefficient plot with external compact legend
    # --------------------------------------------------
    if plot_coefficients:
        fig_coef, ax_coef = plt.subplots(figsize=figsize_coef)

        ax_coef.bar(
            0,
            intercept,
            color="lightblue",
            edgecolor="black",
            width=0.4,
            label="Intercept",
        )
        ax_coef.errorbar(0, intercept, yerr=intercept_ci, fmt="none",
                         ecolor="black", capsize=4)

        ax_coef.errorbar(
            x_vals, rewarded_coeffs, yerr=rewarded_ci,
            fmt="-o", color="blue", label="Rewarded"
        )
        ax_coef.errorbar(
            x_vals, unrewarded_coeffs, yerr=unrewarded_ci,
            fmt="-o", color="red", label="Unrewarded"
        )

        # Axis formatting
        ax_coef.set_xticks(np.arange(0, lag + 1))
        ax_coef.set_xticklabels(["0"] + [str(i) for i in range(1, lag + 1)],
                                fontsize=label_font_size)
        ax_coef.set_xlabel("Lag [0=Intercept]", fontsize=label_font_size)
        ax_coef.set_ylabel("Coefficient Value", fontsize=label_font_size)
        ax_coef.set_title(
            f"Logistic Regression Coefficients (bias = {bias_val:.3f}, 95% CI)",
            fontsize=title_font_size,
        )

        ax_coef.grid(True, alpha=0.3)
        ax_coef.tick_params(axis="both", labelsize=label_font_size)

        # Legend
        handles, labels = ax_coef.get_legend_handles_labels()
        ax_coef.legend(
            handles, labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=legend_font_size,
            frameon=True,
            borderpad=0.2,
            labelspacing=0.2,
            handlelength=1.1,
            handletextpad=0.3,
            markerscale=0.6,
            title="Predictors",
            title_fontsize=legend_font_size,
        )

        # GOF block
        gof_lines = [
            f"N used:    {gof['nobs']}",
            f"Pseudo R²: {gof['pseudo_r2']:.4f}",
            f"LogLik:    {gof['llf']:.2f}",
            f"Null LL:   {gof['llnull']:.2f}",
            f"LLR p:     {gof['llr_pvalue']:.2e}",
            f"AIC:       {gof['aic']:.1f}",
            f"BIC:       {gof['bic']:.1f}",
            f"Conv.:     {gof['converged']}",
        ]
        ax_coef.text(
            1.02, 0.0, "\n".join(gof_lines),
            transform=ax_coef.transAxes,
            ha="left", va="bottom",
            fontsize=gof_font_size,
            family="monospace",
        )

        fig_coef.tight_layout(rect=[0.0, 0.0, 0.75, 1.0])
        out["coefficients"] = (fig_coef, ax_coef)

    # --------------------------------------------------
    # 2. Predicted vs Actual plot
    # --------------------------------------------------
    if plot_predictions:
        pred_p_right = result.predict()
        actual = result.model.endog

        fig_pred, ax_pred = plt.subplots(figsize=figsize_pred)

        ax_pred.plot(
            used_indices, pred_p_right,
            "-o", markersize=4, color="blue", label="Pred P(Right)"
        )
        ax_pred.scatter(
            used_indices, actual,
            s=16, color="red", alpha=0.5, label="Actual (0/1)"
        )

        ax_pred.set_xlabel("Trial Index (Regression Used Trials)",
                           fontsize=label_font_size)
        ax_pred.set_ylabel("Probability / Choice",
                           fontsize=label_font_size)
        ax_pred.set_title("Predicted vs Actual Choices",
                          fontsize=title_font_size)

        ax_pred.grid(True, alpha=0.3)
        ax_pred.tick_params(axis="both", labelsize=label_font_size)

        # Legend
        handles, labels = ax_pred.get_legend_handles_labels()
        ax_pred.legend(
            handles, labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=legend_font_size,
            frameon=True,
            borderpad=0.2,
            labelspacing=0.2,
            handlelength=1.2,
            handletextpad=0.3,
            markerscale=0.7,
        )

        fig_pred.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
        out["predictions"] = (fig_pred, ax_pred)

    return out



import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from scipy.optimize import minimize


def fit_compare_to_threshold_model_different_learning_rate(
    nwb_behavior_data: Any,
    model_name: str = "compare_to_threshold_stay_diff_lr",
    *,
    reset_on_switch: bool = False,
    # Bias terms
    include_stay_bias: bool = False,
    include_side_bias: bool = False,
    # Reset behavior
    # - "prior_fixed": reset v to reset_value_fixed
    # - "prior_fit":   fit reset_value as an extra parameter
    # - "threshold":   reset v to CURRENT threshold (time-varying if adaptive threshold enabled)
    reset_mode: str = "threshold",
    reset_value_fixed: float = 0.4,
    # Different learning rates for value (reward vs no reward)
    fit_separate_learning_rates: bool = False,
    reward_eps: float = 0.0,  # For binary rewards (0/1), keep reward_eps=0.0
    alpha_reward_fixed: float = 0.2,
    alpha_noreward_fixed: float = 0.2,
    # Adaptive threshold (NEW)
    # If adaptive_threshold=True, threshold becomes a latent updated each trial using RW-like rule:
    #   threshold <- threshold + eta * (r(t) - threshold)
    # You can choose to fit separate etas for reward/no-reward, or tie them to value learning rates.
    adaptive_threshold: bool = False,
    fit_separate_threshold_learning_rates: bool = False,
    tie_threshold_lrs_to_value_lrs: bool = False,
    eta_reward_fixed: float = 0.05,
    eta_noreward_fixed: float = 0.05,
    # Initial threshold handling
    # - If adaptive_threshold=True, the fitted "threshold" parameter is interpreted as threshold_0 (initial).
    # - If adaptive_threshold=False, threshold is a static parameter as before.
    save_results: bool = False,
    save_folder: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fit a compare-to-threshold STAY/SWITCH (patch-leaving) model with:

    (A) Asymmetric learning for the latent patch value v
        - alpha_reward:   learning rate used on rewarded trials (r(t)=1)
        - alpha_noreward: learning rate used on non-rewarded trials (r(t)=0)

    (B) Optional adaptive (time-varying) decision threshold
        - threshold(t) is treated as a latent state that can also be updated each trial
            using a Rescorla–Wagner-like rule, with learning rates:
            - eta_reward   (rewarded trials)
            - eta_noreward (non-rewarded trials)

    This model is designed for a two-choice task in which “STAY” means repeating the
    previous choice (same side as t-1) and “SWITCH” means choosing the opposite side.

    ---------------------------------------------------------------------------
    Causal order per trial t (IMPORTANT)
    ---------------------------------------------------------------------------

    The model is strictly causal: decisions on trial t are computed from latent
    states available BEFORE observing the outcome (reward) of trial t.

    We distinguish three “moments” of the latent value:
    - v_pre(t):         value estimate at the start of trial t (carried from t-1)
    - v_after_reset(t): value after applying a reset on switch at trial t (if any)
    - v_post(t):        value after updating with the outcome of trial t

    If adaptive_threshold=True, we similarly track:
    - threshold_pre(t):  threshold at the start of trial t
    - threshold_post(t): threshold after updating with the outcome of trial t

    ---------------------------------------------------------------------------
    1) Decision stage (pre-reset, pre-outcome)
    ---------------------------------------------------------------------------

    At the START of trial t, the animal has:
    - v_pre(t): current patch-value estimate
    - threshold_pre(t): current decision threshold (static if adaptive_threshold=False)

    Compute a decision variable:
        d(t) = v_pre(t) - threshold_pre(t)

    Interpretation:
    - d(t) > 0: current patch is estimated ABOVE threshold → tendency to STAY
    - d(t) < 0: current patch is estimated BELOW threshold → tendency to SWITCH

    Convert d(t) into a probability of STAY using a logistic function:
        P(stay_t) = sigmoid( beta * d(t) + stay_bias )

    Parameters:
    - beta: controls sensitivity to d(t). Larger |beta| → more deterministic choices.
    - stay_bias: value-independent tendency to STAY (stickiness/perseveration).
        * stay_bias > 0 increases baseline staying
        * stay_bias < 0 increases baseline switching

    Important: This is a STAY/SWITCH model. There is no inherent left/right preference
    at this stage; the probability is for repeating the previous action.

    ---------------------------------------------------------------------------
    2) Map STAY probability to RIGHT-choice probability (+ optional side_bias)
    ---------------------------------------------------------------------------

    Because the observed data are left/right choices (0=Left, 1=Right), we map the
    stay probability onto P(Right) conditional on the PREVIOUS choice:

    Let prev_choice = choice(t-1):
    - If prev_choice == Right (1):
            base_P(Right_t) = P(stay_t)
        (staying means choosing Right again)

    - If prev_choice == Left (0):
            base_P(Right_t) = 1 - P(stay_t)
        (staying means choosing Left again, so Right corresponds to switching)

    For the first valid trial (t==0), there is no previous choice, so the model uses:
            base_P(Right_0) = 0.5

    Optional TRUE side bias (persistent Right/Left preference):
    If include_side_bias=True, we apply side_bias in log-odds space:
        logit(P(Right_t)) = logit(base_P(Right_t)) + side_bias

    Interpretation:
    - side_bias > 0 shifts choices toward Right regardless of previous choice
    - side_bias < 0 shifts choices toward Left regardless of previous choice

    This is different from stay_bias:
    - stay_bias affects repeating vs switching
    - side_bias affects Right vs Left preference

    ---------------------------------------------------------------------------
    3) Likelihood evaluation (bookkeeping only)
    ---------------------------------------------------------------------------

    Using P(Right_t), we evaluate the log-likelihood of the observed choice on trial t.
    This affects parameter fitting but does NOT alter latent states; it is purely
    for optimization.

    ---------------------------------------------------------------------------
    4) Reset on SWITCH (state transition before outcome update)
    ---------------------------------------------------------------------------

    If the observed choice at trial t differs from trial t-1, we interpret this as
    entering a NEW patch at trial t (a state transition). If reset_on_switch=True,
    we reset the value BEFORE incorporating the reward outcome of trial t.

    Reset options:
    - reset_mode="prior_fixed":
            v_after_reset(t) <- reset_value_fixed

    - reset_mode="prior_fit":
            v_after_reset(t) <- reset_value (estimated from data)

    - reset_mode="threshold":
            v_after_reset(t) <- threshold_pre(t)

    Important when adaptive_threshold=True:
    - threshold_pre(t) can change over time, so “reset to threshold” means
        reset to the CURRENT threshold on that trial (not a constant).

    If no switch occurs (or reset_on_switch=False):
    - v_after_reset(t) = v_pre(t)

    Interpretation:
    - This reset represents the animal’s prior belief about the new patch at the
        moment it enters it, before seeing any evidence (reward) from that patch.

    ---------------------------------------------------------------------------
    5) Outcome observation and learning update (value and optional threshold)
    ---------------------------------------------------------------------------

    After the choice is made and (if applicable) the reset is applied, the animal
    observes the reward outcome r(t) on the chosen side.

    In your task:
    - r(t) is binary: r(t) ∈ {0, 1}
        * r(t)=1 means rewarded
        * r(t)=0 means not rewarded

    The code uses:
        if r(t) > reward_eps: rewarded else non-rewarded
    For binary rewards, set reward_eps = 0.0 so:
        r(t)=1 → rewarded, r(t)=0 → non-rewarded

    (A) Update value v with asymmetric learning rates:
        If rewarded (r(t)=1):
            v_post(t) = v_after_reset(t) + alpha_reward * ( r(t) - v_after_reset(t) )
        If non-rewarded (r(t)=0):
            v_post(t) = v_after_reset(t) + alpha_noreward * ( r(t) - v_after_reset(t) )

    Interpretation:
    - alpha_reward governs how strongly the animal increases value after success
    - alpha_noreward governs how strongly the animal decreases value after omission

    (B) Optional adaptive threshold update (RW-like):
    If adaptive_threshold=True, threshold is also updated after observing r(t):

        If rewarded:
            threshold_post(t) = threshold_pre(t) + eta_reward   * ( r(t) - threshold_pre(t) )
        If non-rewarded:
            threshold_post(t) = threshold_pre(t) + eta_noreward * ( r(t) - threshold_pre(t) )

    Interpretation:
    - eta_reward / eta_noreward control how quickly the criterion (threshold)
        adapts based on recent outcomes.
    - This can capture behavior where animals modulate their “leaving criterion”
        over time rather than keeping it fixed.

    Finally, trial t+1 begins with:
    - v_pre(t+1) = v_post(t)
    - threshold_pre(t+1) = threshold_post(t)  (if adaptive_threshold=True)
        or threshold_pre(t+1) = threshold_pre(t) (if adaptive_threshold=False)

    ---------------------------------------------------------------------------
    Notes and key distinctions
    ---------------------------------------------------------------------------

    - stay_bias (stickiness) ≠ side_bias (Right/Left preference).
    - Reset happens AFTER the choice at trial t is observed (switch detected),
    but BEFORE learning from r(t), ensuring causality.
    - With adaptive_threshold=True and reset_mode="threshold", resetting v to the
    threshold means resetting to the current (time-varying) threshold on that trial.
    - For your binary reward case, keep reward_eps=0.0 to make the rewarded/no-rewarded
    split exactly match r(t)=1 vs r(t)=0.
    """


    # ------------------------------------------------------------------
    # 0) Basic metadata
    # ------------------------------------------------------------------
    session_id = getattr(nwb_behavior_data, "session_id", None) or "session_unknown"
    try:
        auto_train_stage = nwb_behavior_data.trials[0]["auto_train_stage"][0]
    except Exception:
        auto_train_stage = "unknown"

    # ------------------------------------------------------------------
    # 1) Validate saving options (JSON-only saving)
    # ------------------------------------------------------------------
    if save_results:
        if save_folder is None:
            raise ValueError("save_folder must be provided when save_results=True.")
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2) Extract trial vectors from NWB
    # ------------------------------------------------------------------
    animal_response = np.asarray(nwb_behavior_data.trials["animal_response"][:])  # 0=L, 1=R, 2=NR
    reward_left = np.asarray(nwb_behavior_data.trials["rewarded_historyL"][:], dtype=float)
    reward_right = np.asarray(nwb_behavior_data.trials["rewarded_historyR"][:], dtype=float)

    # ------------------------------------------------------------------
    # 3) Remove no-response trials
    # ------------------------------------------------------------------
    valid_mask = animal_response != 2
    animal_response = animal_response[valid_mask]
    reward_left = reward_left[valid_mask]
    reward_right = reward_right[valid_mask]

    n_trials = int(len(animal_response))
    if n_trials == 0:
        print("No valid trials after removing 'no response'.")
        return None

    # ------------------------------------------------------------------
    # 4) Experienced reward r(t) from chosen side
    # ------------------------------------------------------------------
    r = np.zeros(n_trials, dtype=float)
    chose_left = animal_response == 0
    chose_right = animal_response == 1
    r[chose_left] = reward_left[chose_left]
    r[chose_right] = reward_right[chose_right]

    # ------------------------------------------------------------------
    # 5) Switch indicator between consecutive valid trials
    # ------------------------------------------------------------------
    switched = np.zeros(n_trials, dtype=bool)
    if n_trials >= 2:
        switched[1:] = animal_response[1:] != animal_response[:-1]

    # ------------------------------------------------------------------
    # 6) Helpers
    # ------------------------------------------------------------------
    def _sigmoid(x: float) -> float:
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    def _logit(p: float) -> float:
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        return float(np.log(p / (1.0 - p)))

    def _clean_for_json(obj: Any) -> Any:
        """Recursively convert NaN/Inf to None so json.dump produces valid JSON."""
        if isinstance(obj, dict):
            return {k: _clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean_for_json(v) for v in obj]
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        return obj

    # ------------------------------------------------------------------
    # 7) Validate reset_mode
    # ------------------------------------------------------------------
    if reset_mode not in ("prior_fixed", "prior_fit", "threshold"):
        raise ValueError("reset_mode must be one of {'prior_fixed','prior_fit','threshold'}.")

    reset_value_bounds = (-1.0, 2.0)

    # ------------------------------------------------------------------
    # 8) Parameter unpacking
    # ------------------------------------------------------------------
    def _unpack_params(params: np.ndarray) -> Dict[str, float]:
        idx = 0

        # Value learning rates
        if fit_separate_learning_rates:
            alpha_reward = float(params[idx]); idx += 1
            alpha_noreward = float(params[idx]); idx += 1
        else:
            alpha_reward = float(alpha_reward_fixed)
            alpha_noreward = float(alpha_noreward_fixed)

        # Threshold (static or initial threshold_0)
        threshold0 = float(params[idx]); idx += 1

        # Decision sensitivity
        beta = float(params[idx]); idx += 1

        # Stay bias
        if include_stay_bias:
            stay_bias = float(params[idx]); idx += 1
        else:
            stay_bias = 0.0

        # Side bias
        if include_side_bias:
            side_bias = float(params[idx]); idx += 1
        else:
            side_bias = 0.0

        # Threshold learning rates (etas)
        # If not adaptive_threshold, these are unused.
        if adaptive_threshold:
            if tie_threshold_lrs_to_value_lrs:
                eta_reward = alpha_reward
                eta_noreward = alpha_noreward
            else:
                if fit_separate_threshold_learning_rates:
                    eta_reward = float(params[idx]); idx += 1
                    eta_noreward = float(params[idx]); idx += 1
                else:
                    eta_reward = float(eta_reward_fixed)
                    eta_noreward = float(eta_noreward_fixed)
        else:
            eta_reward = 0.0
            eta_noreward = 0.0

        # Reset value (if prior_fit)
        if reset_mode == "prior_fit":
            reset_value = float(params[idx]); idx += 1
        elif reset_mode == "prior_fixed":
            reset_value = float(reset_value_fixed)
        else:
            reset_value = float("nan")

        return {
            "alpha_reward": alpha_reward,
            "alpha_noreward": alpha_noreward,
            "threshold0": threshold0,
            "beta": beta,
            "stay_bias": stay_bias,
            "side_bias": side_bias,
            "eta_reward": eta_reward,
            "eta_noreward": eta_noreward,
            "reset_value": reset_value,
        }

    # ------------------------------------------------------------------
    # 9) NLL and per-trial log-likelihood (causal order)
    # ------------------------------------------------------------------
    def _run_sequence(params: np.ndarray, *, return_ll_per_trial: bool) -> Dict[str, Any]:
        p = _unpack_params(params)

        alpha_reward = p["alpha_reward"]
        alpha_noreward = p["alpha_noreward"]
        threshold = p["threshold0"]  # current threshold (static or adaptive)
        beta = p["beta"]
        stay_bias = p["stay_bias"]
        side_bias = p["side_bias"]
        eta_reward = p["eta_reward"]
        eta_noreward = p["eta_noreward"]
        reset_value = p["reset_value"]

        # Initialize v
        if reset_mode == "threshold":
            v = threshold  # if adaptive, this uses threshold0 at start
        else:
            v = reset_value  # prior_fixed/prior_fit

        eps = 1e-12
        ll_per_trial = np.zeros(n_trials, dtype=float) if return_ll_per_trial else None
        nll = 0.0

        for t in range(n_trials):
            # 1) Decision from current v and current threshold (pre-reset, pre-outcome)
            d = v - threshold
            p_stay = _sigmoid(beta * d + stay_bias)

            if t == 0:
                base_p_right = 0.5
            else:
                prev_choice = int(animal_response[t - 1])  # 0=L, 1=R
                base_p_right = p_stay if prev_choice == 1 else (1.0 - p_stay)

            if include_side_bias:
                p_right = _sigmoid(_logit(base_p_right) + side_bias)
            else:
                p_right = base_p_right

            # 2) Score observed choice at t
            if animal_response[t] == 1:
                p_obs = p_right
            else:
                p_obs = 1.0 - p_right

            ll_t = float(np.log(p_obs + eps))
            nll -= ll_t
            if ll_per_trial is not None:
                ll_per_trial[t] = ll_t

            # 3) Reset value on switch before outcome update
            if reset_on_switch and switched[t]:
                if reset_mode == "threshold":
                    v = threshold  # NOTE: current threshold (adaptive if enabled)
                else:
                    v = reset_value

            # 4) Outcome update with different learning rates (binary rewards: reward_eps=0.0)
            lr_v = alpha_reward if (r[t] > reward_eps) else alpha_noreward
            v = v + lr_v * (r[t] - v)

            # 5) Optional adaptive threshold update (RW-like)
            if adaptive_threshold:
                lr_th = eta_reward if (r[t] > reward_eps) else eta_noreward
                threshold = threshold + lr_th * (r[t] - threshold)

        out: Dict[str, Any] = {"nll": float(nll)}
        if ll_per_trial is not None:
            out["ll_per_trial"] = ll_per_trial
        return out

    def _neg_log_lik(params: np.ndarray) -> float:
        return float(_run_sequence(params, return_ll_per_trial=False)["nll"])

    # ------------------------------------------------------------------
    # 10) Optimization setup
    # ------------------------------------------------------------------
    init_params_list = []
    bounds_list = []
    param_names = []

    # Value learning rates
    if fit_separate_learning_rates:
        init_params_list.append(float(alpha_reward_fixed))
        bounds_list.append((0.0, 1.0))
        param_names.append("alpha_reward")

        init_params_list.append(float(alpha_noreward_fixed))
        bounds_list.append((0.0, 1.0))
        param_names.append("alpha_noreward")

    # threshold (static or initial threshold0)
    init_params_list.append(0.5)
    bounds_list.append((-1.0, 1.0))
    param_names.append("threshold0")

    # beta
    init_params_list.append(5.0)
    bounds_list.append((-200.0, 200.0))
    param_names.append("beta")

    # stay_bias
    if include_stay_bias:
        init_params_list.append(0.0)
        bounds_list.append((-5.0, 5.0))
        param_names.append("stay_bias")

    # side_bias
    if include_side_bias:
        init_params_list.append(0.0)
        bounds_list.append((-5.0, 5.0))
        param_names.append("side_bias")

    # Threshold learning rates (etas)
    if adaptive_threshold and (not tie_threshold_lrs_to_value_lrs) and fit_separate_threshold_learning_rates:
        init_params_list.append(float(eta_reward_fixed))
        bounds_list.append((0.0, 1.0))
        param_names.append("eta_reward")

        init_params_list.append(float(eta_noreward_fixed))
        bounds_list.append((0.0, 1.0))
        param_names.append("eta_noreward")

    # reset_value (only if prior_fit)
    if reset_mode == "prior_fit":
        init_params_list.append(float(reset_value_fixed))
        bounds_list.append(reset_value_bounds)
        param_names.append("reset_value")

    init_params = np.asarray(init_params_list, dtype=float)
    bounds = list(bounds_list)

    # ------------------------------------------------------------------
    # 11) Optimize
    # ------------------------------------------------------------------
    result = minimize(_neg_log_lik, init_params, bounds=bounds, method="L-BFGS-B")
    if not result.success:
        print("Compare-to-threshold (diff LR + adaptive threshold) optimisation failed:", result.message)
        return None

    p_fit = _unpack_params(np.asarray(result.x, dtype=float))

    fitted_params: Dict[str, float] = {
        "alpha_reward": float(p_fit["alpha_reward"]),
        "alpha_noreward": float(p_fit["alpha_noreward"]),
        "threshold0": float(p_fit["threshold0"]),
        "beta": float(p_fit["beta"]),
        "stay_bias": float(p_fit["stay_bias"]) if include_stay_bias else 0.0,
        "side_bias": float(p_fit["side_bias"]) if include_side_bias else 0.0,
        "adaptive_threshold": bool(adaptive_threshold),
        "tie_threshold_lrs_to_value_lrs": bool(tie_threshold_lrs_to_value_lrs),
        "eta_reward": float(p_fit["eta_reward"]) if adaptive_threshold else 0.0,
        "eta_noreward": float(p_fit["eta_noreward"]) if adaptive_threshold else 0.0,
    }

    if reset_mode == "prior_fit":
        fitted_params["reset_value"] = float(p_fit["reset_value"])
    elif reset_mode == "prior_fixed":
        fitted_params["reset_value"] = float(reset_value_fixed)
    else:
        fitted_params["reset_value"] = float("nan")  # will be converted to null in JSON output

    # ------------------------------------------------------------------
    # 12) LL / AIC / BIC + per-trial log-likelihood
    # ------------------------------------------------------------------
    seq_out = _run_sequence(np.asarray(result.x, dtype=float), return_ll_per_trial=True)
    neg_log_likelihood = float(seq_out["nll"])
    ll_per_trial = np.asarray(seq_out["ll_per_trial"], dtype=float)

    log_likelihood = -neg_log_likelihood
    log_likelihood_per_trial = (float(log_likelihood) / float(n_trials)) if n_trials > 0 else float("nan")

    k = int(len(param_names))
    n = int(n_trials)
    aic = 2.0 * float(k) - 2.0 * float(log_likelihood)
    bic = float(k) * float(np.log(n)) - 2.0 * float(log_likelihood)

    # ------------------------------------------------------------------
    # 13) Forward simulation with latents (aligned; includes threshold trajectory if adaptive)
    # ------------------------------------------------------------------
    alpha_reward = fitted_params["alpha_reward"]
    alpha_noreward = fitted_params["alpha_noreward"]
    threshold = fitted_params["threshold0"]
    beta = fitted_params["beta"]
    stay_bias = fitted_params["stay_bias"]
    side_bias = fitted_params["side_bias"]
    eta_reward = fitted_params["eta_reward"]
    eta_noreward = fitted_params["eta_noreward"]

    if reset_mode == "threshold":
        reset_value = float("nan")
        v = threshold
    else:
        reset_value = fitted_params["reset_value"]
        v = reset_value

    v_pre = np.zeros(n_trials, dtype=float)
    v_after_reset = np.zeros(n_trials, dtype=float)
    v_post = np.zeros(n_trials, dtype=float)

    th_pre = np.zeros(n_trials, dtype=float)
    th_post = np.zeros(n_trials, dtype=float)

    d_vals = np.zeros(n_trials, dtype=float)
    p_stay_vals = np.zeros(n_trials, dtype=float)
    base_p_right_vals = np.zeros(n_trials, dtype=float)
    p_right_vals = np.zeros(n_trials, dtype=float)
    lr_value_used = np.zeros(n_trials, dtype=float)
    lr_thresh_used = np.zeros(n_trials, dtype=float)

    for t in range(n_trials):
        th_pre[t] = threshold
        v_pre[t] = v

        # Decision
        d = v - threshold
        d_vals[t] = d

        p_stay = _sigmoid(beta * d + stay_bias)
        p_stay_vals[t] = p_stay

        if t == 0:
            base_p_right = 0.5
        else:
            prev_choice = int(animal_response[t - 1])
            base_p_right = p_stay if prev_choice == 1 else (1.0 - p_stay)
        base_p_right_vals[t] = base_p_right

        if include_side_bias:
            p_right = _sigmoid(_logit(base_p_right) + side_bias)
        else:
            p_right = base_p_right
        p_right_vals[t] = p_right

        # Reset on switch
        v_tmp = v
        if reset_on_switch and switched[t]:
            if reset_mode == "threshold":
                v_tmp = threshold  # current threshold (adaptive if enabled)
            else:
                v_tmp = reset_value
        v_after_reset[t] = v_tmp

        # Value update
        lr_v = alpha_reward if (r[t] > reward_eps) else alpha_noreward
        lr_value_used[t] = lr_v
        v = v_tmp + lr_v * (r[t] - v_tmp)
        v_post[t] = v

        # Threshold update
        if adaptive_threshold:
            lr_th = eta_reward if (r[t] > reward_eps) else eta_noreward
        else:
            lr_th = 0.0
        lr_thresh_used[t] = lr_th
        if adaptive_threshold:
            threshold = threshold + lr_th * (r[t] - threshold)
        th_post[t] = threshold

    p_left_vals = 1.0 - p_right_vals

    fitted_latent_variables = {
        "value_pre": v_pre.tolist(),
        "value_after_reset": v_after_reset.tolist(),
        "value_post": v_post.tolist(),
        "threshold_pre": th_pre.tolist(),
        "threshold_post": th_post.tolist(),
        "decision_variable": d_vals.tolist(),
        "p_stay": p_stay_vals.tolist(),
        "base_p_right": base_p_right_vals.tolist(),
        "choice_prob": [p_left_vals.tolist(), p_right_vals.tolist()],
        "learning_rate_value_used": lr_value_used.tolist(),
        "learning_rate_threshold_used": lr_thresh_used.tolist(),
    }

    # ------------------------------------------------------------------
    # 14) Output
    # ------------------------------------------------------------------
    output: Dict[str, Any] = {
        "model_name": model_name,
        "session_id": session_id,
        "auto_train_stage": auto_train_stage,
        "fitted_params": fitted_params,
        "neg_log_likelihood": neg_log_likelihood,
        "log_likelihood": float(log_likelihood),
        "log_likelihood_per_trial": float(log_likelihood_per_trial),
        "log_likelihood_by_trial": ll_per_trial.tolist(),
        "aic": float(aic),
        "bic": float(bic),
        "n_trials_used": int(n_trials),
        "n_parameters": int(k),
        "fitted_latent_variables": fitted_latent_variables,
        "success": True,
        "metadata": {
            "reset_on_switch": bool(reset_on_switch),
            "include_stay_bias": bool(include_stay_bias),
            "include_side_bias": bool(include_side_bias),
            "reset_mode": reset_mode,
            "reset_value_fixed": float(reset_value_fixed),
            "fit_separate_learning_rates": bool(fit_separate_learning_rates),
            "reward_eps": float(reward_eps),
            "adaptive_threshold": bool(adaptive_threshold),
            "fit_separate_threshold_learning_rates": bool(fit_separate_threshold_learning_rates),
            "tie_threshold_lrs_to_value_lrs": bool(tie_threshold_lrs_to_value_lrs),
            "eta_reward_fixed": float(eta_reward_fixed),
            "eta_noreward_fixed": float(eta_noreward_fixed),
            "note": (
                "Causal order per trial: decision from current v and current threshold -> score choice -> "
                "if switched, reset v (to prior or current threshold) -> update v with alpha_reward/alpha_noreward -> "
                "optional update of threshold with eta_reward/eta_noreward using RW-like rule."
            ),
        },
    }

    # ------------------------------------------------------------------
    # 15) Optional saving (JSON only, strict JSON compliance)
    # ------------------------------------------------------------------
    if save_results:
        safe_session = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(session_id))
        prefix = f"{safe_session}__{model_name}"
        json_path = Path(save_folder) / f"{prefix}_fit_results.json"
        with open(json_path, "w") as f:
            json.dump(_clean_for_json(output), f, indent=2, allow_nan=False)

    return output








def visualize_compare_to_threshold_model(
    fit_output: Dict[str, Any],
    *,
    title_font_size: int = 16,
    label_font_size: int = 12,
    legend_font_size: Optional[int] = 12,
    figsize_value: Tuple[int, int] = (13, 4),
    figsize_pred: Tuple[int, int] = (13, 4),
) -> Dict[str, Any]:
    """
    Visualize the compare-to-threshold model output.

    Produces two figures:
      1) Value trajectory v(t) with threshold line
      2) Predicted P(Right) over trials with actual choices overlay

    Expected fields in fit_output:
        - fitted_params: contains threshold
        - fitted_latent_variables: value (len n+1), choice_prob (len n)
        - metadata: may contain options
    """

    fitted_params = fit_output["fitted_params"]
    threshold = float(fitted_params["threshold"])

    latents = fit_output["fitted_latent_variables"]
    v_vals = np.asarray(latents["value"], dtype=float)                 # len n+1
    p_left, p_right = latents["choice_prob"]
    p_right = np.asarray(p_right, dtype=float)                         # len n
    n_trials = int(len(p_right))

    if legend_font_size is None:
        legend_font_size = max(6, label_font_size - 2)

    out: Dict[str, Any] = {"value": None, "predictions": None}

    # --------------------------------------------------
    # 1) Value vs threshold
    # --------------------------------------------------
    fig_v, ax_v = plt.subplots(figsize=figsize_value)
    ax_v.plot(np.arange(len(v_vals)), v_vals, "-o", markersize=3, label="Value v(t)")
    ax_v.axhline(threshold, linestyle="--", label=f"Threshold = {threshold:.3f}")

    ax_v.set_xlabel("t (value index; includes t=0)", fontsize=label_font_size)
    ax_v.set_ylabel("Value", fontsize=label_font_size)
    ax_v.set_title("Compare-to-Threshold: Value Trajectory", fontsize=title_font_size)
    ax_v.grid(True, alpha=0.3)
    ax_v.tick_params(axis="both", labelsize=label_font_size)
    ax_v.legend(loc="upper left", fontsize=legend_font_size, frameon=True)
    fig_v.tight_layout()
    out["value"] = (fig_v, ax_v)

    # --------------------------------------------------
    # 2) Predicted P(Right) and actual choice (if provided)
    # --------------------------------------------------
    fig_p, ax_p = plt.subplots(figsize=figsize_pred)
    ax_p.plot(np.arange(n_trials), p_right, "-o", markersize=3, label="Pred P(Right)")

    # If caller also saved actual choices in metadata (optional), overlay them.
    # This keeps the visualizer flexible without requiring NWB access here.
    actual = fit_output.get("actual_choice_valid", None)
    if actual is not None:
        actual = np.asarray(actual, dtype=float)
        if len(actual) == n_trials:
            ax_p.scatter(np.arange(n_trials), actual, s=16, alpha=0.5, label="Actual (0/1)")

    ax_p.set_xlabel("Valid trial index", fontsize=label_font_size)
    ax_p.set_ylabel("Probability / Choice", fontsize=label_font_size)
    ax_p.set_title("Compare-to-Threshold: Predicted vs Actual", fontsize=title_font_size)
    ax_p.grid(True, alpha=0.3)
    ax_p.tick_params(axis="both", labelsize=label_font_size)
    ax_p.legend(loc="upper left", fontsize=legend_font_size, frameon=True)
    fig_p.tight_layout()
    out["predictions"] = (fig_p, ax_p)

    return out
