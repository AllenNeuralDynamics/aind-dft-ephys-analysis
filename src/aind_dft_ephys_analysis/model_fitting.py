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


def fit_compare_to_threshold_model(
    nwb_behavior_data: Any,
    model_name: str = "compare_to_threshold_stay",
    *,
    reset_on_switch: bool = True,
    include_bias: bool = True,
    save_results: bool = False,
    save_folder: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fit a "compare-to-threshold" STAY/SWITCH (patch-leaving) model to a two-choice task stored in NWB.

    Model summary (stay/switch form)
    --------------------------------
    1) Experienced reward r(t) is binary (0/1) and comes from the chosen side.

    2) Single latent value update (EWMA / delta rule):
        v(t+1) = v(t) + alpha * (r(t) - v(t))

    3) Compare to threshold:
        d(t) = v(t) - threshold

    4) STAY probability:
        P(stay at t) = sigmoid( beta * d(t) + bias )

       Choice mapping (for t >= 1):
         - If previous choice was Right:
             P(Right) = P(stay)
         - If previous choice was Left:
             P(Right) = 1 - P(stay)

       For t == 0 (no previous choice):
         - P(Right at t=0) = sigmoid(bias) if include_bias else 0.5

    Free parameters:
      - include_bias=True  => alpha, threshold, beta, bias
      - include_bias=False => alpha, threshold, beta (bias fixed to 0)

    Reset option
    ------------
    reset_on_switch:
        If True, whenever choice(t) != choice(t-1) between consecutive valid trials,
        reset v(t) to threshold at the start of trial t (before computing P(stay)).
    """

    # ------------------------------------------------------------------
    # 0) Extract session_id and auto_train_stage (best-effort)
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
    # 3) Remove no-response trials (animal_response == 2)
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
    # 4) Build experienced reward r(t) from chosen side
    # ------------------------------------------------------------------
    r = np.zeros(n_trials, dtype=float)
    chose_left = animal_response == 0
    chose_right = animal_response == 1
    r[chose_left] = reward_left[chose_left]
    r[chose_right] = reward_right[chose_right]

    # ------------------------------------------------------------------
    # 5) Detect switching between consecutive valid trials
    # ------------------------------------------------------------------
    switched = np.zeros(n_trials, dtype=bool)
    if n_trials >= 2:
        switched[1:] = animal_response[1:] != animal_response[:-1]

    # ------------------------------------------------------------------
    # 6) Numerically stable sigmoid helper
    # ------------------------------------------------------------------
    def _sigmoid(x: float) -> float:
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # ------------------------------------------------------------------
    # 7) Negative log-likelihood (NLL) for optimization
    # ------------------------------------------------------------------
    def _neg_log_lik(params: np.ndarray) -> float:
        if include_bias:
            alpha, threshold, beta, bias = params
        else:
            alpha, threshold, beta = params
            bias = 0.0

        v = 0.0
        nll = 0.0
        eps = 1e-12

        for t in range(n_trials):
            # Reset at the start of a switched trial (generative stay/switch interpretation)
            if reset_on_switch and switched[t]:
                v = threshold

            d = v - threshold
            p_stay = _sigmoid(beta * d + bias)

            # Map p_stay -> P(Right) using previous choice (t >= 1 only)
            if t == 0:
                # No previous choice: initial side bias only
                p_right = _sigmoid(bias) if include_bias else 0.5
            else:
                prev_choice = int(animal_response[t - 1])  # 0=L, 1=R
                if prev_choice == 1:
                    p_right = p_stay
                else:
                    p_right = 1.0 - p_stay

            # Likelihood of observed choice
            if animal_response[t] == 1:
                nll -= np.log(p_right + eps)
            else:
                nll -= np.log(1.0 - p_right + eps)

            # Update after reward
            v = v + alpha * (r[t] - v)

        return float(nll)

    # ------------------------------------------------------------------
    # 8) Optimization configuration
    # ------------------------------------------------------------------
    if include_bias:
        init_params = np.array([0.2, 0.5, 5.0, 0.0], dtype=float)
        bounds = [(0.0, 1.0), (-1.0, 1.0), (1e-3, 50.0), (-5.0, 5.0)]
        param_names = ["alpha", "threshold", "beta", "bias"]
    else:
        init_params = np.array([0.2, 0.5, 5.0], dtype=float)
        bounds = [(0.0, 1.0), (-1.0, 1.0), (1e-3, 200.0)]
        param_names = ["alpha", "threshold", "beta"]

    # ------------------------------------------------------------------
    # 9) Run optimizer (L-BFGS-B)
    # ------------------------------------------------------------------
    result = minimize(_neg_log_lik, init_params, bounds=bounds, method="L-BFGS-B")
    if not result.success:
        print("Compare-to-threshold (stay) optimisation failed:", result.message)
        return None

    fitted_params = dict(zip(param_names, map(float, result.x)))
    if not include_bias:
        fitted_params["bias"] = 0.0

    # ------------------------------------------------------------------
    # 10) Compute LL / AIC / BIC
    # ------------------------------------------------------------------
    neg_log_likelihood = float(result.fun)
    log_likelihood = -neg_log_likelihood

    k = int(len(param_names))
    n = int(n_trials)

    aic = 2.0 * float(k) - 2.0 * float(log_likelihood)
    bic = float(k) * float(np.log(n)) - 2.0 * float(log_likelihood)

    # ------------------------------------------------------------------
    # 11) Forward simulation to generate latents
    # ------------------------------------------------------------------
    alpha = float(fitted_params["alpha"])
    threshold = float(fitted_params["threshold"])
    beta = float(fitted_params["beta"])
    bias = float(fitted_params["bias"])

    v_vals = np.zeros(n_trials + 1, dtype=float)  # includes v(0)
    d_vals = np.zeros(n_trials, dtype=float)
    p_stay_vals = np.zeros(n_trials, dtype=float)
    p_right_vals = np.zeros(n_trials, dtype=float)

    v = 0.0
    for t in range(n_trials):
        if reset_on_switch and switched[t]:
            v = threshold

        d = v - threshold
        d_vals[t] = d
        p_stay = _sigmoid(beta * d + bias)
        p_stay_vals[t] = p_stay

        if t == 0:
            p_right = _sigmoid(bias) if include_bias else 0.5
        else:
            prev_choice = int(animal_response[t - 1])
            p_right = p_stay if prev_choice == 1 else (1.0 - p_stay)

        p_right_vals[t] = p_right

        v = v + alpha * (r[t] - v)
        v_vals[t + 1] = v

    p_left_vals = 1.0 - p_right_vals

    # ------------------------------------------------------------------
    # 12) Assemble results (JSON-friendly)
    # ------------------------------------------------------------------
    fitted_latent_variables = {
        "value": v_vals.tolist(),
        "decision_variable": d_vals.tolist(),
        "p_stay": p_stay_vals.tolist(),
        "choice_prob": [p_left_vals.tolist(), p_right_vals.tolist()],
    }

    output: Dict[str, Any] = {
        "model_name": model_name,
        "session_id": session_id,
        "auto_train_stage": auto_train_stage,
        "fitted_params": fitted_params,
        "neg_log_likelihood": neg_log_likelihood,
        "log_likelihood": float(log_likelihood),
        "aic": float(aic),
        "bic": float(bic),
        "n_trials_used": n,
        "n_parameters": k,
        "fitted_latent_variables": fitted_latent_variables,
        "success": True,
        "metadata": {
            "reset_on_switch": bool(reset_on_switch),
            "include_bias": bool(include_bias),
            "note": (
                "Stay/switch model: P(stay)=sigmoid(beta*(v-threshold)+bias). "
                "For t>=1, P(Right)=P(stay) if previous choice was Right else 1-P(stay). "
                "For t=0, P(Right)=sigmoid(bias) (or 0.5 if no bias). "
                "Reward r(t) is experienced reward on the chosen side; no-response trials removed. "
                "If reset_on_switch is True, value is reset to threshold at the start of switched trials."
            ),
        },
    }

    # ------------------------------------------------------------------
    # 13) Optional saving (JSON only)
    # ------------------------------------------------------------------
    if save_results:
        safe_session = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(session_id))
        prefix = f"{safe_session}__{model_name}"
        json_path = Path(save_folder) / f"{prefix}_fit_results.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)

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
