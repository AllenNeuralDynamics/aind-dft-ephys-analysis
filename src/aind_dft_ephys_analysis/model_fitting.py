import numpy as np
from typing import Any, Optional, Tuple, Dict, List
from scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import BinaryResults 



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



def fit_choice_logistic_regression(
    nwb_data: Any,
    lag: int = 10,
    pad_mode: str = "zero",
) -> Optional[Tuple[BinaryResults, Dict[str, Any]]]:
    """
    Fit a two‑kernel logistic regression to an animal's left/right choices in
    an NWB file, **excluding** all no‑response trials (``animal_response == 2``)
    and **retaining the original number of choice trials**.

    *History padding*: for the first ``lag`` trials where a full history window
    is not available, we pad missing values with ``0`` (neutral influence).
    Pass ``pad_mode=None`` to revert to the previous behaviour (skip those
    trials and return a design matrix of length ``n_valid - lag``).

    Mathematical model
    ------------------
    ::

        logit(P(c_r(t)=1)) = β0
                             + sum_{i=1}^{L} β_i^R   * (R_r(t−i) − R_l(t−i))
                             + sum_{i=1}^{L} β_i^{NR} * (N_r(t−i) − N_l(t−i))

    *R_r/L* and *N_r/L* are indicators for rewarded or unrewarded right/left
    choices; *L* = ``lag``.

    Parameters
    ----------
    nwb_data : Any
        Open NWB handle whose ``trials`` table includes the columns
        ``animal_response``, ``rewarded_historyL``, and ``rewarded_historyR``.
    lag : int, default 10
        Number of past trials to include in each kernel.
    pad_mode : {"zero", None}, default "zero"
        * ``"zero"`` – pad missing history with 0 so every choice trial
          contributes one row (length == ``n_valid``).
        * ``None`` – drop the first ``lag`` trials (length == ``n_valid-lag``).

    Returns
    -------
    (BinaryResults, dict) | None
        ``BinaryResults`` – fitted result from :pyclass:`statsmodels.Logit`.
        ``dict`` – serialisable summary; *None* if optimisation fails or the
        session lacks sufficient analysable trials.
    """

    # 1. Extract trial‑level arrays ----------------------------------------
    trials = nwb_data.trials
    resp = np.asarray(trials["animal_response"], dtype=int)
    rL = np.asarray(trials["rewarded_historyL"], dtype=int)
    rR = np.asarray(trials["rewarded_historyR"], dtype=int)

    # Keep only genuine choices (0 = left, 1 = right)
    valid_mask = (resp == 0) | (resp == 1)
    valid_idx = np.where(valid_mask)[0]

    if valid_idx.size <= lag and pad_mode is None:
        print("[log-reg] ✖ not enough choice trials after removing no‑responses")
        return None

    # Compact arrays (no no‑response trials remain)
    resp_c = resp[valid_mask]
    rL_c = rL[valid_mask]
    rR_c = rR[valid_mask]

    # 2. Pre‑compute Boolean kernels ---------------------------------------
    Rl = (resp_c == 0) & (rL_c == 1)  # rewarded LEFT
    Rr = (resp_c == 1) & (rR_c == 1)  # rewarded RIGHT
    Nl = (resp_c == 0) & (rL_c == 0)  # unrewarded LEFT
    Nr = (resp_c == 1) & (rR_c == 0)  # unrewarded RIGHT

    # 3. Build design matrix X and target y -------------------------------
    X_rows: List[List[int]] = []
    y_list: List[int] = []
    used_orig_idx: List[int] = []

    n_valid = resp_c.size
    start_t = 0 if pad_mode == "zero" else lag

    for t in range(start_t, n_valid):
        if pad_mode is None and t < lag:
            # should never happen because we start at lag
            continue

        row: List[int] = []
        # rewarded kernel
        for i in range(1, lag + 1):
            idx = t - i
            if idx < 0:
                row.append(0)  # padding
            else:
                row.append(int(Rr[idx]) - int(Rl[idx]))
        # unrewarded kernel
        for i in range(1, lag + 1):
            idx = t - i
            if idx < 0:
                row.append(0)
            else:
                row.append(int(Nr[idx]) - int(Nl[idx]))

        X_rows.append(row)
        y_list.append(int(resp_c[t]))
        used_orig_idx.append(valid_idx[t])

    if not X_rows:
        print("[log-reg] ✖ no analysable windows after filtering")
        return None

    X = sm.add_constant(np.asarray(X_rows, dtype=float))
    y = np.asarray(y_list, dtype=int)

    # 4. Fit the logistic model -------------------------------------------
    try:
        result = sm.Logit(y, X).fit(disp=False)
    except Exception as err:
        print(f"[log-reg] ✖ fit failed: {err}")
        return None

    # 5. Package results ---------------------------------------------------
    p_right = result.predict()
    p_left = 1.0 - p_right

    fit_dict: Dict[str, Any] = {
        "fit_result": result,
        "model_summary": result.summary().as_text(),
        "fitted_latent_variables": {"choice_prob": [p_left, p_right]},
        "used_trial_indices": np.asarray(used_orig_idx, dtype=int),
    }

    return result, fit_dict


def visualize_choice_logistic_regression(
    result: BinaryResults,
    *,
    lag: Optional[int] = None,
    used_trial_indices: Optional[np.ndarray] = None,
    title_font_size: int = 16,
    label_font_size: int = 12,
) -> None:
    """Plot kernel coefficients and fit quality for a fitted logistic model.

    Parameters
    ----------
    result : BinaryResults
        Object returned by :pyclass:`statsmodels.Logit.fit`.
    lag : int, optional
        History length used when fitting. If *None*, it is inferred from the
        parameter vector length, which must follow the pattern ``1 + 2·lag``.
    used_trial_indices : np.ndarray, optional
        Trial indices corresponding to rows in the design matrix – used for
        the x‑axis on the probability/choice panel.
    title_font_size / label_font_size : int, optional
        Font sizes for figure elements.
    """

    # ─── Infer lag automatically if not provided ──────────────────────────
    if lag is None:
        n_coeff = len(result.params)
        lag = (n_coeff - 1) // 2
        if 1 + 2 * lag != n_coeff or lag <= 0:
            raise ValueError(
                "Could not infer lag – parameter vector length does not match 1 + 2·lag pattern"
            )

    # ─── Extract coefficients and CI half‑widths ──────────────────────────
    params = result.params
    conf_int = result.conf_int(alpha=0.05)
    ci_half = (conf_int[:, 1] - conf_int[:, 0]) / 2.0

    intercept = params[0]
    intercept_ci = ci_half[0]

    rewarded = params[1 : lag + 1]
    rewarded_ci = ci_half[1 : lag + 1]

    unrewarded = params[lag + 1 : 2 * lag + 1]
    unrewarded_ci = ci_half[lag + 1 : 2 * lag + 1]

    x_lags = np.arange(1, lag + 1)

    # ─── Coefficient plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(0, intercept, width=0.4, color="lightblue", edgecolor="black", label="Intercept")
    ax.errorbar(0, intercept, yerr=intercept_ci, fmt="none", ecolor="black", capsize=4)

    ax.errorbar(
        x_lags,
        rewarded,
        yerr=rewarded_ci,
        fmt="-o",
        color="blue",
        ecolor="blue",
        capsize=4,
        label="Rewarded",
    )

    ax.errorbar(
        x_lags,
        unrewarded,
        yerr=unrewarded_ci,
        fmt="-o",
        color="red",
        ecolor="red",
        capsize=4,
        label="Unrewarded",
    )

    ax.set_xticks(np.arange(0, lag + 1))
    ax.set_xticklabels(["0"] + [str(i) for i in x_lags])
    ax.set_xlabel("Lag (0 = intercept)", fontsize=label_font_size)
    ax.set_ylabel("Coefficient value", fontsize=label_font_size)
    ax.set_title("Logistic regression coefficients (95% CI)", fontsize=title_font_size)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ─── Probability vs. choice plot ──────────────────────────────────────
    pred_p_right = result.predict()
    actual_choice = result.model.endog

    if used_trial_indices is None or len(used_trial_indices) != len(pred_p_right):
        x_axis = np.arange(len(pred_p_right))
    else:
        x_axis = used_trial_indices

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_axis, pred_p_right, "-o", color="blue", label="Predicted P(Right)", alpha=0.7)
    ax.scatter(x_axis, actual_choice, color="red", label="Actual choice", alpha=0.5)
    ax.set_xlabel("Trial index (filtered)")
    ax.set_ylabel("Probability / choice")
    ax.set_title("Predicted probability vs. actual choice")
    ax.legend()
    plt.tight_layout()
    plt.show()



