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


def build_choice_design_matrix(
    nwb_data: Optional[Union[Any, Sequence[Any]]] = None,
    session_names: Optional[Union[str, Sequence[str]]] = None,
    *,
    lag: int = 10,
    pad_mode: str = "zero",
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build the design matrix **X**, outcome vector **y**, and a trial-index
    array for a history-kernel logistic-regression model of choice behavior.

    Either **nwb_data** *or* **session_names** can be supplied:

    * If *session_names* is provided, each session is opened with
      :func:`NWBUtils.read_ophys_or_behavior_nwb`, processed, and closed
      immediately to conserve memory; *nwb_data* is ignored.
    * If *session_names* is *None*, *nwb_data* must be a single NWB object or
      an iterable of pre-opened NWB objects.

    Mathematical model
    ------------------
    ::

        logit(P(c_r(t)=1)) = β0
                             + sum_{i=1}^{L} β_i^R   * (R_r(t−i) − R_l(t−i))
                             + sum_{i=1}^{L} β_i^{NR} * (N_r(t−i) − N_l(t−i))

    Parameters
    ----------
    nwb_data : NWBFile or Sequence[NWBFile], optional
        One or more pre-opened NWB handles. Ignored when *session_names* is
        provided.
    session_names : str or Sequence[str], optional
        Session identifiers such as ``"776293_2025-02-14_15-19-17"``. May be a
        single string or a list/tuple of strings.
    lag : int, default 10
        Number of past trials (:math:`L`) to include in the history kernel.
    pad_mode : {"zero", "repeat"}, default "zero"
        Strategy for padding the first *lag* trials, where full history is not
        available. ``"zero"`` pads with zeros; ``"repeat"`` duplicates the
        earliest valid history vector.

    Returns
    -------
    X : ndarray, shape (n_trials, 2*lag + 1)
        Design matrix with an intercept column followed by, for each lag
        1 … *L*, the contrast terms ``R_r − R_l`` and ``N_r − N_l``.
    y : ndarray, shape (n_trials,)
        Binary outcome vector (1 = right choice, 0 = left choice).
    trial_idx : ndarray, shape (n_trials,)
        Absolute trial index within each session, useful for mapping results
        back to raw data.

    Notes
    -----
    * Trials with no response are excluded from **X** and **y**.
    * The function treats reward as a binary indicator.
    * When multiple sessions are supplied, **X**, **y**, and **trial_idx**
      are concatenated in session order.
    """

    # Decide which source to use
    if session_names is not None:
        items: List[Union[Any, str]] = (
            [session_names] if isinstance(session_names, str) else list(session_names)
        )
    else:
        if nwb_data is None:
            raise ValueError("Either nwb_data or session_names must be provided")
        items = list(nwb_data) if isinstance(nwb_data, (list, tuple)) else [nwb_data]

    X_all: List[List[int]] = []
    y_all: List[int] = []
    idx_all: List[int] = []

    for sess_id, item in enumerate(items):
        opened_here = False
        if isinstance(item, str):
            nwb = NWBUtils.read_ophys_or_behavior_nwb(session_name=item)
            opened_here = True
        else:
            nwb = item

        if nwb is None:
            continue

        try:
            trials = nwb.trials
            resp = np.asarray(trials["animal_response"], dtype=int)
            rL = np.asarray(trials["rewarded_historyL"], dtype=int)
            rR = np.asarray(trials["rewarded_historyR"], dtype=int)

            valid_mask = (resp == 0) | (resp == 1)
            valid_idx = np.where(valid_mask)[0]
            if valid_idx.size <= lag and pad_mode is None:
                continue

            resp_c, rL_c, rR_c = resp[valid_mask], rL[valid_mask], rR[valid_mask]
            Rl = (resp_c == 0) & (rL_c == 1)
            Rr = (resp_c == 1) & (rR_c == 1)
            Nl = (resp_c == 0) & (rL_c == 0)
            Nr = (resp_c == 1) & (rR_c == 0)

            n_valid = resp_c.size
            start_t = 0 if pad_mode == "zero" else lag

            for t in range(start_t, n_valid):
                row: List[int] = []
                for i in range(1, lag + 1):
                    idx = t - i
                    row.append(0 if idx < 0 else int(Rr[idx]) - int(Rl[idx]))
                for i in range(1, lag + 1):
                    idx = t - i
                    row.append(0 if idx < 0 else int(Nr[idx]) - int(Nl[idx]))

                X_all.append(row)
                y_all.append(int(resp_c[t]))
                idx_all.append((sess_id << 32) + int(valid_idx[t]))
        except Exception as err:
            # Catch and report any error for this session, then continue
            print(f"[log-reg] ✖ error processing session {item}: {err}")
            # Skip to the next item without aborting the whole routine
            continue
        finally:
            if opened_here and hasattr(nwb, "io"):
                try:
                    nwb.io.close()
                except Exception:
                    pass

    if not X_all:
        print("[log-reg] ✖ no analysable rows across sessions")
        return None

    X_arr = sm.add_constant(np.asarray(X_all, dtype=float))
    y_arr = np.asarray(y_all, dtype=int)
    idx_arr = np.asarray(idx_all, dtype=int)
    return X_arr, y_arr, idx_arr

def fit_choice_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    trial_indices: Optional[np.ndarray] = None,
    **glm_kwargs,
) -> Optional[Tuple[BinaryResults, Dict[str, Any]]]:
    """Fit a logistic regression on the provided design matrix."""

    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching first dimension (rows)")

    try:
        result = sm.Logit(y, X).fit(disp=False, **glm_kwargs)
    except Exception as err:
        print(f"[log-reg] ✖ fit failed: {err}")
        return None

    p_right = result.predict()
    p_left = 1.0 - p_right

    if trial_indices is None or len(trial_indices) != len(y):
        trial_indices = np.arange(len(y))

    fit_dict: Dict[str, Any] = {
        "fit_result": result,
        "model_summary": result.summary().as_text(),
        "fitted_latent_variables": {"choice_prob": [p_left, p_right]},
        "used_trial_indices": trial_indices,
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



