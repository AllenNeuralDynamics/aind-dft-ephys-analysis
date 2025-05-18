import numpy as np
from typing import Any, Optional, Tuple, Dict, List
from scipy.optimize import minimize

import numpy as np
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



__all__ = ["fit_choice_logistic_regression"]


def fit_choice_logistic_regression(
    nwb_data: Any,
    lag: int = 10,
) -> Optional[Tuple[BinaryResults, Dict[str, Any]]]:
    """
    Utility helper to fit a two‑kernel logistic regression model to
    left/right choice behaviour captured in NWB files.

    The model estimates how past rewarded and unrewarded choices bias the
    probability of selecting **right** on the current trial.

    Mathematical form
    -----------------
    ::

        logit(P(c_r(t)=1)) = β0
                             + sum_{i=1}^{L} β_i^R   * (R_r(t−i) − R_l(t−i))
                             + sum_{i=1}^{L} β_i^{NR} * (N_r(t−i) − N_l(t−i))

    where
      * *R_r*, *R_l* – rewarded right/left choice indicators;
      * *N_r*, *N_l* – **un**rewarded right/left indicators;
      * *L* – history length (``lag``).

    Parameters
    ----------
    nwb_data : Any
        Open NWB file handle whose ``trials`` table contains at least the
        columns ``animal_response``, ``rewarded_historyL``, and
        ``rewarded_historyR``.
    lag : int, default 10
        Number of past trials to include for each kernel.

    Returns
    -------
    (BinaryResults, Dict[str, Any]) | None
        ``BinaryResults``
            The fitted logistic‑regression results object returned by
            :class:`statsmodels.discrete.discrete_model.Logit`.
        ``Dict[str, Any]``
            Serializable summary with keys:

            * ``model_summary`` – text from ``result.summary()``.
            * ``fitted_latent_variables['choice_prob']`` – list ``[P(left), P(right)]``.
            * ``used_trial_indices`` – indices of trials contributing to the
              regression.

        The function returns **None** if the session is too short, lacks
        analysable trials, or the optimiser fails.

    Notes
    -----
    * Trials with ``animal_response == 2`` (no response) are skipped.
    * To fit a single model across multiple sessions, concatenate the design
      matrices produced here and call :func:`statsmodels.Logit` once on the
      combined arrays.
    """

    # 1. Extract arrays
    trials = nwb_data.trials
    resp: np.ndarray = np.asarray(trials["animal_response"][:])
    rL: np.ndarray = np.asarray(trials["rewarded_historyL"][:], dtype=int)
    rR: np.ndarray = np.asarray(trials["rewarded_historyR"][:], dtype=int)

    n_tot: int = len(resp)
    if n_tot <= lag + 1:
        print(f"[log-reg] ✖ session too short (n={n_tot}) for lag={lag}")
        return None

    # 2. Pre-compute Boolean flags
    Rl: np.ndarray = (resp == 0) & (rL == 1)  # rewarded LEFT
    Rr: np.ndarray = (resp == 1) & (rR == 1)  # rewarded RIGHT
    Nl: np.ndarray = (resp == 0) & (rL == 0)  # unrewarded LEFT
    Nr: np.ndarray = (resp == 1) & (rR == 0)  # unrewarded RIGHT

    # 3. Construct X and y
    X_rows: List[List[float]] = []
    y_list: List[int] = []
    used_idx: List[int] = []

    for t in range(lag, n_tot):
        this_resp = resp[t]
        if this_resp not in (0, 1):
            continue  # skip no-response trials

        row: List[int] = []

        # Rewarded kernel
        for i in range(1, lag + 1):
            row.append(int(Rr[t - i]) - int(Rl[t - i]))

        # Unrewarded kernel
        for i in range(1, lag + 1):
            row.append(int(Nr[t - i]) - int(Nl[t - i]))

        X_rows.append(row)
        y_list.append(int(this_resp))  # 0 = left, 1 = right
        used_idx.append(t)

    if not X_rows:
        print("[log-reg] ✖ no analysable trials after filtering")
        return None

    X = sm.add_constant(np.asarray(X_rows, dtype=float))
    y = np.asarray(y_list, dtype=int)

    # 4. Fit the model
    try:
        result = sm.Logit(y, X).fit(disp=False)
    except Exception as err:  # pragma: no cover
        print(f"[log-reg] ✖ fit failed: {err}")
        return None

    # 5. Build companion summary dict
    p_right: np.ndarray = result.predict()
    p_left: np.ndarray = 1.0 - p_right

    fit_dict: Dict[str, Any] = {
        "fit_result": result,
        "model_summary": result.summary().as_text(),
        "fitted_latent_variables": {
            "choice_prob": [p_left, p_right],
        },
        "used_trial_indices": np.asarray(used_idx, dtype=int),
    }

    return fit_dict, result 



