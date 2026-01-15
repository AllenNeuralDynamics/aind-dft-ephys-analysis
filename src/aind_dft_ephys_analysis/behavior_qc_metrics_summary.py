import os
import ast
from typing import Optional, Dict, Sequence, Iterable, Union, List

import numpy as np
import pandas as pd

from behavior_qc_visualization import (
    plot_rpe_history_regression_from_nwb,
)

from behavior_utils import get_fitted_latent
from model_fitting import fit_choice_logistic_regression_from_nwb
from nwb_utils import NWBUtils


def collect_behavior_model_summary(
    sessions: Optional[Iterable[str]] = None,
    models: Union[str, Iterable[str]] = [
        "QLearning_L1F1_CK1_softmax",
        "QLearning_L2F1_softmax",
        "QLearning_L2F1_CK1_softmax",
        "QLearning_L2F1_CKfull_softmax",
        "ForagingCompareThreshold",
        "QLearning_L1F0_CKfull_softmax",
        "QLearning_L1F1_CKfull_softmax",
    ],
    session_paths: Optional[List[str]] = None,
    *,
    reward_coef_sum_n: Union[int, Sequence[int]] = [1, 2, 3, 4, 5, 6],
    spread_eps_mode: str = "auto",
    spread_eps_value: float = 0.05,
    spread_entropy_bins: int = 30,
    spread_entropy_base: float = np.e,
) -> pd.DataFrame:
    """
    Collect fitted reinforcement-learning (Q-learning) parameters, model-level
    goodness-of-fit metrics, and RPE history regression summaries across sessions
    and models. In addition, this function computes multiple distributional
    “spread” diagnostics from fitted latent variables to quantify how well
    values and learning signals are expressed within each session.

    Specifically, spread metrics are computed for three latent signals:
      - deltaQ = Q_left − Q_right
            Measures trial-by-trial value separation between actions and
            directly governs choice under softmax decision rules.
      - sumQ   = Q_left + Q_right
            Captures the overall value magnitude / motivational baseline
            independent of choice direction.
      - RPE    = reward prediction error
            Reflects the magnitude and diversity of learning signals driving
            value updates.

    For each signal, the following categories of spread metrics are computed
    per session × per model:
      - Variance-based metrics (std)
      - Robust spread metrics (IQR, MAD, P95–P5)
      - Tie / near-zero fraction (fraction of trials where |signal| < ε),
        which captures value collapse or indifference
      - Distributional entropy and effective number of occupied states,
        which quantify how broadly the signal explores its state space

    Assumptions about fitted latent variables
    -----------------------------------------
    The fitted latent variables are retrieved from:
        metrics['latent_variables']

    and are assumed to have the following structure:
        Q_left  = metrics['latent_variables']['q_value'][0]
        Q_right = metrics['latent_variables']['q_value'][1]
        RPE     = metrics['latent_variables']['rpe']

    where each entry is a 1D array indexed by trial.

    Interpretation notes
    --------------------
    - High deltaQ_tie_frac indicates weak value discrimination, even if
      choice behavior appears stochastic or biased.
    - High sumQ_tie_frac indicates collapse of overall value magnitude,
      often due to low learning rate, strong forgetting, or over-regularization.
    - High RPE_tie_frac combined with low deltaQ spread suggests under-learning,
      whereas high RPE spread with unstable deltaQ may indicate noisy learning.
    - Entropy complements variance-based metrics by capturing whether values
      explore multiple states or collapse into a narrow regime.

    Dependency notes
    ----------------
    This function assumes the following symbols are available in the runtime
    environment:
      - NWBUtils (behavior NWB loading)
      - plot_rpe_history_regression_from_nwb (RPE history regression)
      - get_fitted_latent (model fitting + latent extraction)
      - fit_choice_logistic_regression_from_nwb (baseline logistic fit)
      - compute_spread_stats (top-level helper for spread / entropy metrics)

    Returns
    -------
    pandas.DataFrame
        A wide-format DataFrame with one row per session and model-specific
        column blocks containing fitted parameters, model metrics, and
        latent-variable spread diagnostics.
    """

    def _normalize_n_list(n_in: Union[int, Sequence[int]]) -> List[int]:
        """Normalize reward_coef_sum_n to a sorted list of unique positive ints."""
        if isinstance(n_in, (int, np.integer)):
            n_list = [int(n_in)]
        else:
            n_list = [int(v) for v in n_in]

        n_list = [v for v in n_list if v > 0]
        n_list = sorted(set(n_list))
        return n_list

    def _sum_first_n(x: object, n: int) -> float:
        """Return sum of first n elements of x, or np.nan if invalid/too short."""
        if x is None:
            return np.nan

        # Parse stringified lists if needed (keeps backward compatibility)
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except Exception:
                return np.nan

        arr = np.asarray(x)

        # Skip scalars / invalid entries
        if arr.ndim == 0:
            return np.nan

        if arr.size < n:
            return np.nan

        return float(arr[:n].sum())

    if isinstance(models, str):
        models = [models]
    else:
        models = list(models)

    n_list = _normalize_n_list(reward_coef_sum_n)
    rows: List[dict] = []

    # If neither sessions nor session_paths is provided, return an empty DataFrame
    if not sessions and not session_paths:
        return pd.DataFrame()

    if sessions:
        session_iters = sessions
        use_paths = False
    else:
        session_iters = session_paths  # type: ignore[assignment]
        use_paths = True

    for session in session_iters:
        session_name = os.path.basename(session)
        row: dict = {"session": session}

        # -----------------------------------------
        # 1) Load NWB safely
        # -----------------------------------------
        try:
            if use_paths:
                nwb_data = NWBUtils.read_behavior_nwb(nwb_full_path=session)
            else:
                nwb_data = NWBUtils.read_behavior_nwb(session_name=session_name)
        except Exception as exc:
            row["nwb_load_error"] = f"read_behavior_nwb exception: {exc}"
            rows.append(row)
            continue

        if nwb_data is None:
            row["nwb_load_error"] = "read_behavior_nwb returned None (missing/invalid behavior NWB?)"
            rows.append(row)
            continue

        # -----------------------------------------
        # 2) RPE history regression once per session
        # -----------------------------------------
        _, _, fitting_results = plot_rpe_history_regression_from_nwb(
            nwb_data=nwb_data,
            session_name=session_name,
            make_figure=False,
            show_figure=False,
        )

        # -----------------------------------------
        # 3) Per-model fitted params/metrics + reg outputs + spread metrics
        # -----------------------------------------
        for model in models:
            prefix = f"{model}_"

            try:
                results = get_fitted_latent(session_name=session_name, model_alias=model)
                params = results.get("params", {})
                metrics = results.get("results", {})

                row.update({
                    f"{prefix}learn_rate_rew": params.get("learn_rate_rew"),
                    f"{prefix}learn_rate_unrew": params.get("learn_rate_unrew"),
                    f"{prefix}forget_rate_unchosen": params.get("forget_rate_unchosen"),
                    f"{prefix}choice_kernel_relative_weight": params.get("choice_kernel_relative_weight"),
                    f"{prefix}choice_kernel_step_size": params.get("choice_kernel_step_size"),
                    f"{prefix}biasL": params.get("biasL"),
                    f"{prefix}softmax_inverse_temperature": params.get("softmax_inverse_temperature"),
                    f"{prefix}learn_rate": params.get("learn_rate"),

                    f"{prefix}log_likelihood": metrics.get("log_likelihood"),
                    f"{prefix}AIC": metrics.get("AIC"),
                    f"{prefix}BIC": metrics.get("BIC"),
                    f"{prefix}LPT": metrics.get("LPT"),
                    f"{prefix}LPT_AIC": metrics.get("LPT_AIC"),
                    f"{prefix}LPT_BIC": metrics.get("LPT_BIC"),
                    f"{prefix}prediction_accuracy": metrics.get("prediction_accuracy"),
                })

                # ---------------------------------
                # Spread stats from fitted latent variables
                # ---------------------------------
                latent = metrics.get("latent_variables", {}) 
                q_value = latent.get("q_value", None)
                rpe = latent.get("rpe", None)

                # Extract Ql, Qr and derive deltaQ and sumQ
                deltaQ = None
                sumQ = None
                try:
                    Ql = None
                    Qr = None

                    if q_value is not None:
                        q_arr = np.asarray(q_value, dtype=float)

                        # Expect shape (2, T) or list-like [Ql, Qr]
                        if q_arr.ndim >= 2 and q_arr.shape[0] >= 2:
                            Ql = np.asarray(q_arr[0], dtype=float).reshape(-1)
                            Qr = np.asarray(q_arr[1], dtype=float).reshape(-1)
                        elif isinstance(q_value, (list, tuple)) and len(q_value) >= 2:
                            Ql = np.asarray(q_value[0], dtype=float).reshape(-1)
                            Qr = np.asarray(q_value[1], dtype=float).reshape(-1)

                    if Ql is not None and Qr is not None:
                        n_min = min(Ql.size, Qr.size)
                        Ql = Ql[:n_min]
                        Qr = Qr[:n_min]
                        deltaQ = Ql - Qr
                        sumQ = Ql + Qr

                except Exception as exc:
                    row[f"{prefix}Q_extract_error"] = f"Q extraction failed: {exc}"
                    deltaQ = None
                    sumQ = None

                # deltaQ spread
                dQ_stats = compute_spread_stats(
                    deltaQ,
                    eps_mode=spread_eps_mode,
                    eps_value=spread_eps_value,
                    entropy_bins=spread_entropy_bins,
                    entropy_base=spread_entropy_base,
                )
                row.update({
                    f"{prefix}deltaQ_n": dQ_stats["n"],
                    f"{prefix}deltaQ_mean": dQ_stats["mean"],
                    f"{prefix}deltaQ_std": dQ_stats["std"],
                    f"{prefix}deltaQ_iqr": dQ_stats["iqr"],
                    f"{prefix}deltaQ_q95_q5": dQ_stats["q95_q5"],
                    f"{prefix}deltaQ_mad": dQ_stats["mad"],
                    f"{prefix}deltaQ_tie_frac": dQ_stats["tie_frac"],
                    f"{prefix}deltaQ_eps_used": dQ_stats["eps_used"],
                    f"{prefix}deltaQ_entropy": dQ_stats["entropy"],
                    f"{prefix}deltaQ_entropy_eff_bins": dQ_stats["entropy_eff_bins"],
                    f"{prefix}deltaQ_entropy_bins_used": dQ_stats["entropy_bins_used"],
                })

                # sumQ spread
                sQ_stats = compute_spread_stats(
                    sumQ,
                    eps_mode=spread_eps_mode,
                    eps_value=spread_eps_value,
                    entropy_bins=spread_entropy_bins,
                    entropy_base=spread_entropy_base,
                )
                row.update({
                    f"{prefix}sumQ_n": sQ_stats["n"],
                    f"{prefix}sumQ_mean": sQ_stats["mean"],
                    f"{prefix}sumQ_std": sQ_stats["std"],
                    f"{prefix}sumQ_iqr": sQ_stats["iqr"],
                    f"{prefix}sumQ_q95_q5": sQ_stats["q95_q5"],
                    f"{prefix}sumQ_mad": sQ_stats["mad"],
                    f"{prefix}sumQ_tie_frac": sQ_stats["tie_frac"],
                    f"{prefix}sumQ_eps_used": sQ_stats["eps_used"],
                    f"{prefix}sumQ_entropy": sQ_stats["entropy"],
                    f"{prefix}sumQ_entropy_eff_bins": sQ_stats["entropy_eff_bins"],
                    f"{prefix}sumQ_entropy_bins_used": sQ_stats["entropy_bins_used"],
                })

                # RPE spread
                rpe_stats = compute_spread_stats(
                    rpe,
                    eps_mode=spread_eps_mode,
                    eps_value=spread_eps_value,
                    entropy_bins=spread_entropy_bins,
                    entropy_base=spread_entropy_base,
                )
                row.update({
                    f"{prefix}rpe_n": rpe_stats["n"],
                    f"{prefix}rpe_mean": rpe_stats["mean"],
                    f"{prefix}rpe_std": rpe_stats["std"],
                    f"{prefix}rpe_iqr": rpe_stats["iqr"],
                    f"{prefix}rpe_q95_q5": rpe_stats["q95_q5"],
                    f"{prefix}rpe_mad": rpe_stats["mad"],
                    f"{prefix}rpe_tie_frac": rpe_stats["tie_frac"],
                    f"{prefix}rpe_eps_used": rpe_stats["eps_used"],
                    f"{prefix}rpe_entropy": rpe_stats["entropy"],
                    f"{prefix}rpe_entropy_eff_bins": rpe_stats["entropy_eff_bins"],
                    f"{prefix}rpe_entropy_bins_used": rpe_stats["entropy_bins_used"],
                })

                # ---------------------------------
                # RPE history regression (same model)
                # ---------------------------------
                reward_coefs = (
                    fitting_results
                    .get(model, {})
                    .get("all", {})
                    .get("reward_coefs")
                )

                if reward_coefs is not None:
                    reward_coefs = np.asarray(reward_coefs)
                    row[f"{prefix}reward_coefs"] = reward_coefs
                else:
                    row[f"{prefix}reward_coefs"] = np.nan

                # Sum first n reward coefs
                for n in n_list:
                    row[f"{prefix}reward_coefs_sum{n}"] = _sum_first_n(row[f"{prefix}reward_coefs"], n)

            except Exception as exc:
                row[f"{prefix}error"] = str(exc)

                # Keep schema consistent: sum columns
                for n in n_list:
                    row.setdefault(f"{prefix}reward_coefs_sum{n}", np.nan)

                # Keep schema consistent: spread columns (deltaQ, sumQ, rpe)
                spread_keys = [
                    # deltaQ
                    "deltaQ_n", "deltaQ_mean", "deltaQ_std", "deltaQ_iqr", "deltaQ_q95_q5",
                    "deltaQ_mad", "deltaQ_tie_frac", "deltaQ_eps_used",
                    "deltaQ_entropy", "deltaQ_entropy_eff_bins", "deltaQ_entropy_bins_used",
                    # sumQ
                    "sumQ_n", "sumQ_mean", "sumQ_std", "sumQ_iqr", "sumQ_q95_q5",
                    "sumQ_mad", "sumQ_tie_frac", "sumQ_eps_used",
                    "sumQ_entropy", "sumQ_entropy_eff_bins", "sumQ_entropy_bins_used",
                    # rpe
                    "rpe_n", "rpe_mean", "rpe_std", "rpe_iqr", "rpe_q95_q5",
                    "rpe_mad", "rpe_tie_frac", "rpe_eps_used",
                    "rpe_entropy", "rpe_entropy_eff_bins", "rpe_entropy_bins_used",
                ]
                for key in spread_keys:
                    row.setdefault(f"{prefix}{key}", np.nan)

        # Logistic regression
        logistic_results = fit_choice_logistic_regression_from_nwb(nwb_data)
        logistic_bias = logistic_results["fit_result"].params[0]
        row["logistic_bias"] = logistic_bias

        # auto_train_stage (last)
        try:
            if len(nwb_data.trials["auto_train_stage"]) > 0:
                row["auto_train_stage"] = nwb_data.trials["auto_train_stage"][-1]
            else:
                row["auto_train_stage"] = np.nan
        except Exception as exc:
            row["auto_train_stage_error"] = str(exc)

        rows.append(row)

    return pd.DataFrame(rows)




def compute_spread_stats(
    values: object,
    *,
    eps_mode: str = "auto",
    eps_value: float = 0.05,
    entropy_bins: int = 30,
    entropy_base: float = np.e,
) -> Dict[str, float]:
    """
    Compute dispersion / spread metrics for a 1D signal (e.g. deltaQ or RPE).

    Metric meanings
    ---------------
    n:
        Number of finite samples.

    mean:
        Average value (for deltaQ: average action preference).

    std:
        Standard deviation (global spread, sensitive to outliers).

    iqr:
        Interquartile range = P75 - P25 (robust central spread).

    q95_q5:
        P95 - P5 (robust wide-range spread).

    mad:
        Median absolute deviation, scaled (1.4826×) to match std for Gaussian data.
        Very robust to outliers.

    tie_frac:
        Fraction of samples close to zero:
            tie_frac = mean(|x| < eps_used)
        For deltaQ:
            High tie_frac → Q-values rarely separate → weak value discrimination.
        For RPE:
            High tie_frac → prediction errors mostly near zero → low learning drive.

    eps_used:
        Threshold ε used for tie_frac.

    entropy:
        Shannon entropy of the empirical distribution (histogram-based).
        High entropy → values spread across many bins.
        Low entropy → values concentrated / narrow.

    entropy_eff_bins:
        Effective number of occupied bins = exp(entropy) (or base**entropy).

    entropy_bins_used:
        Number of histogram bins actually used.

    Parameters
    ----------
    values : array-like or None
        Input signal.
    eps_mode : {"auto", "fixed"}, default "auto"
        - "auto": eps = 0.1 * std(values)
        - "fixed": eps = eps_value
    eps_value : float
        Fixed or fallback epsilon.
    entropy_bins : int
        Target number of bins for entropy.
    entropy_base : float
        Log base (np.e = nats, 2 = bits).

    Returns
    -------
    Dict[str, float]
    """
    empty = {
        "n": 0.0,
        "mean": np.nan,
        "std": np.nan,
        "iqr": np.nan,
        "q95_q5": np.nan,
        "mad": np.nan,
        "tie_frac": np.nan,
        "eps_used": np.nan,
        "entropy": np.nan,
        "entropy_eff_bins": np.nan,
        "entropy_bins_used": np.nan,
    }

    if values is None:
        return empty

    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return empty

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))

    p25, p75 = np.percentile(arr, [25, 75])
    p5, p95 = np.percentile(arr, [5, 95])

    med = np.median(arr)
    mad = float(1.4826 * np.median(np.abs(arr - med)))

    # epsilon for tie fraction
    if eps_mode == "fixed":
        eps_used = float(eps_value)
    elif eps_mode == "auto":
        eps_used = float(0.1 * std) if std > 0 else float(eps_value)
    else:
        raise ValueError(f"Unknown eps_mode={eps_mode!r}")

    tie_frac = float(np.mean(np.abs(arr) < eps_used))

    # Entropy (histogram-based)
    bins_used = min(entropy_bins, max(1, int(np.sqrt(arr.size))))
    try:
        counts, _ = np.histogram(arr, bins=bins_used)
        p = counts / counts.sum()
        p = p[p > 0]

        if p.size > 0:
            if entropy_base == np.e:
                entropy = float(-np.sum(p * np.log(p)))
                eff_bins = float(np.exp(entropy))
            else:
                entropy = float(-np.sum(p * np.log(p) / np.log(entropy_base)))
                eff_bins = float(entropy_base ** entropy)
        else:
            entropy = np.nan
            eff_bins = np.nan
    except Exception:
        entropy = np.nan
        eff_bins = np.nan

    return {
        "n": float(arr.size),
        "mean": mean,
        "std": std,
        "iqr": float(p75 - p25),
        "q95_q5": float(p95 - p5),
        "mad": mad,
        "tie_frac": tie_frac,
        "eps_used": eps_used,
        "entropy": entropy,
        "entropy_eff_bins": eff_bins,
        "entropy_bins_used": float(bins_used),
    }