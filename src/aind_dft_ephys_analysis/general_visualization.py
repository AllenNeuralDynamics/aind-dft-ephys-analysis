import numpy as np
from typing import Any, Dict, Optional

from aind_dynamic_foraging_basic_analysis.plot import plot_foraging_session
from behavior_utils import extract_fitted_data


def plot_behavior_session(
    nwb_data      : Any,
    model_alias   : str  = "QLearning_L2F1_softmax",
    latent_name   : str  = "right_choice_probability",
    include_fitted: bool = True,
    fitted_latent : Optional[Dict[str, Any]] = None
) -> None:
    """
    Visualise one behavioural session, optionally overlaying a model-derived
    latent time-series.

    Notes on the two mutually supportive inputs
    -------------------------------------------
    • **`fitted_latent` provided**  
      → The function uses that dictionary directly; `model_alias` is ignored
      for latent extraction (but is still printed in the status message).

    • **`fitted_latent` is *None***  
      → `extract_fitted_data` will try to fetch the required fit from the
      remote behaviour-analysis database, using  
      `session_name = nwb_data.session_id` and the supplied `model_alias`.

    Parameters
    ----------
    nwb_data : Any
        An open NWB object that contains a `.trials` DynamicTable with at least
        the columns:
            'animal_response', 'rewarded_historyL', 'rewarded_historyR',
            'reward_probabilityL', 'reward_probabilityR'.
    model_alias : str, default "QLearning_L2F1_softmax"
        The model identifier that should be fetched *if* `fitted_latent`
        is not supplied.  Has no effect when `fitted_latent` is given.
    latent_name : str, default "right_choice_probability"
        Which latent variable to plot.  Must be recognised by
        `extract_fitted_data`.
    include_fitted : bool, default True
        If False, only raw behavioural data are plotted (no model overlay).
    fitted_latent : dict, optional
        A pre-computed fit dictionary containing the key
        ``"fitted_latent_variables"`` (exact structure expected by
        `extract_fitted_data`).  When this argument is supplied, the function
        will *not* attempt any database call.

    Returns
    -------
    None – the function shows a Matplotlib figure via
    ``plot_foraging_session.plot_foraging_session``.
    """
    # ──────────────────────────────────────────────────────────────
    # 1. Fetch trial-level arrays
    # ──────────────────────────────────────────────────────────────
    trials = nwb_data.trials
    animal_resp = np.asarray(trials["animal_response"][:])           # 0 / 1 / 2
    reward_hist = (
        trials["rewarded_historyL"][:] | trials["rewarded_historyR"][:]
    ).astype(int)
    p_reward = np.vstack((
        np.asarray(trials["reward_probabilityL"][:]),
        np.asarray(trials["reward_probabilityR"][:])
    ))

    # ──────────────────────────────────────────────────────────────
    # 2. Obtain fitted latent series (single unified call)
    # ──────────────────────────────────────────────────────────────
    fitted_series = None
    if include_fitted:
        session_full = getattr(nwb_data, "session_id", None)  # may be None
        fitted_series = extract_fitted_data(
            nwb_behavior_data = nwb_data,
            fitted_latent     = fitted_latent,   # if None, helper will fetch
            session_name      = session_full,
            model_alias       = model_alias,
            latent_name       = latent_name
        )

    # ──────────────────────────────────────────────────────────────
    # 3. Build vectors for plotting
    # ──────────────────────────────────────────────────────────────
    if include_fitted and fitted_series is not None:
        valid       = animal_resp != 2        # keep only responded trials
        choice_hist = animal_resp[valid]
    else:
        valid       = np.ones_like(animal_resp, dtype=bool)
        # mark no-response trials as NaN for the raster plot
        choice_hist = np.where(animal_resp == 2, np.nan, animal_resp)

    reward_use   = reward_hist[valid]
    p_reward_use = p_reward[:, valid]

    # ──────────────────────────────────────────────────────────────
    # 4. Delegate to the AIND plotting utility
    # ──────────────────────────────────────────────────────────────
    print(f"Plotting alias '{model_alias}' (latent: {latent_name})")
    plot_foraging_session.plot_foraging_session(
        choice_history = choice_hist,
        reward_history = reward_use,
        p_reward       = p_reward_use,
        fitted_data    = fitted_series
    )
