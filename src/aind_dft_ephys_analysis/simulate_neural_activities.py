# Standard library
from pathlib import Path

# Data handling
import numpy as np
import pandas as pd
import cloudpickle

# Statistics
from scipy.stats import linregress

# Plotting
import matplotlib.pyplot as plt
# -----------------------------
# Core simulation function (per-trial Poisson neurons)
# -----------------------------
def simulate_neurons_with_poisson(
    latent_values: np.ndarray,
    n_neurons: int = 50,
    rho_mean: float = 0.5,
    rho_std: float = 0.1,
    baseline_range: tuple = (1, 5),
    gain_range: tuple = (5, 15),
    plot_example: bool = True,
    n_example: int = 5
) -> np.ndarray:
    """
    Simulate neurons whose firing is correlated with a latent value using Poisson spiking.
    
    Parameters
    ----------
    latent_values : np.ndarray
        1D array of latent values (per trial).
    n_neurons : int
        Number of neurons to simulate.
    rho_mean : float
        Mean correlation strength of neurons with latent value (0-1).
    rho_std : float
        Standard deviation of correlation strength across neurons.
    baseline_range : tuple
        Min and max baseline firing rate (Hz).
    gain_range : tuple
        Min and max gain (scaling) factor for firing rate.
    plot_example : bool
        Whether to plot example neurons.
    n_example : int
        Number of neurons to plot if plot_example=True.
        
    Returns
    -------
    simulated_firing : np.ndarray
        Array of shape (n_trials, n_neurons) with Poisson spike counts.
    """
    # Normalize latent values to [0,1]
    latent_norm = (latent_values - np.min(latent_values)) / (np.ptp(latent_values) + 1e-12)
    
    simulated_firing = np.zeros((len(latent_values), n_neurons))
    
    for n in range(n_neurons):
        rho = np.clip(np.random.normal(rho_mean, rho_std), 0, 1)
        baseline = np.random.uniform(*baseline_range)
        gain = np.random.uniform(*gain_range)
        
        # firing rate per trial
        firing_rate = baseline + gain * (rho * latent_norm + (1 - rho) * np.random.rand(len(latent_values)))
        simulated_firing[:, n] = np.random.poisson(firing_rate)
    
    if plot_example:
        plt.figure(figsize=(10, 4))
        plt.plot(simulated_firing[:, :n_example])
        plt.xlabel('Trial')
        plt.ylabel('Spike count')
        plt.title(f'Simulated {n_example} neurons correlated with latent value')
        plt.show()
    
    return simulated_firing


# -----------------------------
# Wrapper function
# -----------------------------
def simulate_neurons(
    session_name: str,
    parent_folder: str,
    behavior_model_path: str,
    simulate_method: str = "poisson",
    simulate_params: dict = None,
    latent_values: np.ndarray = None,
    plot_example: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper to simulate neurons based on latent values from a behavior model.
    The first trial value is removed before simulation.
    
    Returns
    -------
    simulated_firing : np.ndarray
        Simulated neurons (n_trials-1 x n_neurons).
    latent_values_trimmed : np.ndarray
        The latent values used for simulation (first value removed).
    """
    # Construct full path to behavior model
    model_file = Path(parent_folder) / session_name / behavior_model_path
    
    # Load fitted agent if latent_values not provided
    if latent_values is None:
        with model_file.open("rb") as f:
            fitted_agent = cloudpickle.load(f)
        latent_values = fitted_agent.get_fitting_result_dict()['fitted_latent_variables']['value']
    
    # Remove the first value
    latent_values_trimmed = latent_values[1:]
    
    # Default simulation parameters
    simulate_params = simulate_params or {}
    
    # Choose method
    if simulate_method == "poisson":
        simulated_firing = simulate_neurons_with_poisson(
            latent_values_trimmed,
            plot_example=plot_example,
            **simulate_params
        )
    else:
        raise NotImplementedError(f"Simulation method '{simulate_method}' not implemented.")
    
    return simulated_firing, latent_values_trimmed




def fit_activity_to_latent_complete(simulated_spikes: np.ndarray,
                                    session_name: str,
                                    parent_folder: str,
                                    behavior_model_path: str,
                                    latent: str = "deltaQ"):
    """
    Fit simulated neuron activity to a latent variable extracted from a behavior model.
    Stores complete correlation and slope results for all neurons.

    Parameters
    ----------
    simulated_spikes : np.ndarray
        Simulated spikes, shape (n_trials, n_neurons)
    session_name : str
        Session name
    parent_folder : str
        Parent folder containing session folders
    behavior_model_path : str
        Path to the fitted_agent.cloudpickle
    latent : str
        Which latent variable to use: "value", "sumQ", or "deltaQ"

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with neuron index, slope, correlation, and latent variable used
    latent_values : np.ndarray
        Extracted latent variable (trimmed first trial)
    """
    # Load behavior model
    model_file = Path(parent_folder) / session_name / behavior_model_path
    with model_file.open("rb") as f:
        fitted_agent = cloudpickle.load(f)
    
    # Extract latent variable
    latent_dict = fitted_agent.get_fitting_result_dict()['fitted_latent_variables']
    
    if latent == "value":
        latent_values = np.array(latent_dict['value'])
    elif latent == "sumQ":
        q_values = np.array(latent_dict['q_value'])
        latent_values = q_values[0] + q_values[1]
    elif latent == "deltaQ":
        q_values = np.array(latent_dict['q_value'])
        latent_values = q_values[1] - q_values[0]
    else:
        raise ValueError(f"Unknown latent variable: {latent}")
    
    # Remove the first trial
    latent_values = latent_values[1:]
    
    # Ensure shapes
    latent_values = np.array(latent_values)
    simulated_spikes = np.array(simulated_spikes)
    
    # Compute slope and correlation for all neurons
    neuron_indices = np.arange(simulated_spikes.shape[1])
    slopes = []
    correlations = []
    
    for idx in neuron_indices:
        y = simulated_spikes[:, idx]
        x = latent_values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        slopes.append(slope)
        correlations.append(r_value)
    
    # Create a complete results DataFrame
    results_df = pd.DataFrame({
        "Neuron": neuron_indices,
        "Slope": slopes,
        "Correlation": correlations,
        "Latent": latent
    })
    
    return results_df, latent_values
def plot_activity_vs_latent(simulated_spikes: np.ndarray,
                            latent_values,
                            selected_neurons: list = None,
                            n_neurons: int = 5,
                            random_seed: int = None):
    """
    Plot activity vs latent variable for selected neurons with linear fit.

    Parameters
    ----------
    simulated_spikes : np.ndarray
        Shape (n_trials, n_neurons)
    latent_values : np.ndarray or list
        Latent variable for fitting (shared across all neurons). If list, will be converted to np.ndarray.
    selected_neurons : list or None
        List of neuron indices to plot. If None, randomly select n_neurons
    n_neurons : int
        Number of neurons to randomly select if selected_neurons is None
    random_seed : int or None
        Random seed for reproducible neuron selection. If None, selection is random each time.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_total_neurons = simulated_spikes.shape[1]

    if selected_neurons is None:
        selected_neurons = np.random.choice(n_total_neurons, size=min(n_neurons, n_total_neurons), replace=False)

    # Convert latent_values to np.ndarray if it's a list
    if isinstance(latent_values, list):
        latent_values = np.array(latent_values)
    elif not isinstance(latent_values, np.ndarray):
        raise TypeError("latent_values must be a np.ndarray or a list")

    fig, axes = plt.subplots(len(selected_neurons), 1, figsize=(6, 4*len(selected_neurons)))
    if len(selected_neurons) == 1:
        axes = [axes]

    for ax, idx in zip(axes, selected_neurons):
        y = simulated_spikes[:, idx]
        x = latent_values
        slope, intercept, r_value, _, _ = linregress(x, y)
        y_fit = intercept + slope * x

        ax.scatter(x, y, alpha=0.6)
        ax.plot(x, y_fit, color='red', linewidth=2, label=f'Fit line (r={r_value:.2f})')
        ax.set_xlabel("Latent")
        ax.set_ylabel("Spike Count")
        ax.set_title(f"Neuron {idx}: Activity vs Latent")
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_latent_relationship(parent_folder: str,
                             session_name: str,
                             models: list,
                             latents: list,
                             agent_filename: str = "fitted_agent.cloudpickle",
                             plot_title: str = None):
    """
    Visualize the relationship between two latent variables from two models.

    Parameters
    ----------
    parent_folder : str
        Parent folder containing session folders
    session_name : str
        Name of the session folder
    models : list of str
        Names of the two model folders, e.g.,
        ["ForagingCompareThreshold_L1_CKnone_ResetT_StayBiasF_FixThrF", "QLearning_L1F1_softmax"]
    latents : list of str
        Latent variables to extract from each model, e.g., ["value", "deltaQ"]
    agent_filename : str
        Filename of the fitted agent inside each model folder (default: "fitted_agent.cloudpickle")
    plot_title : str or None
        Optional title for the scatter plot

    Returns
    -------
    latent1, latent2 : np.ndarray
        Extracted latent variables (first trial removed)
    """
    if len(models) != 2 or len(latents) != 2:
        raise ValueError("Exactly two models and two latent variables must be provided.")

    latent_values = []

    for model_folder, latent_name in zip(models, latents):
        model_file = Path(parent_folder) / session_name / model_folder / agent_filename
        with model_file.open("rb") as f:
            fitted_agent = cloudpickle.load(f)

        latent_dict = fitted_agent.get_fitting_result_dict()['fitted_latent_variables']

        # Extract latent variable
        if latent_name == "value":
            latent = np.array(latent_dict['value'])
        elif latent_name == "sumQ":
            q_values = np.array(latent_dict['q_value'])
            latent = q_values[0] + q_values[1]
        elif latent_name == "deltaQ":
            q_values = np.array(latent_dict['q_value'])
            latent = q_values[1] - q_values[0]
        else:
            raise ValueError(f"Unknown latent variable: {latent_name}")

        # Remove the first trial
        latent = latent[1:]
        latent_values.append(latent)

    latent1, latent2 = latent_values

    # Plot scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(latent1, latent2, alpha=0.6)
    plt.xlabel(f"{latents[0]} from {models[0]}")
    plt.ylabel(f"{latents[1]} from {models[1]}")
    title = plot_title or f"Relationship between {latents[0]} and {latents[1]}"
    plt.title(title)

    # Fit line
    slope, intercept = np.polyfit(latent1, latent2, 1)
    x_vals = np.linspace(np.min(latent1), np.max(latent1), 100)
    plt.plot(x_vals, intercept + slope * x_vals, color='red', linewidth=2, label=f"Fit line (slope={slope:.2f})")
    plt.legend()
    plt.show()

    return latent1, latent2