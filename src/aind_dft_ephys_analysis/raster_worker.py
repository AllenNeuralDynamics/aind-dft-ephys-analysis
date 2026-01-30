# raster_worker.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless in workers

from general_utils import smart_read_csv
from behavior_utils import find_trials, get_fitted_model_names
from nwb_utils import NWBUtils
from create_psth import load_zarr
from plot_raster import plot_raster_and_quantile_psth_by_latent

# ---- Shared config (edit if needed) ----
PSTH_DIR    = Path("/root/capsule/scratch/psth_results")
RESULTS_DIR = Path("/root/capsule/scratch/behavior_summary")
OUTDIR      = Path("/root/capsule/scratch/raster_plot")

# ============================================================
# LATENTS and Latent_NAMES
# ============================================================

LATENTS = []
Latent_NAMES = []

# --------------------------------------------------
# Model-based latents (existing)
# --------------------------------------------------
model_latents = [
    ("ForagingCompareThreshold-value-1", "Foraging-value-1"),
    ("ForagingCompareThreshold-RPE", "Foraging-RPE"),
]

for latent, name in model_latents:
    LATENTS.append(latent)
    Latent_NAMES.append(name)

# --------------------------------------------------
# no_model — running-window reward rate (window = 1–30)
# --------------------------------------------------
WINDOWS = range(1, 31)
RUN_TYPES = [
    ("running_experienced", "experienced"),
    ("running_left_reward", "left"),
    ("running_right_reward", "right"),
]

for w in WINDOWS:
    for suffix, short_name in RUN_TYPES:
        LATENTS.append(
            f"no_model-reward_rate_window_{w}-{suffix}"
        )
        Latent_NAMES.append(
            f"rr_window_{w}_{short_name}"
        )

# --------------------------------------------------
# no_model — EWMA reward rate (alpha-based)
# --------------------------------------------------
ALPHAS = [
    0.05, 0.1, 0.15, 0.2, 0.25,
    0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.75, 0.8, 0.85, 0.9
]

EWMA_TYPES = [
    ("ewma_experienced", "experienced"),
    ("ewma_left_reward", "left"),
    ("ewma_right_reward", "right"),
]

for a in ALPHAS:
    a_str = str(a).rstrip("0").rstrip(".")
    for suffix, short_name in EWMA_TYPES:
        LATENTS.append(
            f"no_model-reward_rate_alpha_{a}-{suffix}"
        )
        Latent_NAMES.append(
            f"rr_alpha_{a_str}_{short_name}"
        )

# --------------------------------------------------
# Sanity check
# --------------------------------------------------
assert len(LATENTS) == len(Latent_NAMES)


ALIGN_TO = "go_cue"
TIME_WIN = (-3, 4)


def process_session(session: str) -> str:
    """
    Run plotting for one session. Returns a short status string.
    This function must be top-level in a module so it’s importable by 'spawn'.
    """
    try:
        zarr_path = PSTH_DIR / f"{session}_0.2s.zarr"
        beh_csv   = RESULTS_DIR / f"behavior_summary-{session}.csv"
        save_dir  = OUTDIR / session

        if not zarr_path.exists():
            return f"[{session}] skip: Zarr not found: {zarr_path}"
        if not beh_csv.exists():
            return f"[{session}] skip: behavior CSV not found: {beh_csv}"

        print(f"\n[{session}] models: {get_fitted_model_names(session_name=session)}")

        nwb, _ = NWBUtils.combine_nwb(session_name=session)
        psth   = load_zarr(str(zarr_path))
        beh    = smart_read_csv(str(beh_csv))
        trials = np.asarray(find_trials(nwb, "response"))

        save_dir.mkdir(parents=True, exist_ok=True)

        for col, lname in zip(LATENTS, Latent_NAMES):

            if col == "QLearning_L2F1_softmax-reward":
                raster_colormap=True
            else:
                raster_colormap=False
            if col not in beh.columns:
                print(f"[{session}] skip: {col} (missing column)")
                continue

            vals = beh[col][0]

            plot_raster_and_quantile_psth_by_latent(
                source=psth,
                latent_values=vals,
                latent_trial_ids=trials,
                unit_ids=None,
                align_to_event=ALIGN_TO,
                time_window=TIME_WIN,
                n_bins=4,
                binning="equal",
                quantile_stat="mean",
                ci="sem",
                title_prefix="",
                figsize=(6, 5),
                save_path=str(save_dir),
                cmap_name="coolwarm",
                raster_colormap=raster_colormap,
                show_colormap=True,
                save_prefix=f"{col}_",
                latent_name=lname,
                show=False,   # save only
            )
            print(f"[{session}] plotted: {col}")

        return f"[{session}] done"

    except Exception as e:
        return f"[{session}] failed: {e}"
