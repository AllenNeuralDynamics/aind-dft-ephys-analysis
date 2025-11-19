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
PSTH_DIR    = Path("/root/capsule/scratch/psth")
RESULTS_DIR = Path("/root/capsule/scratch")
OUTDIR      = Path("/root/capsule/scratch/raster_plot")

LATENTS      = [
    "QLearning_L2F1_softmax-deltaQ-1",
    "QLearning_L2F1_softmax-reward",
    "QLearning_L2F1_softmax-sumQ-1",
]
LATENT_NAMES = ["deltaQ-1", "reward", "sumQ-1"]


LATENTS      = [
    "QLearning_L2F1_softmax-deltaQ-1",
    "QLearning_L2F1_softmax-sumQ-1"
]
LATENT_NAMES = [ "deltaQ-1","sumQ-1"]


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

        for col, lname in zip(LATENTS, LATENT_NAMES):

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
