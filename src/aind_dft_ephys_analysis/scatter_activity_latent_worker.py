# raster_worker.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gc
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless in workers
import matplotlib.pyplot as plt

from general_utils import smart_read_csv
from behavior_utils import find_trials, get_fitted_model_names
from nwb_utils import NWBUtils
from create_psth import load_zarr

# IMPORTANT:
# - Put the new scatter function in an importable module.
# - If you placed it in `plot_raster.py`, keep this import as-is.
from plot_raster import plot_trial_mean_activity_vs_latent_per_unit


# ---- Shared config (edit if needed) ----
PSTH_DIR = Path("/root/capsule/scratch/psth_results")
RESULTS_DIR = Path("/root/capsule/scratch/behavior_summary")
OUTDIR = Path("/root/capsule/scratch/scatter_plot")  # you can keep the same OUTDIR

# ============================================================
# LATENTS and Latent_NAMES
# ============================================================

LATENTS: list[str] = []
Latent_NAMES: list[str] = []

# --------------------------------------------------
# Model-based latents (existing)
# --------------------------------------------------
model_latents: list[tuple[str, str]] = [
    ("ForagingCompareThreshold-value-1", "Foraging-value-1"),
    ("ForagingCompareThreshold-RPE", "Foraging-RPE"),
    ("QLearning_L2F1_softmax-chosenQ", "QLearning_L2F1-chosenQ"),
    ("QLearning_L2F1_softmax-sumQ-1", "QLearning_L2F1-sumQ-1"),
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
        LATENTS.append(f"no_model-reward_rate_window_{w}-{suffix}")
        Latent_NAMES.append(f"rr_window_{w}_{short_name}")

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
        LATENTS.append(f"no_model-reward_rate_alpha_{a}-{suffix}")
        Latent_NAMES.append(f"rr_alpha_{a_str}_{short_name}")

# --------------------------------------------------
# Sanity check
# --------------------------------------------------
assert len(LATENTS) == len(Latent_NAMES)

ALIGN_TO = "go_cue"

# Optional: this time window is only used to limit loaded PSTH data (faster IO)
# It MUST include ACTIVITY_WINDOW.
TIME_WIN = (-3, 4)

# NEW: define the time range to average activity within each trial (relative to ALIGN_TO)
# Example: mean firing rate from 0 to 0.2 s after go_cue.
ACTIVITY_WINDOW = (-1, 0)


def _as_1d_float_array(x: Any) -> np.ndarray:
    """
    Convert a behavior cell value into a compact 1D float array.
    This helps avoid pandas object-dtype retention and reduces memory overhead.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1).astype(np.float32, copy=False)
    return arr


def process_session(session: str) -> str:
    """
    Run plotting for one session. Returns a short status string.
    This function must be top-level in a module so it’s importable by 'spawn'.

    Memory notes:
      - We explicitly close Matplotlib figures after each plot to prevent
        figure-manager accumulation.
      - We delete large temporaries and run gc periodically to reduce peak usage.
      - OS-reported RSS may still not fall due to Python/NumPy allocator behavior.
    """
    nwb: Optional[Any] = None
    psth: Optional[Any] = None
    beh = None
    trials: Optional[np.ndarray] = None

    try:
        zarr_path = PSTH_DIR / f"{session}_0.2s.zarr"
        beh_csv = RESULTS_DIR / f"behavior_summary-{session}.csv"
        save_dir = OUTDIR / session  # session-level folder

        if not zarr_path.exists():
            return f"[{session}] skip: Zarr not found: {zarr_path}"
        if not beh_csv.exists():
            return f"[{session}] skip: behavior CSV not found: {beh_csv}"

        print(f"\n[{session}] models: {get_fitted_model_names(session_name=session)}")

        # Load session-level resources once per worker/session
        nwb, _ = NWBUtils.combine_nwb(session_name=session)
        psth = load_zarr(str(zarr_path))
        beh = smart_read_csv(str(beh_csv))
        trials = np.asarray(find_trials(nwb, "response"))

        save_dir.mkdir(parents=True, exist_ok=True)

        for i, (col, lname) in enumerate(zip(LATENTS, Latent_NAMES), start=1):
            if col not in beh.columns:
                print(f"[{session}] skip: {col} (missing column)")
                continue

            # Pull the latent vector out of the "single-row object column"
            vals = _as_1d_float_array(beh.at[0, col])

            try:
                plot_trial_mean_activity_vs_latent_per_unit(
                    source=psth,
                    latent_values=vals,
                    latent_trial_ids=trials,
                    activity_window=ACTIVITY_WINDOW,
                    unit_ids=None,
                    align_to_event=ALIGN_TO,
                    figsize=(5.5, 4.5),
                    dpi=300,
                    consolidated=True,
                    save_path=str(save_dir),
                    title_prefix="",
                    save_prefix=f"{col}_",          # keeps your subfolder + filename conventions
                    latent_name=lname,              # x-axis label
                    activity_name="Trial mean FR (spk/s)",
                    show=False,                     # save only in workers
                    overwrite=True,
                    fit_kind="linear",
                    show_identity=False,
                )
                print(f"[{session}] plotted scatter: {col}")
            finally:
                # Close any figures created inside the plotting function.
                plt.close("all")

                # Drop the per-latent array immediately.
                del vals

                # Periodic GC to limit peak memory in long loops.
                if (i % 10) == 0:
                    gc.collect()

        return f"[{session}] done"

    except Exception as e:
        return f"[{session}] failed: {e}"

    finally:
        # Session-level cleanup
        try:
            plt.close("all")
        except Exception:
            pass

        for name in ("trials", "beh", "psth", "nwb"):
            try:
                del locals()[name]  # best-effort; safe even if missing
            except Exception:
                pass

        gc.collect()
