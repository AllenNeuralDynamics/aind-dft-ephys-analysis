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
from plot_raster import plot_raster_and_quantile_psth_by_latent


# ---- Shared config (edit if needed) ----
PSTH_DIR = Path("/root/capsule/scratch/psth_results")
RESULTS_DIR = Path("/root/capsule/scratch/behavior_summary")
OUTDIR = Path("/root/capsule/scratch/raster_plot")

# ============================================================
# LATENTS and Latent_NAMES
# ============================================================

LATENTS: list[str] = []
Latent_NAMES: list[str] = []

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
TIME_WIN = (-3, 4)


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
        save_dir = OUTDIR / session

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
            raster_colormap = (col == "QLearning_L2F1_softmax-reward")

            if col not in beh.columns:
                print(f"[{session}] skip: {col} (missing column)")
                continue

            # Pull the latent vector out of the "single-row object column"
            # Convert to compact float array to avoid pandas object retention.
            vals = _as_1d_float_array(beh.at[0, col])

            try:
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
                    overwrite=False,
                )
                print(f"[{session}] plotted: {col}")
            finally:
                # Critical: close any figures created inside the plotting function.
                # If the plotting function already closes figures, this is harmless.
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
