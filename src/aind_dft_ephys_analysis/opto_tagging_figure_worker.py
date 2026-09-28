"""Top-level worker for parallel per-unit opto-tagging figure generation.

Use with ``concurrent.futures.ProcessPoolExecutor`` and a ``spawn`` context.
Each worker process builds ONE ``OpticalTagging`` object (via the initializer,
so the NWB handle is opened once per process) and then renders a figure for
every unit id it is handed. Rendering runs on the headless ``Agg`` backend.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend for worker processes

MODULE_PATH = Path("/root/capsule/src/aind_dft_ephys_analysis")
if str(MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH))

from optical_tagging import OpticalTagging

# Per-process global state populated by ``init_worker``.
_STATE = {}


def init_worker(behavior_json_file, ephys_nwb_file, out_dir, plot_kwargs, save_formats):
    """Build the OpticalTagging object once per worker process."""
    os.makedirs(out_dir, exist_ok=True)
    _STATE["ot"] = OpticalTagging(
        behavior_json_file=behavior_json_file,
        ephys_nwb_file=ephys_nwb_file,
    )
    _STATE["out_dir"] = out_dir
    _STATE["plot_kwargs"] = dict(plot_kwargs)
    _STATE["save_formats"] = list(save_formats)


def render_unit(unit_id):
    """Render and save one unit's figure. Returns (unit_id, ok, message)."""
    try:
        ot = _STATE["ot"]
        save_path = os.path.join(_STATE["out_dir"], f"unit_{unit_id}")
        ot.plot_raster_graph(
            unit_index=int(unit_id),
            save_path=save_path,
            save_formats=_STATE["save_formats"],
            **_STATE["plot_kwargs"],
        )
        return (unit_id, True, "ok")
    except Exception as e:  # noqa: BLE001 - report failures per unit, keep pool alive
        return (unit_id, False, repr(e))
