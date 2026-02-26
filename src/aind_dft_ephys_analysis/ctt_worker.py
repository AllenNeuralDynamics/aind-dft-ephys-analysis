
from __future__ import annotations

import gc
import traceback
from pathlib import Path
from typing import Dict, Any, List

from nwb_utils import NWBUtils
from model_fitting import fit_compare_to_threshold_model_different_learning_rate

RESET_MODE = "threshold"


def make_model_name(cfg: Dict[str, Any]) -> str:
    reset_tag = f"reset{int(cfg['reset_on_switch'])}"
    stay_tag = f"stayB{int(cfg['include_stay_bias'])}"
    side_tag = f"sideB{int(cfg['include_side_bias'])}"
    sepLR_tag = f"sepLR{int(cfg['fit_separate_learning_rates'])}"
    adapt_tag = f"adaptTh{int(cfg['adaptive_threshold'])}"
    tie_tag = f"tieThLR{int(cfg['tie_threshold_lrs_to_value_lrs'])}"
    sepEta_tag = (
        f"sepEta{int(cfg['fit_separate_threshold_learning_rates'])}"
        if cfg["adaptive_threshold"]
        else "sepEtaNA"
    )
    return f"CTT_threshold_{reset_tag}_{stay_tag}_{side_tag}_{sepLR_tag}_{adapt_tag}_{sepEta_tag}_{tie_tag}"


def run_one_file_process(nwb_path_str: str, out_root_str: str, param_grid: List[Dict[str, Any]]) -> Dict[str, Any]:
    nwb_path = Path(nwb_path_str)
    out_root = Path(out_root_str)

    animal_id = nwb_path.parent.name
    file_key = f"{animal_id}:{nwb_path.name}"

    out_dir = out_root / str(animal_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    status: Dict[str, Any] = dict(
        file_key=file_key,
        file=str(nwb_path),
        animal_id=animal_id,
        ok=True,
        errors=[],
        n_attempted=0,
        n_skipped_existing=0,
        n_fit_ok=0,
        n_models_completed=0,
    )

    nwb_data = None
    try:
        nwb_data = NWBUtils.read_behavior_nwb(nwb_full_path=str(nwb_path))
    except Exception as e:
        status["ok"] = False
        status["errors"].append(
            {"stage": "read_nwb", "error": repr(e), "traceback": traceback.format_exc()}
        )
        status["n_models_completed"] = len(param_grid)
        return status

    try:
        for cfg in param_grid:
            model_name = make_model_name(cfg)
            status["n_attempted"] += 1

            try:
                fit_out = fit_compare_to_threshold_model_different_learning_rate(
                    nwb_data,
                    model_name=model_name,
                    reset_on_switch=cfg["reset_on_switch"],
                    include_stay_bias=cfg["include_stay_bias"],
                    include_side_bias=cfg["include_side_bias"],
                    reset_mode=RESET_MODE,
                    reset_value_fixed=0.0,
                    fit_separate_learning_rates=cfg["fit_separate_learning_rates"],
                    adaptive_threshold=cfg["adaptive_threshold"],
                    fit_separate_threshold_learning_rates=cfg["fit_separate_threshold_learning_rates"],
                    tie_threshold_lrs_to_value_lrs=cfg["tie_threshold_lrs_to_value_lrs"],
                    save_results=True,
                    save_folder=out_dir,
                    overwrite=False,
                )

                if fit_out is None:
                    status["n_skipped_existing"] += 1
                else:
                    status["n_fit_ok"] += 1
                    del fit_out

            except Exception as e:
                status["ok"] = False
                status["errors"].append(
                    {
                        "stage": "fit_exception",
                        "cfg": cfg,
                        "model_name": model_name,
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                    }
                )

            status["n_models_completed"] += 1

        return status

    finally:
        if nwb_data is not None:
            if hasattr(nwb_data, "close") and callable(getattr(nwb_data, "close")):
                try:
                    nwb_data.close()
                except Exception:
                    pass
            del nwb_data
        gc.collect()
