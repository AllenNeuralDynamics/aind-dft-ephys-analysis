"""Top-level worker for batch opto-tagging figure generation (one SESSION per call).

Designed for ``concurrent.futures.ProcessPoolExecutor`` with a ``spawn`` context so
that **whole sessions** can be processed in parallel. Each call to :func:`run_session`
runs the full single-session pipeline end to end:

    load -> compute tagging metrics -> select tagged units -> render per-unit figures

Rendering inside a session is **sequential** (headless ``Agg`` backend). This avoids
nesting a second process pool inside each session worker. If instead you want to
process ONE session with per-unit parallelism, use ``opto_tagging_figure_worker``
(``init_worker`` / ``render_unit``) directly.

``run_session`` is a module-level function (not a closure) so it is importable and
picklable by ``spawn`` workers.
"""
import os
import sys
import glob
from ast import literal_eval
from pathlib import Path

MODULE_PATH = Path("/root/capsule/src/aind_dft_ephys_analysis")
if str(MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH))


def resolve_session_paths(session_date, data_root="/root/capsule/data"):
    """Resolve the behavior JSON and sorted ephys NWB paths for a session_date.

    Returns ``(behavior_json_file, ephys_nwb_file)``. Raises ``FileNotFoundError``
    if either cannot be located (the sorted folder carries an extra timestamp, so
    the NWB is found by glob).

    The sorted asset may carry a plain ``_sorted_`` suffix OR a variant such as
    ``_sorted-opto_`` / ``_sorted-bandpass_``; all are matched. When both a plain
    ``_sorted_`` and a variant exist, the plain one is preferred. A few
    multi-recording sessions store units in a recording other than recording1;
    those are handled via ``EPHYS_RECORDING_OVERRIDES`` so the correct file is
    returned.
    """
    behavior_json_file = (
        f"{data_root}/ecephys_{session_date}/behavior/{session_date}.json"
    )
    if not os.path.exists(behavior_json_file):
        raise FileNotFoundError(f"behavior JSON not found: {behavior_json_file}")

    # Which recording holds the units (default recording1; a few multi-recording
    # sessions put the units table in a different recording).
    from nwb_utils import EPHYS_RECORDING_OVERRIDES
    recording_token = EPHYS_RECORDING_OVERRIDES.get(
        session_date, "experiment1_recording1")

    # Match "_sorted_", "_sorted-opto_", "_sorted-bandpass_", etc.
    matches = sorted(glob.glob(
        f"{data_root}/ecephys_{session_date}_sorted*/nwb/"
        f"ecephys_{session_date}_{recording_token}.nwb"
    ))
    if not matches:
        raise FileNotFoundError(
            f"no sorted NWB ({recording_token}) for session {session_date} "
            f"under {data_root}")
    # Prefer a plain "_sorted_" asset over a "_sorted-<variant>_" one.
    plain = [m for m in matches if "_sorted_" in m]
    ephys_nwb_file = plain[0] if plain else matches[0]
    return behavior_json_file, ephys_nwb_file


def _as_tuple(c):
    """Normalise a condition entry (possibly a CSV-round-tripped string) to a tuple."""
    if isinstance(c, str):
        try:
            c = literal_eval(c)
        except (ValueError, SyntaxError):
            return None
    return tuple(c) if isinstance(c, (tuple, list)) else None


def _normalize_targets(target_cond):
    """Accept either a single 6-tuple condition or a list of candidate 6-tuples.

    Returns a list of tuples. A session matches if it satisfies ANY candidate,
    so the same config works across sessions that use different powers (e.g.
    5.0 mW in one session, 10.0 mW in another).
    """
    if len(target_cond) and isinstance(target_cond[0], (tuple, list)):
        return [tuple(t) for t in target_cond]
    return [tuple(target_cond)]


def select_common_units(metrics_df, tagged_df, qc_index, target_cond,
                        select_pulse_index=-1, ignore_laser_idx=2):
    """Replicate the notebook's selection: tagged units matching ``target_cond``
    (ignoring ``laser_name`` at ``ignore_laser_idx``) at ``select_pulse_index``,
    intersected with QC-passing units.

    ``target_cond`` may be a single 6-tuple or a list of candidate 6-tuples; a
    condition matches if it satisfies ANY candidate (so one config can cover
    sessions with different powers).

    Returns ``(common_units, target_exists, available_powers, matched_targets)``.
    """
    match_idx = [i for i in range(6) if i != ignore_laser_idx]
    targets = _normalize_targets(target_cond)

    def match_any(c):
        return any(all(c[i] == t[i] for i in match_idx) for t in targets)

    unique_conds = {t for t in (_as_tuple(c) for c in metrics_df["condition"].unique()) if t}
    target_exists = any(match_any(c) for c in unique_conds)
    available_powers = sorted({c[0] for c in unique_conds})
    matched_targets = sorted({t for t in targets
                              for c in unique_conds
                              if all(c[i] == t[i] for i in match_idx)})

    def matches(c):
        c = _as_tuple(c)
        if c is None or len(c) != 6:
            return False
        return match_any(c)

    cond_mask = tagged_df["condition"].apply(matches)
    pulse_mask = tagged_df["pulse_index"] == select_pulse_index
    sigrows = tagged_df[cond_mask & pulse_mask & (tagged_df["tagged"] == True)]  # noqa: E712
    common_units = sorted(set(qc_index) & set(sigrows["unit_id"]))
    return common_units, target_exists, available_powers, matched_targets


def run_session(cfg):
    """Run the full pulse/train figure pipeline for a SINGLE session.

    Parameters
    ----------
    cfg : dict
        Required keys:
          - ``session_date``        : str
        Optional keys (sensible defaults applied):
          - ``data_root``           : str  (default "/root/capsule/data")
          - ``behavior_json_file``  : str  (else resolved from session_date)
          - ``ephys_nwb_file``      : str  (else resolved from session_date)
          - ``metrics_csv_path``    : str  (else scratch/opto_tagging/metrics_<date>.csv)
          - ``save_metrics``        : bool (default True)
          - ``compute_kwargs``      : dict passed to compute_tagging_metrics
          - ``select_kwargs``       : dict passed to select_tagged_units
          - ``target_cond``         : 6-tuple condition to select on
          - ``select_pulse_index``  : int  (default -1)
          - ``ignore_laser_idx``    : int  (default 2)
          - ``alignments``          : dict name -> {"out_dir": str, "plot_kwargs": dict}
          - ``save_formats``        : list (default ["png"])

    Returns
    -------
    dict
        Summary with session, n_units, per-alignment saved/failed counts, status
        and any error message. Never raises (errors are captured in the result).
    """
    import matplotlib
    matplotlib.use("Agg")  # headless; this is a fresh process in session-parallel mode
    import time

    session_date = cfg["session_date"]
    _progress = cfg.get("progress", True)

    def _log(msg):
        if _progress:
            print(f"[{session_date}] {msg}", flush=True)

    _t_start = time.perf_counter()
    result = {"session_date": session_date, "status": "ok", "error": None,
              "n_units": 0, "common_units": [], "target_exists": None,
              "available_powers": None, "matched_targets": None,
              "alignments": {}}
    try:
        from optical_tagging import OpticalTagging, select_tagged_units

        data_root = cfg.get("data_root", "/root/capsule/data")
        beh = cfg.get("behavior_json_file")
        nwb = cfg.get("ephys_nwb_file")
        if beh is None or nwb is None:
            beh, nwb = resolve_session_paths(session_date, data_root=data_root)

        metrics_csv_path = cfg.get(
            "metrics_csv_path",
            f"/root/capsule/scratch/opto_tagging/metrics_{session_date}.csv",
        )

        _log("loading session (OpticalTagging)...")
        _t = time.perf_counter()
        ot = OpticalTagging(behavior_json_file=beh, ephys_nwb_file=nwb)
        _log(f"loaded in {time.perf_counter() - _t:.1f}s")

        # ---- compute metrics (or reuse an existing CSV) ----
        compute_kwargs = cfg.get("compute_kwargs", {})
        skip_existing = cfg.get("skip_existing_metrics", False)
        if skip_existing and os.path.exists(metrics_csv_path):
            import pandas as pd
            metrics_df = pd.read_csv(metrics_csv_path)
            result["metrics_reused"] = True
            _log(f"reused existing metrics ({len(metrics_df)} row(s)) from "
                 f"{metrics_csv_path}; skipping recompute")
        else:
            _t = time.perf_counter()
            metrics_df = ot.compute_tagging_metrics(**compute_kwargs)
            result["metrics_reused"] = False
            _log(f"computed {len(metrics_df)} metric row(s) in "
                 f"{time.perf_counter() - _t:.1f}s")
            if cfg.get("save_metrics", True):
                os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
                metrics_df.to_csv(metrics_csv_path, index=False)
        result["metrics_csv_path"] = metrics_csv_path
        result["n_metric_rows"] = int(len(metrics_df))

        # ---- select tagged units ----
        select_kwargs = cfg.get("select_kwargs", {})
        tagged_df = select_tagged_units(metrics_df, **select_kwargs)

        target_cond = cfg["target_cond"]
        common_units, target_exists, available_powers, matched_targets = select_common_units(
            metrics_df, tagged_df, ot.units_passing_qc.index, target_cond,
            select_pulse_index=cfg.get("select_pulse_index", -1),
            ignore_laser_idx=cfg.get("ignore_laser_idx", 2),
        )
        result["common_units"] = common_units
        result["n_units"] = len(common_units)
        result["target_exists"] = bool(target_exists)
        result["available_powers"] = available_powers
        result["matched_targets"] = matched_targets
        _log(f"selected {len(common_units)} tagged unit(s) "
             f"(target_exists={bool(target_exists)})")

        if not target_exists:
            result["status"] = "no_condition"
        if not common_units:
            # still report; nothing to render
            if result["status"] == "ok":
                result["status"] = "no_units"
            return result

        # ---- render figures, sequentially, per alignment ----
        save_formats = cfg.get("save_formats", ["png"])
        alignments = cfg.get("alignments", {})
        n_cu = len(common_units)
        for name, spec in alignments.items():
            out_dir = spec["out_dir"]
            plot_kwargs = dict(spec.get("plot_kwargs", {}))
            # Inject this session's own metrics_df where requested (so the pulse
            # panels can be annotated); the sentinel keeps cfg picklable.
            for _k, _v in list(plot_kwargs.items()):
                if isinstance(_v, str) and _v == "__SELF_METRICS__":
                    plot_kwargs[_k] = metrics_df
            os.makedirs(out_dir, exist_ok=True)
            _log(f"rendering {n_cu} '{name}' figure(s) -> {out_dir}")
            _t = time.perf_counter()
            _rstep = max(1, n_cu // 10)
            saved, failed, errors = 0, 0, []
            for j, unit_id in enumerate(common_units):
                try:
                    ot.plot_raster_graph(
                        unit_index=int(unit_id),
                        save_path=os.path.join(out_dir, f"unit_{unit_id}"),
                        save_formats=save_formats,
                        **plot_kwargs,
                    )
                    saved += 1
                except Exception as e:  # noqa: BLE001 - per-unit failures are non-fatal
                    failed += 1
                    errors.append((unit_id, repr(e)))
                if _progress and ((j + 1) % _rstep == 0 or (j + 1) == n_cu):
                    _log(f"  '{name}': {j + 1}/{n_cu} done "
                         f"({saved} ok, {failed} failed) | "
                         f"{time.perf_counter() - _t:.1f}s")
            result["alignments"][name] = {
                "out_dir": out_dir, "saved": saved, "failed": failed,
                "errors": errors,
            }
        _log(f"finished in {time.perf_counter() - _t_start:.1f}s")
        return result

    except Exception as e:  # noqa: BLE001 - never kill the pool on one session
        result["status"] = "error"
        result["error"] = repr(e)
        _log(f"ERROR after {time.perf_counter() - _t_start:.1f}s: {e!r}")
        return result
