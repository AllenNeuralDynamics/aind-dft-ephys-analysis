"""
Append optotagging results to the NWB ``units`` table and select tagged units.

This module takes the per-probe laser-response metric DataFrames produced by
:class:`optotagging_Anna_nwb.OptotaggingAnalysisNWB` (one row per QC unit, keyed
by the NWB unit index in the ``unit_id`` column) and writes the results back onto
every unit of the NWB ``units`` table under an ``opto_tagging_Anna`` namespace:

- ``opto_tagging_Anna_tagged``   (bool) : whether the unit passed the tagging criteria
- ``opto_tagging_Anna_tag_type`` (str)  : ``''`` / ``'external_red'`` / ``'external_blue'`` / ...
- ``opto_tagging_Anna_criteria`` (str)  : the exact query used to tag the unit
- ``opto_tagging_Anna``          (str)  : JSON blob with every metric value (the CSV row)

Units that were not analyzed (failed QC, or on a probe with no stimulation) get
``tagged=False``, empty strings, and an empty JSON blob.

Two persistence paths are provided:

1. :func:`append_opto_tagging_columns` — adds the columns to the **in-memory**
   ``nwb_data.units`` table so the rest of the session can query them. Lightweight.
2. :func:`export_nwb_with_opto_tagging` — writes a **new NWB copy** with the columns
   baked in (use when you need the results persisted on disk; copies the whole file).
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PREFIX = "opto_tagging_Anna"

# Column type for the values returned by ``build_opto_tagging_columns``:
# name -> (data list of length n_units, description string)
Columns = Dict[str, Tuple[List[Any], str]]

METRICS_CSV_SUFFIX = "_laser_response_metrics.csv"


# ---------------------------------------------------------------------------
# Load pre-generated metric CSVs from a folder
# ---------------------------------------------------------------------------
def load_metrics_from_csv(
    csv_folder: str,
    session: str,
    suffix: str = METRICS_CSV_SUFFIX,
) -> Dict[str, pd.DataFrame]:
    """
    Load the per-probe ``*_laser_response_metrics.csv`` files for one session.

    Matches files named ``{session}_{probe}{suffix}`` (the naming written in
    Step 4) and returns ``{probe: metrics DataFrame}``. The probe label is
    recovered from the portion of the filename between the session prefix and
    the suffix.
    """
    import glob
    import os

    all_metrics: Dict[str, pd.DataFrame] = {}
    pattern = os.path.join(csv_folder, f"{session}_*{suffix}")
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        probe = name[len(session) + 1 : -len(suffix)]
        all_metrics[probe] = pd.read_csv(path)
    return all_metrics


def infer_trial_types(
    all_metrics: Dict[str, pd.DataFrame],
    token: str = "_train_max_num_sig_pulses",
) -> List[str]:
    """Infer the trial/emission types present from the metric column names."""
    types = set()
    for df in all_metrics.values():
        for col in df.columns:
            if col.endswith(token):
                types.add(col[: -len(token)])
    return sorted(types)


def append_opto_tagging_from_csv(
    nwb_data: Any,
    csv_folder: str,
    session: str,
    trial_types: Optional[Iterable[str]] = None,
    red_min_sig_pulses: int = 4,
    blue_min_sig_pulses: int = 5,
    max_jitter: float = 0.006,
    max_isi: float = 0.5,
    prefix: str = PREFIX,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end: read the session's pre-generated metric CSVs from ``csv_folder``
    and append the opto-tagging results onto ``nwb_data``'s units table.

    This is the standalone workflow: metrics were computed once and saved to CSV
    (Step 4); later you load the NWB and point this at the CSV folder. Unit rows
    are matched by the ``unit_id`` column (the NWB unit index), so the same NWB
    the metrics were computed from must be used.

    Returns a dict with ``all_metrics``, ``trial_types``, ``criteria_by_type``,
    ``tag_by_probe``, ``columns`` and ``added`` (the column names written).
    """
    all_metrics = load_metrics_from_csv(csv_folder, session)
    if not all_metrics:
        raise FileNotFoundError(
            f"No '{session}_*{METRICS_CSV_SUFFIX}' files found in {csv_folder}."
        )
    trial_types = list(trial_types) if trial_types is not None else infer_trial_types(all_metrics)

    tag_by_probe, criteria_by_type = assign_opto_tags(
        all_metrics,
        trial_types,
        red_min_sig_pulses=red_min_sig_pulses,
        blue_min_sig_pulses=blue_min_sig_pulses,
        max_jitter=max_jitter,
        max_isi=max_isi,
    )
    columns = build_opto_tagging_columns(nwb_data, all_metrics, tag_by_probe, prefix=prefix)
    added = append_opto_tagging_columns(nwb_data, columns, overwrite=overwrite)

    return {
        "all_metrics": all_metrics,
        "trial_types": trial_types,
        "criteria_by_type": criteria_by_type,
        "tag_by_probe": tag_by_probe,
        "columns": columns,
        "added": added,
    }


# ---------------------------------------------------------------------------
# Tag assignment (mirrors the selection logic in the notebook / Anna's main.py)
# ---------------------------------------------------------------------------
def _tag_conditions(
    metrics: pd.DataFrame,
    trial_type: str,
    min_sig_pulses: int,
    max_jitter: float,
    max_isi: float,
) -> List[str]:
    """Build the list of ``DataFrame.query`` conditions that apply to this table."""
    conds: List[str] = []
    if f"{trial_type}_train_max_num_sig_pulses" in metrics.columns:
        conds.append(f"{trial_type}_train_max_num_sig_pulses >= {min_sig_pulses}")
    if f"{trial_type}_train_best_mean_jitter" in metrics.columns:
        conds.append(f"{trial_type}_train_best_mean_jitter < {max_jitter}")
    if "pre_stim_isi_ratio" in metrics.columns:
        conds.append(f"pre_stim_isi_ratio < {max_isi}")
    return conds


def assign_opto_tags(
    all_metrics: Dict[str, pd.DataFrame],
    trial_types: Iterable[str],
    red_min_sig_pulses: int = 4,
    blue_min_sig_pulses: int = 5,
    max_jitter: float = 0.006,
    max_isi: float = 0.5,
) -> Tuple[Dict[str, Dict[int, Tuple[str, str]]], Dict[str, str]]:
    """
    Assign a tag to each QC unit, mirroring Anna's ``main.py`` selection.

    Red types are evaluated first (>= ``red_min_sig_pulses`` significant pulses);
    blue types require ``blue_min_sig_pulses`` significant pulses and exclude any
    unit already tagged red.

    Returns
    -------
    tag_by_probe : dict
        ``{probe: {unit_index: (tag_type, criteria_str)}}`` for tagged units only.
    criteria_by_type : dict
        ``{trial_type: criteria_str}`` describing the query applied per type.
    """
    trial_types = list(trial_types)
    red_types = [t for t in trial_types if "red" in t]
    blue_types = [t for t in trial_types if "blue" in t]

    tag_by_probe: Dict[str, Dict[int, Tuple[str, str]]] = {}
    criteria_by_type: Dict[str, str] = {}

    for probe, metrics in all_metrics.items():
        assigns: Dict[int, Tuple[str, str]] = {}
        red_index_labels = set()

        for trial_type in red_types:
            conds = _tag_conditions(metrics, trial_type, red_min_sig_pulses, max_jitter, max_isi)
            criteria = " and ".join(conds) if conds else "(no applicable criteria)"
            criteria_by_type[trial_type] = criteria
            if not conds:
                continue
            tagged = metrics.query(" and ".join(conds))
            for idx in tagged.index:
                red_index_labels.add(idx)
                unit = int(metrics.at[idx, "unit_id"])
                assigns[unit] = (trial_type, criteria)

        for trial_type in blue_types:
            conds = _tag_conditions(metrics, trial_type, blue_min_sig_pulses, max_jitter, max_isi)
            criteria = (" and ".join(conds) + " and not red-tagged") if conds else "(no applicable criteria)"
            criteria_by_type[trial_type] = criteria
            if not conds:
                continue
            tagged = metrics.query(" and ".join(conds))
            tagged = tagged[~tagged.index.isin(red_index_labels)]
            for idx in tagged.index:
                unit = int(metrics.at[idx, "unit_id"])
                assigns.setdefault(unit, (trial_type, criteria))

        tag_by_probe[probe] = assigns

    return tag_by_probe, criteria_by_type


# ---------------------------------------------------------------------------
# Build the per-unit columns
# ---------------------------------------------------------------------------
def _native(value: Any) -> Any:
    """Convert numpy / pandas scalars to JSON-serialisable Python types (NaN -> None)."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def _n_units(nwb_data: Any) -> int:
    return len(nwb_data.units.id.data)


def build_opto_tagging_columns(
    nwb_data: Any,
    all_metrics: Dict[str, pd.DataFrame],
    tag_by_probe: Dict[str, Dict[int, Tuple[str, str]]],
    prefix: str = PREFIX,
) -> Columns:
    """
    Build full-length (one entry per NWB unit) columns encoding the opto results.

    Parameters
    ----------
    nwb_data : NWB file handle (used only for the total unit count).
    all_metrics : ``{probe: metrics DataFrame}`` with a ``unit_id`` = NWB unit index.
    tag_by_probe : output of :func:`assign_opto_tags`.
    prefix : namespace for the new columns.
    """
    n = _n_units(nwb_data)
    tagged = [False] * n
    tag_type = [""] * n
    criteria = [""] * n
    info = [""] * n

    # metric blobs (the CSV row for every analyzed unit)
    for probe, metrics in all_metrics.items():
        for _, row in metrics.iterrows():
            unit = int(row["unit_id"])
            if not 0 <= unit < n:
                continue
            blob = {k: _native(v) for k, v in row.items() if k != "unit_id"}
            blob["probe"] = str(probe)
            info[unit] = json.dumps(blob)

    # tag flags
    for probe, assigns in tag_by_probe.items():
        for unit, (ttype, crit) in assigns.items():
            unit = int(unit)
            if not 0 <= unit < n:
                continue
            tagged[unit] = True
            tag_type[unit] = str(ttype)
            criteria[unit] = str(crit)

    return {
        f"{prefix}_tagged": (tagged, "Optotagging (Anna pipeline): True if the unit passed the tagging criteria."),
        f"{prefix}_tag_type": (tag_type, "Optotagging tag type, e.g. 'external_red'/'external_blue' ('' if untagged)."),
        f"{prefix}_criteria": (criteria, "Exact selection query used to tag the unit ('' if untagged)."),
        prefix: (info, "JSON blob of all laser-response metrics for the unit ('' if not analyzed)."),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def append_opto_tagging_columns(
    nwb_data: Any,
    columns: Columns,
    overwrite: bool = False,
) -> List[str]:
    """
    Add ``columns`` to the **in-memory** ``nwb_data.units`` table.

    A DynamicTable read from disk cannot drop an existing column, so if a column
    already exists this skips it (unless ``overwrite`` is True and the backing
    array is writable). Re-read the NWB to get a clean table before re-appending.
    """
    units = nwb_data.units
    added: List[str] = []
    for name, (data, desc) in columns.items():
        if name in units.colnames:
            if overwrite:
                try:
                    units[name].data[:] = data
                    added.append(name)
                    continue
                except Exception as exc:  # noqa: BLE001 - report and skip
                    print(f"Could not overwrite existing column '{name}' ({exc}); re-read the NWB.")
                    continue
            print(f"Column '{name}' already exists — skipping (re-read the NWB to refresh).")
            continue
        units.add_column(name=name, description=desc, data=list(data))
        added.append(name)
    return added


def get_nwb_source_path(nwb_data: Any) -> Optional[str]:
    """Best-effort path of the on-disk NWB backing ``nwb_data`` (or None)."""
    io = getattr(nwb_data, "io", None)
    return getattr(io, "path", None) if io is not None else None


def export_nwb_with_opto_tagging(
    source_nwb_path: str,
    out_path: str,
    columns: Columns,
    overwrite: bool = False,
) -> str:
    """
    Write a **new NWB copy** at ``out_path`` with the opto-tagging columns added.

    Uses the hdmf-zarr export mechanism (the supported way to add columns to an
    already-written table). This copies the entire NWB, so it can be large/slow.
    """
    import os

    from hdmf_zarr import NWBZarrIO

    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass overwrite=True to replace it.")

    with NWBZarrIO(source_nwb_path, mode="r") as read_io:
        nwbfile = read_io.read()
        for name, (data, desc) in columns.items():
            if name in nwbfile.units.colnames:
                print(f"Column '{name}' already present in source NWB — skipping.")
                continue
            nwbfile.units.add_column(name=name, description=desc, data=list(data))
        with NWBZarrIO(out_path, mode="w") as export_io:
            export_io.export(src_io=read_io, nwbfile=nwbfile, write_args={"link_data": False})
    print(f"Exported NWB with opto-tagging columns -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------
def select_tagged_units(
    nwb_data: Any,
    tag_type: Optional[str] = None,
    prefix: str = PREFIX,
) -> np.ndarray:
    """
    Return the NWB unit indices tagged by the opto pipeline.

    Parameters
    ----------
    tag_type : optional
        Restrict to a single tag type (e.g. ``'external_red'``). If None, returns
        all tagged units regardless of type.
    """
    units = nwb_data.units
    col = f"{prefix}_tagged"
    if col not in units.colnames:
        raise KeyError(
            f"'{col}' not found in units table. Run append_opto_tagging_columns "
            "(or read an NWB exported with the opto-tagging columns) first."
        )
    tagged = np.asarray(units[col].data, dtype=bool)
    mask = tagged.copy()
    if tag_type is not None:
        ttypes = np.asarray(units[f"{prefix}_tag_type"].data).astype(str)
        mask &= ttypes == tag_type
    return np.nonzero(mask)[0]


def get_opto_tagging_table(
    nwb_data: Any,
    prefix: str = PREFIX,
    tagged_only: bool = True,
) -> pd.DataFrame:
    """
    Return a tidy DataFrame of opto-tagging results (one row per unit).

    The JSON metric blob is expanded into individual columns. Set
    ``tagged_only=False`` to include every analyzed unit.
    """
    units = nwb_data.units
    col = f"{prefix}_tagged"
    if col not in units.colnames:
        raise KeyError(
            f"'{col}' not found in units table. Run append_opto_tagging_columns first."
        )
    tagged = np.asarray(units[col].data, dtype=bool)
    ttypes = np.asarray(units[f"{prefix}_tag_type"].data).astype(str)
    criteria = np.asarray(units[f"{prefix}_criteria"].data).astype(str)
    info = np.asarray(units[prefix].data).astype(str)

    rows: List[Dict[str, Any]] = []
    for unit in range(len(tagged)):
        analyzed = bool(info[unit])
        if tagged_only and not tagged[unit]:
            continue
        if not tagged_only and not analyzed and not tagged[unit]:
            continue
        record: Dict[str, Any] = {
            "unit_index": unit,
            "tagged": bool(tagged[unit]),
            "tag_type": ttypes[unit],
            "criteria": criteria[unit],
        }
        if analyzed:
            record.update(json.loads(info[unit]))
        rows.append(record)
    return pd.DataFrame(rows)
