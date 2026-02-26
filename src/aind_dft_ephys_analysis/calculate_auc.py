from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import re
import numpy as np
import pandas as pd
import xarray as xr

from ephys_dimension_reduction_CD import (
    extract_trial_unit_rates,
    _ids_to_indices,
    _roc_auc_binary,
)


def per_unit_auc_from_psth(
    psth_da: xr.Dataset,
    trial_ids_typeA: Union[np.ndarray, List[int]],
    trial_ids_typeB: Union[np.ndarray, List[int]],
    *,
    align: str = "go_cue",
    time_window: Tuple[float, float] = (0.0, 0.5),
    n_trials_per_class: Optional[int] = None,
    n_repeats: int = 1000,
    min_trials_per_class: int = 5,
    random_state: int = 0,
    unit_ids: Optional[Union[np.ndarray, List[int]]] = None,
    metadata: Optional[pd.DataFrame] = None,
    sorted_session_name: Optional[str] = None,
    metadata_unit_col: str = "unit_index",
    save_path: Optional[Union[str, Path]] = None,
    save_format: str = "zarr",
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Compute per-unit ROC-AUC between two trial sets (typeA vs typeB), with
    optional per-unit metadata attached to the output and saved file.

    For each repetition, this function randomly samples `n_trials_per_class`
    trials from the provided `trial_ids_typeA` and `trial_ids_typeB` (without
    replacement), computes the ROC-AUC for each unit, and repeats this process
    `n_repeats` times. The final output reports the mean and standard deviation
    of the AUC per unit across repetitions.

    If the minimal trial requirement (`min_trials_per_class`) is not met for
    either class, the function returns a dict with `skipped=True` and does not
    compute or save any AUC values.

    If `metadata` and `sorted_session_name` are provided, `metadata` is first
    filtered to that session via `metadata["sorted_session_name"] ==
    sorted_session_name`, then aligned to the selected `unit_ids` (or all
    units) using `metadata_unit_col`. The aligned metadata is returned as
    `out["metadata_per_unit"]` and also saved as additional variables.

    Parameters
    ----------
    psth_da : xr.Dataset
        PSTH dataset (same structure assumed as in `coding_direction_from_psth`).
        Must contain the variables and coordinates required by
        `extract_trial_unit_rates`.

    trial_ids_typeA : array-like of int
        Trial IDs for class A, mapped via the appropriate trial index coordinate
        for the chosen `align` (e.g. `trial_index_go_cue`).

    trial_ids_typeB : array-like of int
        Trial IDs for class B, mapped similarly to `trial_ids_typeA`.

    align : {'go_cue', 'reward_go_cue_start'}, default 'go_cue'
        Alignment/event used to select the PSTH variable and trial dimension.

    time_window : tuple of float, default (0.0, 0.5)
        Time window [t0, t1) in seconds (relative to `align`) over which to
        average the PSTH for AUC computation.

    n_trials_per_class : int or None, optional
        Number of trials to sample for each class per repetition.
        - If None, use as many trials as possible: min(n_typeA_total, n_typeB_total).
        - If not None, the actual number used is
          `min(n_trials_per_class, n_typeA_total, n_typeB_total)`.

    n_repeats : int, default 1000
        Number of random resampling repetitions used to estimate AUC
        variability.

    min_trials_per_class : int, default 5
        Minimal number of AVAILABLE trials required for each class before
        any subsampling. If either class has fewer trials than this, the
        function returns `skipped=True`.

    random_state : int, default 0
        Seed for the random number generator used for subsampling trials.

    unit_ids : array-like of int or None, optional
        Optional subset of units, specified in terms of `psth_da["unit_index"]`.
        If provided, only these units are used. If None, all units are used.

    metadata : pandas.DataFrame or None, optional
        Per-unit metadata table. If provided, it must contain at least:
        - Column `metadata_unit_col` (default `"unit_index"`) with unit IDs
          that match `psth_da["unit_index"]`.
        - If `sorted_session_name` is provided, a column `"sorted_session_name"`
          is also required and will be used to subset metadata to that session.

    sorted_session_name : str or None, optional
        Session identifier used to filter `metadata`:
        `metadata["sorted_session_name"] == sorted_session_name`.
        If None, no session-based filtering is applied to `metadata`.

    metadata_unit_col : str, default "unit_index"
        Column name in `metadata` containing unit IDs that match
        `psth_da["unit_index"]` (or the subset specified by `unit_ids`).

    save_path : str or Path or None, default None
        Optional path where results will be saved. If None, nothing is written
        to disk and only the dictionary is returned.

    save_format : {'npz', 'nc', 'zarr'}, default 'zarr'
        File format used when `save_path` is not None:
        - 'npz'  : compressed NumPy archive
        - 'nc'   : NetCDF via xarray
        - 'zarr' : Zarr store via xarray

    overwrite : bool, default True
        If True and `save_format='zarr'`, an existing Zarr store at `save_path`
        will be removed before writing.

    Returns
    -------
    out : dict
        If minimal trial requirement is not met:
            {
                "skipped": True,
                "reason": str,
                "n_typeA_total": int,
                "n_typeB_total": int,
            }

        Otherwise:
            {
                "skipped": False,
                "auc_mean": np.ndarray (N_units,),
                "auc_std": np.ndarray (N_units,),
                "auc_repeats": np.ndarray (n_repeats, N_units),
                "unit_ids": np.ndarray (N_units,),
                "metadata_per_unit": pandas.DataFrame (optional, aligned),
                "align": str,
                "time_window": (float, float),
                "n_typeA_total": int,
                "n_typeB_total": int,
                "n_trials_per_class_used": int,
                "n_repeats": int,
                "min_trials_per_class": int,
                "random_state": int,
                "saved_to": str (optional),
                "saved_format": str (optional),
            }
    """
    # ------------------------------------------------------------------
    # 0) Filter and prepare metadata (if provided)
    # ------------------------------------------------------------------
    metadata_filtered: Optional[pd.DataFrame] = None
    if metadata is not None:
        if sorted_session_name is not None:
            if "sorted_session_name" not in metadata.columns:
                raise KeyError("metadata must contain column 'sorted_session_name'")
            metadata_filtered = metadata[
                metadata["sorted_session_name"] == sorted_session_name
            ].copy()
        else:
            metadata_filtered = metadata.copy()

        if metadata_unit_col not in metadata_filtered.columns:
            raise KeyError(
                f"metadata_unit_col='{metadata_unit_col}' not found in metadata columns"
            )

    # ------------------------------------------------------------------
    # 1) Extract trial × unit matrix in the specified time window
    # ------------------------------------------------------------------
    fit_ext = extract_trial_unit_rates(
        psth_da=psth_da,
        align=align,
        time_window=time_window,
        zscore_units=False,
        unit_ids=unit_ids,
    )
    R = fit_ext["R"]  # (T_full × N_units)
    trial_ids_full = fit_ext["trial_ids"]
    unit_ids_selected = fit_ext["unit_ids"]

    # ------------------------------------------------------------------
    # 1b) Align metadata to unit_ids_selected (if available)
    # ------------------------------------------------------------------
    metadata_aligned: Optional[pd.DataFrame] = None
    if metadata_filtered is not None:
        meta_tmp = (
            metadata_filtered
            .drop_duplicates(subset=[metadata_unit_col])
            .set_index(metadata_unit_col)
        )
        metadata_aligned = meta_tmp.reindex(unit_ids_selected)
        metadata_aligned.index.name = metadata_unit_col

    # ------------------------------------------------------------------
    # 2) Map trial IDs to indices, check trial counts
    # ------------------------------------------------------------------
    idx_a_all = _ids_to_indices(
        trial_ids_full, np.asarray(trial_ids_typeA), require_all=False
    )
    idx_b_all = _ids_to_indices(
        trial_ids_full, np.asarray(trial_ids_typeB), require_all=False
    )

    nA_total = len(idx_a_all)
    nB_total = len(idx_b_all)

    if nA_total < min_trials_per_class or nB_total < min_trials_per_class:
        return {
            "skipped": True,
            "reason": (
                "Not enough trials for one or both classes: "
                f"n_typeA_total={nA_total}, n_typeB_total={nB_total}, "
                f"min_trials_per_class={min_trials_per_class}"
            ),
            "n_typeA_total": int(nA_total),
            "n_typeB_total": int(nB_total),
        }

    # Determine number of trials used per class in each repetition
    if n_trials_per_class is None:
        n_use = min(nA_total, nB_total)
    else:
        n_use = int(min(n_trials_per_class, nA_total, nB_total))

    if n_use < min_trials_per_class:
        return {
            "skipped": True,
            "reason": (
                "After applying n_trials_per_class, the usable trial number "
                "is below the minimal requirement: "
                f"n_use={n_use}, min_trials_per_class={min_trials_per_class}, "
                f"n_typeA_total={nA_total}, n_typeB_total={nB_total}"
            ),
            "n_typeA_total": int(nA_total),
            "n_typeB_total": int(nB_total),
        }

    # ------------------------------------------------------------------
    # 3) Random resampling and per-unit AUC computation
    # ------------------------------------------------------------------
    rng = np.random.RandomState(int(random_state))
    n_units = R.shape[1]
    auc_repeats = np.empty((n_repeats, n_units), dtype=float)

    # Labels: first A (positive), then B (negative)
    labels_pm1 = np.concatenate(
        [np.ones(n_use, dtype=float), -np.ones(n_use, dtype=float)]
    )

    for rep in range(n_repeats):
        # Sample trial indices without replacement
        idx_a = rng.choice(idx_a_all, size=n_use, replace=False)
        idx_b = rng.choice(idx_b_all, size=n_use, replace=False)

        idx_rep = np.concatenate([idx_a, idx_b])  # (2 * n_use,)
        R_rep = R[idx_rep]  # (2 * n_use × N_units)

        # Per-unit AUC
        for u in range(n_units):
            scores_u = R_rep[:, u]
            auc_repeats[rep, u] = _roc_auc_binary(labels_pm1, scores_u)

    auc_mean = np.nanmean(auc_repeats, axis=0)
    auc_std = np.nanstd(auc_repeats, axis=0)

    out: Dict[str, Any] = {
        "skipped": False,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_repeats": auc_repeats,
        "unit_ids": unit_ids_selected,
        "align": align,
        "time_window": time_window,
        "n_typeA_total": int(nA_total),
        "n_typeB_total": int(nB_total),
        "n_trials_per_class_used": int(n_use),
        "n_repeats": int(n_repeats),
        "min_trials_per_class": int(min_trials_per_class),
        "random_state": int(random_state),
    }

    if metadata_aligned is not None:
        out["metadata_per_unit"] = metadata_aligned

    # ------------------------------------------------------------------
    # 4) Optional save to disk (including metadata if present)
    # ------------------------------------------------------------------
    if save_path is not None and not out["skipped"]:
        path = Path(save_path)
        fmt = str(save_format).lower()
        if fmt not in {"npz", "nc", "zarr"}:
            raise ValueError("save_format must be one of {'npz','nc','zarr'}")

        attrs_payload = {
            "align": align,
            "time_window": time_window,
            "sorted_session_name": sorted_session_name,
            "n_typeA_total": int(nA_total),
            "n_typeB_total": int(nB_total),
            "n_trials_per_class_used": int(n_use),
            "n_repeats": int(n_repeats),
            "min_trials_per_class": int(min_trials_per_class),
            "random_state": int(random_state),
            "unit_ids_subset": unit_ids is not None,
            "has_metadata": metadata_aligned is not None,
            "metadata_unit_col": metadata_unit_col,
        }

        if fmt == "npz":
            save_dict: Dict[str, Any] = dict(
                auc_mean=auc_mean,
                auc_std=auc_std,
                auc_repeats=auc_repeats,
                unit_ids=np.asarray(unit_ids_selected, dtype=int),
                attrs_str=json.dumps(attrs_payload),
            )
            if metadata_aligned is not None:
                for col in metadata_aligned.columns:
                    save_dict[f"meta_{col}"] = metadata_aligned[col].to_numpy()
            np.savez_compressed(path, **save_dict)

        else:
            # Base AUC variables
            data_vars: Dict[str, Any] = {
                "auc_mean": (("unit",), auc_mean),
                "auc_std": (("unit",), auc_std),
                "auc_repeats": (("repeat", "unit"), auc_repeats),
            }

            # Add metadata columns, making sure they are serializable by xarray
            if metadata_aligned is not None:
                for col in metadata_aligned.columns:
                    series = metadata_aligned[col]
                    # If dtype is object (e.g., dicts), cast to string
                    if series.dtype == "O":
                        arr = series.astype(str).to_numpy()
                    else:
                        arr = series.to_numpy()
                    data_vars[f"meta_{col}"] = (("unit",), arr)

            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "unit_id": ("unit", np.asarray(unit_ids_selected, dtype=int)),
                    "repeat": ("repeat", np.arange(n_repeats, dtype=int)),
                },
                attrs=attrs_payload,
            )

            if fmt == "nc":
                ds.to_netcdf(path)
            elif fmt == "zarr":
                import shutil

                if path.exists() and overwrite:
                    shutil.rmtree(path)
                ds.to_zarr(path, mode="w")

        out["saved_to"] = str(path)
        out["saved_format"] = fmt

    return out




def combine_auc_zarr_along_unit(
    root_dir: Union[str, Path] = "/root/capsule/scratch/AUC_results",
    pattern: str = "right_choice_trials_left_choice_trials_TW_-1_0.zarr",
    save_name: str = "AUC_combined_right_choice_trials_left_choice_trials_TW_-1_0.zarr",
) -> xr.Dataset:
    """
    Find all AUC Zarr files for a given filename pattern, load them,
    attach session labels per unit, concatenate them along the `unit`
    dimension, and save a combined Zarr dataset.

    Parameters
    ----------
    root_dir : str or Path, optional
        Directory containing the individual AUC Zarr result files.

    pattern : str, optional
        Filename substring to match AUC result files. Only files whose
        names contain this substring will be collected.

    save_name : str, optional
        Name of the combined Zarr output file written inside `root_dir`.

    Returns
    -------
    xr.Dataset
        Combined xarray Dataset where all units from all matching
        sessions are concatenated along the `unit` dimension. A new
        coordinate `session` is added to indicate the source session
        for each unit.
    """
    root = Path(root_dir)
    save_path = root / save_name

    # Find all matching Zarr files
    zarr_files: List[Path] = sorted(root.rglob(f"*{pattern}"))

    print(f"Found {len(zarr_files)} matching AUC files:")
    for f in zarr_files:
        print("  -", f)

    if not zarr_files:
        raise RuntimeError(f"No files containing '{pattern}' found under {root}")

    datasets: List[xr.Dataset] = []

    # Regex to extract session name from filename
    session_regex = re.compile(r"(ecephys_[^_]+_[^_]+_[^_]+_[^_]+)")

    for zpath in zarr_files:
        print(f"\nLoading: {zpath}")
        ds = xr.open_zarr(zpath)

        # Try to get session name from attrs first
        session_attr = ds.attrs.get("sorted_session_name", None)

        if isinstance(session_attr, str) and len(session_attr) > 0:
            session_name = session_attr
        else:
            # Fallback: parse from filename
            m = session_regex.search(zpath.name)
            if m:
                session_name = m.group(1)
            else:
                session_name = (
                    zpath.name.replace("AUC_", "").replace(pattern, "").strip("_")
                )

        print(f"  → session: {session_name}")

        if "unit" not in ds.dims:
            raise KeyError(f"Dataset {zpath} does not have 'unit' dimension.")

        # Use sizes instead of dims to avoid FutureWarning
        n_unit: int = ds.sizes["unit"]

        # Assign session coordinate per unit
        session_coord = xr.DataArray(
            [session_name] * n_unit,
            dims=("unit",),
            name="session",
        )
        ds = ds.assign_coords(session=session_coord)

        datasets.append(ds)

    # Concatenate along the `unit` dimension
    print("\nConcatenating along 'unit'...")
    combined: xr.Dataset = xr.concat(datasets, dim="unit")

    print("Combined dataset variables and shapes:")
    for name, da in combined.data_vars.items():
        print(f"  {name}: {da.shape} (dtype={da.dtype})")

    # Cast object-dtype vars/coords to string for Zarr compatibility
    for name, da in list(combined.data_vars.items()):
        if da.dtype == "O":
            print(f"  Casting data_var '{name}' to str (object dtype detected)")
            combined[name] = da.astype(str)

    for cname, ca in list(combined.coords.items()):
        if ca.dtype == "O":
            print(f"  Casting coord '{cname}' to str (object dtype detected)")
            combined = combined.assign_coords({cname: ca.astype(str)})

    # Save combined dataset
    print(f"\nSaving combined dataset to: {save_path}")
    if save_path.exists():
        import shutil
        shutil.rmtree(save_path)
    combined.to_zarr(save_path, mode="w")
    print("✔ Combined file saved successfully.")

    return combined


def auto_combine_all_auc_patterns(
    root_dir: Union[str, Path] = "/root/capsule/scratch/AUC_results",
    overwrite: bool = False,
) -> Dict[str, Path]:
    """
    Automatically detect AUC filename patterns in a folder and run
    `combine_auc_zarr_along_unit` once for each pattern.

    A pattern is defined as the part of the filename *after*
    `sorted_YYYY-MM-DD_HH-MM-SS_`.

    Example
    -------
    Filename:
        AUC_ecephys_753124_2024-12-10_17-24-56_sorted_2024-12-13_09-48-25
        _right_choice_trials_left_choice_trials_TW_-1_0.zarr

    Pattern:
        right_choice_trials_left_choice_trials_TW_-1_0.zarr

    For each unique pattern, this function will call:
        combine_auc_zarr_along_unit(
            root_dir=root_dir,
            pattern=pattern,
            save_name=f"AUC_combined_{pattern_without_zarr}_by_unit.zarr",
        )

    Parameters
    ----------
    root_dir : str or Path, optional
        Directory where all AUC Zarr files are stored.
        Default: "/root/capsule/scratch/AUC_results".

    overwrite : bool, default False
        If False and a combined file for a pattern already exists,
        the pattern is skipped. If True, combined files are re-created.

    Returns
    -------
    dict
        Mapping from pattern string to the Path of the combined Zarr file.
        Patterns that were skipped due to existing files will still be
        included, pointing to the existing combined file.
    """
    root = Path(root_dir)

    # Collect all base AUC files (ignore already combined ones)
    all_files: List[Path] = sorted(
        f
        for f in root.glob("AUC_*.zarr")
        if not f.name.startswith("AUC_combined_")
    )

    if not all_files:
        raise RuntimeError(f"No AUC_ecephys Zarr files found under {root}")

    # Group by detected pattern
    pattern_to_files: Dict[str, List[Path]] = {}

    for f in all_files:
        name = f.name

        # We expect "..._sorted_YYYY-MM-DD_HH-MM-SS_<pattern>"
        if "sorted_" not in name:
            print(f"⚠️  Skip (no 'sorted_' token): {name}")
            continue

        after_sorted = name.split("sorted_", 1)[1]
        # after_sorted: "2024-12-13_09-48-25_right_choice_trials_left_choice_trials_TW_-1_0.zarr"
        tokens = after_sorted.split("_")

        if len(tokens) < 3:
            print(f"⚠️  Skip (unexpected format after 'sorted_'): {name}")
            continue

        # Remove date ("YYYY-MM-DD") and time ("HH-MM-SS") → keep the tail as pattern
        pattern = "_".join(tokens[2:])  # → "right_choice_trials_left_choice_trials_TW_-1_0.zarr"

        pattern_to_files.setdefault(pattern, []).append(f)

    if not pattern_to_files:
        raise RuntimeError("No valid patterns could be parsed from filenames.")

    print("\nDetected patterns:")
    for pat, files in pattern_to_files.items():
        print(f"  Pattern: {pat}  (n_files={len(files)})")

    combined_paths: Dict[str, Path] = {}

    # For each pattern, run combine_auc_zarr_along_unit
    for pattern, files in pattern_to_files.items():
        pattern_no_zarr = pattern[:-5] if pattern.endswith(".zarr") else pattern
        save_name = f"AUC_combined_{pattern_no_zarr}_by_unit.zarr"
        combined_path = root / save_name

        if combined_path.exists() and not overwrite:
            print(f"\nSkip pattern '{pattern}' (combined file already exists): {combined_path}")
            combined_paths[pattern] = combined_path
            continue

        print(f"\n=== Combining pattern: {pattern} ===")
        print(f"  Files ({len(files)}):")
        for f in files:
            print(f"    - {f.name}")

        ds_combined = combine_auc_zarr_along_unit(
            root_dir=root,
            pattern=pattern,
            save_name=save_name,
        )

        combined_paths[pattern] = combined_path
        print(f"  ✔ Finished combining pattern '{pattern}' → {combined_path}")

    return combined_paths
