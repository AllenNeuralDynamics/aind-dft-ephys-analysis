from __future__ import annotations

# ==============================
# Standard library
# ==============================
import math
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Literal,
)

# ==============================
# Third-party
# ==============================
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ==============================
# Local imports
# ==============================
from create_psth import load_psth_raster_subset
from ephys_utils import append_units_locations
from general_utils import extract_session_name_core

def summarize_monotonic_unit_df_by_latent_quantile(
    source: Union[str, Path, xr.DataArray, xr.Dataset],
    *,
    latent_values: Sequence[float],
    latent_trial_ids: Sequence[int],
    activity_window: Tuple[float, float],
    calculation_min_window: Optional[Tuple[float, float]] = (-4.0, 3),  # NEW
    unit_ids: Optional[Sequence[int]] = None,
    align_to_event: Optional[str] = None,
    time_window: Optional[Tuple[float, float]] = None,
    n_bins: int = 5,
    binning: Literal["quantile", "equal"] = "quantile",
    bin_range: Optional[Tuple[float, float]] = None,
    quantile_stat: Literal["mean", "median"] = "mean",
    ci: Literal["sem", "iqr", "none"] = "sem",
    monotonic_tol: float = 0.0,
    dropna_latent: bool = True,
    dropna_activity: bool = True,
    activity_min_threshold: float = 0.0,
    min_average_firing_rate: float = 0.0,
    session_name: Optional[str] = None,
    latent_name: Optional[str] = None,
    unit_metadata: Optional[pd.DataFrame] = None,
    unit_key_in_metadata: str = "unit_index",
    nwb_data: Optional[Any] = None,
    consolidated: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
    save_filename: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    save_format: Literal["csv", "parquet"] = "csv",
    overwrite: bool = True,
) -> Tuple[pd.DataFrame, Optional[Path]]:
    """
    Parameters
    ----------
    source
        PSTH/raster data source. Can be:
        - str or Path: path to a zarr/npz/h5/etc supported by `load_psth_raster_subset`
        - xr.DataArray or xr.Dataset: already-loaded raster/PSTH object
        Must contain trial and time dimensions/coords compatible with `load_psth_raster_subset`.

    latent_values
        1D array-like of latent values (float) for each trial id in `latent_trial_ids`.
        Length must equal `latent_trial_ids`. These are used to bin trials (quantile or equal-width).

    latent_trial_ids
        1D array-like of trial identifiers (int) corresponding one-to-one with `latent_values`.
        Must be unique (no duplicates). Used to align latent values to trials present in `source`.

    activity_window
        Tuple (start, end) in seconds defining the time window (relative to `align_to_event`)
        over which per-trial mean activity is computed for:
        - per-bin activity summaries (q{b}_* columns)
        - monotonicity checks (RAW and _gt_thr)
        - Spearman correlation computations (RAW and _gt_thr)
        Must satisfy end > start.

    calculation_min_window
        Optional tuple (start, end) in seconds defining the time window (relative to `align_to_event`)
        used ONLY for thresholding/exclusion logic:
        (1) Applying `activity_min_threshold`: a trial is included in the *_gt_thr analyses if
            its mean activity in `calculation_min_window` is > `activity_min_threshold`.
        (2) Applying `min_average_firing_rate`: a unit is excluded entirely if its average
            activity across trials (mean of per-trial means) in `calculation_min_window` is
            < `min_average_firing_rate`.
        If None, defaults to `activity_window`.
        Must satisfy end > start.

    unit_ids
        Optional list/array of unit indices to include. If None, all units in `source` are processed.
        Unit indices must match the `unit_index` coordinate in the PSTH data returned by
        `load_psth_raster_subset`.

    align_to_event
        Optional name of the alignment event used by `load_psth_raster_subset` (e.g. "stim_onset",
        "go_cue", etc.). This defines the reference point for `time_window`, `activity_window`,
        and `calculation_min_window`. If None, uses the default alignment behavior in your loader.

    time_window
        Optional tuple (start, end) in seconds to crop the loaded PSTH/raster in time before any
        downstream computations. This is passed to `load_psth_raster_subset`.
        This controls what time samples are available, but DOES NOT by itself define which time
        range is used for activity summaries (that is `activity_window`) or thresholding
        (`calculation_min_window`).
        If provided, it must contain both `activity_window` and `calculation_min_window` ranges;
        otherwise you may get a “selects no time points” error.

    n_bins
        Requested number of latent bins. Must be >= 2.
        For quantile binning, the effective number of bins may be smaller due to ties/duplicates
        (see `duplicates="drop"` behavior); this is recorded in `n_bins_effective`.

    binning
        Binning mode for latent values:
        - "quantile": bins have (approximately) equal counts; uses `pd.qcut(..., duplicates="drop")`
        - "equal": equal-width bins over [min, max] or `bin_range` if provided
        Determines how trials are grouped into bins.

    bin_range
        Optional (lo, hi) for equal-width binning when `binning="equal"`.
        If None, uses (min(latent_values), max(latent_values)) computed after dropping NaNs (if enabled).
        Ignored when `binning="quantile"`.

    quantile_stat
        Summary statistic computed within each bin from per-trial mean activity:
        - "mean": uses np.nanmean
        - "median": uses np.nanmedian
        This statistic populates q{b}_{quantile_stat}_activity and *_gt_thr counterparts, and
        is the value sequence used for monotonicity checking.

    ci
        Uncertainty summary per bin:
        - "sem": standard error of the mean (std(ddof=1) / sqrt(n)); if n<2, sem=0.0
        - "iqr": interquartile range (25th and 75th percentiles)
        - "none": no CI columns are added
        Controls which CI columns are emitted for RAW and *_gt_thr.

    monotonic_tol
        Tolerance for the monotonicity check. Successive diffs dv are evaluated as:
        - increasing ok if dv >= -monotonic_tol
        - decreasing ok if dv <= +monotonic_tol
        Larger values allow small violations due to noise to still count as monotonic.

    dropna_latent
        If True (default), trials with NaN latent values are excluded from all computations.
        If False and NaNs exist in latent, the function raises (because NaNs cannot be binned).

    dropna_activity
        If True (default), trials with NaN per-trial mean activity in `activity_window` are excluded
        from computations (RAW and *_gt_thr summaries/correlations).
        If False, NaNs may propagate to summaries and reduce valid-bin counts.

    activity_min_threshold
        Trial exclusion threshold applied in `calculation_min_window`:
        - A trial is included in *_gt_thr analyses if trial_mean_calc > activity_min_threshold.
        This affects:
        - n_trials_used_gt_thr
        - all *_gt_thr per-bin summaries
        - is_monotonic_gt_thr / related monotonic annotation fields
        - spearman_rho_gt_thr / spearman_p_gt_thr
        Note: the activities summarized in *_gt_thr bins come from `activity_window`, but the
        inclusion decision comes from `calculation_min_window`.

    min_average_firing_rate
        Unit exclusion threshold applied in `calculation_min_window`:
        - Compute avg_firing_rate_window = mean over trials of trial_mean_calc
            (ignoring NaNs).
        - If avg_firing_rate_window < min_average_firing_rate, the unit is skipped and
            does not appear in `unit_df`.
        This changes the denominator for any downstream “fraction of units” computation
        performed on `unit_df`.

    session_name
        Optional session identifier stored in output rows.
        If None and `nwb_data` is provided, it is derived from `nwb_data.session_id` via
        `extract_session_name_core`. Otherwise it may remain None.

    latent_name
        Optional label/name of the latent variable stored in output rows.

    unit_metadata
        Optional DataFrame with per-unit metadata to attach to each row (e.g. quality metrics,
        probe info, etc.). Must include a unit identifier column named by `unit_key_in_metadata`.

    unit_key_in_metadata
        Column name in `unit_metadata` that identifies units (default "unit_index").
        This column is used to index/join metadata onto output rows.

    nwb_data
        Optional NWB object. If provided, the function will:
        - derive a session core id using `extract_session_name_core(nwb_data.session_id)`
        - call `append_units_locations(nwb_data, session_id_core)` to populate CCF fields
        - build a per-unit metadata table containing:
            * brain_region (string)
            * ccf_location (dict or structured object)
        and attach these to output rows (also merged with `unit_metadata` if provided).

    consolidated
        Passed through to `load_psth_raster_subset`. Typically indicates whether to read from a
        consolidated zarr store (faster metadata access) when applicable.

    save_dir
        Optional directory to save the output table. Used when `save_filename` is provided or when
        `save_path` is a directory-like path (no suffix) is not used.

    save_filename
        Optional output filename (with or without extension). If provided, the table is saved under
        `save_dir` (or current directory if `save_dir` is None). If no suffix is provided, one is
        added based on `save_format`.

    save_path
        Optional path controlling saving behavior:
        - If it has a file suffix, it is used directly as the output file.
        - If it has no suffix, it is treated as an output directory and an auto-generated
            filename is used (based on session/latent/threshold/bins/window and `save_format`).

    save_format
        Output file format when saving:
        - "csv": saved with `DataFrame.to_csv(index=False)`
        - "parquet": saved with `DataFrame.to_parquet(index=False)`
        Also controls how early-load works when overwrite=False and the file exists.

    overwrite
        If False and the resolved output file already exists:
        - the file is loaded from disk and returned immediately (early exit),
            skipping recomputation.
        If True, existing files are overwritten.
    """

    import math  # Local import to avoid relying on module-level imports

    # -----------------------------
    # 0) Resolve intended output path EARLY, so we can skip work if it exists.
    # -----------------------------
    def _sanitize_token(s: Optional[str]) -> str:
        if s is None:
            return "NA"
        s2 = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".", "+")) else "_" for ch in str(s))
        s2 = s2.strip("_")
        return s2 if s2 else "NA"

    thr = float(activity_min_threshold)
    w0, w1 = float(activity_window[0]), float(activity_window[1])

    # NEW: calculation window for min-threshold and min-average-FR
    if calculation_min_window is None:
        cw0, cw1 = w0, w1
    else:
        cw0, cw1 = float(calculation_min_window[0]), float(calculation_min_window[1])

    min_fr = float(min_average_firing_rate)

    saved_file: Optional[Path] = None

    if save_path is not None:
        p = Path(save_path)
        if p.suffix:
            saved_file = p
        else:
            out_dir = p
            sess_tok = _sanitize_token(session_name) if session_name else "session"
            lat_tok = _sanitize_token(latent_name) if latent_name else "latent"
            thr_tok = _sanitize_token(f"thr{thr:g}")
            saved_file = out_dir / f"{sess_tok}__{lat_tok}__{thr_tok}__bins{int(n_bins)}__win_{w0:g}_{w1:g}.{save_format}"

    elif save_filename is not None:
        out_dir = Path(save_dir) if save_dir is not None else Path(".")
        saved_file = out_dir / save_filename
        if not saved_file.suffix:
            saved_file = saved_file.with_suffix(f".{save_format}")

    elif save_dir is not None:
        out_dir = Path(save_dir)
        sess_tok = _sanitize_token(session_name) if session_name else "session"
        lat_tok = _sanitize_token(latent_name) if latent_name else "latent"
        thr_tok = _sanitize_token(f"thr{thr:g}")
        saved_file = out_dir / f"{sess_tok}__{lat_tok}__{thr_tok}__bins{int(n_bins)}__win_{w0:g}_{w1:g}.{save_format}"

    if saved_file is not None and saved_file.exists() and (not overwrite):
        if save_format == "csv":
            unit_df = pd.read_csv(saved_file)
        elif save_format == "parquet":
            unit_df = pd.read_parquet(saved_file)
        else:
            raise ValueError(f"Unsupported save_format={save_format!r} for early-load.")
        return unit_df, saved_file

    # -----------------------------
    # 1) Validate inputs
    # -----------------------------
    latent_values = np.asarray(latent_values, dtype=np.float64)
    latent_trial_ids = np.asarray(latent_trial_ids, dtype=np.int64)

    if latent_values.shape[0] != latent_trial_ids.shape[0]:
        raise ValueError("`latent_values` and `latent_trial_ids` must have the same length.")
    if latent_trial_ids.size != np.unique(latent_trial_ids).size:
        raise ValueError("`latent_trial_ids` contains duplicates; must be one-to-one.")

    if w1 <= w0:
        raise ValueError("`activity_window` must satisfy activity_window[1] > activity_window[0].")

    # NEW: validate calculation window
    if cw1 <= cw0:
        raise ValueError("`calculation_min_window` must satisfy calculation_min_window[1] > calculation_min_window[0].")

    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")

    if ci not in ("sem", "iqr", "none"):
        raise ValueError("ci must be one of {'sem','iqr','none'}.")

    if min_fr < 0.0:
        raise ValueError("min_average_firing_rate must be >= 0.")

    # -----------------------------
    # 1b) Optional: build/augment unit metadata from NWB
    # -----------------------------
    nwb_meta_df: Optional[pd.DataFrame] = None
    if nwb_data is not None:
        sess_raw = getattr(nwb_data, "session_id", None)
        sess_core = extract_session_name_core(sess_raw)
        if session_name is None:
            session_name = sess_core

        nwb_data = append_units_locations(nwb_data, sess_core)

        unit_rows = []
        n_units_total = len(nwb_data.units)
        for u in range(n_units_total):
            loc = nwb_data.units["ccf_location"][u] if "ccf_location" in nwb_data.units.colnames else None
            loc = loc or {}
            region = ""
            if isinstance(loc, dict):
                region = str(loc.get("brain_region", "")) if loc.get("brain_region", "") is not None else ""
            unit_rows.append({"unit_index": int(u), "brain_region": region, "ccf_location": loc})
        nwb_meta_df = pd.DataFrame(unit_rows).set_index("unit_index", drop=False)

    meta_df: Optional[pd.DataFrame] = None
    if unit_metadata is not None:
        if unit_key_in_metadata not in unit_metadata.columns:
            raise ValueError(f"unit_metadata must contain column {unit_key_in_metadata!r}.")
        meta_df = unit_metadata.copy().set_index(unit_key_in_metadata, drop=False)

        if nwb_meta_df is not None:
            for col in ("brain_region", "ccf_location"):
                if col not in meta_df.columns:
                    meta_df[col] = np.nan
            join_cols = ["brain_region", "ccf_location"]
            meta_df = meta_df.join(nwb_meta_df[join_cols], how="left", rsuffix="_nwb")
            for col in join_cols:
                col_nwb = f"{col}_nwb"
                meta_df[col] = meta_df[col].where(meta_df[col].notna(), meta_df[col_nwb])
                meta_df = meta_df.drop(columns=[col_nwb], errors="ignore")
    else:
        if nwb_meta_df is not None:
            meta_df = nwb_meta_df

    # -----------------------------
    # 2) Load PSTH subset
    # -----------------------------
    psth_da, _ = load_psth_raster_subset(
        source,
        trial_ids=None,
        unit_ids=unit_ids,
        align_to_event=align_to_event,
        time_window=time_window,
        consolidated=consolidated,
    )

    trial_dim = next(d for d in psth_da.dims if d.startswith("trial_"))
    trial_coord = next(c for c in psth_da.coords if c.startswith("trial_index_"))

    all_trial_ids_in_ds = psth_da.coords[trial_coord].values.astype(np.int64)
    unit_indices = psth_da.coords["unit_index"].values.astype(np.int64)
    times = np.asarray(psth_da.coords["time"].values, dtype=np.float64)

    # -----------------------------
    # 3) Align latent to trials present in dataset
    # -----------------------------
    present_mask = np.isin(latent_trial_ids, all_trial_ids_in_ds)
    if not np.any(present_mask):
        raise ValueError("None of `latent_trial_ids` are present in the dataset.")

    latent_trial_ids = latent_trial_ids[present_mask]
    latent_values = latent_values[present_mask]

    pos_idx = pd.Index(all_trial_ids_in_ds).get_indexer(latent_trial_ids)
    keep = pos_idx >= 0

    latent_trial_ids = latent_trial_ids[keep]
    latent_values = latent_values[keep]

    psth_da = psth_da.isel({trial_dim: pos_idx[keep]})

    # -----------------------------
    # 4) Time masks for trial-mean activity
    # -----------------------------
    tmask_activity = (times >= w0) & (times <= w1)
    if not np.any(tmask_activity):
        raise ValueError(
            f"`activity_window`={activity_window} selects no time points. "
            f"Dataset time range is [{float(np.nanmin(times)):.3f}, {float(np.nanmax(times)):.3f}]."
        )

    # NEW: separate window for thresholding and min-average-FR
    tmask_calc = (times >= cw0) & (times <= cw1)
    if not np.any(tmask_calc):
        raise ValueError(
            f"`calculation_min_window`={(cw0, cw1)} selects no time points. "
            f"Dataset time range is [{float(np.nanmin(times)):.3f}, {float(np.nanmax(times)):.3f}]."
        )

    # -----------------------------
    # 5) Build latent bins (shared across units)
    # -----------------------------
    lat = latent_values.copy()
    ok_lat = np.isfinite(lat)

    if (not dropna_latent) and (not np.all(ok_lat)):
        raise ValueError("latent has NaNs but dropna_latent=False; cannot bin NaNs.")

    lat_finite = lat[ok_lat]
    if lat_finite.size == 0:
        raise ValueError("All latent values are NaN; cannot build bins.")

    n_bins_effective = int(n_bins)

    if binning == "quantile":
        s = pd.Series(lat_finite, dtype="float64")
        try:
            q = pd.qcut(s, q=n_bins, duplicates="drop")
        except Exception:
            r = s.rank(method="average")
            q = pd.qcut(r, q=n_bins, duplicates="drop")

        n_bins_eff = int(q.cat.categories.size)
        if n_bins_eff < 2:
            raise ValueError("Quantile binning collapsed (insufficient latent variability).")

        codes = q.cat.codes.to_numpy(dtype=np.int32)

        bin_idx = np.full(lat.shape[0], -1, dtype=np.int32)
        bin_idx[ok_lat] = codes

        n_bins_effective = n_bins_eff

        q_perc = np.linspace(0, 100, n_bins_effective + 1)
        bin_edges = np.unique(np.nanpercentile(lat_finite, q_perc))

    else:
        lo, hi = (
            (float(np.nanmin(lat_finite)), float(np.nanmax(lat_finite)))
            if bin_range is None
            else (float(bin_range[0]), float(bin_range[1]))
        )
        if hi <= lo:
            raise ValueError("Invalid bin_range or latent range.")
        edges = np.linspace(lo, hi, n_bins + 1)
        edges[-1] = np.nextafter(edges[-1], np.inf)

        bin_idx = np.full(lat.shape[0], -1, dtype=np.int32)
        bin_idx[ok_lat] = np.digitize(lat[ok_lat], edges[1:-1], right=False).astype(np.int32)
        bin_idx = np.clip(bin_idx, -1, n_bins - 1)

        n_bins_effective = int(n_bins)
        bin_edges = edges

    # -----------------------------
    # 6) Helpers
    # -----------------------------
    def _safe_center(vals: np.ndarray) -> float:
        if vals.size == 0:
            return np.nan
        if quantile_stat == "median":
            return float(np.nanmedian(vals))
        return float(np.nanmean(vals))

    def _ci_summary(vals: np.ndarray) -> Dict[str, float]:
        if vals.size == 0:
            if ci == "sem":
                return {"sem": np.nan}
            if ci == "iqr":
                return {"q25": np.nan, "q75": np.nan}
            return {}
        if ci == "sem":
            if vals.size < 2:
                return {"sem": 0.0}
            sem = float(np.nanstd(vals, ddof=1) / np.sqrt(max(vals.size, 1)))
            return {"sem": sem}
        if ci == "iqr":
            q25 = float(np.nanpercentile(vals, 25))
            q75 = float(np.nanpercentile(vals, 75))
            return {"q25": q25, "q75": q75}
        return {}

    def _monotonic_annotation(per_bin_means: List[float], tol: float) -> Dict[str, Any]:
        v = np.asarray(per_bin_means, dtype=np.float64)
        finite_bins = np.where(np.isfinite(v))[0].astype(int)
        v2 = v[finite_bins]

        out: Dict[str, Any] = {
            "monotonic_check_values": [float(x) if np.isfinite(x) else np.nan for x in v.tolist()],
            "monotonic_check_bins_valid": [int(b) for b in finite_bins.tolist()],
            "monotonic_n_valid_bins": int(v2.size),
            "monotonic_diffs": [],
            "monotonic_violation_count": 0,
            "monotonic_violation_diffs": [],
            "monotonic_violation_pairs": [],
            "monotonic_direction_increasing_ok": False,
            "monotonic_direction_decreasing_ok": False,
            "is_monotonic": False,
            "monotonic_direction": "insufficient_bins",
        }

        if v2.size < 2:
            return out

        dv = np.diff(v2)
        out["monotonic_diffs"] = [float(x) for x in dv.tolist()]

        inc_ok = dv >= -float(tol)
        dec_ok = dv <= +float(tol)

        out["monotonic_direction_increasing_ok"] = bool(np.all(inc_ok))
        out["monotonic_direction_decreasing_ok"] = bool(np.all(dec_ok))

        inc_viol_idx = np.where(~inc_ok)[0].astype(int)
        dec_viol_idx = np.where(~dec_ok)[0].astype(int)

        if out["monotonic_direction_increasing_ok"] and not out["monotonic_direction_decreasing_ok"]:
            direction = "increasing"
            is_m = True
            viol_idx = inc_viol_idx
        elif out["monotonic_direction_decreasing_ok"] and not out["monotonic_direction_increasing_ok"]:
            direction = "decreasing"
            is_m = True
            viol_idx = dec_viol_idx
        elif out["monotonic_direction_increasing_ok"] and out["monotonic_direction_decreasing_ok"]:
            direction = "flat"
            is_m = True
            viol_idx = np.array([], dtype=int)
        else:
            direction = "non_monotonic"
            is_m = False
            viol_idx = (
                np.unique(np.concatenate([inc_viol_idx, dec_viol_idx]))
                if (inc_viol_idx.size + dec_viol_idx.size) > 0
                else np.array([], dtype=int)
            )

        out["is_monotonic"] = bool(is_m)
        out["monotonic_direction"] = str(direction)

        if viol_idx.size > 0:
            pairs = []
            diffs = []
            for i in viol_idx.tolist():
                b0 = int(finite_bins[i])
                b1 = int(finite_bins[i + 1])
                d = float(dv[i])
                pairs.append((b0, b1, d))
                diffs.append(d)
            out["monotonic_violation_count"] = int(len(pairs))
            out["monotonic_violation_diffs"] = diffs
            out["monotonic_violation_pairs"] = pairs

        return out

    def _build_bins_for_latent_values(
        vals: np.ndarray,
        *,
        n_bins_requested: int,
        binning_mode: str,
        bin_range_local: Optional[Tuple[float, float]],
        dropna: bool,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        v = np.asarray(vals, dtype=np.float64)
        ok = np.isfinite(v)

        if (not dropna) and (not np.all(ok)):
            raise ValueError("latent has NaNs but dropna_latent=False; cannot bin NaNs.")

        v_finite = v[ok]
        if v_finite.size == 0:
            return np.full(v.shape[0], -1, dtype=np.int32), np.asarray([], dtype=np.float64), 0

        if binning_mode == "quantile":
            s = pd.Series(v_finite, dtype="float64")
            try:
                q = pd.qcut(s, q=int(n_bins_requested), duplicates="drop")
            except Exception:
                r = s.rank(method="average")
                q = pd.qcut(r, q=int(n_bins_requested), duplicates="drop")

            n_bins_eff = int(q.cat.categories.size)
            if n_bins_eff < 2:
                return np.full(v.shape[0], -1, dtype=np.int32), np.asarray([], dtype=np.float64), 0

            codes = q.cat.codes.to_numpy(dtype=np.int32)

            bin_idx_local = np.full(v.shape[0], -1, dtype=np.int32)
            bin_idx_local[ok] = codes

            q_perc = np.linspace(0, 100, n_bins_eff + 1)
            bin_edges = np.unique(np.nanpercentile(v_finite, q_perc))

            return bin_idx_local, np.asarray(bin_edges, dtype=np.float64), n_bins_eff

        lo, hi = (
            (float(np.nanmin(v_finite)), float(np.nanmax(v_finite)))
            if bin_range_local is None
            else (float(bin_range_local[0]), float(bin_range_local[1]))
        )
        if hi <= lo:
            return np.full(v.shape[0], -1, dtype=np.int32), np.asarray([], dtype=np.float64), 0

        edges = np.linspace(lo, hi, int(n_bins_requested) + 1)
        edges[-1] = np.nextafter(edges[-1], np.inf)

        bin_idx_local = np.full(v.shape[0], -1, dtype=np.int32)
        bin_idx_local[ok] = np.digitize(v[ok], edges[1:-1], right=False).astype(np.int32)
        bin_idx_local = np.clip(bin_idx_local, -1, int(n_bins_requested) - 1)

        return bin_idx_local, np.asarray(edges, dtype=np.float64), int(n_bins_requested)

    def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        if np.sum(ok) < 3:
            return np.nan, np.nan
        try:
            from scipy.stats import spearmanr  # type: ignore
            r, p = spearmanr(x[ok], y[ok])
            return float(r), float(p)
        except Exception:
            rx = pd.Series(x[ok]).rank(method="average").to_numpy()
            ry = pd.Series(y[ok]).rank(method="average").to_numpy()
            r = np.corrcoef(rx, ry)[0, 1]
            n = float(np.sum(ok))
            if not np.isfinite(r) or n <= 3:
                return float(r), np.nan
            z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(max(n - 3.0, 1.0))
            p = float(math.erfc(abs(z) / math.sqrt(2.0)))
            return float(r), float(p)

    # -----------------------------
    # 7) Loop units -> build unit_df rows
    # -----------------------------
    unit_list = list(unit_indices) if unit_ids is None else list(unit_ids)
    unit_rows: List[Dict[str, Any]] = []

    for unit in unit_list:
        where = np.where(unit_indices == int(unit))[0]
        if where.size == 0:
            continue
        upos = int(where[0])

        unit_psth_np = np.asarray(psth_da.isel(unit=upos).values, dtype=np.float64)

        # Activity used for per-bin summaries/monotonicity (activity_window)
        trial_mean_activity = np.nanmean(unit_psth_np[:, tmask_activity], axis=1)

        # NEW: activity used for thresholding and min_average_firing_rate (calculation_min_window)
        trial_mean_calc = np.nanmean(unit_psth_np[:, tmask_calc], axis=1)

        # NEW: exclude low-firing units based on calc-window average firing rate
        avg_fr_window = (
            float(np.nanmean(trial_mean_calc[np.isfinite(trial_mean_calc)]))
            if np.any(np.isfinite(trial_mean_calc))
            else np.nan
        )
        if np.isfinite(min_fr) and (min_fr > 0.0):
            if (not np.isfinite(avg_fr_window)) or (avg_fr_window < min_fr):
                continue

        ok_raw = np.ones(trial_mean_activity.shape[0], dtype=bool)
        if dropna_latent:
            ok_raw &= np.isfinite(lat)
        ok_raw &= (bin_idx >= 0)
        if dropna_activity:
            ok_raw &= np.isfinite(trial_mean_activity)

        lat_ok = lat[ok_raw]
        bin_ok = bin_idx[ok_raw]
        act_ok = trial_mean_activity[ok_raw]

        # NEW: threshold decision uses calc-window values, but selected trials keep activity-window values
        ok_thr = ok_raw & np.isfinite(trial_mean_calc) & (trial_mean_calc > thr)
        lat_ok_thr = lat[ok_thr]
        act_ok_thr = trial_mean_activity[ok_thr]

        bin_idx_thr_local, bin_edges_thr, n_bins_effective_thr = _build_bins_for_latent_values(
            lat_ok_thr,
            n_bins_requested=int(n_bins),
            binning_mode=str(binning),
            bin_range_local=bin_range,
            dropna=bool(dropna_latent),
        )

        ok_thr_binned = (bin_idx_thr_local >= 0)
        lat_ok_thr_binned = lat_ok_thr[ok_thr_binned]
        act_ok_thr_binned = act_ok_thr[ok_thr_binned]
        bin_ok_thr_re = bin_idx_thr_local[ok_thr_binned]

        q_lists, q_counts, q_centers, q_ci = [], [], [], []
        q_lists_thr, q_counts_thr, q_centers_thr, q_ci_thr = [], [], [], []

        for b in range(n_bins_effective):
            vals = act_ok[bin_ok == b]
            vals = vals[np.isfinite(vals)]
            q_lists.append([float(v) for v in vals])
            q_counts.append(int(vals.size))
            q_centers.append(_safe_center(vals))
            q_ci.append(_ci_summary(vals))

        for b in range(n_bins_effective):
            if (n_bins_effective_thr >= 2) and (b < n_bins_effective_thr):
                vals2 = act_ok_thr_binned[bin_ok_thr_re == b]
                vals2 = vals2[np.isfinite(vals2)]
            else:
                vals2 = np.asarray([], dtype=np.float64)

            q_lists_thr.append([float(v) for v in vals2])
            q_counts_thr.append(int(vals2.size))
            q_centers_thr.append(_safe_center(vals2))
            q_ci_thr.append(_ci_summary(vals2))

        ann_raw = _monotonic_annotation(q_centers, tol=monotonic_tol)
        ann_thr = _monotonic_annotation(q_centers_thr, tol=monotonic_tol)

        rho_raw, p_raw = _spearman(lat_ok, act_ok)
        rho_thr, p_thr = _spearman(lat_ok_thr, act_ok_thr)

        row: Dict[str, Any] = {
            "unit_index": int(unit),
            "session_name": session_name,
            "latent_name": latent_name,
            "activity_min_threshold": float(thr),
            "min_average_firing_rate": float(min_fr),
            "avg_firing_rate_window": float(avg_fr_window),

            # NEW: store calculation window used for thresholding / min-FR
            "calculation_window_start": float(cw0),
            "calculation_window_end": float(cw1),

            "n_trials_used": int(act_ok.size),
            "n_trials_used_gt_thr": int(act_ok_thr.size),

            "n_bins_requested": int(n_bins),
            "n_bins_effective": int(n_bins_effective),
            "binning": str(binning),

            "activity_window_start": float(w0),
            "activity_window_end": float(w1),

            "quantile_stat": str(quantile_stat),
            "ci": str(ci),
            "monotonic_tol": float(monotonic_tol),

            "is_monotonic": bool(ann_raw["is_monotonic"]),
            "monotonic_direction": str(ann_raw["monotonic_direction"]),
            "monotonic_n_valid_bins": int(ann_raw["monotonic_n_valid_bins"]),
            "monotonic_check_bins_valid": ann_raw["monotonic_check_bins_valid"],
            "monotonic_check_values": ann_raw["monotonic_check_values"],
            "monotonic_diffs": ann_raw["monotonic_diffs"],
            "monotonic_violation_count": int(ann_raw["monotonic_violation_count"]),
            "monotonic_violation_diffs": ann_raw["monotonic_violation_diffs"],
            "monotonic_violation_pairs": ann_raw["monotonic_violation_pairs"],
            "monotonic_direction_increasing_ok": bool(ann_raw["monotonic_direction_increasing_ok"]),
            "monotonic_direction_decreasing_ok": bool(ann_raw["monotonic_direction_decreasing_ok"]),

            "is_monotonic_gt_thr": bool(ann_thr["is_monotonic"]),
            "monotonic_direction_gt_thr": str(ann_thr["monotonic_direction"]),
            "monotonic_n_valid_bins_gt_thr": int(ann_thr["monotonic_n_valid_bins"]),
            "monotonic_check_bins_valid_gt_thr": ann_thr["monotonic_check_bins_valid"],
            "monotonic_check_values_gt_thr": ann_thr["monotonic_check_values"],
            "monotonic_diffs_gt_thr": ann_thr["monotonic_diffs"],
            "monotonic_violation_count_gt_thr": int(ann_thr["monotonic_violation_count"]),
            "monotonic_violation_diffs_gt_thr": ann_thr.get("monotonic_violation_diffs", []),
            "monotonic_violation_pairs_gt_thr": ann_thr.get("monotonic_violation_pairs", []),
            "monotonic_direction_increasing_ok_gt_thr": bool(ann_thr["monotonic_direction_increasing_ok"]),
            "monotonic_direction_decreasing_ok_gt_thr": bool(ann_thr["monotonic_direction_decreasing_ok"]),

            "spearman_rho": float(rho_raw) if np.isfinite(rho_raw) else np.nan,
            "spearman_p": float(p_raw) if np.isfinite(p_raw) else np.nan,
            "spearman_rho_gt_thr": float(rho_thr) if np.isfinite(rho_thr) else np.nan,
            "spearman_p_gt_thr": float(p_thr) if np.isfinite(p_thr) else np.nan,

            "bin_edges": np.array2string(np.asarray(bin_edges, dtype=np.float64), precision=6, separator=","),
        }

        for b in range(n_bins_effective):
            row[f"q{b}_n"] = q_counts[b]
            row[f"q{b}_{quantile_stat}_activity"] = q_centers[b]
            row[f"q{b}_trial_mean_activity_list"] = q_lists[b]
            if ci == "sem":
                row[f"q{b}_sem_activity"] = float(q_ci[b].get("sem", np.nan))
            elif ci == "iqr":
                row[f"q{b}_q25_activity"] = float(q_ci[b].get("q25", np.nan))
                row[f"q{b}_q75_activity"] = float(q_ci[b].get("q75", np.nan))

            row[f"q{b}_n_gt_thr"] = q_counts_thr[b]
            row[f"q{b}_{quantile_stat}_activity_gt_thr"] = q_centers_thr[b]
            row[f"q{b}_trial_mean_activity_list_gt_thr"] = q_lists_thr[b]
            if ci == "sem":
                row[f"q{b}_sem_activity_gt_thr"] = float(q_ci_thr[b].get("sem", np.nan))
            elif ci == "iqr":
                row[f"q{b}_q25_activity_gt_thr"] = float(q_ci_thr[b].get("q25", np.nan))
                row[f"q{b}_q75_activity_gt_thr"] = float(q_ci_thr[b].get("q75", np.nan))

        if meta_df is not None and int(unit) in meta_df.index:
            meta_row = meta_df.loc[int(unit)]
            for col in meta_df.columns:
                if col == unit_key_in_metadata:
                    continue
                row[col] = meta_row[col]

        unit_rows.append(row)

    unit_df = pd.DataFrame(unit_rows)
    if not unit_df.empty:
        unit_df = unit_df.sort_values(
            ["is_monotonic_gt_thr", "is_monotonic", "monotonic_direction_gt_thr", "spearman_rho_gt_thr"],
            ascending=[False, False, True, False],
        ).reset_index(drop=True)

    # -----------------------------
    # 8) Save output (optional)
    # -----------------------------
    if saved_file is not None:
        if ("__bins" in saved_file.name) and (f"__bins{int(n_bins)}__" in saved_file.name) and (n_bins_effective != int(n_bins)):
            out_dir = saved_file.parent
            sess_tok = _sanitize_token(session_name) if session_name else "session"
            lat_tok = _sanitize_token(latent_name) if latent_name else "latent"
            thr_tok = _sanitize_token(f"thr{thr:g}")
            saved_file = out_dir / f"{sess_tok}__{lat_tok}__{thr_tok}__bins{int(n_bins_effective)}__win_{w0:g}_{w1:g}.{save_format}"

            if saved_file.exists() and (not overwrite):
                if save_format == "csv":
                    unit_df2 = pd.read_csv(saved_file)
                elif save_format == "parquet":
                    unit_df2 = pd.read_parquet(saved_file)
                else:
                    raise ValueError(f"Unsupported save_format={save_format!r} for early-load.")
                return unit_df2, saved_file

        saved_file.parent.mkdir(parents=True, exist_ok=True)

        if saved_file.exists() and (not overwrite):
            if save_format == "csv":
                unit_df2 = pd.read_csv(saved_file)
            elif save_format == "parquet":
                unit_df2 = pd.read_parquet(saved_file)
            else:
                raise ValueError(f"Unsupported save_format={save_format!r} for early-load.")
            return unit_df2, saved_file

        if save_format == "csv":
            unit_df.to_csv(saved_file, index=False)
        elif save_format == "parquet":
            unit_df.to_parquet(saved_file, index=False)
        else:
            raise ValueError(f"Unsupported save_format={save_format!r}.")

        print(f"Saved unit_df to: {saved_file}")

    return unit_df, saved_file





def load_and_combine_monotonic_unit_dfs(
    folder: Union[str, Path],
    *,
    pattern: str = "*.csv",
    recursive: bool = False,
    add_source_file: bool = True,
    source_file_col: str = "source_file",
    enforce_same_columns: bool = False,
    preferred_column_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Load all CSV files produced by `summarize_monotonic_unit_df_by_latent_quantile`
    from a folder and combine them into a single DataFrame.

    Parameters
    ----------
    folder
        Directory containing per-session/unit summary CSV files.
    pattern
        Glob pattern to match files. Default "*.csv".
        You can pass a narrower pattern if you only want the outputs
        from this function, e.g. "*__bins*__win_*_*.csv".
    recursive
        If True, search recursively (uses rglob). If False, uses glob.
    add_source_file
        If True, add a column indicating which CSV each row came from.
    source_file_col
        Column name for the source filename/path.
    enforce_same_columns
        If True, raise if the set of columns differs across CSV files.
        If False, take the union of columns and fill missing with NaN.
    preferred_column_order
        Optional list of columns to put first (if present). Remaining columns
        appear after, in their original order where possible.

    Returns
    -------
    combined_df
        Concatenated DataFrame across all matched CSV files.

    Notes
    -----
    - Columns containing list-like data were saved to CSV as strings (e.g. "[0.1, 0.2]").
      This loader does NOT automatically parse those strings back into Python lists
      because parsing can be slow and ambiguous. If you want parsing, add it downstream.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a directory: {folder}")

    files: List[Path] = list(folder.rglob(pattern) if recursive else folder.glob(pattern))
    files = [p for p in files if p.is_file()]

    if len(files) == 0:
        return pd.DataFrame()

    dfs: List[pd.DataFrame] = []
    ref_cols: Optional[List[str]] = None

    for fp in sorted(files):
        df = pd.read_csv(fp)

        if add_source_file:
            # Store relative path when possible (more compact than absolute)
            try:
                df[source_file_col] = str(fp.relative_to(folder))
            except Exception:
                df[source_file_col] = str(fp)

        if enforce_same_columns:
            cols = list(df.columns)
            if ref_cols is None:
                ref_cols = cols
            else:
                if set(cols) != set(ref_cols):
                    missing = sorted(set(ref_cols) - set(cols))
                    extra = sorted(set(cols) - set(ref_cols))
                    raise ValueError(
                        f"Column mismatch for file {fp}\n"
                        f"Missing columns: {missing}\n"
                        f"Extra columns: {extra}"
                    )

        dfs.append(df)

    combined = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    # Optional reorder
    if preferred_column_order is not None:
        preferred = [c for c in preferred_column_order if c in combined.columns]
        remaining = [c for c in combined.columns if c not in preferred]
        combined = combined[preferred + remaining]

    return combined



def summarize_significant_and_monotonic_fractions(
    combined_df: pd.DataFrame,
    *,
    p_col: str = "spearman_p",
    alpha: float = 0.05,
    monotonic_col: str = "is_monotonic",
    require_valid_monotonic: bool = True,
    brain_region_col: str = "brain_region",
    brain_region_filter: Optional[Sequence[Sequence[str]]] = None,
    include_overall: bool = True,
    include_group_dataframes: bool = True,
    # --- session filtering via summary ---
    summary: Optional[pd.DataFrame] = None,
    criteria_col: str = "QLearning_L1F1_CK1_softmax_pass_all_criteria",
    summary_session_col: str = "session_name",
    combined_session_col: str = "session_name",
) -> Dict[str, Any]:
    """
    Compute:
      - fraction of significant rows (p <= alpha) among rows with finite p
      - fraction of monotonic rows among significant rows

    Optionally filter `combined_df` by sessions listed in `summary` where `criteria_col` is True.
    Session names are not standardized, so we normalize both sides using
    `extract_session_name_core(basename)`.

    Additionally, if `brain_region_filter` is provided, compute the same summary for each *group*
    of brain regions.

    Returns:
      - optional "overall"
      - "by_brain_region_group" (always present)
      - session filtering metadata if summary is provided
    """

    if p_col not in combined_df.columns:
        raise KeyError(f"Missing p-value column: {p_col!r}")
    if monotonic_col not in combined_df.columns:
        raise KeyError(f"Missing monotonic column: {monotonic_col!r}")
    if brain_region_col not in combined_df.columns:
        raise KeyError(f"Missing brain region column: {brain_region_col!r}")

    # Default filter requested by user
    if brain_region_filter is None:
        brain_region_filter = [
            ["MD"],
            ["SI", "MA"],
            ["PL5", "PL6a", "ILA5", "ILA6a"],
            ["MOs2/3", "MOs5", "MOs6a"],
            ["ORBm1", "ORBm5", "ORBm2/3", "ORBm6a"],
        ]

    # -----------------------------
    # Filter combined_df by sessions in summary that pass criteria
    # -----------------------------
    filtered_df = combined_df
    session_filter_info: Dict[str, Any] = {"session_filter_applied": False}

    if summary is not None:
        if summary_session_col not in summary.columns:
            raise KeyError(f"Missing summary session column: {summary_session_col!r}")
        if criteria_col not in summary.columns:
            raise KeyError(f"Missing criteria column in summary: {criteria_col!r}")
        if combined_session_col not in combined_df.columns:
            raise KeyError(f"Missing combined_df session column: {combined_session_col!r}")

        # Normalize criteria to bool
        crit = summary[criteria_col]
        if crit.dtype == object:
            crit_bool = (
                crit.astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                        "yes": True,
                        "no": False,
                        "t": True,
                        "f": False,
                    }
                )
                .fillna(False)
                .astype(bool)
            )
        else:
            crit_bool = crit.fillna(False).astype(bool)

        # Normalize summary session names -> core
        summary_sessions = summary.loc[crit_bool, summary_session_col].astype(str)

        pass_session_cores = {
            extract_session_name_core(s.rstrip("/").split("/")[-1])
            for s in summary_sessions.tolist()
            if len(s) > 0
        }

        # Normalize combined_df session names -> core
        combined_sessions = combined_df[combined_session_col].astype(str)
        combined_session_cores = combined_sessions.map(
            lambda s: extract_session_name_core(s.rstrip("/").split("/")[-1])
        )

        mask = combined_session_cores.isin(pass_session_cores)
        filtered_df = combined_df.loc[mask].copy()

        session_filter_info = {
            "session_filter_applied": True,
            "criteria_col": criteria_col,
            "summary_session_col": summary_session_col,
            "combined_session_col": combined_session_col,
            "n_sessions_passing_criteria": int(len(pass_session_cores)),
            "n_rows_before_session_filter": int(len(combined_df)),
            "n_rows_after_session_filter": int(len(filtered_df)),
        }

    # -----------------------------
    # Core summarizer
    # -----------------------------
    def _summarize_one(df: pd.DataFrame) -> Dict[str, Any]:
        p = pd.to_numeric(df[p_col], errors="coerce").to_numpy(dtype=np.float64)
        p_valid_mask = np.isfinite(p)

        n_total = int(len(df))
        n_p_valid = int(np.sum(p_valid_mask))

        sig_mask = p_valid_mask & (p <= float(alpha))
        n_sig = int(np.sum(sig_mask))

        frac_sig_among_p_valid = (n_sig / n_p_valid) if n_p_valid > 0 else np.nan

        sig_df = df.loc[sig_mask].copy()

        # Monotonic within significant
        m = sig_df[monotonic_col]

        # Normalize to a pandas nullable boolean series
        if m.dtype == object:
            m_norm = (
                m.astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                        "yes": True,
                        "no": False,
                        "t": True,
                        "f": False,
                    }
                )
                .astype("boolean")
            )
        else:
            m_norm = m.astype("boolean")

        if require_valid_monotonic:
            m_valid = m_norm.notna().to_numpy()
            n_sig_monot_denom = int(np.sum(m_valid))
            n_sig_monot = int(np.sum(m_norm[m_valid].to_numpy(dtype=bool)))

            sig_monot_mask = (
                m_norm.fillna(False).to_numpy(dtype=bool)
                & m_norm.notna().to_numpy(dtype=bool)
            )
            significant_monotonic_df = sig_df.loc[sig_monot_mask].copy()
        else:
            m_filled = m_norm.fillna(False).to_numpy(dtype=bool)
            n_sig_monot_denom = int(len(sig_df))
            n_sig_monot = int(np.sum(m_filled))

            significant_monotonic_df = sig_df.loc[m_filled].copy()

        frac_monotonic_among_sig = (
            (n_sig_monot / n_sig_monot_denom) if n_sig_monot_denom > 0 else np.nan
        )

        return {
            "alpha": float(alpha),
            "p_col": p_col,
            "monotonic_col": monotonic_col,
            "n_total_rows": n_total,
            "n_rows_with_valid_p": n_p_valid,
            "n_significant": n_sig,
            "frac_significant_among_valid_p": frac_sig_among_p_valid,
            "n_significant_with_valid_monotonic": n_sig_monot_denom,
            "n_significant_and_monotonic": n_sig_monot,
            "frac_monotonic_among_significant": frac_monotonic_among_sig,
            "significant_df": sig_df,
            "significant_monotonic_df": significant_monotonic_df,
        }

    # -----------------------------
    # Output structure
    # -----------------------------
    out: Dict[str, Any] = {
        "brain_region_col": brain_region_col,
        "brain_region_filter": [list(g) for g in brain_region_filter],
        "by_brain_region_group": {},
        **session_filter_info,
    }

    if include_overall:
        out["overall"] = _summarize_one(filtered_df)

    region_series = filtered_df[brain_region_col].astype(str)
    for group in brain_region_filter:
        group_list = list(group)
        label = "[" + ",".join(group_list) + "]"

        mask = region_series.isin(group_list)
        group_df = filtered_df.loc[mask].copy()

        group_summary = _summarize_one(group_df)
        group_summary["brain_regions"] = group_list

        if include_group_dataframes:
            group_summary["group_df"] = group_df

        out["by_brain_region_group"][label] = group_summary

    return out









def show_unit_figures_from_df_inline(
    df: pd.DataFrame,
    *,
    root: Union[str, Path] = "/root/capsule/scratch/raster_plot",
    session_col: str = "session_name",
    latent_col: str = "latent_name",
    unit_col: str = "unit_index",
    n: Optional[int] = None,
    random_state: Optional[int] = 0,
    filename_template: str = "{latent}_unit_{unit}.png",
    folder_template: str = "{root}/{session}/{latent}",
    figsize: tuple[float, float] = (10.0, 6.0),
    dpi: int = 120,
    title_mode: str = "fullpath",  # "basename" | "fullpath"
    warn_missing: bool = True,
) -> list[Path]:
    """
    Visualize raster plot PNG figures based on rows in a DataFrame.

    This function reconstructs file paths using:
        root / session_name / latent_name / <latent>_unit_<unit>.png

    It then opens and displays the corresponding PNG files inline
    (using IPython.display.display), which is reliable in Jupyter / Code Ocean.

    ------------------------------------------------------------------------
    Expected directory structure
    ------------------------------------------------------------------------
    root/
        <session_name>/
            <latent_name>/
                <latent_name>_unit_<unit_index>.png

    Example:
    /root/capsule/scratch/raster_plot/
        ecephys_753124_2024-12-10_17-24-56_sorted_2024-12-13_09-48-25/
            QLearning_L2F1_softmax-sumQ-1/
                QLearning_L2F1_softmax-sumQ-1_unit_35.png

    ------------------------------------------------------------------------
    Parameters
    ------------------------------------------------------------------------
    df : pd.DataFrame
        DataFrame containing at least:
            - session_name
            - latent_name
            - unit_index

        Typically this is your:
            significant_monotonic_df

    root : str | Path
        Base directory containing all raster plot outputs.

    session_col : str
        Column name in df that contains session_name.

    latent_col : str
        Column name in df that contains latent_name.

    unit_col : str
        Column name in df that contains unit_index.

    n : int | None
        If provided:
            Randomly select n rows to display.
        If None:
            Display all rows.

    random_state : int | None
        Random seed for reproducible sampling when n is specified.

    filename_template : str
        Template for filename.
        Available fields:
            {latent}
            {unit}

    folder_template : str
        Template for folder structure.
        Available fields:
            {root}
            {session}
            {latent}

    figsize : tuple
        (Currently unused in inline display; kept for compatibility.)

    dpi : int
        (Currently unused in inline display; kept for compatibility.)

    title_mode : str
        "fullpath"  → show full file path above image
        "basename"  → show only file name

    warn_missing : bool
        If True:
            Print missing file paths.

    ------------------------------------------------------------------------
    Returns
    ------------------------------------------------------------------------
    opened_paths : list[Path]
        List of successfully opened figure paths.
    """

    # Import here to ensure this works only when running in notebook environments
    try:
        from IPython.display import display
        from PIL import Image
    except Exception as e:
        raise RuntimeError(
            "Inline display requires IPython and Pillow. "
            "If running outside a notebook, use matplotlib-based viewer instead."
        ) from e

    # ---------------------------------------------------------------------
    # 1) Validate required columns exist
    # ---------------------------------------------------------------------
    for col in (session_col, latent_col, unit_col):
        if col not in df.columns:
            raise KeyError(f"Missing required column in df: {col!r}")

    root = Path(root)

    # ---------------------------------------------------------------------
    # 2) Extract only required columns and sanitize types
    # ---------------------------------------------------------------------
    sub = df[[session_col, latent_col, unit_col]].copy()

    sub[session_col] = sub[session_col].astype(str)
    sub[latent_col] = sub[latent_col].astype(str)

    # Ensure unit_index numeric
    sub[unit_col] = pd.to_numeric(sub[unit_col], errors="coerce")
    sub = sub.dropna(subset=[unit_col]).copy()
    sub[unit_col] = sub[unit_col].astype(int)

    if len(sub) == 0:
        return []

    # ---------------------------------------------------------------------
    # 3) Random sampling (if requested)
    # ---------------------------------------------------------------------
    if n is not None:
        if n <= 0:
            return []
        n = min(int(n), len(sub))
        sub = sub.sample(n=n, replace=False, random_state=random_state).reset_index(drop=True)
    else:
        sub = sub.reset_index(drop=True)

    opened: list[Path] = []
    missing: list[Path] = []

    # ---------------------------------------------------------------------
    # 4) Loop through rows and open figures
    # ---------------------------------------------------------------------
    for i in range(len(sub)):
        session = sub.loc[i, session_col]
        latent = sub.loc[i, latent_col]
        unit = int(sub.loc[i, unit_col])

        # Build folder path using template
        folder_str = folder_template.format(
            root=str(root),
            session=session,
            latent=latent
        )

        # Build filename
        file_str = filename_template.format(
            latent=latent,
            unit=unit
        )

        fp = Path(folder_str) / file_str

        # Check existence
        if not fp.exists():
            missing.append(fp)
            continue

        # Print header for clarity
        title = str(fp) if title_mode == "fullpath" else fp.name
        print(title)

        # Display image inline
        display(Image.open(fp))

        opened.append(fp)

    # ---------------------------------------------------------------------
    # 5) Report missing files (if any)
    # ---------------------------------------------------------------------
    if warn_missing and len(missing) > 0:
        print(f"[WARN] Missing {len(missing)} files (showing up to 10):")
        for p in missing[:10]:
            print("  ", p)

    return opened
