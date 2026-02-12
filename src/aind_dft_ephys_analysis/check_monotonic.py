from __future__ import annotations

# Standard library
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

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Local imports
from create_psth import load_psth_raster_subset
from ephys_utils import append_units_locations

def summarize_monotonic_unit_df_by_latent_quantile(
    source: Union[str, Path, xr.DataArray, xr.Dataset],
    *,
    latent_values: Sequence[float],
    latent_trial_ids: Sequence[int],
    activity_window: Tuple[float, float],
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
    # Filter for the "gt threshold" monotonic check
    activity_min_threshold: float = 0.0,
    # Metadata
    session_name: Optional[str] = None,
    latent_name: Optional[str] = None,
    unit_metadata: Optional[pd.DataFrame] = None,
    unit_key_in_metadata: str = "unit_index",
    # NEW: if provided, append_units_locations and attach brain_region/ccf_location
    nwb_data: Optional[Any] = None,
    # Loader
    consolidated: bool = True,
    # Saving (flexible)
    save_dir: Optional[Union[str, Path]] = None,
    save_filename: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    save_format: Literal["csv", "parquet"] = "csv",
    overwrite: bool = True,
) -> Tuple[pd.DataFrame, Optional[Path]]:
    """
    Per-unit monotonicity summary vs latent (binned), with an additional monotonic check
    after removing trials whose trial-mean activity <= activity_min_threshold.

    NEW:
    If `nwb_data` is provided, this function will:
      1) call `append_units_locations(nwb_data, session_id_core)` to populate CCF fields
      2) attach `brain_region` and `ccf_location` to the output rows (per unit)

    OUTPUT (unit_df): one row per unit, includes:
    --------------------------------------------------------------------------
    Core identifiers / metadata
      - unit_index
      - session_name
      - latent_name
      - brain_region           (if nwb_data provided or in unit_metadata)
      - ccf_location           (if nwb_data provided or in unit_metadata)

    Binning + window config
      - binning, n_bins_requested, n_bins_effective, bin_edges
      - activity_window_start, activity_window_end
      - quantile_stat, ci
      - monotonic_tol
      - activity_min_threshold

    Trial usage counts
      - n_trials_used               (after dropna_latent/dropna_activity/bin>=0)
      - n_trials_used_gt_thr        (above, AND trial_mean_activity > threshold)

    Per-bin outputs (RAW)
      - q{b}_n
      - q{b}_{quantile_stat}_activity
      - q{b}_trial_mean_activity_list
      - CI columns depending on ci:
          * ci="sem": q{b}_sem_activity
          * ci="iqr": q{b}_q25_activity, q{b}_q75_activity

    Per-bin outputs (FILTERED: trial_mean_activity > threshold)
      - q{b}_n_gt_thr
      - q{b}_{quantile_stat}_activity_gt_thr
      - q{b}_trial_mean_activity_list_gt_thr
      - CI columns with _gt_thr suffix

    Detailed monotonic annotation (RAW)
      - is_monotonic
      - monotonic_direction                 ("increasing"|"decreasing"|"flat"|...)
      - monotonic_n_valid_bins
      - monotonic_check_values              length=n_bins_effective (NaN for empty bins)
      - monotonic_check_bins_valid          bin indices used in diffs
      - monotonic_diffs                     successive diffs of valid-bin values
      - monotonic_violation_count           # of diffs violating increasing/decreasing (tol-aware)
      - monotonic_violation_diffs           the violating diffs (float)
      - monotonic_violation_pairs           list of (bin_left, bin_right, diff)
      - monotonic_direction_increasing_ok   bool
      - monotonic_direction_decreasing_ok   bool

    Detailed monotonic annotation (FILTERED)
      - same fields, suffixed with _gt_thr

    Trial-wise Spearman correlation (RAW and FILTERED)
      - spearman_rho, spearman_p
      - spearman_rho_gt_thr, spearman_p_gt_thr

    Saving behavior:
      - If save_path is a file (has suffix), use it directly.
      - Else if save_filename is provided, save under save_dir (or ".").
      - Else if save_dir or save_path (directory) is provided, auto-generate filename.
    """
    import math  # Local import to avoid relying on module-level imports

    # -----------------------------
    # 1) Validate inputs
    # -----------------------------
    latent_values = np.asarray(latent_values, dtype=np.float64)
    latent_trial_ids = np.asarray(latent_trial_ids, dtype=np.int64)

    if latent_values.shape[0] != latent_trial_ids.shape[0]:
        raise ValueError("`latent_values` and `latent_trial_ids` must have the same length.")
    if latent_trial_ids.size != np.unique(latent_trial_ids).size:
        raise ValueError("`latent_trial_ids` contains duplicates; must be one-to-one.")

    w0, w1 = float(activity_window[0]), float(activity_window[1])
    if w1 <= w0:
        raise ValueError("`activity_window` must satisfy activity_window[1] > activity_window[0].")

    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")

    if ci not in ("sem", "iqr", "none"):
        raise ValueError("ci must be one of {'sem','iqr','none'}.")

    thr = float(activity_min_threshold)

    # -----------------------------
    # 1b) Optional: build/augment unit metadata from NWB
    # -----------------------------
    nwb_meta_df: Optional[pd.DataFrame] = None
    if nwb_data is not None:
        # Determine session name core for anatomy lookup
        sess_raw = getattr(nwb_data, "session_id", None)
        sess_core = extract_session_name_core(sess_raw)
        # If user didn't pass session_name, fill it
        if session_name is None:
            session_name = sess_core

        # Populate/attach CCF fields onto nwb_data
        nwb_data = append_units_locations(nwb_data, sess_core)

        # Build a minimal metadata table keyed by unit_index
        # Assumption: unit_index corresponds to row index in nwb_data.units
        unit_rows = []
        n_units_total = len(nwb_data.units)
        for u in range(n_units_total):
            loc = nwb_data.units["ccf_location"][u] if "ccf_location" in nwb_data.units.colnames else None
            loc = loc or {}
            region = ""
            if isinstance(loc, dict):
                region = str(loc.get("brain_region", "")) if loc.get("brain_region", "") is not None else ""
            unit_rows.append(
                {
                    "unit_index": int(u),
                    "brain_region": region,
                    "ccf_location": loc,
                }
            )
        nwb_meta_df = pd.DataFrame(unit_rows).set_index("unit_index", drop=False)

    # If user provided unit_metadata, prefer it but fill missing from NWB metadata
    meta_df: Optional[pd.DataFrame] = None
    if unit_metadata is not None:
        if unit_key_in_metadata not in unit_metadata.columns:
            raise ValueError(f"unit_metadata must contain column {unit_key_in_metadata!r}.")
        meta_df = unit_metadata.copy().set_index(unit_key_in_metadata, drop=False)

        # Fill missing anatomy columns from nwb_data if available
        if nwb_meta_df is not None:
            for col in ("brain_region", "ccf_location"):
                if col not in meta_df.columns:
                    meta_df[col] = np.nan
            # Left-join by unit index
            join_cols = ["brain_region", "ccf_location"]
            meta_df = meta_df.join(nwb_meta_df[join_cols], how="left", rsuffix="_nwb")
            # Prefer existing values; fall back to NWB-derived values
            for col in join_cols:
                col_nwb = f"{col}_nwb"
                meta_df[col] = meta_df[col].where(meta_df[col].notna(), meta_df[col_nwb])
                meta_df = meta_df.drop(columns=[col_nwb], errors="ignore")

    else:
        # No user metadata: if NWB metadata exists, use it
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
    _trial_ids_arr = psth_da.coords[trial_coord].values.astype(np.int64)  # kept for completeness

    # -----------------------------
    # 4) Time mask for trial-mean activity
    # -----------------------------
    tmask = (times >= w0) & (times <= w1)
    if not np.any(tmask):
        raise ValueError(
            f"`activity_window`={activity_window} selects no time points. "
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
        trial_mean = np.nanmean(unit_psth_np[:, tmask], axis=1)

        # Base validity (raw)
        ok_raw = np.ones(trial_mean.shape[0], dtype=bool)
        if dropna_latent:
            ok_raw &= np.isfinite(lat)
        ok_raw &= (bin_idx >= 0)
        if dropna_activity:
            ok_raw &= np.isfinite(trial_mean)

        lat_ok = lat[ok_raw]
        bin_ok = bin_idx[ok_raw]
        act_ok = trial_mean[ok_raw]

        # Thresholded validity (trial_mean > thr)
        ok_thr = ok_raw & np.isfinite(trial_mean) & (trial_mean > thr)
        lat_ok_thr = lat[ok_thr]
        bin_ok_thr = bin_idx[ok_thr]
        act_ok_thr = trial_mean[ok_thr]

        # Per-bin lists/stats: RAW and THR
        q_lists, q_counts, q_centers, q_ci = [], [], [], []
        q_lists_thr, q_counts_thr, q_centers_thr, q_ci_thr = [], [], [], []

        for b in range(n_bins_effective):
            vals = act_ok[bin_ok == b]
            vals = vals[np.isfinite(vals)]
            q_lists.append([float(v) for v in vals])
            q_counts.append(int(vals.size))
            q_centers.append(_safe_center(vals))
            q_ci.append(_ci_summary(vals))

            vals2 = act_ok_thr[bin_ok_thr == b]
            vals2 = vals2[np.isfinite(vals2)]
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

            # RAW: core + detailed
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

            # THR: core + detailed
            "is_monotonic_gt_thr": bool(ann_thr["is_monotonic"]),
            "monotonic_direction_gt_thr": str(ann_thr["monotonic_direction"]),
            "monotonic_n_valid_bins_gt_thr": int(ann_thr["monotonic_n_valid_bins"]),
            "monotonic_check_bins_valid_gt_thr": ann_thr["monotonic_check_bins_valid"],
            "monotonic_check_values_gt_thr": ann_thr["monotonic_check_values"],
            "monotonic_diffs_gt_thr": ann_thr["monotonic_diffs"],
            "monotonic_violation_count_gt_thr": int(ann_thr["monotonic_violation_count"]),
            "monotonic_violation_diffs_gt_thr": ann_thr["monotonic_violation_diffs"],
            "monotonic_violation_pairs_gt_thr": ann_thr["monotonic_violation_pairs"],
            "monotonic_direction_increasing_ok_gt_thr": bool(ann_thr["monotonic_direction_increasing_ok"]),
            "monotonic_direction_decreasing_ok_gt_thr": bool(ann_thr["monotonic_direction_decreasing_ok"]),

            # Spearman
            "spearman_rho": float(rho_raw) if np.isfinite(rho_raw) else np.nan,
            "spearman_p": float(p_raw) if np.isfinite(p_raw) else np.nan,
            "spearman_rho_gt_thr": float(rho_thr) if np.isfinite(rho_thr) else np.nan,
            "spearman_p_gt_thr": float(p_thr) if np.isfinite(p_thr) else np.nan,

            "bin_edges": np.array2string(np.asarray(bin_edges, dtype=np.float64), precision=6, separator=","),
        }

        # Per-bin columns
        for b in range(n_bins_effective):
            # RAW
            row[f"q{b}_n"] = q_counts[b]
            row[f"q{b}_{quantile_stat}_activity"] = q_centers[b]
            row[f"q{b}_trial_mean_activity_list"] = q_lists[b]
            if ci == "sem":
                row[f"q{b}_sem_activity"] = float(q_ci[b].get("sem", np.nan))
            elif ci == "iqr":
                row[f"q{b}_q25_activity"] = float(q_ci[b].get("q25", np.nan))
                row[f"q{b}_q75_activity"] = float(q_ci[b].get("q75", np.nan))

            # THR
            row[f"q{b}_n_gt_thr"] = q_counts_thr[b]
            row[f"q{b}_{quantile_stat}_activity_gt_thr"] = q_centers_thr[b]
            row[f"q{b}_trial_mean_activity_list_gt_thr"] = q_lists_thr[b]
            if ci == "sem":
                row[f"q{b}_sem_activity_gt_thr"] = float(q_ci_thr[b].get("sem", np.nan))
            elif ci == "iqr":
                row[f"q{b}_q25_activity_gt_thr"] = float(q_ci_thr[b].get("q25", np.nan))
                row[f"q{b}_q75_activity_gt_thr"] = float(q_ci_thr[b].get("q75", np.nan))

        # Optional unit metadata merge (including NWB-derived anatomy)
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
    def _sanitize_token(s: str) -> str:
        s2 = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".", "+")) else "_" for ch in s)
        return s2.strip("_") if s2.strip("_") else "NA"

    saved_file: Optional[Path] = None

    if save_path is not None:
        p = Path(save_path)
        if p.suffix:
            saved_file = p
        else:
            out_dir = p
            out_dir.mkdir(parents=True, exist_ok=True)
            sess_tok = _sanitize_token(session_name) if session_name else "session"
            lat_tok = _sanitize_token(latent_name) if latent_name else "latent"
            thr_tok = _sanitize_token(f"thr{thr:g}")
            saved_file = out_dir / f"{sess_tok}__{lat_tok}__{thr_tok}__bins{n_bins_effective}__win_{w0:g}_{w1:g}.{save_format}"

    elif save_filename is not None:
        out_dir = Path(save_dir) if save_dir is not None else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_file = out_dir / save_filename
        if not saved_file.suffix:
            saved_file = saved_file.with_suffix(f".{save_format}")

    elif save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sess_tok = _sanitize_token(session_name) if session_name else "session"
        lat_tok = _sanitize_token(latent_name) if latent_name else "latent"
        thr_tok = _sanitize_token(f"thr{thr:g}")
        saved_file = out_dir / f"{sess_tok}__{lat_tok}__{thr_tok}__bins{n_bins_effective}__win_{w0:g}_{w1:g}.{save_format}"

    if saved_file is not None:
        if saved_file.exists() and not overwrite:
            raise FileExistsError(f"Output exists and overwrite=False: {saved_file}")

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
) -> Dict[str, Any]:
    """
    Compute:
      - fraction of significant rows (p <= alpha) among rows with finite p
      - fraction of monotonic rows among significant rows

    Returns a dict with fractions + counts + DataFrames:
      - significant_df: rows with p <= alpha
      - significant_monotonic_df: rows with p <= alpha AND monotonic==True
    """

    if p_col not in combined_df.columns:
        raise KeyError(f"Missing p-value column: {p_col!r}")
    if monotonic_col not in combined_df.columns:
        raise KeyError(f"Missing monotonic column: {monotonic_col!r}")

    p = pd.to_numeric(combined_df[p_col], errors="coerce").to_numpy(dtype=np.float64)
    p_valid_mask = np.isfinite(p)

    n_total = int(len(combined_df))
    n_p_valid = int(np.sum(p_valid_mask))

    sig_mask = p_valid_mask & (p <= float(alpha))
    n_sig = int(np.sum(sig_mask))

    frac_sig_among_p_valid = (n_sig / n_p_valid) if n_p_valid > 0 else np.nan

    # Significant subset
    sig_df = combined_df.loc[sig_mask].copy()

    # Monotonic within significant
    m = sig_df[monotonic_col]
    # Handle cases where monotonic is stored as bool / int / string
    if m.dtype == object:
        m_bool = m.astype(str).str.lower().map({"true": True, "false": False})
    else:
        m_bool = m.astype("boolean")

    if require_valid_monotonic:
        m_valid = m_bool.notna().to_numpy()
        n_sig_monot_denom = int(np.sum(m_valid))
        n_sig_monot = int(np.sum(m_bool[m_valid].to_numpy(dtype=bool)))

        # Only count True among valid; exclude NA rows from fraction denom
        sig_monot_mask = m_bool.fillna(False).to_numpy(dtype=bool) & m_bool.notna().to_numpy(dtype=bool)
        significant_monotonic_df = sig_df.loc[sig_monot_mask].copy()
    else:
        # Treat NA as False
        m_filled = m_bool.fillna(False).to_numpy(dtype=bool)
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
