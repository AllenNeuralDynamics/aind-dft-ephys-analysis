from __future__ import annotations

# ==============================
# Standard library
# ==============================
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal

# ==============================
# Third-party libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist


def _as_1d_array(x: Any, *, dtype: Any = float) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, np.ndarray):
        if x.ndim != 1 or x.size == 0:
            return None
        return x.astype(dtype, copy=False)
    if isinstance(x, (list, tuple)):
        arr = np.asarray(x, dtype=dtype).ravel()
        if arr.size == 0:
            return None
        return arr
    return None


def _detect_q_values(df: pd.DataFrame, *, quantile_prefix: str = "q") -> List[int]:
    pat = re.compile(rf"^{re.escape(quantile_prefix)}(\d+)_")
    q_set: set[int] = set()
    for c in df.columns:
        m = pat.match(c)
        if m:
            q_set.add(int(m.group(1)))
    return sorted(q_set)


def _get_value_col(
    q: int,
    *,
    value_kind: Literal["mean_rate", "mean_rate_gt_thr", "zscore", "zscore_gt_thr"],
    quantile_prefix: str,
) -> str:
    return f"{quantile_prefix}{q}_{value_kind}"


def _build_matrix_for_indices(
    df: pd.DataFrame,
    *,
    idx_order: Sequence[Any],
    col: str,
    dtype: Any = float,
    fill_missing: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build matrix following idx_order exactly.
    Missing/invalid arrays -> filled with NaNs.
    """
    rows: List[np.ndarray] = []
    lengths: List[int] = []

    # First pass: collect valid arrays to infer T (modal length)
    valid_arrays: List[np.ndarray] = []
    for idx in idx_order:
        arr = _as_1d_array(df.at[idx, col], dtype=dtype) if (col in df.columns and idx in df.index) else None
        if arr is not None:
            valid_arrays.append(arr)
            lengths.append(arr.size)

    if len(valid_arrays) == 0:
        return np.zeros((0, 0), dtype=dtype), np.asarray([], dtype=df.index.dtype)

    uniq, counts = np.unique(lengths, return_counts=True)
    T = int(uniq[np.argmax(counts)])

    for idx in idx_order:
        arr = _as_1d_array(df.at[idx, col], dtype=dtype) if (col in df.columns and idx in df.index) else None
        if arr is None or arr.size != T:
            rows.append(np.full((T,), fill_missing, dtype=dtype))
        else:
            rows.append(arr.astype(dtype, copy=False))

    mat = np.vstack(rows)
    idx_keep = np.asarray(list(idx_order), dtype=df.index.dtype)
    return mat, idx_keep



def plot_clustered_psth_heatmaps_multi_panel(
    df_with_psth: pd.DataFrame,
    *,
    quantiles: Optional[Sequence[int]] = None,
    reference_quantile: int = 0,
    value_kind: Literal[
        "mean_rate",
        "mean_rate_gt_thr",
        "zscore",
        "zscore_gt_thr",
    ] = "zscore",
    quantile_prefix: str = "q",
    time: Optional[np.ndarray] = None,
    time_col_name: str = "psth_time",
    brain_region_col: str = "brain_region",
    brain_regions: Optional[Sequence[str]] = None,
    # --- significance / direction filters ---
    significant: bool = False,
    alpha: float = 0.05,
    p_col: Optional[str] = None,
    direction: Literal["positive", "negative", "both"] = "both",
    rho_col: Optional[str] = None,
    # --- monotonic filter ---
    monotonic_only: bool = False,
    monotonic_col: Optional[str] = None,
    # --- selection behavior across quantiles ---
    require_all_quantiles: bool = True,
    # --- clustering (rows / units) ---
    cluster_rows: bool = True,
    linkage_method: str = "average",
    distance_metric: str = "correlation",
    # --- clustering (cols / time bins) (NEW) ---
    cluster_cols: bool = False,
    col_linkage_method: str = "average",
    col_distance_metric: str = "correlation",
    # --- color scaling ---
    color_range: Optional[Tuple[float, float]] = None,
    color_percentile: Optional[Tuple[float, float]] = (1, 99),
    cmap: str = "RdBu_r",
    # --- plotting ---
    figsize: Tuple[float, float] = (12, 6),
    dpi: int = 150,
    go_cue_label: str = "Go cue",
    go_cue_linestyle: str = "--",
    go_cue_linewidth: float = 1.2,
    go_cue_alpha: float = 0.9,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Dict[str, Any]:
    """
    Plot multiple quantile PSTH heatmaps on the SAME FIGURE with:
      - shared clustering order using `reference_quantile`
      - optional hierarchical clustering on rows (units) and columns (time bins)
      - shared colorbar range across panels
      - external colorbar (outside panels)
      - optional filters: brain region, significance, correlation direction, monotonic-only

    Notes
    -----
    - `value_kind` determines default p/rho/monotonic columns:
        * if value_kind ends with "_gt_thr":
            p_col default         = "spearman_p_gt_thr"
            rho_col default       = "spearman_rho_gt_thr"
            monotonic_col default = "is_monotonic_gt_thr"
        * else:
            p_col default         = "spearman_p"
            rho_col default       = "spearman_rho"
            monotonic_col default = "is_monotonic"

    - If `time` is provided (or found in df), time=0 is marked with a dashed line and labeled as `go_cue_label`.

    Parameters (NEW)
    ----------------
    cluster_cols : bool
        If True, perform hierarchical clustering along the time-bin dimension (columns).

    col_linkage_method : str
        Linkage method for column clustering (e.g., "average", "complete", "single", "ward").

    col_distance_metric : str
        Distance metric for column clustering (e.g., "correlation", "euclidean").
        If `col_linkage_method == "ward"`, then `col_distance_metric` must be "euclidean".
    """
    df = df_with_psth.copy()

    # ---------------------------------------
    # Resolve p/rho/monotonic columns
    # ---------------------------------------
    if p_col is None:
        p_col = "spearman_p_gt_thr" if value_kind.endswith("_gt_thr") else "spearman_p"

    if rho_col is None:
        rho_col = "spearman_rho_gt_thr" if value_kind.endswith("_gt_thr") else "spearman_rho"

    if monotonic_col is None:
        monotonic_col = "is_monotonic_gt_thr" if value_kind.endswith("_gt_thr") else "is_monotonic"

    # ---------------------------------------
    # Detect quantiles
    # ---------------------------------------
    if quantiles is None:
        pat = re.compile(rf"^{re.escape(quantile_prefix)}(\d+)_")
        quantiles = sorted(
            {
                int(m.group(1))
                for c in df.columns
                for m in [pat.match(c)]
                if m
            }
        )

    if quantiles is None or len(quantiles) == 0:
        raise ValueError("No quantiles found. Provide `quantiles` or ensure q#_* columns exist.")

    if reference_quantile not in quantiles:
        raise ValueError("reference_quantile must be in quantiles")

    # ---------------------------------------
    # Filters
    # ---------------------------------------
    mask = pd.Series(True, index=df.index)

    if brain_regions is not None:
        if brain_region_col not in df.columns:
            raise ValueError(f"brain_region_col '{brain_region_col}' not in DataFrame.")
        mask &= df[brain_region_col].isin(brain_regions)

    if significant:
        if p_col not in df.columns:
            raise ValueError(f"p_col '{p_col}' not in DataFrame (needed for significant=True).")
        mask &= pd.to_numeric(df[p_col], errors="coerce") <= float(alpha)

    if direction != "both":
        if rho_col not in df.columns:
            raise ValueError(f"rho_col '{rho_col}' not in DataFrame (needed for direction filter).")
        rho = pd.to_numeric(df[rho_col], errors="coerce")
        if direction == "positive":
            mask &= rho > 0
        elif direction == "negative":
            mask &= rho < 0
        else:
            raise ValueError(f"Unknown direction: {direction}")

    if monotonic_only:
        if monotonic_col not in df.columns:
            raise ValueError(f"monotonic_col '{monotonic_col}' not in DataFrame (needed for monotonic_only=True).")
        mask &= df[monotonic_col].astype(bool)

    df = df.loc[mask]
    if len(df) == 0:
        raise ValueError("No rows remain after filters. Relax filters and try again.")

    # ---------------------------------------
    # Get time axis
    # ---------------------------------------
    if time is None and time_col_name in df.columns:
        for x in df[time_col_name]:
            if isinstance(x, np.ndarray) and x.ndim == 1 and x.size > 0:
                time = x
                break

    if time is not None:
        time = np.asarray(time, dtype=float).ravel()

    # ---------------------------------------
    # Collect valid indices per quantile
    # ---------------------------------------
    valid_by_q: Dict[int, pd.Index] = {}
    for q in quantiles:
        col = f"{quantile_prefix}{q}_{value_kind}"
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

        valid_by_q[q] = df.index[df[col].apply(lambda x: isinstance(x, np.ndarray) and np.asarray(x).ndim == 1)]

    ref_idx = set(valid_by_q[reference_quantile])

    if require_all_quantiles:
        shared_idx = ref_idx.copy()
        for q in quantiles:
            shared_idx &= set(valid_by_q[q])
        shared_idx = list(shared_idx)
    else:
        shared_idx = list(ref_idx)

    if len(shared_idx) == 0:
        raise ValueError("No shared units found (check require_all_quantiles and array validity).")

    # ---------------------------------------
    # Build reference matrix (units x time)
    # ---------------------------------------
    ref_col = f"{quantile_prefix}{reference_quantile}_{value_kind}"
    ref_mat = np.vstack(df.loc[shared_idx, ref_col].to_numpy())

    # ---------------------------------------
    # Cluster order: rows (units)
    # ---------------------------------------
    if cluster_rows and ref_mat.shape[0] > 1:
        tmp = ref_mat.copy()
        tmp[~np.isfinite(tmp)] = 0.0

        if linkage_method == "ward" and distance_metric != "euclidean":
            raise ValueError("linkage_method='ward' requires distance_metric='euclidean'.")

        d = pdist(tmp, metric=distance_metric)
        Z = linkage(d, method=linkage_method)
        row_order = leaves_list(Z)
    else:
        row_order = np.arange(ref_mat.shape[0])

    shared_idx = np.array(shared_idx, dtype=object)[row_order]
    ref_mat_ordered = ref_mat[row_order, :]

    # ---------------------------------------
    # Cluster order: columns (time bins) (NEW)
    # ---------------------------------------
    T = ref_mat_ordered.shape[1]
    if cluster_cols and T > 1:
        tmpc = ref_mat_ordered.copy()
        tmpc[~np.isfinite(tmpc)] = 0.0

        if col_linkage_method == "ward" and col_distance_metric != "euclidean":
            raise ValueError("col_linkage_method='ward' requires col_distance_metric='euclidean'.")

        # Cluster columns by treating each column as a vector over units: (T x N_units)
        d_c = pdist(tmpc.T, metric=col_distance_metric)
        Zc = linkage(d_c, method=col_linkage_method)
        col_order = leaves_list(Zc)
    else:
        col_order = np.arange(T)

    # Reorder time axis if present
    if time is not None and time.size == T:
        time_ordered = time[col_order]
    else:
        time_ordered = time

    # ---------------------------------------
    # Build matrices for all quantiles (in shared row order)
    # ---------------------------------------
    matrices: List[np.ndarray] = []
    for q in quantiles:
        col = f"{quantile_prefix}{q}_{value_kind}"
        mat = np.vstack(
            [
                df.at[idx, col]
                if isinstance(df.at[idx, col], np.ndarray)
                else np.full((T,), np.nan, dtype=float)
                for idx in shared_idx
            ]
        )
        # Apply column order (time clustering) to every panel
        mat = mat[:, col_order]
        matrices.append(mat)

    # ---------------------------------------
    # Determine shared color range
    # ---------------------------------------
    all_vals = np.concatenate([m.ravel() for m in matrices])

    if color_range is not None:
        vmin, vmax = float(color_range[0]), float(color_range[1])
    else:
        if color_percentile is None:
            raise ValueError("Provide either color_range or color_percentile.")
        low, high = color_percentile
        vmin, vmax = np.nanpercentile(all_vals, [low, high])
        vmin, vmax = float(vmin), float(vmax)

    # ---------------------------------------
    # Create figure and panels
    # ---------------------------------------
    n = len(quantiles)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=figsize,
        dpi=dpi,
        sharey=True,
    )
    if n == 1:
        axes = [axes]

    ims = []
    for ax, q, mat in zip(axes, quantiles, matrices):
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ims.append(im)
        ax.set_title(f"q{q}")

        # ---------------------------------------
        # X-axis ticks / go-cue line
        # ---------------------------------------
        if time_ordered is not None and time_ordered.size == mat.shape[1]:
            # Mark time=0 (closest bin in the REORDERED time axis)
            zero_idx = int(np.argmin(np.abs(time_ordered)))
            ax.axvline(
                zero_idx,
                color="k",
                linestyle=go_cue_linestyle,
                linewidth=go_cue_linewidth,
                alpha=go_cue_alpha,
                zorder=10,
            )

            # If columns are clustered, the time axis is no longer monotonic.
            # We can still label integer seconds by mapping each integer second's
            # original nearest bin to its new position (via col_order).
            t_min = int(np.ceil(np.nanmin(time_ordered)))
            t_max = int(np.floor(np.nanmax(time_ordered)))
            tick_seconds = np.arange(t_min, t_max + 1, 1)

            # Build ticks at positions of closest bins in the REORDERED axis
            tick_positions = [int(np.argmin(np.abs(time_ordered - t))) for t in tick_seconds]

            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{int(t)}" for t in tick_seconds])
            ax.set_xlabel("Time (s; align to go cue)" + ("; clustered" if cluster_cols else ""))
        else:
            ax.set_xlabel("Time bins")

    # ---------------------------------------
    # Shrink subplot region (reserve space for external colorbar)
    # ---------------------------------------
    fig.subplots_adjust(
        left=0.07,
        right=0.86,
        bottom=0.12,
        top=0.92,
        wspace=0.05,
    )

    # ---------------------------------------
    # Add external colorbar aligned with panels
    # ---------------------------------------
    bbox = axes[-1].get_position()
    cbar_ax = fig.add_axes([bbox.x1 + 0.015, bbox.y0, 0.02, bbox.height])
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.set_label(value_kind, rotation=90, labelpad=12)

    # ---------------------------------------
    # Save / show
    # ---------------------------------------
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return dict(
        fig=fig,
        axes=axes,
        matrices=matrices,
        shared_index=shared_idx,
        row_order=row_order,
        col_order=col_order,
        time=time_ordered,
        vmin=vmin,
        vmax=vmax,
        filters=dict(
            brain_regions=brain_regions,
            significant=significant,
            alpha=alpha,
            p_col=p_col,
            direction=direction,
            rho_col=rho_col,
            monotonic_only=monotonic_only,
            monotonic_col=monotonic_col,
            require_all_quantiles=require_all_quantiles,
        ),
        clustering=dict(
            cluster_rows=cluster_rows,
            linkage_method=linkage_method,
            distance_metric=distance_metric,
            cluster_cols=cluster_cols,
            col_linkage_method=col_linkage_method,
            col_distance_metric=col_distance_metric,
        ),
    )

