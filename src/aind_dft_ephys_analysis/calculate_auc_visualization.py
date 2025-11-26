from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, Dict, List

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def plot_auc_cdf_by_region(
    zarr_path: Union[str, Path],
    *,
    auc_var: str = "auc_mean",
    region_var: str = "meta_brain_region",
    brain_regions: Optional[Sequence[Sequence[str]]] = None,
    flip_auc: bool = False,
    flip_mode: str = "simple",
    min_units_per_region: int = 10,
    figsize: Tuple[float, float] = (6.0, 4.0),
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, int]]:
    """
    Plot AUC cumulative distribution functions (CDFs) grouped by brain region.

    Supports:
        - region grouping via list-of-lists (e.g., [["MD", "MOs5"], ["PL"]])
        - optional AUC flipping for direction-invariant discriminability
        - saving to a file

    Parameters
    ----------
    zarr_path : str or Path
        Path to the combined AUC zarr dataset.

    auc_var : str, default "auc_mean"
        The name of the AUC variable inside the dataset.

    region_var : str, default "meta_brain_region"
        The dataset variable that stores per-unit region labels.

    brain_regions : list of list of str, optional
        Specifies grouping for plotting:
            None  → each region appears as its own line.
            [["MD", "MOs5"], ["PL"]]
                → plot (MD+MOs5) together as one line, PL as another.

        Each sub-list defines a group of regions combined into one curve.

    flip_auc : bool, default False
        If True, remove directionality from AUC values.
        Two modes available:

            - "simple":
                AUC < 0.5 → 1 - AUC
                AUC >= 0.5 unchanged
                (keeps range [0.5, 1])

            - "symmetric":
                Convert to direction-agnostic discriminability strength:
                AUC → abs(AUC - 0.5) * 2
                Range becomes [0, 1], with:
                    0 = no discriminability
                    1 = perfect discriminability

        This is useful when direction (A-prefer vs B-prefer) should be ignored.

    flip_mode : {"simple", "symmetric"}, default "simple"
        The mode used when flip_auc=True.

        "simple"     — reflect values < 0.5 across 0.5  
        "symmetric"  — convert to discriminability strength in [0, 1]

    min_units_per_region : int, default 10
        Minimum number of units required to plot a region group.
        Groups with fewer units are skipped.

    figsize : (float, float), default (6, 4)
        Size of the generated figure.

    ax : matplotlib.axes.Axes, optional
        Axes object to draw on.
        If None, a new figure + axes are created.

    title : str, optional
        Title of the plot. If None, a default title is generated.

    save_path : str or Path, optional
        If provided, save the plot to this file (png, pdf, etc.).
        Directory must exist.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.

    ax : matplotlib.axes.Axes
        The axes containing the plot.

    counts_per_group : dict
        Dictionary mapping each region-group label → number of units used.

        Example:
            {
              "MD + MOs5": 185,
              "PL": 92,
            }

    Notes
    -----
    AUC interpretation:
        0.5  = chance
        1.0  = perfect discrimination
        0.0  = perfect discrimination in opposite direction

    When flip_auc=True:
        - "simple" makes AUC directional but >0.5
        - "symmetric" transforms AUC into a symmetric discriminability score
    """

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    zarr_path = Path(zarr_path)
    ds = xr.open_zarr(zarr_path)

    if auc_var not in ds:
        raise KeyError(f"AUC variable '{auc_var}' not found.")
    if region_var not in ds:
        raise KeyError(f"Region variable '{region_var}' not found.")

    auc = ds[auc_var].values
    regions = ds[region_var].values.astype(str)

    # Filter invalid AUCs
    mask = np.isfinite(auc)
    auc = auc[mask]
    regions = regions[mask]

    # ---------------------------------------------------------
    # Flip AUC values (optional)
    # ---------------------------------------------------------
    if flip_auc:
        if flip_mode == "simple":
            auc = np.where(auc < 0.5, 1 - auc, auc)

        elif flip_mode == "symmetric":
            auc = np.abs(auc - 0.5) * 2

        else:
            raise ValueError("flip_mode must be 'simple' or 'symmetric'.")

    # ---------------------------------------------------------
    # Region grouping
    # ---------------------------------------------------------
    if brain_regions is None:
        region_groups = [[r] for r in np.unique(regions)]
    else:
        # Ensure list-of-lists
        region_groups = []
        for g in brain_regions:
            region_groups.append(list(g) if isinstance(g, (list, tuple)) else [g])

    # ---------------------------------------------------------
    # Prepare figure
    # ---------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    counts_per_group: Dict[str, int] = {}

    # ---------------------------------------------------------
    # Plot CDF for each group
    # ---------------------------------------------------------
    for group in region_groups:

        group = [str(r) for r in group]
        mask_group = np.isin(regions, group)
        auc_group = auc[mask_group]
        n = int(mask_group.sum())

        if n < min_units_per_region:
            print(f"Skip group {group} (n={n} < {min_units_per_region})")
            continue

        label = " + ".join(group)
        counts_per_group[label] = n

        # CDF
        x = np.sort(auc_group)
        y = np.linspace(1/n, 1, n)

        ax.plot(x, y, label=f"{label} (n={n})")

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------
    xlabel = "Per-unit AUC"
    if flip_auc:
        xlabel += f" (flipped-{flip_mode})"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative fraction of units")

    if title is None:
        title = f"CDF of per-unit AUC by region\n{zarr_path.name}"
    ax.set_title(title)

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig, ax, counts_per_group
