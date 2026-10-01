"""Trial-outcome PSTH analyses for opto-tagged units.

Reusable helpers behind ``response_reward_psth_tagged_units.ipynb``. A single
``Contrast`` (resolved from the name ``"response"`` or ``"reward"``) drives every
downstream analysis:

- per-unit PSTH (group A vs group B)
- population-average PSTH across units
- per-unit scatter of windowed mean rate
- paired Wilcoxon signed-rank test + difference histogram
- scatter colored by signed difference
- trial-count-matched control (subsample the larger group)

``"response"`` compares response (``animal_response != 2``) vs no-response
(``== 2``); ``"reward"`` compares rewarded (either side) vs unrewarded (responded,
no reward; no-response trials excluded).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from behavior_utils import find_trials
from create_psth import load_psth_raster_subset
from plot_raster import plot_psth_raster_for_units

__all__ = [
    "CONTRASTS",
    "Contrast",
    "resolve_contrast",
    "PopulationPSTH",
    "load_population_psth",
    "build_groups",
    "plot_perunit_psth",
    "plot_population_psth",
    "window_rates",
    "plot_scatter",
    "paired_diff",
    "print_paired_diff",
    "plot_paired_diff",
    "plot_scatter_colored",
    "trialcount_matched",
]


# --- Contrast definitions ----------------------------------------------------
CONTRASTS: Dict[str, Dict[str, str]] = {
    "response": dict(
        type_a="response", type_b="no_response",
        label_a="response", label_b="no-response",
        color_a="tab:blue", color_b="tab:orange",
        tag="response_vs_noresponse",
    ),
    "reward": dict(
        type_a="rewarded", type_b="unrewarded",
        label_a="reward", label_b="no-reward",
        color_a="tab:green", color_b="tab:red",
        tag="reward_vs_noreward",
    ),
}


@dataclass
class Contrast:
    """Resolved description of a two-group trial-outcome comparison."""

    name: str
    type_a: str       # find_trials name for group A
    type_b: str       # find_trials name for group B
    label_a: str      # display label for group A
    label_b: str      # display label for group B
    color_a: str      # plot color for group A
    color_b: str      # plot color for group B
    tag: str          # filename / subfolder tag


def resolve_contrast(name: str) -> Contrast:
    """Return the :class:`Contrast` for ``name`` (``"response"`` or ``"reward"``)."""
    if name not in CONTRASTS:
        raise ValueError(f"CONTRAST must be one of {list(CONTRASTS)}; got {name!r}")
    return Contrast(name=name, **CONTRASTS[name])


# --- Population PSTH container -----------------------------------------------
@dataclass
class PopulationPSTH:
    """Loaded PSTH subset for a set of units plus the coordinates we reuse."""

    da: Any                 # xr.DataArray with dims (unit, <trial_dim>, time)
    trial_dim: str
    trial_coord: str
    times: np.ndarray
    avail_ids: np.ndarray   # trial index values present along trial_dim
    unit_ids: np.ndarray    # unit_index values present along unit
    n_units: int


def load_population_psth(
    psth: Any,
    unit_ids: Sequence[int],
    align_to_event: str,
    time_window: Tuple[float, float],
) -> PopulationPSTH:
    """Load the PSTH for ``unit_ids`` (all trials) into a :class:`PopulationPSTH`."""
    da, _ = load_psth_raster_subset(
        psth,
        trial_ids=None,
        unit_ids=unit_ids,
        align_to_event=align_to_event,
        time_window=time_window,
        consolidated=True,
    )
    trial_dim = next(d for d in da.dims if d.startswith("trial_"))
    trial_coord = next(c for c in da.coords if c.startswith("trial_index_"))
    return PopulationPSTH(
        da=da,
        trial_dim=trial_dim,
        trial_coord=trial_coord,
        times=da.coords["time"].values,
        avail_ids=np.asarray(da.coords[trial_coord].values, dtype=np.int64),
        unit_ids=np.asarray(da.coords["unit_index"].values, dtype=np.int64),
        n_units=int(da.sizes["unit"]),
    )


def build_groups(
    nwb: Any,
    contrast: Contrast,
    avail_ids: Sequence[int],
    opto_trial_ids: Optional[Sequence[int]] = None,
    exclude_opto: bool = True,
) -> Dict[str, np.ndarray]:
    """Return ``{label_a: ids, label_b: ids}`` restricted to available trials.

    Optogenetics (laser-on) trials are dropped when ``exclude_opto`` is True.
    """
    opto = (
        np.asarray(opto_trial_ids, dtype=np.int64)
        if (exclude_opto and opto_trial_ids is not None)
        else np.array([], dtype=np.int64)
    )
    avail = np.asarray(avail_ids, dtype=np.int64)
    groups = {
        contrast.label_a: np.asarray(find_trials(nwb, contrast.type_a), dtype=np.int64),
        contrast.label_b: np.asarray(find_trials(nwb, contrast.type_b), dtype=np.int64),
    }
    groups = {k: np.setdiff1d(v, opto) for k, v in groups.items()}
    groups = {k: np.intersect1d(v, avail) for k, v in groups.items()}
    return groups


# --- Small internal helpers --------------------------------------------------
def _excl_note(exclude_opto: bool, opto_n: int) -> str:
    return f"; opto excluded (n={opto_n})" if (exclude_opto and opto_n) else ""


def _save(fig: plt.Figure, save_path: Optional[Any]) -> None:
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"saved -> {save_path}")


# --- Per-unit PSTH -----------------------------------------------------------
def plot_perunit_psth(
    psth: Any,
    nwb: Any,
    unit_ids: Sequence[int],
    contrast: Contrast,
    align_to_event: str,
    time_window: Tuple[float, float],
    opto_trial_ids: Optional[Sequence[int]] = None,
    outdir: Optional[Any] = None,
    figsize: Tuple[float, float] = (6.0, 4.0),
    show: bool = False,
) -> None:
    """Per-unit mean +/- SEM PSTH overlaying the two contrast groups."""
    save_path = None
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        save_path = str(outdir)
    plot_psth_raster_for_units(
        source=psth,
        unit_ids=unit_ids,
        trial_types=[contrast.type_a, contrast.type_b],
        nwb_data=nwb,
        exclude_trial_ids=opto_trial_ids,
        align_to_event=align_to_event,
        time_window=time_window,
        plot_type="mean",
        group_labels=[contrast.label_a, contrast.label_b],
        figsize=figsize,
        save_path=save_path,
        show=show,
    )
    print(
        f"Per-unit PSTH ({contrast.label_a} vs {contrast.label_b}) saved for "
        f"{len(unit_ids)} units -> {outdir}"
    )


# --- Population PSTH ----------------------------------------------------------
def plot_population_psth(
    pop: PopulationPSTH,
    groups: Dict[str, np.ndarray],
    contrast: Contrast,
    align_to_event: str,
    exclude_opto: bool = True,
    opto_n: int = 0,
    save_path: Optional[Any] = None,
    figsize: Tuple[float, float] = (6.0, 4.0),
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Mean +/- SEM (across units) population PSTH for each contrast group."""
    colors = {contrast.label_a: contrast.color_a, contrast.label_b: contrast.color_b}
    fig, ax = plt.subplots(figsize=figsize)
    for label, tids in groups.items():
        if tids.size == 0:
            print(f"skip: {label} (no trials)")
            continue
        sel = np.where(np.isin(pop.avail_ids, tids))[0]
        sub = pop.da.isel({pop.trial_dim: sel})          # unit, trial, time
        per_unit_mean = sub.mean(dim=pop.trial_dim)       # unit, time
        pop_mean = per_unit_mean.mean(dim="unit").values
        pop_sem = per_unit_mean.std(dim="unit").values / np.sqrt(max(pop.n_units, 1))
        color = colors.get(label)
        ax.plot(pop.times, pop_mean, color=color, label=f"{label} (n_trials={tids.size})")
        ax.fill_between(pop.times, pop_mean - pop_sem, pop_mean + pop_sem, color=color, alpha=0.3)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(f"Time from {align_to_event} (s)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(
        f"Population PSTH {contrast.label_a} vs {contrast.label_b}, "
        f"{pop.n_units} tagged units{_excl_note(exclude_opto, opto_n)}"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig, ax


# --- Windowed per-unit rates -------------------------------------------------
def window_rates(
    pop: PopulationPSTH,
    groups: Dict[str, np.ndarray],
    contrast: Contrast,
    scatter_win: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-unit mean rate over ``scatter_win`` for group A (x) and group B (y)."""
    time_idx = np.where((pop.times >= scatter_win[0]) & (pop.times <= scatter_win[1]))[0]

    def _rate(tids: np.ndarray) -> np.ndarray:
        sel = np.where(np.isin(pop.avail_ids, tids))[0]
        sub = pop.da.isel({pop.trial_dim: sel}).isel(time=time_idx)
        return sub.mean(dim=[pop.trial_dim, "time"]).values

    x = np.asarray(_rate(groups[contrast.label_a]), dtype=float)
    y = np.asarray(_rate(groups[contrast.label_b]), dtype=float)
    return x, y


# --- Scatter -----------------------------------------------------------------
def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    contrast: Contrast,
    scatter_win: Tuple[float, float],
    n_units: int,
    exclude_opto: bool = True,
    opto_n: int = 0,
    save_path: Optional[Any] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Per-unit scatter of group-A (x) vs group-B (y) windowed mean rate."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=30, c="tab:purple", edgecolor="k", linewidth=0.4, alpha=0.8)
    lim = float(np.nanmax([np.nanmax(x), np.nanmax(y), 1e-6])) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="unity")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{contrast.label_a} rate (Hz)  [{scatter_win[0]}-{scatter_win[1]} s]")
    ax.set_ylabel(f"{contrast.label_b} rate (Hz)  [{scatter_win[0]}-{scatter_win[1]} s]")
    ax.set_title(
        f"{contrast.label_a} vs {contrast.label_b}, {n_units} tagged units"
        f"{_excl_note(exclude_opto, opto_n)}"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig, ax


# --- Paired difference + Wilcoxon --------------------------------------------
def paired_diff(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Signed per-unit difference ``B - A`` plus a Wilcoxon signed-rank test."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    diff = b - a
    valid = np.isfinite(a) & np.isfinite(b)
    diff_v = diff[valid]
    nonzero = diff_v[diff_v != 0]
    if nonzero.size >= 1:
        w_stat, p_val = wilcoxon(nonzero)
    else:
        w_stat, p_val = np.nan, np.nan
    return dict(
        diff=diff,
        diff_valid=diff_v,
        n_valid=int(valid.sum()),
        n_up=int(np.sum(diff_v > 0)),     # more on group B
        n_down=int(np.sum(diff_v < 0)),   # more on group A
        median=float(np.median(diff_v)) if diff_v.size else float("nan"),
        mean=float(np.mean(diff_v)) if diff_v.size else float("nan"),
        w_stat=float(w_stat),
        p_val=float(p_val),
    )


def print_paired_diff(stats: Dict[str, Any], contrast: Contrast) -> None:
    """Pretty-print the result dict from :func:`paired_diff`."""
    print(f"n units (valid)          : {stats['n_valid']}")
    print(f"more on {contrast.label_b:<14}: {stats['n_up']}")
    print(f"more on {contrast.label_a:<14}: {stats['n_down']}")
    print(f"median ({contrast.label_b} - {contrast.label_a}) : {stats['median']:+.3f} Hz")
    print(f"mean   ({contrast.label_b} - {contrast.label_a}) : {stats['mean']:+.3f} Hz")
    print(f"Wilcoxon signed-rank     : W={stats['w_stat']:.1f}, p={stats['p_val']:.3g}")


def plot_paired_diff(
    stats: Dict[str, Any],
    contrast: Contrast,
    scatter_win: Tuple[float, float],
    exclude_opto: bool = True,
    opto_n: int = 0,
    save_path: Optional[Any] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Histogram of per-unit ``B - A`` differences with the median marked."""
    diff_v = stats["diff_valid"]
    fig, ax = plt.subplots(figsize=(6, 4))
    lim = float(np.nanmax(np.abs(diff_v))) * 1.05 if diff_v.size else 1.0
    bins = np.linspace(-lim, lim, 31)
    ax.hist(diff_v, bins=bins, color="tab:gray", edgecolor="k", linewidth=0.4)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.axvline(stats["median"], color="tab:red", lw=1.5,
               label=f"median = {stats['median']:+.2f} Hz")
    ax.set_xlabel(
        f"{contrast.label_b} - {contrast.label_a} rate (Hz)  "
        f"[{scatter_win[0]}-{scatter_win[1]} s]"
    )
    ax.set_ylabel("Unit count")
    ax.set_title(
        f"Paired diff, {stats['n_valid']} units (Wilcoxon p={stats['p_val']:.3g})"
        f"{_excl_note(exclude_opto, opto_n)}"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig, ax


# --- Scatter colored by signed difference ------------------------------------
def plot_scatter_colored(
    x: np.ndarray,
    y: np.ndarray,
    diff: np.ndarray,
    contrast: Contrast,
    scatter_win: Tuple[float, float],
    n_valid: int,
    exclude_opto: bool = True,
    opto_n: int = 0,
    save_path: Optional[Any] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Group-A vs group-B scatter with points colored by signed ``B - A``."""
    fig, ax = plt.subplots(figsize=(5.6, 5))
    cmax = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-cmax, vcenter=0.0, vmax=cmax)
    sc = ax.scatter(x, y, c=diff, cmap="coolwarm", norm=norm,
                    s=34, edgecolor="k", linewidth=0.4, alpha=0.9)
    lim = float(np.nanmax([np.nanmax(x), np.nanmax(y), 1e-6])) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="unity")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{contrast.label_a} rate (Hz)  [{scatter_win[0]}-{scatter_win[1]} s]")
    ax.set_ylabel(f"{contrast.label_b} rate (Hz)  [{scatter_win[0]}-{scatter_win[1]} s]")
    ax.set_title(
        f"{contrast.label_a} vs {contrast.label_b}, {n_valid} units"
        f"{_excl_note(exclude_opto, opto_n)}"
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{contrast.label_b} - {contrast.label_a} (Hz)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig, ax


# --- Trial-count matched control ---------------------------------------------
def trialcount_matched(
    pop: PopulationPSTH,
    groups: Dict[str, np.ndarray],
    contrast: Contrast,
    scatter_win: Tuple[float, float],
    align_to_event: str,
    n_boot: int = 200,
    seed: int = 0,
    exclude_opto: bool = True,
    opto_n: int = 0,
    save_path: Optional[Any] = None,
    show: bool = True,
) -> Tuple[Dict[str, Any], Tuple[plt.Figure, plt.Axes]]:
    """Subsample the larger group to the smaller group's trial count.

    Returns a summary dict and the ``(fig, ax)`` of the matched population PSTH.
    """
    rng = np.random.default_rng(seed)
    time_idx = np.where((pop.times >= scatter_win[0]) & (pop.times <= scatter_win[1]))[0]

    ids_a = np.asarray(groups[contrast.label_a], dtype=np.int64)
    ids_b = np.asarray(groups[contrast.label_b], dtype=np.int64)
    if ids_a.size >= ids_b.size:
        big_ids, big_label, big_color = ids_a, contrast.label_a, contrast.color_a
        small_ids, small_label, small_color = ids_b, contrast.label_b, contrast.color_b
    else:
        big_ids, big_label, big_color = ids_b, contrast.label_b, contrast.color_b
        small_ids, small_label, small_color = ids_a, contrast.label_a, contrast.color_a
    n_match = int(small_ids.size)

    def _curve(tids: np.ndarray) -> np.ndarray:
        sel = np.where(np.isin(pop.avail_ids, tids))[0]
        sub = pop.da.isel({pop.trial_dim: sel})
        return sub.mean(dim=pop.trial_dim).mean(dim="unit").values

    def _win(curve: np.ndarray) -> float:
        return float(np.mean(curve[time_idx]))

    small_curve = _curve(small_ids)
    small_win = _win(small_curve)

    boot_curves = np.empty((n_boot, pop.times.size), dtype=float)
    boot_win = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(big_ids, size=n_match, replace=False)
        c = _curve(pick)
        boot_curves[b] = c
        boot_win[b] = _win(c)

    big_mean = boot_curves.mean(axis=0)
    big_std = boot_curves.std(axis=0)
    frac_big_higher = float(np.mean(boot_win > small_win))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pop.times, small_curve, color=small_color, label=f"{small_label} (n={n_match})")
    ax.plot(pop.times, big_mean, color=big_color, label=f"{big_label} (matched n={n_match})")
    ax.fill_between(pop.times, big_mean - big_std, big_mean + big_std, color=big_color, alpha=0.25)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.axvspan(scatter_win[0], scatter_win[1], color="gray", alpha=0.12, lw=0)
    ax.set_xlabel(f"Time from {align_to_event} (s)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(
        f"Trial-count matched, {pop.n_units} units ({n_boot} subsamples)"
        f"{_excl_note(exclude_opto, opto_n)}"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()

    results = dict(
        n_match=n_match,
        small_label=small_label,
        big_label=big_label,
        small_win=small_win,
        big_win_mean=float(boot_win.mean()),
        big_win_std=float(boot_win.std()),
        frac_big_higher=frac_big_higher,
        n_boot=n_boot,
    )
    return results, (fig, ax)
