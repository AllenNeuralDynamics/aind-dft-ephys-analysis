# -*- coding: utf-8 -*-
"""
Project stimulation trials onto behavioural coding directions (CD).

Hypothesis tooling for: *does stimulation move population activity along the
natural disengagement axis?*

Workflow
--------
1. Build a population coding direction (CD) from **unstimulated** trials only,
   for one of two contrasts:
     - "response" : engaged-response (+) vs no-response (-)
     - "reward"   : rewarded (+)         vs unrewarded (-)
2. Project **stimulation** trials of a chosen condition onto that unstim axis.
3. Ask whether stimulation pushes the population toward the natural no-response
   (disengaged) pole, and whether the per-trial shift predicts behaviour.

This leans on ``ephys_dimension_reduction_CD.coding_direction_from_psth`` which
fits the CD on the supplied A/B trials and already returns projections for
*every* trial in the PSTH (including trials not used for fitting, i.e. the
stimulation trials).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from behavior_utils import find_trials
from ephys_dimension_reduction_CD import coding_direction_from_psth


# Axis definitions: name -> (class A type, class B type, labels, engaged pole).
# The CD is built as (class A - class B). We put the *disengaged / no-reward*
# pole as class A so the axis is literally ``no_response - response`` and
# ``no_reward - reward`` (positive projection = toward the disengaged pole).
# ``engaged`` records which class ('a' or 'b') is the engaged/rewarded pole so
# the disengagement index and stats keep a fixed meaning regardless of sign.
AXES: Dict[str, Dict[str, str]] = {
    "response": dict(
        type_a="no_response", type_b="response",
        label_a="disengaged (no-response)", label_b="engaged (response)",
        engaged="b",
    ),
    "reward": dict(
        type_a="unrewarded", type_b="rewarded",
        label_a="no-reward", label_b="reward",
        engaged="b",
    ),
}


def resolve_axis(name: str) -> Dict[str, str]:
    """Return the axis spec dict for ``name`` (adds a 'name' key)."""
    if name not in AXES:
        raise KeyError(f"Unknown axis {name!r}; choose from {list(AXES)}.")
    return dict(name=name, **AXES[name])


# ---------------------------------------------------------------------------
# NWB helpers
# ---------------------------------------------------------------------------

def laser_trial_ids(nwb) -> np.ndarray:
    """0-based absolute trial IDs where ``laser_on_trial == 1`` (empty if absent)."""
    if "laser_on_trial" in nwb.trials.colnames:
        laser_on = np.asarray(nwb.trials["laser_on_trial"][:])
        return np.where(laser_on == 1)[0].astype(np.int64)
    return np.array([], dtype=np.int64)


def responded_flags(nwb) -> np.ndarray:
    """Per-trial responded flag (1 = animal responded, 0 = no-response)."""
    resp = np.asarray(nwb.trials["animal_response"][:])
    return (resp != 2).astype(int)


def stim_ids_for_axis(nwb, axis_name, trial_ids, *, exclude_no_response=None) -> np.ndarray:
    """
    Filter a set of stimulation trials for projection onto a given axis.

    No-response trials have no reward outcome, so they are meaningless on the
    ``reward`` axis. When ``exclude_no_response`` is ``None`` (default) they are
    dropped automatically for the reward axis and kept otherwise; pass a bool to
    force the behaviour. Returns the (possibly reduced) trial ids.
    """
    ids = np.asarray(trial_ids, dtype=np.int64).ravel()
    name = axis_name["name"] if isinstance(axis_name, dict) else str(axis_name)
    drop = (name == "reward") if exclude_no_response is None else bool(exclude_no_response)
    if not drop or ids.size == 0:
        return ids
    resp = responded_flags(nwb)
    keep = [int(t) for t in ids if int(t) < resp.size and resp[int(t)] == 1]
    return np.asarray(keep, dtype=np.int64)


def select_units(
    nwb,
    psth,
    *,
    use_tagged: bool = False,
    probes: Optional[Sequence[str]] = None,
    metrics_csv_path=None,
    target_cond: Optional[Sequence] = None,
    pulse_index: int = -1,
) -> list:
    """
    Select units by combinable filters, mirroring the PSTH notebooks.

    Starts from the default-QC units, then optionally keeps only opto-tagged
    units (``use_tagged``) and/or only units on the given NWB ``device_name``
    ``probes``. If both are set, a unit must satisfy BOTH. The result is
    intersected with the units present in ``psth``.
    """
    device_names = np.asarray(nwb.units["device_name"][:]).astype(str)

    from ephys_behavior import get_units_passed_default_qc

    sel = set(int(u) for u in np.asarray(get_units_passed_default_qc(nwb), dtype=np.int64))

    if use_tagged:
        import re
        import pandas as pd
        from ast import literal_eval
        from optical_tagging import select_tagged_units

        if metrics_csv_path is None or target_cond is None:
            raise ValueError("use_tagged=True requires metrics_csv_path and target_cond.")
        tagged_df = select_tagged_units(
            pd.read_csv(str(metrics_csv_path)),
            alpha=0.05, effect_ratio=2.0, min_abs_increase=0.0,
            min_reliability=0.2, max_latency=0.006, max_jitter=0.003, correction="fdr_bh",
        )

        def _as_tuple(c):
            if isinstance(c, str):
                c = re.sub(r"np\.\w+\(([^()]*)\)", r"\1", c)
                try:
                    c = literal_eval(c)
                except (ValueError, SyntaxError):
                    return None
            return tuple(c) if isinstance(c, (tuple, list)) else None

        _match_idx = [i for i in range(6) if i != 2]  # ignore laser_name (index 2)

        def _matches(c):
            c = _as_tuple(c)
            return c is not None and len(c) == 6 and all(c[i] == target_cond[i] for i in _match_idx)

        rows = tagged_df[
            tagged_df["condition"].apply(_matches)
            & (tagged_df["pulse_index"] == pulse_index)
            & (tagged_df["tagged"] == True)
        ]
        sel &= set(int(u) for u in rows["unit_id"].unique())

    if probes:
        want = [str(p) for p in probes]
        sel &= set(int(u) for u in np.where(np.isin(device_names, want))[0])

    unit_ids = sorted(sel)
    if "unit_index" in psth.coords:
        pset = {int(u) for u in np.asarray(psth.coords["unit_index"].values)}
        unit_ids = [u for u in unit_ids if u in pset]
    return unit_ids


def _psth_trial_ids(psth, align: str) -> np.ndarray:
    """Absolute trial IDs available in the PSTH for the chosen alignment."""
    coord = f"trial_index_{align}"
    if coord in psth.coords:
        return np.asarray(psth.coords[coord].values, dtype=np.int64)
    for c in psth.coords:
        if str(c).startswith("trial_index_"):
            return np.asarray(psth.coords[c].values, dtype=np.int64)
    raise KeyError("No 'trial_index_<align>' coordinate found in the PSTH.")


def _snap_window(psth, window, *, verbose: bool = True):
    """
    Return a time window guaranteed to contain >= 1 PSTH sample.

    If ``window`` already captures at least one bin (``time >= t0 & time < t1``)
    it is returned unchanged. Otherwise (e.g. a sub-bin window on a coarse PSTH)
    it is snapped to the single bin nearest the window midpoint, using the median
    bin width to set the returned edges. Returns ``(snapped_window, n_bins)``.
    """
    time = np.asarray(psth["time"].values, dtype=float)
    t0, t1 = float(window[0]), float(window[1])
    n = int(((time >= t0) & (time < t1)).sum())
    if n >= 1:
        return (t0, t1), n
    mid = 0.5 * (t0 + t1)
    j = int(np.argmin(np.abs(time - mid)))
    dt = float(np.median(np.diff(time))) if time.size >= 2 else 1.0
    lo, hi = float(time[j] - dt / 2.0), float(time[j] + dt / 2.0)
    if verbose:
        print(
            f"  [snap] window {tuple(window)} spans no PSTH bin (bin width ~{dt:.3g}s); "
            f"using nearest bin centered at {time[j]:.3g}s -> ({lo:.3g}, {hi:.3g})"
        )
    return (lo, hi), 1


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AxisResult:
    """Fitted CD axis plus per-trial projections for every PSTH trial."""

    axis: Dict[str, str]
    unit_ids: np.ndarray
    time: np.ndarray
    align: str
    fit_window: Tuple[float, float]
    proj_window: Optional[Tuple[float, float]]
    ids_a: np.ndarray                     # unstim class-A trial ids used for the fit
    ids_b: np.ndarray                     # unstim class-B trial ids used for the fit
    proj_by_id: Dict[int, float]          # trial id -> scalar projection (fit window, unbiased)
    trace_by_id: Dict[int, np.ndarray]    # trial id -> time-resolved projection (unbiased)
    mean_a: float                         # mean projection of unstim class A (type_a pole)
    mean_b: float                         # mean projection of unstim class B (type_b pole)
    dprime: float
    auc: float
    engaged: str = "a"                     # which class ('a'/'b') is the engaged pole
    raw: Optional[dict] = field(repr=False, default=None)

    # -- pole accessors (engaged vs disengaged, independent of CD sign) -----
    @property
    def ids_engaged(self) -> np.ndarray:
        return self.ids_b if self.engaged == "b" else self.ids_a

    @property
    def ids_diseng(self) -> np.ndarray:
        return self.ids_a if self.engaged == "b" else self.ids_b

    @property
    def mean_engaged(self) -> float:
        return self.mean_b if self.engaged == "b" else self.mean_a

    @property
    def mean_diseng(self) -> float:
        return self.mean_a if self.engaged == "b" else self.mean_b

    @property
    def label_engaged(self) -> str:
        return self.axis["label_b"] if self.engaged == "b" else self.axis["label_a"]

    @property
    def label_diseng(self) -> str:
        return self.axis["label_a"] if self.engaged == "b" else self.axis["label_b"]

    # -- projection lookups ------------------------------------------------
    def project(self, trial_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Scalar (fit-window) projections for ``trial_ids`` present in the PSTH."""
        tids = np.asarray(trial_ids, dtype=np.int64).ravel()
        tids = np.array([t for t in tids if int(t) in self.proj_by_id], dtype=np.int64)
        scal = np.array([self.proj_by_id[int(t)] for t in tids], dtype=float)
        return tids, scal

    def project_traces(self, trial_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Time-resolved projections for ``trial_ids`` present in the PSTH."""
        tids = np.asarray(trial_ids, dtype=np.int64).ravel()
        tids = np.array([t for t in tids if int(t) in self.trace_by_id], dtype=np.int64)
        tr = np.array([self.trace_by_id[int(t)] for t in tids], dtype=float)
        return tids, tr

    def disengagement_index(self, scal) -> np.ndarray:
        """Rescale projections so 0 = engaged pole, 1 = disengaged pole."""
        scal = np.asarray(scal, dtype=float)
        denom = self.mean_diseng - self.mean_engaged
        if abs(denom) < 1e-12:
            return np.full_like(scal, np.nan)
        return (scal - self.mean_engaged) / denom


# ---------------------------------------------------------------------------
# Build a CD axis from unstimulated trials
# ---------------------------------------------------------------------------

def build_axis(
    psth,
    nwb,
    unit_ids,
    axis_name: str,
    *,
    align: str = "go_cue",
    fit_window: Tuple[float, float] = (0.0, 0.5),
    proj_window: Optional[Tuple[float, float]] = None,
    exclude_trial_ids: Optional[Sequence[int]] = None,
    zscore_units: bool = False,
    random_state: int = 0,
) -> AxisResult:
    """
    Fit a coding direction on **unstimulated** trials for the given contrast.

    Parameters
    ----------
    psth : xr.Dataset
        PSTH with ``psth_<align>`` data vars and ``trial_index_<align>`` coords.
    nwb : NWB-like
        Provides ``trials`` (for class membership and laser flags).
    unit_ids : array-like of int
        Units to build the axis from (absolute NWB unit indices).
    axis_name : {"response", "reward"}
        Which behavioural contrast defines the axis.
    align, fit_window, proj_window : alignment + windows (see CD module).
    exclude_trial_ids : array-like of int, optional
        Trials excluded from the FIT (defaults to all laser-on trials). The
        stimulation trials you later project should be in this excluded set so
        the axis is purely unstimulated.
    zscore_units : bool
        Per-unit z-score inside the CD fit (default False = raw-rate CD).
    random_state : int
        Seed for the balanced half-split.
    """
    axis = resolve_axis(axis_name)
    avail = set(int(t) for t in _psth_trial_ids(psth, align))

    # Snap sub-bin windows to the nearest PSTH bin so coarse PSTHs (e.g. 0.2 s
    # bins) don't silently select zero samples and abort the CD fit.
    fit_window, _ = _snap_window(psth, fit_window)
    if proj_window is not None:
        proj_window, _ = _snap_window(psth, proj_window)

    if exclude_trial_ids is None:
        excl = set(int(t) for t in laser_trial_ids(nwb))
    else:
        excl = set(int(t) for t in np.asarray(exclude_trial_ids, dtype=np.int64))

    ids_a = np.array(
        sorted((set(int(t) for t in find_trials(nwb, axis["type_a"])) & avail) - excl),
        dtype=np.int64,
    )
    ids_b = np.array(
        sorted((set(int(t) for t in find_trials(nwb, axis["type_b"])) & avail) - excl),
        dtype=np.int64,
    )
    if ids_a.size < 2 or ids_b.size < 2:
        raise ValueError(
            f"Too few unstimulated trials for axis {axis_name!r}: "
            f"A({axis['type_a']})={ids_a.size}, B({axis['type_b']})={ids_b.size}."
        )

    res = coding_direction_from_psth(
        psth,
        trial_ids_typeA=ids_a,
        trial_ids_typeB=ids_b,
        align=align,
        time_window=fit_window,
        projection_time_window=proj_window,
        random_state=random_state,
        two_fold_cv=True,
        norm_mode="divide_sqrtN",
        unit_ids=np.asarray(unit_ids, dtype=np.int64),
        zscore_units=zscore_units,
        save_path=None,
    )

    tid_all = np.asarray(res["trial_ids_all_trials"], dtype=np.int64)
    proj_all = np.asarray(res["projection_unbiased_all_trials"], dtype=float)
    trace_all = np.asarray(res["projection_trace_unbiased_all_trials"], dtype=float)
    proj_by_id = {int(t): float(p) for t, p in zip(tid_all, proj_all)}
    trace_by_id = {int(t): trace_all[i] for i, t in enumerate(tid_all)}

    mean_a = float(np.nanmean([proj_by_id[int(t)] for t in ids_a if int(t) in proj_by_id]))
    mean_b = float(np.nanmean([proj_by_id[int(t)] for t in ids_b if int(t) in proj_by_id]))

    return AxisResult(
        axis=axis,
        unit_ids=np.asarray(unit_ids, dtype=np.int64),
        time=np.asarray(res["time_for_projection"], dtype=float),
        align=align,
        fit_window=fit_window,
        proj_window=proj_window,
        ids_a=ids_a,
        ids_b=ids_b,
        proj_by_id=proj_by_id,
        trace_by_id=trace_by_id,
        mean_a=mean_a,
        mean_b=mean_b,
        dprime=float(res["metrics"]["overall"]["dprime"]),
        auc=float(res["metrics"]["overall"]["auc"]),
        engaged=axis.get("engaged", "a"),
        raw=res,
    )


# ---------------------------------------------------------------------------
# Stimulation-condition projection + stats
# ---------------------------------------------------------------------------

def _auc_binary(labels01, scores) -> float:
    """ROC-AUC with labels in {0, 1}; tie-aware (average ranks)."""
    labels01 = np.asarray(labels01, dtype=int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    m = np.isfinite(scores)
    labels01, scores = labels01[m], scores[m]
    pos = labels01 == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # 1-based average rank
        i = j + 1
    u = ranks[pos].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def condition_shift(
    res: AxisResult,
    cond_trial_ids: Sequence[int],
    nwb,
    *,
    exclude_no_response=None,
) -> Dict:
    """
    Quantify how far a stimulation condition shifts along the CD axis and
    whether that shift predicts behaviour on the stimulated trials.

    Returns a dict with:
      - n_stim, mean_di, sem_di : disengagement index (0=engaged, 1=disengaged)
      - p_toward_disengaged     : Mann-Whitney one-sided p (stim more disengaged
                                  than unstim engaged-pole trials)
      - behavior_auc            : AUC using the per-trial projection to predict
                                  no-response on the stimulated trials
      - trial_ids, proj, di     : per-trial arrays for plotting
    """
    from scipy.stats import mannwhitneyu

    cond_trial_ids = stim_ids_for_axis(
        nwb, res.axis, cond_trial_ids, exclude_no_response=exclude_no_response
    )
    tids, scal = res.project(cond_trial_ids)
    _, a_scal = res.project(res.ids_engaged)
    di = res.disengagement_index(scal)
    a_di = res.disengagement_index(a_scal)

    if di.size and a_di.size:
        try:
            u, p = mannwhitneyu(di, a_di, alternative="greater")
        except ValueError:
            u, p = float("nan"), float("nan")
    else:
        u, p = float("nan"), float("nan")

    resp = responded_flags(nwb)
    no_resp01 = np.array([1 - int(resp[int(t)]) for t in tids], dtype=int)  # 1 = no-response
    beh_auc = _auc_binary(no_resp01, di)  # higher disengagement -> no-response?

    return dict(
        axis=res.axis["name"],
        n_stim=int(tids.size),
        mean_di=float(np.nanmean(di)) if di.size else float("nan"),
        sem_di=float(np.nanstd(di) / np.sqrt(max(di.size, 1))) if di.size else float("nan"),
        mean_a=res.mean_a,
        mean_b=res.mean_b,
        mannwhitney_u=float(u) if np.isfinite(u) else float("nan"),
        p_toward_disengaged=float(p) if np.isfinite(p) else float("nan"),
        n_no_response=int(no_resp01.sum()),
        behavior_auc=beh_auc,
        trial_ids=tids,
        proj=scal,
        di=di,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_axis_distributions(
    res: AxisResult,
    cond_trial_ids: Sequence[int],
    *,
    stim_label: str = "stim",
    nwb=None,
    exclude_no_response=None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (6, 4),
):
    """Strip plot of unstim-A, unstim-B and stim fit-window projections."""
    import matplotlib.pyplot as plt

    if nwb is not None:
        cond_trial_ids = stim_ids_for_axis(
            nwb, res.axis, cond_trial_ids, exclude_no_response=exclude_no_response
        )

    rng = np.random.RandomState(0)
    _, eng = res.project(res.ids_engaged)
    _, dis = res.project(res.ids_diseng)
    _, s = res.project(cond_trial_ids)
    groups = [eng, dis, s]
    names = [res.label_engaged, res.label_diseng, stim_label]
    colors = ["tab:blue", "tab:red", "tab:green"]

    fig, ax = plt.subplots(figsize=figsize)
    for i, (vals, c) in enumerate(zip(groups, colors)):
        if vals.size == 0:
            continue
        ax.scatter(np.full(vals.size, i) + rng.uniform(-0.09, 0.09, vals.size),
                   vals, s=12, alpha=0.4, color=c)
        ax.hlines(np.nanmean(vals), i - 0.22, i + 0.22, color=c, lw=2.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylabel(f"CD projection  ({res.label_diseng} +, {res.label_engaged} −)")
    ax.set_title(f"{res.axis['name']} axis — d'={res.dprime:.2f}, AUC={res.auc:.2f}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_axis_traces(
    res: AxisResult,
    cond_trial_ids: Sequence[int],
    *,
    stim_label: str = "stim",
    nwb=None,
    exclude_no_response=None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (6, 4),
):
    """Time-resolved mean ± SEM CD projection for unstim-A, unstim-B and stim."""
    import matplotlib.pyplot as plt

    if nwb is not None:
        cond_trial_ids = stim_ids_for_axis(
            nwb, res.axis, cond_trial_ids, exclude_no_response=exclude_no_response
        )

    def mean_sem(ids):
        _, tr = res.project_traces(ids)
        if tr.size == 0:
            return None, None
        return np.nanmean(tr, axis=0), np.nanstd(tr, axis=0) / np.sqrt(max(tr.shape[0], 1))

    t = res.time
    fig, ax = plt.subplots(figsize=figsize)
    for ids, nm, c in [
        (res.ids_engaged, res.label_engaged, "tab:blue"),
        (res.ids_diseng, res.label_diseng, "tab:red"),
        (np.asarray(cond_trial_ids), stim_label, "tab:green"),
    ]:
        m, sem = mean_sem(ids)
        if m is None:
            continue
        ax.plot(t, m, color=c, label=nm)
        ax.fill_between(t, m - sem, m + sem, color=c, alpha=0.2)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(f"Time from {res.align} (s)")
    ax.set_ylabel("CD projection")
    ax.set_title(f"{res.axis['name']} axis — time-resolved")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_trace_grid(
    axis_cache: dict,
    stim_conditions: dict,
    *,
    axis_name: str,
    condition,
    sel_order: Sequence[str],
    window_order: Sequence[str],
    stim_label: Optional[str] = None,
    nwb=None,
    exclude_no_response=None,
    sharey: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Grid of time-resolved CD projections across unit-selection x fit-window.

    One combined figure for a single ``axis_name`` and stim ``condition``: rows
    are unit selections, columns are fit windows. Each panel shows mean ± SEM
    traces for unstim class A, unstim class B and the stimulation trials, reusing
    the per-combo :class:`AxisResult` objects stored in ``axis_cache`` (keyed by
    ``(selection, window_label, axis_name)``).
    """
    import matplotlib.pyplot as plt

    sel_order = list(sel_order)
    window_order = list(window_order)
    nrow, ncol = len(sel_order), len(window_order)
    if figsize is None:
        figsize = (3.4 * ncol, 2.7 * nrow)
    cond_ids = np.asarray(stim_conditions.get(condition, []), dtype=np.int64)
    if nwb is not None:
        cond_ids = stim_ids_for_axis(
            nwb, axis_name, cond_ids, exclude_no_response=exclude_no_response
        )
    slabel = stim_label if stim_label is not None else f"stim (cond {condition})"

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=sharey, squeeze=False)

    def mean_sem(res, ids):
        _, tr = res.project_traces(ids)
        if tr.size == 0:
            return None, None
        return np.nanmean(tr, axis=0), np.nanstd(tr, axis=0) / np.sqrt(max(tr.shape[0], 1))

    for r, sel in enumerate(sel_order):
        for c, win in enumerate(window_order):
            ax = axes[r][c]
            res = axis_cache.get((sel, win, axis_name))
            if res is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, color="0.6")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                t = res.time
                for ids, nm, col in [
                    (res.ids_engaged, res.label_engaged, "tab:blue"),
                    (res.ids_diseng, res.label_diseng, "tab:red"),
                    (cond_ids, slabel, "tab:green"),
                ]:
                    m, sem = mean_sem(res, ids)
                    if m is None:
                        continue
                    ax.plot(t, m, color=col, lw=1.2, label=nm)
                    ax.fill_between(t, m - sem, m + sem, color=col, alpha=0.18)
                ax.axvline(0, color="k", ls="--", lw=0.7)
                if r == 0 and c == ncol - 1:
                    ax.legend(frameon=False, fontsize=7, loc="best")
            if r == 0:
                ax.set_title(win, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{sel}\nCD proj", fontsize=8)
            if r == nrow - 1:
                ax.set_xlabel("Time from go cue (s)", fontsize=8)

    fig.suptitle(
        f"Time-resolved projection — axis={axis_name}, stim condition {condition}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def plot_strip_grid(
    axis_cache: dict,
    stim_conditions: dict,
    *,
    axis_name: str,
    condition,
    sel_order: Sequence[str],
    window_order: Sequence[str],
    stim_label: Optional[str] = None,
    nwb=None,
    exclude_no_response=None,
    sharey: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Grid of fit-window projection strip plots across unit-selection x fit-window.

    One combined figure for a single ``axis_name`` and stim ``condition``: rows
    are unit selections, columns are fit windows. Each panel is the strip plot
    (engaged / disengaged / stim fit-window projections with mean bars) that
    :func:`plot_axis_distributions` draws, reusing the per-combo
    :class:`AxisResult` objects in ``axis_cache`` (keyed by
    ``(selection, window_label, axis_name)``).
    """
    import matplotlib.pyplot as plt

    sel_order = list(sel_order)
    window_order = list(window_order)
    nrow, ncol = len(sel_order), len(window_order)
    if figsize is None:
        figsize = (3.1 * ncol, 2.7 * nrow)
    cond_ids = np.asarray(stim_conditions.get(condition, []), dtype=np.int64)
    if nwb is not None:
        cond_ids = stim_ids_for_axis(
            nwb, axis_name, cond_ids, exclude_no_response=exclude_no_response
        )
    slabel = stim_label if stim_label is not None else f"stim (cond {condition})"
    rng = np.random.RandomState(0)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=sharey, squeeze=False)

    for r, sel in enumerate(sel_order):
        for c, win in enumerate(window_order):
            ax = axes[r][c]
            res = axis_cache.get((sel, win, axis_name))
            if res is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, color="0.6")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                _, eng = res.project(res.ids_engaged)
                _, dis = res.project(res.ids_diseng)
                _, s = res.project(cond_ids)
                for i, (vals, col) in enumerate(
                    [(eng, "tab:blue"), (dis, "tab:red"), (s, "tab:green")]
                ):
                    if vals.size == 0:
                        continue
                    ax.scatter(np.full(vals.size, i) + rng.uniform(-0.09, 0.09, vals.size),
                               vals, s=9, alpha=0.35, color=col)
                    ax.hlines(np.nanmean(vals), i - 0.22, i + 0.22, color=col, lw=2.2)
                ax.set_xticks(range(3))
                ax.set_xticklabels(["eng", "diseng", "stim"], fontsize=7)
                ax.set_title(f"d'={res.dprime:.2f}", fontsize=7)
            if r == 0:
                ax.annotate(win, xy=(0.5, 1.12), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{sel}\nCD proj", fontsize=8)

    fig.suptitle(
        f"Fit-window projections — axis={axis_name}, stim condition {condition} "
        f"(blue=engaged, red=disengaged, green=stim)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def plot_sweep_summary(
    df,
    *,
    metric: str = "mean_di",
    err: Optional[str] = "sem_di",
    axis_order: Optional[Sequence[str]] = None,
    window_order: Optional[Sequence[str]] = None,
    sel_order: Optional[Sequence[str]] = None,
    cond_order: Optional[Sequence] = None,
    title: str = "Stim projection onto unstim CD — disengagement index",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Collapse a full parameter sweep into one grid figure.

    Expects a tidy DataFrame with columns ``axis``, ``window_label``,
    ``selection``, ``condition`` and the ``metric`` (+ optional ``err``).
    Grid: one row per axis, one column per fit window; within each panel the
    x-axis is the unit selection and bars are grouped by stimulation condition.
    Dashed guides mark the engaged (0) and disengaged (1) poles when plotting
    the disengagement index.
    """
    import matplotlib.pyplot as plt

    axis_order = list(axis_order) if axis_order is not None else sorted(df["axis"].unique())
    window_order = list(window_order) if window_order is not None else list(dict.fromkeys(df["window_label"]))
    sel_order = list(sel_order) if sel_order is not None else list(dict.fromkeys(df["selection"]))
    cond_order = list(cond_order) if cond_order is not None else sorted(df["condition"].unique())

    nrows, ncols = len(axis_order), len(window_order)
    if figsize is None:
        figsize = (3.6 * ncols, 3.1 * nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharey=True)

    cmap = plt.get_cmap("tab10")
    x = np.arange(len(sel_order))
    bw = 0.8 / max(len(cond_order), 1)
    is_di = metric == "mean_di"

    for r, ax_name in enumerate(axis_order):
        for c, win in enumerate(window_order):
            ax = axs[r][c]
            sub = df[(df["axis"] == ax_name) & (df["window_label"] == win)]
            for k, cond in enumerate(cond_order):
                vals, errs = [], []
                for s in sel_order:
                    row = sub[(sub["selection"] == s) & (sub["condition"] == cond)]
                    vals.append(float(row[metric].iloc[0]) if len(row) else np.nan)
                    errs.append(float(row[err].iloc[0]) if (err and len(row)) else 0.0)
                ax.bar(
                    x + (k - (len(cond_order) - 1) / 2) * bw, vals, width=bw,
                    yerr=errs, capsize=2, color=cmap(k), label=f"cond {cond}",
                )
            if is_di:
                ax.axhline(0, color="tab:blue", ls="--", lw=0.8)
                ax.axhline(1, color="tab:red", ls="--", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(sel_order, rotation=30, ha="right", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{ax_name}\n{metric}")
            if r == 0:
                ax.set_title(win, fontsize=9)
    axs[0][-1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axs
