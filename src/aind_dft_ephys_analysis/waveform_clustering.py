# ============================================================
# Future annotations (must be first)
# ============================================================
from __future__ import annotations

# ============================================================
# Standard library
# ============================================================
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# ============================================================
# Third-party libraries
# ============================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# ============================================================
# Local / project imports
# ============================================================
from nwb_utils import NWBUtils
from ephys_utils import append_units_locations
from ephys_behavior import get_units_passed_default_qc
from general_utils import extract_session_name_core


# ============================================================
# Configuration defaults
# ============================================================
#: Waveform sampling rate (Neuropixels standard). All duration features scale
#: with this value, so confirm it matches your recording.
DEFAULT_SAMPLING_RATE_HZ = 30_000.0

#: Fixed trough-aligned extraction window (in ms). ``PRE_MS`` before the trough
#: and ``POST_MS`` after, so waveforms from probes/sessions with different
#: sample counts can be pooled together.
DEFAULT_PRE_MS = 1.0
DEFAULT_POST_MS = 2.0

#: Morphology features produced by :func:`extract_features`.
FEATURE_COLS: List[str] = [
    "trough_to_peak_ms",
    "half_width_ms",
    "peak_trough_ratio",
    "pre_peak_ratio",
    "pre_peak_ms",
    "repolarization_slope",
    "recovery_slope",
]


# ============================================================
# Time-axis helpers
# ============================================================
def ms_per_sample(sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ) -> float:
    """Milliseconds per sample for the given sampling rate."""
    return 1_000.0 / float(sampling_rate_hz)


def make_time_axis(
    pre_ms: float = DEFAULT_PRE_MS,
    post_ms: float = DEFAULT_POST_MS,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> Tuple[np.ndarray, int, int, int]:
    """Build the trough-aligned time axis and window sizes.

    Returns
    -------
    time_ms : np.ndarray
        Time (ms) for each sample in the window, with 0 ms == trough.
    pre_samples, post_samples : int
        Number of samples before / after the trough.
    win_len : int
        Total window length in samples (``pre_samples + post_samples + 1``).
    """
    mps = ms_per_sample(sampling_rate_hz)
    pre_samples = int(round(pre_ms / mps))
    post_samples = int(round(post_ms / mps))
    win_len = pre_samples + post_samples + 1
    time_ms = (np.arange(win_len) - pre_samples) * mps
    return time_ms, pre_samples, post_samples, win_len


# ============================================================
# Waveform extraction
# ============================================================
def get_region_from_loc(loc: Any) -> Optional[str]:
    """Return the ``brain_region`` from a CCF-location dict, else ``None``."""
    if loc is not None and isinstance(loc, dict):
        return loc.get("brain_region", None)
    return None


def extract_peak_window(
    wf: Optional[np.ndarray],
    pre_samples: int,
    post_samples: int,
) -> Optional[np.ndarray]:
    """Trough-aligned peak-channel trace of fixed length, or ``None`` if invalid.

    Parameters
    ----------
    wf : np.ndarray, shape (n_samples, n_channels)
        Mean waveform for one unit.
    pre_samples, post_samples : int
        Samples kept before / after the trough.

    Notes
    -----
    The trace is baseline-subtracted and sign-oriented (largest deflection made
    negative) *before* trough alignment, so every returned waveform has its true
    spike trough at t=0 regardless of original polarity.
    """
    win_len = pre_samples + post_samples + 1
    if wf is None or not np.all(np.isfinite(wf)):
        return None
    troughs = wf.min(axis=0)
    peaks = wf.max(axis=0)
    best_ch = int(np.argmax(peaks - troughs))  # channel with largest trough-to-peak
    trace = wf[:, best_ch].astype(float)
    # Baseline to the pre-spike segment, then orient so the largest deflection
    # (the spike) is negative -> a genuine trough at argmin.
    trace = trace - np.median(trace[: max(1, int(0.1 * len(trace)))])
    if abs(trace.min()) < abs(trace.max()):
        trace = -trace
    if np.ptp(trace) == 0:
        return None
    trough_idx = int(np.argmin(trace))
    # Pad with edge values so a fixed window centered on the trough always fits.
    padded = np.pad(trace, (pre_samples, post_samples), mode="edge")
    return padded[trough_idx: trough_idx + win_len]


def normalize_waveform(trace: np.ndarray) -> np.ndarray:
    """Amplitude-normalize a (baseline-corrected, trough-aligned) waveform."""
    peak_amp = np.abs(trace).max()
    if peak_amp > 0:
        trace = trace / peak_amp
    return trace


def extract_unit_waveforms(
    nwb_data: Any,
    unit_indices: Iterable[int],
    pre_ms: float = DEFAULT_PRE_MS,
    post_ms: float = DEFAULT_POST_MS,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> Tuple[np.ndarray, List[int], List[Optional[str]]]:
    """Extract trough-aligned peak-channel waveforms for specific units.

    This is the building block for analyzing an arbitrary set of units (e.g.
    opto-tagged neurons): pass their unit indices and get back a matrix of
    fixed-length peak waveforms.

    Parameters
    ----------
    nwb_data : NWB file handle
        Ephys NWB with a ``units`` table containing ``waveform_mean`` and
        (optionally) ``ccf_location``.
    unit_indices : iterable of int
        Within-session unit indices to extract.
    pre_ms, post_ms, sampling_rate_hz : float
        Extraction-window and timing parameters.

    Returns
    -------
    waveforms : np.ndarray, shape (n_valid, win_len)
        Peak-channel waveforms for the units that produced a valid trace.
    valid_indices : list of int
        Unit indices (aligned with ``waveforms`` rows) that were kept.
    regions : list of str or None
        Brain region per kept unit (``None`` if CCF location is unavailable).
    """
    _, pre_samples, post_samples, win_len = make_time_axis(pre_ms, post_ms, sampling_rate_hz)
    waveform_mean = nwb_data.units["waveform_mean"][:]
    try:
        ccf_locations = nwb_data.units["ccf_location"][:]
    except Exception:
        ccf_locations = None

    waveforms: List[np.ndarray] = []
    valid_indices: List[int] = []
    regions: List[Optional[str]] = []
    for u in unit_indices:
        u = int(u)
        trace = extract_peak_window(waveform_mean[u], pre_samples, post_samples)
        if trace is None or len(trace) != win_len:
            continue
        region = get_region_from_loc(ccf_locations[u]) if ccf_locations is not None else None
        waveforms.append(trace)
        valid_indices.append(u)
        regions.append(region)

    if waveforms:
        waveforms_arr = np.vstack(waveforms)
    else:
        waveforms_arr = np.empty((0, win_len))
    return waveforms_arr, valid_indices, regions


def collect_region_waveforms_from_sessions(
    sessions_to_use: Sequence[str],
    target_regions: Sequence[str],
    pre_ms: float = DEFAULT_PRE_MS,
    post_ms: float = DEFAULT_POST_MS,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    qc_only: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Pool QC-passing, region-matched peak waveforms across many sessions.

    Loops over ``sessions_to_use``; for each unit that (optionally) passes
    default QC and lies in ``target_regions``, extracts the trough-aligned
    peak-channel waveform.

    Returns
    -------
    result : dict with keys
        ``waveforms`` : np.ndarray (n_pooled, win_len)
        ``sessions``  : np.ndarray of session-core ids per unit
        ``unit_indices`` : list of within-session unit indices
        ``regions``   : np.ndarray of brain region per unit
        ``regions_seen`` : Counter of all regions among QC units (diagnostic)
        ``time_ms``   : np.ndarray window time axis
    """
    time_ms, pre_samples, post_samples, win_len = make_time_axis(
        pre_ms, post_ms, sampling_rate_hz
    )

    peak_waveforms: List[np.ndarray] = []
    kept_sessions: List[str] = []
    kept_indices: List[int] = []
    kept_regions: List[Optional[str]] = []
    regions_seen: "Counter[Optional[str]]" = Counter()

    for si, session_name in enumerate(sessions_to_use):
        try:
            nwb_data = NWBUtils.read_ephys_nwb(session_name=session_name)
            if nwb_data is None:
                if verbose:
                    print(f"[skip] {session_name}: no ephys NWB")
                continue
            session_core = extract_session_name_core(session_name)
            if session_core is None:
                session_core = Path(nwb_data.session_id).stem
            nwb_data = append_units_locations(nwb_data, session_name=session_core)
        except Exception as e:
            if verbose:
                print(f"[skip] {session_name}: {e}")
            continue

        if qc_only:
            try:
                qc_units = set(int(u) for u in get_units_passed_default_qc(nwb_data).tolist())
            except Exception as e:
                if verbose:
                    print(f"[skip] {session_core}: QC unavailable ({e})")
                _close_nwb(nwb_data)
                continue
        else:
            qc_units = None

        waveform_mean = nwb_data.units["waveform_mean"][:]
        ccf_locations = nwb_data.units["ccf_location"][:]
        n_units = waveform_mean.shape[0]

        n_added = 0
        for u in range(n_units):
            if qc_units is not None and u not in qc_units:
                continue
            region = get_region_from_loc(ccf_locations[u])
            regions_seen[region] += 1
            if region not in target_regions:
                continue
            trace = extract_peak_window(waveform_mean[u], pre_samples, post_samples)
            if trace is None or len(trace) != win_len:
                continue
            peak_waveforms.append(trace)
            kept_sessions.append(session_core)
            kept_indices.append(u)
            kept_regions.append(region)
            n_added += 1

        if verbose:
            print(
                f"[{si + 1}/{len(sessions_to_use)}] {session_core}: +{n_added} units "
                f"(pooled total {len(peak_waveforms)})"
            )
        _close_nwb(nwb_data)

    waveforms_arr = np.vstack(peak_waveforms) if peak_waveforms else np.empty((0, win_len))
    return {
        "waveforms": waveforms_arr,
        "sessions": np.array(kept_sessions),
        "unit_indices": kept_indices,
        "regions": np.array(kept_regions),
        "regions_seen": regions_seen,
        "time_ms": time_ms,
    }


def _close_nwb(nwb_data: Any) -> None:
    """Best-effort release of an NWB file handle."""
    if getattr(nwb_data, "io", None) is not None:
        try:
            nwb_data.io.close()
        except Exception:
            pass


# ============================================================
# Feature extraction
# ============================================================
def extract_features(
    trace: np.ndarray,
    time_ms: np.ndarray,
    mps: float,
) -> Dict[str, float]:
    """Compute morphology features from a trough-aligned, normalized waveform.

    Parameters
    ----------
    trace : np.ndarray
        Single trough-aligned (ideally normalized) waveform.
    time_ms : np.ndarray
        Time axis (ms) matching ``trace``.
    mps : float
        Milliseconds per sample (see :func:`ms_per_sample`).
    """
    n = len(trace)
    trough_idx = int(np.argmin(trace))
    trough_val = trace[trough_idx]

    # Post-trough peak (repolarization peak)
    if trough_idx < n - 1:
        rel_peak = int(np.argmax(trace[trough_idx:])) + trough_idx
    else:
        rel_peak = trough_idx
    peak_val = trace[rel_peak]

    trough_to_peak_ms = (rel_peak - trough_idx) * mps
    peak_trough_ratio = peak_val / (abs(trough_val) + 1e-12)

    # Pre-trough positive peak (captures triphasic waveforms).
    if trough_idx > 0:
        pre_peak_idx = int(np.argmax(trace[: trough_idx + 1]))
        pre_peak_val = max(trace[pre_peak_idx], 0.0)
        pre_peak_ms = (trough_idx - pre_peak_idx) * mps
    else:
        pre_peak_val = 0.0
        pre_peak_ms = 0.0
    pre_peak_ratio = pre_peak_val / (abs(trough_val) + 1e-12)

    # Half-width: width where trace <= half of trough depth
    half_level = trough_val / 2.0
    below = np.where(trace <= half_level)[0]
    half_width_ms = (below.max() - below.min()) * mps if below.size > 1 else 0.0

    # Repolarization slope: slope over ~0.15 ms right after trough
    win = max(2, int(0.15 / mps))
    end = min(n, trough_idx + win)
    if end - trough_idx >= 2:
        repolarization_slope = np.polyfit(time_ms[trough_idx:end], trace[trough_idx:end], 1)[0]
    else:
        repolarization_slope = 0.0

    # Recovery slope: slope over ~0.15 ms right after the post-trough peak
    end2 = min(n, rel_peak + win)
    if end2 - rel_peak >= 2:
        recovery_slope = np.polyfit(time_ms[rel_peak:end2], trace[rel_peak:end2], 1)[0]
    else:
        recovery_slope = 0.0

    return {
        "trough_to_peak_ms": trough_to_peak_ms,
        "half_width_ms": half_width_ms,
        "peak_trough_ratio": peak_trough_ratio,
        "pre_peak_ratio": pre_peak_ratio,
        "pre_peak_ms": pre_peak_ms,
        "repolarization_slope": repolarization_slope,
        "recovery_slope": recovery_slope,
    }


def compute_features_table(
    waveforms: np.ndarray,
    time_ms: np.ndarray,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    sessions: Optional[Sequence[str]] = None,
    unit_indices: Optional[Sequence[int]] = None,
    regions: Optional[Sequence[Optional[str]]] = None,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Compute a per-unit feature table from a stack of peak waveforms.

    Parameters
    ----------
    waveforms : np.ndarray, shape (n_units, win_len)
        Trough-aligned peak-channel waveforms.
    time_ms : np.ndarray
        Window time axis (from :func:`make_time_axis`).
    sampling_rate_hz : float
        Used to convert samples to ms.
    sessions, unit_indices, regions : optional sequences
        Metadata columns appended to the feature table when provided.
    normalize : bool
        Amplitude-normalize each waveform before feature extraction.

    Returns
    -------
    features : pd.DataFrame
        One row per unit with :data:`FEATURE_COLS` plus any metadata columns.
    norm_waveforms : np.ndarray
        The (optionally normalized) waveforms used for feature extraction.
    """
    mps = ms_per_sample(sampling_rate_hz)
    if normalize:
        norm_waveforms = (
            np.vstack([normalize_waveform(w) for w in waveforms])
            if len(waveforms)
            else waveforms
        )
    else:
        norm_waveforms = waveforms

    rows = [extract_features(w, time_ms, mps) for w in norm_waveforms]
    features = pd.DataFrame(rows, columns=FEATURE_COLS)
    if sessions is not None:
        features["session"] = list(sessions)
    if unit_indices is not None:
        features["unit_index"] = list(unit_indices)
    if regions is not None:
        features["region"] = list(regions)
    return features, norm_waveforms


# ============================================================
# Clustering
# ============================================================
def scale_features(
    features: pd.DataFrame,
    feature_cols: Sequence[str] = FEATURE_COLS,
) -> Tuple[np.ndarray, StandardScaler]:
    """Standardize the feature columns (NaN/inf-safe).

    Returns the scaled matrix and the fitted :class:`StandardScaler`.
    """
    X = features[list(feature_cols)].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def estimate_n_clusters(
    X_scaled: np.ndarray,
    k_range: Iterable[int] = range(2, 8),
    random_state: int = 0,
) -> Dict[str, Any]:
    """Estimate cluster count via elbow (inertia) and silhouette scores.

    Returns a dict with ``ks``, ``inertias``, ``silhouettes`` and the
    silhouette-optimal ``best_k``.
    """
    ks = list(k_range)
    inertias: List[float] = []
    silhouettes: List[float] = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    best_k = ks[int(np.argmax(silhouettes))]
    return {"ks": ks, "inertias": inertias, "silhouettes": silhouettes, "best_k": best_k}


def cluster_waveforms(
    features: pd.DataFrame,
    n_clusters: int,
    feature_cols: Sequence[str] = FEATURE_COLS,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Cluster units with KMeans and a Gaussian Mixture on standardized features.

    Adds ``cluster_kmeans`` and ``cluster_gmm`` columns to ``features`` in place.

    Returns
    -------
    result : dict with keys
        ``features`` (the same DataFrame), ``kmeans``, ``gmm``, ``scaler``,
        ``X_scaled``.
    """
    X = features[list(feature_cols)].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(X_scaled)
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(X_scaled)

    features["cluster_kmeans"] = kmeans.labels_
    features["cluster_gmm"] = gmm.predict(X_scaled)

    return {
        "features": features,
        "kmeans": kmeans,
        "gmm": gmm,
        "scaler": scaler,
        "X_scaled": X_scaled,
    }


# ============================================================
# CSV export
# ============================================================
def save_features_csv(
    features: pd.DataFrame,
    out_dir: Union[str, Path],
    session_name: Optional[str] = None,
    filename: Optional[str] = None,
    suffix: str = "_waveform_features.csv",
) -> Path:
    """Write a feature table to CSV, tagging every row with the session name.

    Parameters
    ----------
    features : pd.DataFrame
        Feature table (e.g. from :func:`compute_features_table`).
    out_dir : str or Path
        Destination folder (created if missing).
    session_name : str, optional
        Session id written into a ``session_name`` column (added/overwritten for
        every row) and used to build the default filename. If ``features``
        already has a per-unit ``session`` column, that is preserved.
    filename : str, optional
        Explicit output filename. Defaults to ``{session_name}{suffix}`` or
        ``waveform_features.csv`` when no session name is given.
    suffix : str
        Filename suffix used when building the default name.

    Returns
    -------
    Path to the written CSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = features.copy()
    if session_name is not None:
        df["session_name"] = session_name

    if filename is None:
        base = session_name if session_name else "waveform_features"
        filename = f"{base}{suffix}"
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    return out_path


# ============================================================
# High-level convenience: analyze an arbitrary set of units
# ============================================================
def analyze_units_waveforms(
    nwb_data: Any,
    unit_indices: Iterable[int],
    session_name: Optional[str] = None,
    pre_ms: float = DEFAULT_PRE_MS,
    post_ms: float = DEFAULT_POST_MS,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    normalize: bool = True,
    out_dir: Optional[Union[str, Path]] = None,
    csv_suffix: str = "_tagged_units_waveform_features.csv",
) -> Dict[str, Any]:
    """Extract waveforms + morphology features for a given set of units.

    Convenience wrapper for the "analyze the waveform of tagged neurons"
    workflow: pass the tagged unit indices and, optionally, an output folder to
    also save a CSV (with the session name embedded).

    Parameters
    ----------
    nwb_data : NWB file handle
        Ephys NWB with ``waveform_mean`` (and optionally ``ccf_location``).
    unit_indices : iterable of int
        Units to analyze (e.g. opto-tagged unit indices).
    session_name : str, optional
        Session id embedded in the CSV and used for its filename. Falls back to
        ``nwb_data.session_id`` when available.
    out_dir : str or Path, optional
        If given, a CSV is written there and its path is returned in the result.

    Returns
    -------
    result : dict with keys
        ``features`` (DataFrame), ``waveforms`` (extracted peak waveforms),
        ``norm_waveforms``, ``valid_indices``, ``regions``, ``time_ms``,
        ``csv_path`` (Path or None).
    """
    if session_name is None:
        session_name = str(getattr(nwb_data, "session_id", None) or "")
        session_name = session_name or None

    time_ms, _, _, _ = make_time_axis(pre_ms, post_ms, sampling_rate_hz)
    waveforms, valid_indices, regions = extract_unit_waveforms(
        nwb_data,
        unit_indices,
        pre_ms=pre_ms,
        post_ms=post_ms,
        sampling_rate_hz=sampling_rate_hz,
    )

    sessions = [session_name] * len(valid_indices) if session_name else None
    features, norm_waveforms = compute_features_table(
        waveforms,
        time_ms,
        sampling_rate_hz=sampling_rate_hz,
        sessions=sessions,
        unit_indices=valid_indices,
        regions=regions,
        normalize=normalize,
    )

    csv_path: Optional[Path] = None
    if out_dir is not None:
        csv_path = save_features_csv(
            features,
            out_dir=out_dir,
            session_name=session_name,
            suffix=csv_suffix,
        )

    return {
        "features": features,
        "waveforms": waveforms,
        "norm_waveforms": norm_waveforms,
        "valid_indices": valid_indices,
        "regions": regions,
        "time_ms": time_ms,
        "csv_path": csv_path,
    }


# ============================================================
# Full dataset: every unit of every session (QC-labelled)
# ============================================================
#: Column-name prefix for the raw trough-aligned waveform samples stored in the
#: big dataset CSV (e.g. ``wf_000``, ``wf_001`` ...). Used to round-trip
#: waveforms through the CSV without a separate file.
WAVEFORM_COL_PREFIX = "wf_"

#: Metadata columns written before the feature / waveform columns.
DATASET_META_COLS: List[str] = [
    "session_name",
    "unit_index",
    "region",
    "qc_pass",
]

#: Timing-context columns (constant per row) so the trough-aligned time axis can
#: be rebuilt on reload.
DATASET_CONTEXT_COLS: List[str] = ["pre_ms", "post_ms", "sampling_rate_hz"]


def build_features_dataset(
    sessions_to_use: Sequence[str],
    pre_ms: float = DEFAULT_PRE_MS,
    post_ms: float = DEFAULT_POST_MS,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    qc_metric_fallback: bool = True,
    normalize: bool = True,
    include_waveform: bool = True,
    wf_prefix: str = WAVEFORM_COL_PREFIX,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build one big table over *every* unit of *every* session.

    Unlike :func:`collect_region_waveforms_from_sessions` (which keeps only
    QC-passing, region-matched units), this keeps **all** units that produce a
    valid trough-aligned waveform and simply *labels* each row with whether it
    passed default QC. The result is a single DataFrame you can save once and
    reload for downstream feature analysis / clustering without re-reading NWBs.

    Each row contains:

    - ``session_name`` : session-core id
    - ``unit_index``   : within-session unit index
    - ``region``       : CCF brain region (or ``None``)
    - ``qc_pass``      : ``True`` if the unit passed default QC, else ``False``
    - the :data:`FEATURE_COLS` morphology features
    - ``pre_ms`` / ``post_ms`` / ``sampling_rate_hz`` : timing context
    - ``wf_000`` ... ``wf_NNN`` : the raw trough-aligned peak waveform samples
      (only when ``include_waveform`` is True)

    Parameters
    ----------
    sessions_to_use : sequence of str
        Session names to load (e.g. from :func:`find_ephys_sessions`).
    pre_ms, post_ms, sampling_rate_hz : float
        Extraction-window and timing parameters.
    qc_metric_fallback : bool
        Passed to :func:`get_units_passed_default_qc` so QC can be rebuilt from
        raw metric columns when the precomputed flag passes zero units.
    normalize : bool
        Amplitude-normalize each waveform before computing features (the raw,
        un-normalized waveform is what gets stored in the ``wf_`` columns).
    include_waveform : bool
        Store the per-sample waveform columns in the table.
    wf_prefix : str
        Prefix for the waveform sample columns.
    verbose : bool
        Print per-session progress.

    Returns
    -------
    pd.DataFrame
        One row per valid unit across all sessions.
    """
    time_ms, pre_samples, post_samples, win_len = make_time_axis(
        pre_ms, post_ms, sampling_rate_hz
    )
    mps = ms_per_sample(sampling_rate_hz)

    rows: List[Dict[str, Any]] = []
    waveforms: List[np.ndarray] = []

    for si, session_name in enumerate(sessions_to_use):
        try:
            nwb_data = NWBUtils.read_ephys_nwb(session_name=session_name)
            if nwb_data is None:
                if verbose:
                    print(f"[skip] {session_name}: no ephys NWB")
                continue
            session_core = extract_session_name_core(session_name)
            if session_core is None:
                session_core = Path(str(getattr(nwb_data, "session_id", session_name))).stem
            nwb_data = append_units_locations(nwb_data, session_name=session_core)
        except Exception as e:
            if verbose:
                print(f"[skip] {session_name}: {e}")
            continue

        try:
            qc_units = set(
                int(u)
                for u in get_units_passed_default_qc(
                    nwb_data, metric_fallback=qc_metric_fallback
                ).tolist()
            )
        except Exception as e:
            if verbose:
                print(f"[warn] {session_core}: QC unavailable ({e}); marking all False")
            qc_units = set()

        waveform_mean = nwb_data.units["waveform_mean"][:]
        try:
            ccf_locations = nwb_data.units["ccf_location"][:]
        except Exception:
            ccf_locations = None
        n_units = waveform_mean.shape[0]

        n_added = 0
        for u in range(n_units):
            trace = extract_peak_window(waveform_mean[u], pre_samples, post_samples)
            if trace is None or len(trace) != win_len:
                continue
            wf_for_features = normalize_waveform(trace) if normalize else trace
            feats = extract_features(wf_for_features, time_ms, mps)
            region = (
                get_region_from_loc(ccf_locations[u])
                if ccf_locations is not None
                else None
            )
            row: Dict[str, Any] = {
                "session_name": session_core,
                "unit_index": u,
                "region": region,
                "qc_pass": u in qc_units,
            }
            row.update(feats)
            row["pre_ms"] = pre_ms
            row["post_ms"] = post_ms
            row["sampling_rate_hz"] = sampling_rate_hz
            rows.append(row)
            waveforms.append(trace)
            n_added += 1

        if verbose:
            n_pass = sum(1 for u in range(n_units) if u in qc_units)
            print(
                f"[{si + 1}/{len(sessions_to_use)}] {session_core}: "
                f"+{n_added} units ({n_pass} QC-pass; total {len(rows)})"
            )
        _close_nwb(nwb_data)

    base_cols = DATASET_META_COLS + FEATURE_COLS + DATASET_CONTEXT_COLS
    df = pd.DataFrame(rows, columns=base_cols) if rows else pd.DataFrame(columns=base_cols)

    if include_waveform:
        wf_cols = [f"{wf_prefix}{i:03d}" for i in range(win_len)]
        wf_arr = np.vstack(waveforms) if waveforms else np.empty((0, win_len))
        wf_df = pd.DataFrame(wf_arr, columns=wf_cols, index=df.index)
        df = pd.concat([df, wf_df], axis=1)

    return df


def save_dataset_csv(
    df: pd.DataFrame,
    out_dir: Union[str, Path],
    filename: str = "all_sessions_waveform_features.csv",
) -> Path:
    """Write the big per-unit dataset to a single CSV. Returns its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    return out_path


def load_features_dataset(
    csv_path: Union[str, Path],
    wf_prefix: str = WAVEFORM_COL_PREFIX,
) -> Dict[str, Any]:
    """Reload a dataset saved by :func:`build_features_dataset`.

    Splits the flat CSV back into a feature/metadata table and a waveform
    matrix, and rebuilds the trough-aligned time axis from the stored timing
    context columns.

    Returns
    -------
    result : dict with keys
        ``features`` : pd.DataFrame with metadata + :data:`FEATURE_COLS`
        (waveform columns dropped).
        ``waveforms`` : np.ndarray (n_units, win_len) of the stored waveforms.
        ``time_ms`` : np.ndarray trough-aligned time axis.
        ``pre_ms`` / ``post_ms`` / ``sampling_rate_hz`` : timing context.
    """
    df = pd.read_csv(csv_path)

    wf_cols = [c for c in df.columns if c.startswith(wf_prefix)]
    wf_cols = sorted(wf_cols, key=lambda c: int(c[len(wf_prefix):]))
    if wf_cols:
        waveforms = df[wf_cols].to_numpy(dtype=float)
    else:
        waveforms = np.empty((len(df), 0))

    features = df.drop(columns=wf_cols) if wf_cols else df.copy()

    def _ctx(col: str, default: float) -> float:
        if col in df.columns and len(df):
            try:
                return float(df[col].iloc[0])
            except Exception:
                return default
        return default

    pre_ms = _ctx("pre_ms", DEFAULT_PRE_MS)
    post_ms = _ctx("post_ms", DEFAULT_POST_MS)
    sampling_rate_hz = _ctx("sampling_rate_hz", DEFAULT_SAMPLING_RATE_HZ)
    time_ms, _, _, _ = make_time_axis(pre_ms, post_ms, sampling_rate_hz)

    return {
        "features": features,
        "waveforms": waveforms,
        "time_ms": time_ms,
        "pre_ms": pre_ms,
        "post_ms": post_ms,
        "sampling_rate_hz": sampling_rate_hz,
    }
