"""Cross-architecture inheritance metrics (WP4, closes G9).

Pure analysis: fingerprint distances, depth-profile topology, carrier-head overlap,
functional-cost vectors, E6B drift trajectories, and the descriptive statistics E6/E7
report. Nothing here loads a model; inputs are plain mappings / arrays extracted from
``FingerprintRecord`` objects (or dicts shaped like them).

Design constraints made mechanical here (see ``new_design_plans/02_MODULE_SPEC_common.md``
section 4):

* ``keys`` is always passed explicitly and always comes from ``mutual_interventions`` --
  no function silently unions or intersects dict keys (design-delta D2).
* NaN policy: a key that is NaN in *either* fingerprint is dropped from all metrics, and
  every report records ``n_keys_used`` and ``dropped_keys``.
* ``weighted_jaccard`` **raises** on unequal head counts rather than truncating -- a
  silent truncation would be an invisible confound.
* The three-seed paired permutation test cannot produce ``p < 0.25``; the contrast helper
  returns ``min_attainable_p`` and a descriptive-not-inferential ``note`` so no table can
  present ``p = 0.25`` as significance.
* There is deliberately **no** composite ``inheritance_score`` (design section 4).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # dual-import idiom: scripts put common/ on sys.path directly
    from . import depth_band as _depth_band
except ImportError:  # pragma: no cover - exercised via direct-path imports
    import depth_band as _depth_band

# --------------------------------------------------------------------------------------
# Intervention categories (design section 4.1)
# --------------------------------------------------------------------------------------

INTERVENTION_CATEGORIES: Dict[str, Tuple[float, float]] = {
    "strong_reduction": (0.00, 0.50),
    "partial_reduction": (0.50, 0.90),
    "negligible": (0.90, 1.10),
    "increase": (1.10, float("inf")),
}


def category_of(r: float) -> str:
    """Category label for a single reduction ratio ``r_j`` (half-open [lo, hi))."""
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "undefined"
    for name, (lo, hi) in INTERVENTION_CATEGORIES.items():
        if lo <= r < hi:
            return name
    return "undefined"


# --------------------------------------------------------------------------------------
# Accessors — tolerate a FingerprintRecord object or a plain mapping
# --------------------------------------------------------------------------------------

def _mapping(record: Any, attr: str) -> Mapping[str, float]:
    if hasattr(record, attr):
        value = getattr(record, attr)
        if value is None:
            raise ValueError(f"record.{attr} is None")
        return value
    if isinstance(record, Mapping):
        if attr in record:
            return record[attr]
        # a bare fingerprint/functional mapping was passed directly
        return record
    raise TypeError(f"cannot extract '{attr}' from {type(record)!r}")


def _gini(cells: np.ndarray) -> float:
    """Reuse the frozen E5 ``_gini`` (dispatch, never reimplement).

    Imported lazily so this module stays torch/matplotlib-free at import time; the E5
    driver pulls heavy deps that pure-analysis callers should not pay for.
    """
    ev_dir = Path(__file__).resolve().parents[1] / "evaluation_robustness"
    if str(ev_dir) not in sys.path:
        sys.path.insert(0, str(ev_dir))
    from evaluation_robustness_analysis import _gini as frozen_gini  # noqa: WPS433

    return float(frozen_gini(np.asarray(cells, dtype=np.float64)))


# --------------------------------------------------------------------------------------
# 4.1 Fingerprint distances
# --------------------------------------------------------------------------------------

def fingerprint_vector(record: Any, keys: Sequence[str]) -> np.ndarray:
    """Vector of ``r_j`` values in ``keys`` order; missing keys become NaN."""
    fp = _mapping(record, "fingerprint")
    return np.array([float(fp.get(k, np.nan)) for k in keys], dtype=np.float64)


def _valid_pair(fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """Boolean mask of positions finite in both vectors (NaN policy)."""
    return np.isfinite(fa) & np.isfinite(fb)


def _drop_info(keys: Sequence[str], mask: np.ndarray) -> Tuple[int, List[str]]:
    dropped = [k for k, ok in zip(keys, mask) if not ok]
    return int(mask.sum()), dropped


def fingerprint_cosine(fa: Mapping, fb: Mapping, keys: Sequence[str]) -> float:
    va, vb = fingerprint_vector(fa, keys), fingerprint_vector(fb, keys)
    mask = _valid_pair(va, vb)
    if mask.sum() < 1:
        return float("nan")
    a, b = va[mask], vb[mask]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def fingerprint_spearman(fa: Mapping, fb: Mapping, keys: Sequence[str]) -> float:
    from scipy.stats import spearmanr

    va, vb = fingerprint_vector(fa, keys), fingerprint_vector(fb, keys)
    mask = _valid_pair(va, vb)
    if mask.sum() < 2:
        return float("nan")
    rho, _ = spearmanr(va[mask], vb[mask])
    return float(rho)


def fingerprint_l1(fa: Mapping, fb: Mapping, keys: Sequence[str],
                   normalise: bool = True) -> float:
    va, vb = fingerprint_vector(fa, keys), fingerprint_vector(fb, keys)
    mask = _valid_pair(va, vb)
    if mask.sum() < 1:
        return float("nan")
    total = float(np.abs(va[mask] - vb[mask]).sum())
    return total / int(mask.sum()) if normalise else total


def category_agreement(fa: Mapping, fb: Mapping, keys: Sequence[str]) -> float:
    va, vb = fingerprint_vector(fa, keys), fingerprint_vector(fb, keys)
    mask = _valid_pair(va, vb)
    if mask.sum() < 1:
        return float("nan")
    agree = sum(category_of(a) == category_of(b)
                for a, b in zip(va[mask], vb[mask]))
    return agree / int(mask.sum())


def fingerprint_report(fa: Mapping, fb: Mapping, keys: Sequence[str]) -> dict:
    va, vb = fingerprint_vector(fa, keys), fingerprint_vector(fb, keys)
    mask = _valid_pair(va, vb)
    n_used, dropped = _drop_info(keys, mask)
    per_key = {k: (float(a) - float(b))
               for k, a, b, ok in zip(keys, va, vb, mask) if ok}
    return {
        "cosine": fingerprint_cosine(fa, fb, keys),
        "spearman": fingerprint_spearman(fa, fb, keys),
        "l1_normalised": fingerprint_l1(fa, fb, keys, normalise=True),
        "category_agreement": category_agreement(fa, fb, keys),
        "per_key_delta": per_key,
        "n_keys_used": n_used,
        "dropped_keys": dropped,
    }


# --------------------------------------------------------------------------------------
# 4.2 Topology (depth profiles)
# --------------------------------------------------------------------------------------

def interp_depth_profile(per_layer_values: Sequence[float],
                         n_points: int = 16) -> np.ndarray:
    """Interpolate a per-layer profile onto ``n_points`` fixed fractional depths.

    ``d = i / (L - 1)``; linear interpolation onto ``linspace(0, 1, n_points)``. A
    single-layer profile has no depth axis, so it maps to a constant profile and emits
    a warning (design section 4.2).
    """
    y = np.asarray(per_layer_values, dtype=np.float64).ravel()
    L = y.size
    if L == 0:
        raise ValueError("per_layer_values is empty")
    grid = np.linspace(0.0, 1.0, n_points)
    if L == 1:
        import warnings
        warnings.warn("interp_depth_profile: single-layer profile -> constant",
                      RuntimeWarning, stacklevel=2)
        return np.full(n_points, y[0], dtype=np.float64)
    depths = _depth_band.layer_depths(L)
    return np.interp(grid, depths, y)


def topology_spearman(pa: Sequence[float], pb: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    a, b = np.asarray(pa, float), np.asarray(pb, float)
    if a.size < 2:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho)


def topology_area_diff(pa: Sequence[float], pb: Sequence[float],
                       normalise: bool = True) -> float:
    a, b = np.asarray(pa, float), np.asarray(pb, float)
    total = float(np.abs(a - b).sum())
    return total / a.size if normalise else total


def topology_wasserstein(pa: Sequence[float],
                         pb: Sequence[float]) -> Tuple[float, Dict[str, float]]:
    """1-D Wasserstein between profiles treated as masses over depth.

    Both profiles are L1-normalised to sum 1 first; returns
    ``(distance, {"sum_a", "sum_b"})`` so a degenerate all-zero profile is reported
    rather than silently producing 0.
    """
    from scipy.stats import wasserstein_distance

    a = np.asarray(pa, dtype=np.float64).ravel()
    b = np.asarray(pb, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"profile length mismatch: {a.size} vs {b.size}")
    a = np.clip(a, 0.0, None)
    b = np.clip(b, 0.0, None)
    sum_a, sum_b = float(a.sum()), float(b.sum())
    support = np.linspace(0.0, 1.0, a.size)
    consts = {"sum_a": sum_a, "sum_b": sum_b}
    if sum_a <= 1e-12 or sum_b <= 1e-12:
        return float("nan"), consts
    dist = float(wasserstein_distance(support, support, a / sum_a, b / sum_b))
    return dist, consts


def topology_report(pa: Sequence[float], pb: Sequence[float]) -> dict:
    dist, consts = topology_wasserstein(pa, pb)
    return {
        "spearman": topology_spearman(pa, pb),
        "area_diff_normalised": topology_area_diff(pa, pb, normalise=True),
        "wasserstein": dist,
        "wasserstein_norm_constants": consts,
    }


# --------------------------------------------------------------------------------------
# 4.3 Carrier heads
# --------------------------------------------------------------------------------------

def carrier_cells(per_head_sink: Sequence[Sequence[float]],
                  num_layers: int, num_heads: int) -> np.ndarray:
    cells = np.asarray(per_head_sink, dtype=np.float64)
    if cells.shape != (num_layers, num_heads):
        raise ValueError(
            f"per_head_sink shape {cells.shape} != ({num_layers}, {num_heads})")
    return cells


def carrier_concentration(cells: np.ndarray) -> float:
    """Gini concentration over the [L, H] layer-head sink cells (reuses frozen _gini)."""
    return _gini(np.asarray(cells, dtype=np.float64))


def top_carriers(cells: np.ndarray, k: int = 5) -> List[Tuple[float, int, float]]:
    """Top-``k`` (depth, head, strength) cells by strength, descending."""
    cells = np.asarray(cells, dtype=np.float64)
    L, H = cells.shape
    depths = _depth_band.layer_depths(L)
    flat = [(depths[l], h, float(cells[l, h])) for l in range(L) for h in range(H)]
    flat.sort(key=lambda t: t[2], reverse=True)
    return [(float(d), int(h), s) for d, h, s in flat[:k]]


def _depth_layer_matching(La: int, Lb: int, tol: float) -> List[Tuple[int, int]]:
    """Match each layer of A to the nearest-depth layer of B within ``tol``.

    Returns the list of (layer_a, layer_b) pairs whose normalised-depth gap <= tol.
    Many-to-one is permitted (nearest depth wins); layers with no match are dropped.
    """
    da, db = _depth_band.layer_depths(La), _depth_band.layer_depths(Lb)
    pairs: List[Tuple[int, int]] = []
    for ia, d in enumerate(da):
        ib = int(np.argmin(np.abs(db - d)))
        if abs(db[ib] - d) <= tol:
            pairs.append((ia, ib))
    return pairs


def weighted_jaccard(cells_a: np.ndarray, cells_b: np.ndarray, *,
                     depth_align: bool = True, tol: float = 0.10,
                     return_mapping: bool = False):
    """Weighted Jaccard over carrier cells: sum(min)/sum(max).

    Head-index comparison requires equal head counts and **raises** otherwise -- a
    silent truncation would be an invisible confound (design section 4.3). When layer
    counts differ and ``depth_align`` is True, layers are matched by nearest normalised
    depth within ``tol``.
    """
    a = np.asarray(cells_a, dtype=np.float64)
    b = np.asarray(cells_b, dtype=np.float64)
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"weighted_jaccard requires equal head counts, got "
            f"{a.shape[1]} and {b.shape[1]}")
    La, Lb = a.shape[0], b.shape[0]
    if La == Lb:
        mapping = [(i, i) for i in range(La)]
    elif depth_align:
        mapping = _depth_layer_matching(La, Lb, tol)
    else:
        raise ValueError(
            f"layer counts differ ({La} vs {Lb}) and depth_align=False")
    if not mapping:
        result = float("nan")
        return (result, mapping) if return_mapping else result
    aa = np.stack([a[ia] for ia, _ in mapping])
    bb = np.stack([b[ib] for _, ib in mapping])
    mn = np.minimum(aa, bb).sum()
    mx = np.maximum(aa, bb).sum()
    result = float(mn / mx) if mx > 1e-12 else float("nan")
    return (result, mapping) if return_mapping else result


def carrier_report(cells_a: np.ndarray, cells_b: np.ndarray,
                   num_heads_a: int, num_heads_b: int) -> dict:
    if num_heads_a != num_heads_b:
        raise ValueError(
            f"carrier_report requires equal head counts, got "
            f"{num_heads_a} and {num_heads_b}")
    jac, mapping = weighted_jaccard(cells_a, cells_b, return_mapping=True)
    return {
        "weighted_jaccard": jac,
        "concentration_a": carrier_concentration(cells_a),
        "concentration_b": carrier_concentration(cells_b),
        "top_carriers_a": top_carriers(cells_a),
        "top_carriers_b": top_carriers(cells_b),
        "layer_mapping": mapping,
    }


# --------------------------------------------------------------------------------------
# 4.4 Function (delta-CE)
# --------------------------------------------------------------------------------------

def functional_vector(record: Any, keys: Sequence[str]) -> np.ndarray:
    dce = _mapping(record, "delta_ce")
    return np.array([float(dce.get(k, np.nan)) for k in keys], dtype=np.float64)


def functional_cosine(ra: Any, rb: Any, keys: Sequence[str]) -> float:
    va, vb = functional_vector(ra, keys), functional_vector(rb, keys)
    mask = _valid_pair(va, vb)
    if mask.sum() < 1:
        return float("nan")
    a, b = va[mask], vb[mask]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def functional_report(ra: Any, rb: Any, keys: Sequence[str]) -> dict:
    va, vb = functional_vector(ra, keys), functional_vector(rb, keys)
    mask = _valid_pair(va, vb)
    n_used, dropped = _drop_info(keys, mask)
    return {
        "cosine": functional_cosine(ra, rb, keys),
        "l1_normalised": (float(np.abs(va[mask] - vb[mask]).sum()) / int(mask.sum())
                          if mask.sum() else float("nan")),
        "n_keys_used": n_used,
        "dropped_keys": dropped,
    }


# --------------------------------------------------------------------------------------
# 4.5 Drift (E6B)
# --------------------------------------------------------------------------------------

def fingerprint_drift(f_t: Any, f_0: Any, keys: Sequence[str]) -> float:
    c = fingerprint_cosine(f_t, f_0, keys)
    return float("nan") if np.isnan(c) else 1.0 - c


def topology_drift(p_t: Sequence[float], p_0: Sequence[float]) -> float:
    dist, _ = topology_wasserstein(p_t, p_0)
    return dist


def carrier_drift(c_t: np.ndarray, c_0: np.ndarray, *, tol: float = 0.10) -> float:
    j = weighted_jaccard(c_t, c_0, tol=tol)
    return float("nan") if np.isnan(j) else 1.0 - j


def drift_trajectory(records_by_step: Mapping[int, Any], base_record: Any,
                     keys: Sequence[str]):
    """DataFrame of drift columns vs the base record, one row per step (sorted)."""
    import pandas as pd

    rows = []
    for step in sorted(records_by_step):
        rec = records_by_step[step]
        row = {"step": step,
               "fingerprint_drift": fingerprint_drift(rec, base_record, keys)}
        # topology / carrier drift only when the record carries those fields
        try:
            row["topology_drift"] = topology_drift(
                _mapping(rec, "depth_profile_16"),
                _mapping(base_record, "depth_profile_16"))
        except (TypeError, ValueError):
            row["topology_drift"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def drift_onset(traj, threshold: float, column: str) -> Optional[int]:
    """First step whose ``column`` value exceeds ``threshold``; None if never."""
    hit = traj[traj[column] > threshold]
    if len(hit) == 0:
        return None
    return int(hit.iloc[0]["step"])


def clean_run_threshold(clean_trajs: Sequence, column: str, k: float = 2.0) -> float:
    """``mean + k*std`` of ``column`` across clean-run trajectories (matched steps)."""
    import pandas as pd

    values = pd.concat([t[column] for t in clean_trajs], ignore_index=True)
    values = values.dropna()
    if len(values) == 0:
        return float("nan")
    return float(values.mean() + k * values.std(ddof=0))


# --------------------------------------------------------------------------------------
# 4.6 Statistics
# --------------------------------------------------------------------------------------

def paired_seed_contrast(values_a: Sequence[float],
                         values_b: Sequence[float]) -> dict:
    """Paired difference with an EXACT sign-flip permutation test.

    With ``n`` seeds there are ``2**n`` sign assignments; the two extreme assignments
    give the smallest two-sided p, so ``min_attainable_p = 2**(1-n)`` (0.25 at n=3).
    The result is descriptive, not inferential -- the ``note`` says so explicitly so no
    downstream table presents ``p = 0.25`` as significance (design section 15.1).
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("values_a and values_b must be 1-D of equal length")
    diffs = a - b
    n = diffs.size
    observed = float(np.abs(diffs.mean()))
    count = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        stat = abs(float(np.mean(np.asarray(signs) * diffs)))
        if stat >= observed - 1e-15:
            count += 1
    p = count / (2 ** n)
    return {
        "mean_diff": float(diffs.mean()),
        "per_seed_diffs": [float(x) for x in diffs],
        "n": int(n),
        "p_value": float(p),
        "min_attainable_p": float(2.0 ** (1 - n)),
        "note": "descriptive, not inferential",
    }


def hierarchical_bootstrap(df, group_cols: Sequence[str], value_col: str,
                           n_boot: int = 10000, seed: int = 0,
                           return_draws: bool = False) -> dict:
    """Two-level cluster bootstrap: resample the first group col, then the second.

    ``group_cols = (outer, inner)`` -- for E7, ``("semantic_id", "language")``. Each
    replicate resamples outer clusters with replacement, then inner clusters within each
    chosen outer, and takes the mean of ``value_col``. Returns mean and a percentile CI.

    ``return_draws`` (WP11, additive; the default reproduces the previous return value
    exactly) adds the replicate means under the key ``"draws"``. E7 needs a p-value from
    the *same* resampling that produced the CI -- reporting a percentile interval from a
    hierarchical bootstrap next to a p-value from some other test would be two different
    inference frameworks in one row.
    """
    import pandas as pd

    if len(group_cols) != 2:
        raise ValueError("hierarchical_bootstrap expects exactly two group columns")
    outer_col, inner_col = group_cols
    rng = np.random.default_rng(seed)
    outer_groups = {key: sub for key, sub in df.groupby(outer_col)}
    outer_keys = list(outer_groups)
    if not outer_keys:
        raise ValueError("empty dataframe")

    # The clustering is a property of the data, not of the replicate. Deriving it once
    # instead of re-running a pandas groupby per outer cluster per replicate is the
    # difference between minutes and seconds at n_boot=10000, and changes nothing: the
    # rng.choice calls below are made in the same order, on the same key lists, so every
    # replicate draws exactly what it drew before.
    inner_cache: Dict[Any, Dict[Any, np.ndarray]] = {
        key: {ik: sub[value_col].to_numpy(dtype=float)
              for ik, sub in group.groupby(inner_col)}
        for key, group in outer_groups.items()}
    inner_key_lists = {key: list(groups) for key, groups in inner_cache.items()}

    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        chosen_outer = rng.choice(outer_keys, size=len(outer_keys), replace=True)
        parts: List[np.ndarray] = []
        for ok in chosen_outer:
            inner_groups = inner_cache[ok]
            inner_keys = inner_key_lists[ok]
            chosen_inner = rng.choice(inner_keys, size=len(inner_keys), replace=True)
            parts.extend(inner_groups[ik] for ik in chosen_inner)
        means[b] = np.mean(np.concatenate(parts)) if parts else np.nan
    means = means[np.isfinite(means)]
    lo, hi = np.percentile(means, [2.5, 97.5]) if means.size else (np.nan, np.nan)
    result = {
        "mean": float(df[value_col].mean()),
        "boot_mean": float(np.mean(means)) if means.size else float("nan"),
        "ci_lo": float(lo), "ci_hi": float(hi),
        "n_boot": int(means.size),
    }
    if return_draws:
        result["draws"] = means
    return result


def bootstrap_two_sided_p(draws, null_value: float = 0.0) -> dict:
    """Two-sided p from bootstrap replicate means, with its own resolution floor.

    ``p = 2 * min(P(draw <= null), P(draw >= null))``, clipped to at least ``1 / n_boot``:
    a bootstrap cannot resolve a p below its own replicate count, and reporting ``0.0``
    would present a resolution limit as evidence. The floor is returned as
    ``min_attainable_p`` for the same reason
    :func:`paired_seed_contrast` returns one.
    """
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {"p_value": float("nan"), "min_attainable_p": float("nan"), "n_boot": 0}
    floor = 1.0 / n
    below = float(np.mean(values <= null_value))
    above = float(np.mean(values >= null_value))
    p = min(1.0, 2.0 * min(below, above))
    return {"p_value": float(max(p, floor)), "min_attainable_p": float(floor),
            "n_boot": n}


def bh_correct(pvalues: Sequence[float], alpha: float = 0.05):
    """Benjamini-Hochberg. Returns (rejected: np.ndarray[bool], qvalues: np.ndarray)."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.array([], dtype=bool), np.array([])
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(1, m + 1))
    # enforce monotonicity from the largest rank down
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    qvalues = np.empty(m, dtype=np.float64)
    qvalues[order] = q
    rejected = qvalues <= alpha
    return rejected, qvalues


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot_means = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
