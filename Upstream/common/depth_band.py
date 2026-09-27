"""Normalised-depth layer band for cross-scale sink comparisons (WP1, closes G6).

The legacy ``compute_band`` (in ``intervention_analysis_legacy``) excludes the first
three and the last layer. That rule is fine for the 12/24/32-layer models of E1-E5 but
degenerate for shallow models: ``compute_band(4, "scaled") -> (3, 4)`` selects a single
layer, so a 4-layer teacher and an 8-layer student would have their sink strengths
averaged over non-comparable depth regions -- silently.

E6/E7 therefore choose the band by *fractional depth* instead. The default
``(0.25, 0.90)`` was picked so that a 12-layer model reproduces the legacy band exactly
(``(3, 11)``), keeping E6B's distilgpt2/gpt2 screening on the same footing as the frozen
E1 numbers, while shallow and deep models get proportional, mutually comparable bands.

This module never modifies ``compute_band``: E1-E5 reproduction keeps calling the legacy
function; E6/E7 call ``normalised_depth_band`` here. See
``new_design_plans/01_REFACTOR_SPEC_existing_code.md`` section 3.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

DEPTH_BAND_DEFAULT: Tuple[float, float] = (0.25, 0.90)  # fractional [start, end)
DEPTH_BAND_VERSION: str = "depth_band_v1"

# Endpoint agreement tolerance for cross-model comparisons (refactor spec section 3.3).
BAND_AGREEMENT_TOL: float = 0.10


def layer_depths(num_layers: int) -> np.ndarray:
    """Fractional depth ``d = i / (num_layers - 1)`` for each layer index ``i``.

    A single-layer model has no non-trivial depth axis, so it maps to ``[0.0]``.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if num_layers == 1:
        return np.array([0.0])
    return np.arange(num_layers, dtype=float) / (num_layers - 1)


def normalised_depth_band(num_layers: int,
                          frac: Tuple[float, float] = DEPTH_BAND_DEFAULT,
                          min_layers: int = 1) -> Tuple[int, int, Dict]:
    """0-indexed ``[start, end)`` chosen by fractional depth, never empty.

    Algorithm (refactor spec section 3.2)::

        start = floor(frac[0] * num_layers)
        end   = max(start + min_layers, ceil(frac[1] * num_layers))
        end   = min(end, num_layers)
        start = min(start, end - min_layers)

    Returns ``(start, end, meta)`` where ``meta`` records ``num_layers``, ``frac``, the
    realised depth interval ``(depth[start], depth[end - 1])``, the number of layers in
    the band, and ``DEPTH_BAND_VERSION``.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if min_layers < 1:
        raise ValueError(f"min_layers must be >= 1, got {min_layers}")
    if min_layers > num_layers:
        raise ValueError(
            f"min_layers={min_layers} exceeds num_layers={num_layers}")
    lo, hi = frac
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(f"frac must satisfy 0 <= lo < hi <= 1, got {frac}")

    start = int(math.floor(lo * num_layers))
    end = max(start + min_layers, int(math.ceil(hi * num_layers)))
    end = min(end, num_layers)
    start = min(start, end - min_layers)
    start = max(start, 0)  # guard: floor of a non-negative product is already >= 0

    depths = layer_depths(num_layers)
    depth_interval = (float(depths[start]), float(depths[end - 1]))
    meta = {
        "num_layers": num_layers,
        "frac": (float(lo), float(hi)),
        "band": (start, end),
        "n_band_layers": end - start,
        "depth_interval": depth_interval,
        "depth_band_version": DEPTH_BAND_VERSION,
    }
    return start, end, meta


def band_agreement_report(num_layers_a: int, num_layers_b: int,
                          frac: Tuple[float, float] = DEPTH_BAND_DEFAULT,
                          tol: float = BAND_AGREEMENT_TOL) -> Dict:
    """Compare the realised depth intervals of two models' normalised bands.

    Cross-model sink comparisons must not average over non-comparable depth regions.
    Callers use this report to raise when the depth-interval mismatch exceeds ``tol``
    (unless ``allow_band_mismatch=True`` is passed and recorded); see refactor spec
    section 3.3.

    Returns a dict with each model's band + realised depth interval, the absolute
    endpoint mismatches (start and end), their maximum, ``tol``, and
    ``exceeds_tolerance`` -- ``True`` when the max mismatch is strictly greater than
    ``tol``.
    """
    start_a, end_a, meta_a = normalised_depth_band(num_layers_a, frac=frac)
    start_b, end_b, meta_b = normalised_depth_band(num_layers_b, frac=frac)
    di_a = meta_a["depth_interval"]
    di_b = meta_b["depth_interval"]
    start_mismatch = abs(di_a[0] - di_b[0])
    end_mismatch = abs(di_a[1] - di_b[1])
    max_mismatch = max(start_mismatch, end_mismatch)
    return {
        "model_a": {"num_layers": num_layers_a, "band": (start_a, end_a),
                    "depth_interval": di_a},
        "model_b": {"num_layers": num_layers_b, "band": (start_b, end_b),
                    "depth_interval": di_b},
        "frac": (float(frac[0]), float(frac[1])),
        "start_mismatch": start_mismatch,
        "end_mismatch": end_mismatch,
        "max_mismatch": max_mismatch,
        "tol": tol,
        "exceeds_tolerance": max_mismatch > tol,
        "depth_band_version": DEPTH_BAND_VERSION,
    }
