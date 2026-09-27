"""WP1 tests for common/depth_band.py (06_TEST_PLAN test_depth_band.py).

Proves: band is never empty; num_layers=12 coincides with compute_band(12,"scaled");
the band start is monotone non-decreasing in num_layers; band_agreement_report reports
the 4L-teacher / 8L-student pair as comparable under the normalised band (the whole
reason (0.25, 0.90) was chosen, closing G6).

Model-free: imports only the pure legacy band helper for the coincidence assertion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "common"))

from depth_band import (  # noqa: E402
    DEPTH_BAND_DEFAULT,
    DEPTH_BAND_VERSION,
    band_agreement_report,
    layer_depths,
    normalised_depth_band,
)

# Realised bands under the default fraction, from refactor spec section 3.2.
EXPECTED_BANDS = {4: (1, 4), 6: (1, 6), 8: (2, 8), 12: (3, 11), 24: (6, 22)}


def test_band_never_empty() -> None:
    for num_layers in range(1, 65):
        start, end, meta = normalised_depth_band(num_layers)
        assert 0 <= start < end <= num_layers, (num_layers, start, end)
        assert end - start >= 1
        assert meta["n_band_layers"] == end - start
        assert meta["depth_band_version"] == DEPTH_BAND_VERSION


def test_expected_realised_bands() -> None:
    for num_layers, expected in EXPECTED_BANDS.items():
        start, end, _ = normalised_depth_band(num_layers)
        assert (start, end) == expected, (num_layers, (start, end), expected)


def test_min_layers_floor_respected() -> None:
    # A very narrow fraction on a shallow model must still yield >= min_layers.
    for num_layers in (1, 2, 3, 4):
        start, end, _ = normalised_depth_band(num_layers, frac=(0.4, 0.45), min_layers=1)
        assert end - start >= 1
    start, end, _ = normalised_depth_band(8, frac=(0.4, 0.45), min_layers=2)
    assert end - start >= 2


def test_coincides_with_legacy_band_at_12() -> None:
    # The design invariant: at 12 layers the normalised band must equal the frozen
    # legacy band, so E6B screening stays on the same footing as E1.
    from intervention_analysis_legacy import compute_band

    start, end, _ = normalised_depth_band(12)
    assert (start, end) == compute_band(12, "scaled") == (3, 11)


def test_band_start_monotone_in_num_layers() -> None:
    starts = [normalised_depth_band(n)[0] for n in range(1, 65)]
    assert all(b >= a for a, b in zip(starts, starts[1:])), starts


def test_layer_depths_edges() -> None:
    assert layer_depths(1).tolist() == [0.0]
    d = layer_depths(12)
    assert d[0] == 0.0 and d[-1] == 1.0
    assert len(d) == 12
    assert np.allclose(np.diff(d), d[1] - d[0])  # uniform spacing


def test_agreement_teacher_student_pair() -> None:
    # 4-layer TinyStories teacher vs 8-layer student: the normalised band makes them
    # comparable (max endpoint mismatch well within tolerance), unlike the legacy band.
    report = band_agreement_report(4, 8, frac=DEPTH_BAND_DEFAULT)
    assert report["model_a"]["band"] == (1, 4)
    assert report["model_b"]["band"] == (2, 8)
    assert report["max_mismatch"] <= report["tol"]
    assert report["exceeds_tolerance"] is False


def test_agreement_flags_incomparable_pair() -> None:
    # A 2-layer model vs a 24-layer model is genuinely incomparable at the shallow end
    # even under the default fraction (2 layers cannot resolve a [0.25, 0.90) window),
    # so the report must flag it and let a caller raise.
    report = band_agreement_report(2, 24, frac=DEPTH_BAND_DEFAULT, tol=0.10)
    assert report["exceeds_tolerance"] is True
    assert report["max_mismatch"] > report["tol"]
