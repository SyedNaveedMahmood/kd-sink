"""WP4 tests for common/inheritance_metrics.py (06_TEST_PLAN test_inheritance_metrics.py).

Proves the invariants that make the four inheritance axes trustworthy:
  * identity  => cosine 1, L1 0, Wasserstein 0, weighted-Jaccard 1;
  * NaN keys are dropped from every metric and reported;
  * weighted_jaccard raises on unequal head counts (no silent truncation);
  * interp_depth_profile handles a single-layer profile;
  * the 3-seed paired permutation test reports min_attainable_p = 0.25.

Pure numpy/scipy; no model, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "common"))

import inheritance_metrics as im  # noqa: E402

KEYS = ["int_a", "int_c", "int_d", "int_f", "int_g"]


def _fp(values):
    return {"fingerprint": dict(zip(KEYS, values))}


def test_identity_fingerprint_is_perfect() -> None:
    fp = _fp([1.0, 0.3, 0.8, 1.2, 0.5])
    assert im.fingerprint_cosine(fp, fp, KEYS) == pytest.approx(1.0, abs=1e-12)
    assert im.fingerprint_l1(fp, fp, KEYS) == pytest.approx(0.0, abs=1e-12)
    assert im.category_agreement(fp, fp, KEYS) == pytest.approx(1.0)
    report = im.fingerprint_report(fp, fp, KEYS)
    assert report["n_keys_used"] == len(KEYS)
    assert report["dropped_keys"] == []


def test_identity_topology_and_carrier() -> None:
    profile = [0.1, 0.4, 0.9, 0.6, 0.2]
    dist, consts = im.topology_wasserstein(profile, profile)
    assert dist == pytest.approx(0.0, abs=1e-12)
    assert consts["sum_a"] == pytest.approx(consts["sum_b"])
    cells = np.array([[0.1, 0.9], [0.5, 0.2], [0.8, 0.0]])
    assert im.weighted_jaccard(cells, cells) == pytest.approx(1.0, abs=1e-12)


def test_nan_keys_dropped_and_reported() -> None:
    fa = _fp([1.0, np.nan, 0.8, 1.2, 0.5])
    fb = _fp([1.0, 0.3, 0.8, np.nan, 0.5])
    report = im.fingerprint_report(fa, fb, KEYS)
    # int_c (nan in fa) and int_f (nan in fb) must be dropped from all metrics.
    assert set(report["dropped_keys"]) == {"int_c", "int_f"}
    assert report["n_keys_used"] == 3
    assert "int_c" not in report["per_key_delta"]
    assert "int_f" not in report["per_key_delta"]
    # cosine over the 3 shared identical keys is 1.
    assert report["cosine"] == pytest.approx(1.0, abs=1e-12)


def test_weighted_jaccard_raises_on_unequal_heads() -> None:
    a = np.zeros((4, 8))
    b = np.zeros((4, 4))
    with pytest.raises(ValueError, match="equal head counts"):
        im.weighted_jaccard(a, b)


def test_weighted_jaccard_depth_aligns_unequal_layers() -> None:
    # 4-layer vs 8-layer, equal heads: identical constant carriers -> Jaccard 1.
    a = np.ones((4, 4))
    b = np.ones((8, 4))
    j, mapping = im.weighted_jaccard(a, b, return_mapping=True)
    assert j == pytest.approx(1.0)
    assert len(mapping) >= 1


def test_interp_depth_profile_handles_single_layer() -> None:
    with pytest.warns(RuntimeWarning):
        prof = im.interp_depth_profile([0.7], n_points=16)
    assert prof.shape == (16,)
    assert np.allclose(prof, 0.7)


def test_interp_depth_profile_endpoints() -> None:
    prof = im.interp_depth_profile([0.0, 1.0], n_points=16)
    assert prof[0] == pytest.approx(0.0)
    assert prof[-1] == pytest.approx(1.0)


def test_permutation_min_attainable_p_at_n3() -> None:
    res = im.paired_seed_contrast([0.9, 0.8, 0.85], [0.1, 0.2, 0.15])
    assert res["n"] == 3
    assert res["min_attainable_p"] == pytest.approx(0.25)
    assert res["note"] == "descriptive, not inferential"
    # A large, consistent separation attains the floor.
    assert res["p_value"] == pytest.approx(0.25)


def test_category_of_bins() -> None:
    assert im.category_of(0.2) == "strong_reduction"
    assert im.category_of(0.7) == "partial_reduction"
    assert im.category_of(1.0) == "negligible"
    assert im.category_of(2.0) == "increase"
    assert im.category_of(float("nan")) == "undefined"


def test_carrier_concentration_reuses_frozen_gini() -> None:
    # Uniform cells -> Gini 0; fully concentrated -> Gini > 0.
    assert im.carrier_concentration(np.ones((3, 4))) == pytest.approx(0.0, abs=1e-9)
    spike = np.zeros((3, 4))
    spike[0, 0] = 1.0
    assert im.carrier_concentration(spike) > 0.5


def test_bh_correct_basic() -> None:
    rejected, q = im.bh_correct([0.001, 0.01, 0.5, 0.9], alpha=0.05)
    assert rejected[0] and rejected[1]
    assert not rejected[2] and not rejected[3]
    assert np.all(q >= 0) and np.all(q <= 1)


def test_bootstrap_ci_brackets_mean() -> None:
    lo, hi = im.bootstrap_ci([1.0, 1.1, 0.9, 1.05, 0.95], n_boot=2000, seed=0)
    assert lo <= 1.0 <= hi


# ── hierarchical bootstrap (WP11 additions) ───────────────────────────────────


def _clustered_frame(shift: float = 0.0):
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 240
    return pd.DataFrame({
        "semantic_id": [f"s{i % 20}" for i in range(n)],
        "target_language": [["de", "zh", "tr"][i % 3] for i in range(n)],
        "value": rng.normal(size=n) + shift,
    })


def test_hierarchical_bootstrap_is_deterministic_from_its_seed() -> None:
    frame = _clustered_frame()
    keys = ("semantic_id", "target_language")
    first = im.hierarchical_bootstrap(frame, keys, "value", n_boot=300, seed=0)
    second = im.hierarchical_bootstrap(frame, keys, "value", n_boot=300, seed=0)
    assert first == second
    other = im.hierarchical_bootstrap(frame, keys, "value", n_boot=300, seed=1)
    assert other["ci_lo"] != first["ci_lo"] or other["ci_hi"] != first["ci_hi"]


def test_return_draws_is_additive_and_consistent_with_the_summary() -> None:
    """The default return value is unchanged; the draws reproduce the reported summary."""
    frame = _clustered_frame()
    keys = ("semantic_id", "target_language")
    plain = im.hierarchical_bootstrap(frame, keys, "value", n_boot=300, seed=0)
    with_draws = im.hierarchical_bootstrap(frame, keys, "value", n_boot=300, seed=0,
                                           return_draws=True)

    assert "draws" not in plain
    draws = with_draws.pop("draws")
    assert with_draws == plain
    assert draws.shape == (plain["n_boot"],)
    assert np.mean(draws) == pytest.approx(plain["boot_mean"])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    assert (lo, hi) == pytest.approx((plain["ci_lo"], plain["ci_hi"]))


def test_bootstrap_two_sided_p_reports_its_own_resolution_floor() -> None:
    """A bootstrap cannot resolve a p below 1/n_boot; reporting 0.0 would hide that."""
    keys = ("semantic_id", "target_language")
    separated = im.hierarchical_bootstrap(_clustered_frame(shift=50.0), keys, "value",
                                          n_boot=500, seed=0, return_draws=True)
    result = im.bootstrap_two_sided_p(separated["draws"])
    assert result["min_attainable_p"] == pytest.approx(1.0 / 500)
    assert result["p_value"] == pytest.approx(1.0 / 500)
    assert result["p_value"] > 0.0

    centred = im.hierarchical_bootstrap(_clustered_frame(), keys, "value",
                                        n_boot=500, seed=0, return_draws=True)
    assert im.bootstrap_two_sided_p(centred["draws"])["p_value"] > 1.0 / 500


def test_bootstrap_two_sided_p_on_no_draws_is_nan_not_zero() -> None:
    result = im.bootstrap_two_sided_p(np.array([]))
    assert np.isnan(result["p_value"])
    assert result["n_boot"] == 0


def test_hierarchical_bootstrap_requires_exactly_two_group_columns() -> None:
    frame = _clustered_frame()
    with pytest.raises(ValueError, match="exactly two group columns"):
        im.hierarchical_bootstrap(frame, ("semantic_id",), "value", n_boot=10)
