# -*- coding: utf-8 -*-
"""test_retrieval_chance.py — WP9 retrieval invariants (06_TEST_PLAN.md §1).

Random representations retrieve at chance; identical ones retrieve perfectly; per-language
centring happens **before** similarity. That last one is the load-bearing test: without
centring, cosine similarity is dominated by the language mean and "retrieval" measures
script identity rather than semantics — a bug that produces a large, smooth, entirely
uninformative number instead of a crash.

No model is built here, so no Windows teardown dance is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "crosslingual_semantics"))

import evaluate_crosslingual_retrieval as ret  # noqa: E402


# ── chance and ceiling ────────────────────────────────────────────────────────


def test_random_representations_retrieve_at_about_chance():
    rng = np.random.default_rng(0)
    n, dim = 300, 64
    metrics = ret.retrieval_metrics(rng.normal(size=(n, dim)),
                                    rng.normal(size=(n, dim)))
    assert metrics["n_candidates"] == n
    assert metrics["chance"] == pytest.approx(1.0 / n)
    # 300 independent draws at p = 1/300: the expected count is 1, so anything up to a
    # handful of hits is ordinary. The assertion is that it is near chance, not exactly it.
    assert metrics["top1"] < 0.05
    assert metrics["top1_over_chance"] < 10.0
    assert metrics["median_rank"] > n / 4


def test_identical_representations_retrieve_perfectly():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(50, 16))
    metrics = ret.retrieval_metrics(matrix, matrix.copy())
    assert metrics["top1"] == pytest.approx(1.0)
    assert metrics["top5"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["median_rank"] == pytest.approx(1.0)
    assert metrics["top1_over_chance"] == pytest.approx(50.0)


def test_a_degenerate_constant_representation_does_not_look_like_perfect_retrieval():
    """Every row identical ⇒ every rank tied ⇒ pessimistic ranking must report chance.

    An optimistic tie-break would report top-1 = 1.0 for a representation that carries no
    information at all.
    """
    constant = np.ones((40, 8))
    metrics = ret.retrieval_metrics(constant, constant.copy())
    assert metrics["top1"] == pytest.approx(0.0)
    assert metrics["median_rank"] == pytest.approx(40.0)


# ── the centring step ─────────────────────────────────────────────────────────


def test_centring_removes_the_language_mean():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(20, 5)) + 100.0
    centred = ret.centre(matrix)
    assert np.allclose(centred.mean(axis=0), 0.0, atol=1e-12)
    # Centring is a translation: pairwise differences are untouched.
    assert np.allclose(centred[1] - centred[0], matrix[1] - matrix[0])


def test_retrieval_centres_each_language_before_similarity():
    """A large per-language offset must not change any retrieval number.

    Both languages get a different constant shift. Uncentred cosine similarity would be
    dominated by those offsets; centred similarity is exactly invariant to them.
    """
    rng = np.random.default_rng(3)
    source = rng.normal(size=(60, 12))
    target = source @ rng.normal(size=(12, 12))

    plain = ret.retrieval_metrics(source, target)
    shifted = ret.retrieval_metrics(source + 25.0, target - 7.5)
    for key in ("top1", "top5", "mrr", "median_rank"):
        assert shifted[key] == pytest.approx(plain[key])


def test_uncentred_similarity_would_have_given_a_different_answer():
    """The guard-rail for the test above: the shift is big enough to matter if ignored.

    If this ever stops holding, the invariance test above has become vacuous and would no
    longer catch a missing centring step.
    """
    rng = np.random.default_rng(4)
    source = rng.normal(size=(60, 12))
    target = source @ rng.normal(size=(12, 12))

    def uncentred(a, b):
        au = a / np.linalg.norm(a, axis=1, keepdims=True)
        bu = b / np.linalg.norm(b, axis=1, keepdims=True)
        similarity = au @ bu.T
        gold = np.diagonal(similarity).copy()
        ranks = (similarity >= gold[:, None]).sum(axis=1)
        return float((ranks <= 1).mean())

    assert uncentred(source, target) != pytest.approx(
        uncentred(source + 25.0, target - 7.5))


# ── Procrustes ────────────────────────────────────────────────────────────────


def test_procrustes_recovers_a_known_rotation_on_held_out_ids():
    rng = np.random.default_rng(5)
    dim = 10
    rotation, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    train = rng.normal(size=(100, dim))
    test = rng.normal(size=(200, dim))

    fitted = ret.procrustes_rotation(train, train @ rotation)
    aligned = ret.centre(test) @ fitted
    metrics = ret.retrieval_metrics(aligned, test @ rotation)
    assert metrics["top1"] == pytest.approx(1.0)


def test_retrieval_metrics_refuses_unequal_candidate_counts():
    with pytest.raises(ValueError, match="unequal candidate counts"):
        ret.retrieval_metrics(np.zeros((5, 3)), np.zeros((6, 3)))


# ── the language probe groups by semantic id ──────────────────────────────────


def test_language_probe_groups_by_semantic_id_not_by_row():
    """A perfectly language-separable feature must still score well under grouped CV.

    The grouping is what stops the *same sentence* appearing in train and test in two
    different languages; this asserts the probe runs and reports on that basis.
    """
    rng = np.random.default_rng(6)
    ids = [f"s{i}" for i in range(20)]
    features = {
        "en": rng.normal(loc=0.0, scale=0.1, size=(20, 4)),
        "de": rng.normal(loc=5.0, scale=0.1, size=(20, 4)),
    }
    result = ret.language_probe(features, ids, n_folds=5, seed=0)
    assert result["status"] == "ok"
    assert result["n_folds"] == 5
    assert result["n_samples"] == 40
    assert result["chance_accuracy"] == pytest.approx(0.5)
    assert result["macro_f1"] > 0.9
    assert result["balanced_accuracy"] > 0.9


def test_language_probe_reports_a_skip_rather_than_crashing_on_one_language():
    rng = np.random.default_rng(7)
    result = ret.language_probe({"en": rng.normal(size=(6, 3))},
                                [f"s{i}" for i in range(6)])
    assert result["status"] == "skipped"
    assert "languages" in result["warning"]
    assert np.isnan(result["macro_f1"])
