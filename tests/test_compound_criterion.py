# -*- coding: utf-8 -*-
"""test_compound_criterion.py — ``compound: {all_of: [...]}`` (WP12, ``08`` §2.3).

Design §18 criteria 3 and 4 are conjunctions whose limbs run in **opposite directions** —
"functional inheritance remains high *while* mechanistic inheritance is low", "similar
function *through* measurably different fingerprints". That opposition is the whole content
of the criterion, so a compound that collapsed to a single number would test nothing.

What is asserted:

* ``met`` is the AND of the limbs, and every limb reports its own observed value,
  threshold, direction and verdict, so a failure names which limb failed;
* a limb with no verdict makes the compound ``None``, **never** ``False`` — unknown is not
  refuted, and a criterion that reported "not met" for want of data would read as evidence
  against the hypothesis;
* ``status: PENDING_*`` on the parent short-circuits the whole thing, as before;
* a limb may be single-armed (``condition_a`` with no ``condition_b``), which is the shape
  design §18 criterion 3 actually uses.

No threshold here is a scientific claim; each test supplies its own and asserts the
*relationship* between it and the verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import aggregate_transformation as ag  # noqa: E402

CORPUS = "tinystories_validation_sink_300"


def _row(condition, seed, *, func, fingerprint, step=1000):
    return {"experiment_id": "e6a", "run_id": f"e6a_{condition}_seed{seed}",
            "condition": condition, "seed": seed, "checkpoint_step": step,
            "corpus_id": CORPUS, "status": "ok", "n_items": 100, "n_failed": 0,
            "intervention_registry_version": "v1", "validation_ce": 2.5,
            "functional_cosine_to_teacher": func,
            "fingerprint_cosine_to_teacher": fingerprint, "n_keys_used": 9}


def _frame(func=0.95, fingerprint=0.40):
    return pd.DataFrame([_row("D1", seed, func=func, fingerprint=fingerprint)
                         for seed in (0, 1, 2)])


def _compound(high, low):
    """§18 criterion 3's shape: functional high AND mechanistic low, single-armed."""
    return {"id": "e6a_3", "text": "functional high while mechanistic low",
            "family": "e6a_go_no_go",
            "compound": {"all_of": [
                {"metric": "functional_cosine_to_teacher", "condition_a": "D1",
                 "corpus_id": CORPUS, "threshold": high, "direction": "greater_equal"},
                {"metric": "fingerprint_cosine_to_teacher", "condition_a": "D1",
                 "corpus_id": CORPUS, "threshold": low, "direction": "less_equal"}]}}


# ── AND semantics ──────────────────────────────────────────────────────────────


def test_both_limbs_satisfied_is_met():
    out = ag.evaluate_criterion(_frame(func=0.95, fingerprint=0.40),
                                _compound(high=0.90, low=0.60), {})
    assert out["met"] is True
    assert out["compound"] == "all_of"
    assert [sub["met"] for sub in out["sub_criteria"]] == [True, True]


@pytest.mark.parametrize("func,fingerprint,expected_failing", [
    (0.80, 0.40, ["all_of[0]"]),      # functional not high enough
    (0.95, 0.80, ["all_of[1]"]),      # mechanistic not low enough
    (0.80, 0.80, ["all_of[0]", "all_of[1]"]),
])
def test_a_failing_limb_is_named(func, fingerprint, expected_failing):
    out = ag.evaluate_criterion(_frame(func=func, fingerprint=fingerprint),
                                _compound(high=0.90, low=0.60), {})
    assert out["met"] is False
    failing = [sub["limb"] for sub in out["sub_criteria"] if sub["met"] is False]
    assert failing == expected_failing
    assert str(expected_failing) in out["reason"]


def test_the_limbs_run_in_opposite_directions():
    """The opposition is the content of the criterion, so it is asserted directly."""
    out = ag.evaluate_criterion(_frame(), _compound(high=0.90, low=0.60), {})
    directions = [sub["direction"] for sub in out["sub_criteria"]]
    assert directions == ["greater_equal", "less_equal"]


def test_every_limb_reports_its_own_observed_threshold_and_direction():
    out = ag.evaluate_criterion(_frame(func=0.95, fingerprint=0.40),
                                _compound(high=0.90, low=0.60), {})
    first, second = out["sub_criteria"]
    assert first["observed"] == pytest.approx(0.95)
    assert first["threshold"] == 0.90
    assert second["observed"] == pytest.approx(0.40)
    assert second["threshold"] == 0.60


# ── unknown is not refuted ─────────────────────────────────────────────────────


def test_a_limb_with_no_threshold_makes_the_compound_none_not_false():
    """The v3 file ships exactly this state: `threshold: null` pending decision D2."""
    out = ag.evaluate_criterion(_frame(), _compound(high=None, low=0.60), {})
    assert out["met"] is None, "an undecided threshold must never read as 'not met'"
    assert [sub["met"] for sub in out["sub_criteria"]] == [None, True]
    assert "all_of[0]" in out["reason"]


def test_a_limb_with_no_data_makes_the_compound_none():
    empty = pd.DataFrame(columns=["seed", "condition", "checkpoint_step", "corpus_id",
                                  "functional_cosine_to_teacher",
                                  "fingerprint_cosine_to_teacher"])
    out = ag.evaluate_criterion(empty, _compound(high=0.90, low=0.60), {})
    assert out["met"] is None
    assert all(sub["met"] is None for sub in out["sub_criteria"])


def test_a_none_limb_beats_a_false_limb():
    """One unresolved and one refuted limb is still unresolved, not refuted."""
    out = ag.evaluate_criterion(_frame(func=0.10, fingerprint=0.40),
                                _compound(high=None, low=0.60), {})
    assert out["met"] is None


# ── the parent's PENDING status still short-circuits ───────────────────────────


def test_a_pending_parent_short_circuits_the_whole_compound():
    entry = {**_compound(high=0.90, low=0.60),
             "status": "PENDING_DECISION_THRESHOLD", "reason": "D2 is undecided"}
    out = ag.evaluate_criterion(_frame(), entry, {})
    assert out["met"] is None
    assert out["status"] == "PENDING_DECISION_THRESHOLD"
    assert out["reason"] == "D2 is undecided"
    assert "sub_criteria" not in out


# ── seed consistency composes with compounds ───────────────────────────────────


def test_seed_consistency_on_the_parent_reaches_every_limb():
    rows = [_row("D1", 0, func=0.95, fingerprint=0.40),
            _row("D1", 1, func=0.95, fingerprint=0.40),
            _row("D1", 2, func=0.20, fingerprint=0.40)]   # one seed fails the first limb
    entry = {**_compound(high=0.90, low=0.60), "seed_consistency": "all"}
    out = ag.evaluate_criterion(pd.DataFrame(rows), entry, {})
    assert out["met"] is False
    assert out["sub_criteria"][0]["n_seeds_meeting"] == 2
    assert out["sub_criteria"][1]["n_seeds_meeting"] == 3


def test_the_compound_verdict_follows_the_yaml_not_the_code():
    frame = _frame(func=0.95, fingerprint=0.40)
    assert ag.evaluate_criterion(frame, _compound(high=0.90, low=0.60), {})["met"] is True
    assert ag.evaluate_criterion(frame, _compound(high=0.99, low=0.60), {})["met"] is False
