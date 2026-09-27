# -*- coding: utf-8 -*-
"""test_seed_consistency.py — ``seed_consistency: all`` (WP12, ``08`` §5.1).

This is a **correctness fix**, not a feature. Design §18 reads "at least one of the
following is observed *across all three seeds*", and the pre-WP12 aggregator set ``met``
from the pooled ``mean_diff`` while recording ``all_seeds`` unused — so a criterion could
pass on one strong seed and two weak ones and nothing in the artefact would say so.

**The example in ``08`` §5.1 does not have the property it claims, and is corrected here
rather than worked around.** It asks for per-seed diffs ``[0.30, 0.05, 0.05]`` against
threshold ``0.15`` to be ``met: true`` without the flag and ``met: false`` with it — but
those three average to ``0.4 / 3 = 0.1333``, which is *below* 0.15, so the pooled reading
is ``False`` too and the case demonstrates nothing. The spec's literal triple is asserted
below exactly as it behaves, and the contrast it was reaching for is made with
``[0.30, 0.10, 0.10]`` (pooled ``0.1667 >= 0.15``; one seed of three clears the bar) — the
smallest change that gives the example its intended shape.

The flag's absence must keep today's behaviour exactly; that half is asserted too, because
a "fix" that changed every criterion silently would be the worse bug.
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


def _row(condition, seed, cos, step=1000):
    return {"experiment_id": "e6a", "run_id": f"e6a_{condition}_seed{seed}",
            "condition": condition, "seed": seed, "checkpoint_step": step,
            "corpus_id": CORPUS, "status": "ok", "n_items": 100, "n_failed": 0,
            "intervention_registry_version": "v1", "validation_ce": 2.5,
            "fingerprint_cosine_to_teacher": cos, "n_keys_used": 9}


def _frame(diffs):
    """A frame whose per-seed D2 − D0 differences are exactly ``diffs``."""
    rows = []
    for seed, diff in enumerate(diffs):
        rows.append(_row("D0", seed, 0.40))
        rows.append(_row("D2", seed, 0.40 + diff))
    return pd.DataFrame(rows)


def _criterion(seed_consistency=None, threshold=0.15):
    entry = {"id": "c", "text": "t", "metric": "fingerprint_cosine_to_teacher",
             "condition_a": "D2", "condition_b": "D0", "corpus_id": CORPUS,
             "threshold": threshold, "direction": "greater_equal"}
    if seed_consistency:
        entry["seed_consistency"] = seed_consistency
    return entry


# ── the case 08 §5.1 names ─────────────────────────────────────────────────────


def test_one_strong_seed_carries_the_pooled_mean_but_not_seed_consistency():
    """``[0.30, 0.10, 0.10]`` vs 0.15: pooled says yes, per-seed says no."""
    frame = _frame([0.30, 0.10, 0.10])

    pooled = ag.evaluate_criterion(frame, _criterion(), {})
    strict = ag.evaluate_criterion(frame, _criterion(seed_consistency="all"), {})

    assert pooled["met"] is True, (
        "without the flag the pooled mean 0.1667 must still be compared to 0.15 the way "
        "it always was — this half asserts the fix changed nothing it should not")
    assert strict["met"] is False
    assert strict["n_seeds_meeting"] == 1
    assert strict["per_seed_met"] == [True, False, False]
    # The pooled reading stays visible so a reader can see *why* the two disagree.
    assert strict["pooled_met"] is True


def test_the_spec_example_is_below_the_bar_under_both_readings():
    """``08`` §5.1's literal triple, asserted as it actually behaves.

    Recorded rather than quietly replaced: ``[0.30, 0.05, 0.05]`` averages to 0.1333, so
    the spec's "met: true without the flag" is not attainable from these numbers. If a
    later revision of `08` fixes the example, this test is the place the discrepancy is
    already written down.
    """
    frame = _frame([0.30, 0.05, 0.05])
    assert ag.evaluate_criterion(frame, _criterion(), {})["met"] is False
    strict = ag.evaluate_criterion(frame, _criterion(seed_consistency="all"), {})
    assert strict["met"] is False and strict["n_seeds_meeting"] == 1


def test_pooled_mean_alone_would_not_have_flagged_the_disagreement():
    """Regression guard: the pre-fix path reported no per-seed detail at all."""
    strict = ag.evaluate_criterion(_frame([0.30, 0.10, 0.10]),
                                   _criterion(seed_consistency="all"), {})
    assert "per_seed_met" in strict and "n_seeds_meeting" in strict


# ── the flag is inert when every seed agrees ───────────────────────────────────


def test_every_seed_clearing_the_bar_is_met_under_both_readings():
    frame = _frame([0.30, 0.20, 0.25])
    assert ag.evaluate_criterion(frame, _criterion(), {})["met"] is True
    strict = ag.evaluate_criterion(frame, _criterion(seed_consistency="all"), {})
    assert strict["met"] is True and strict["n_seeds_meeting"] == 3


def test_no_seed_clearing_the_bar_is_unmet_under_both_readings():
    frame = _frame([0.05, 0.04, 0.03])
    assert ag.evaluate_criterion(frame, _criterion(), {})["met"] is False
    strict = ag.evaluate_criterion(frame, _criterion(seed_consistency="all"), {})
    assert strict["met"] is False and strict["n_seeds_meeting"] == 0


# ── absence reproduces today's behaviour ───────────────────────────────────────


def test_a_criterion_without_the_flag_records_no_seed_consistency_fields():
    out = ag.evaluate_criterion(_frame([0.30, 0.05, 0.05]), _criterion(), {})
    assert "seed_consistency" not in out
    assert "per_seed_met" not in out


def test_the_flag_is_read_from_the_yaml_not_the_code():
    """Toggling only the YAML field must flip the verdict on identical data."""
    frame = _frame([0.30, 0.10, 0.10])
    prereg_off = {"contrasts": [], "analysis": {}, "e6b": {}, "version": "t",
                  "go_no_go": [_criterion()]}
    prereg_on = {"contrasts": [], "analysis": {}, "e6b": {}, "version": "t",
                 "go_no_go": [_criterion(seed_consistency="all")]}
    assert ag.build_go_no_go(frame, prereg_off, {})["criteria"][0]["met"] is True
    assert ag.build_go_no_go(frame, prereg_on, {})["criteria"][0]["met"] is False


def test_seed_consistency_applies_to_the_absolute_difference_when_asked():
    """A directionless criterion tests |diff| per seed, not the signed value."""
    frame = _frame([0.30, -0.30, 0.30])
    entry = {**_criterion(seed_consistency="all", threshold=0.10),
             "absolute_difference": True}
    out = ag.evaluate_criterion(frame, entry, {})
    assert out["met"] is True and out["n_seeds_meeting"] == 3
    # The signed mean is ~0.10 and would have hidden the sign flip; the magnitudes did not.
    assert out["all_seeds"] == pytest.approx([0.30, 0.30, 0.30])
