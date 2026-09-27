# -*- coding: utf-8 -*-
"""test_claim_gate_no_double_count.py — ``claim_gate_component`` (WP12, ``08`` §9).

Design §18's E7 criteria 4 and 5 — "the direction replicates in 0.5B and 1.5B models" and
"the result holds in at least five target languages" — **are** the two halves of the §10.11
claim gate rather than additional to it. Evaluating them independently does two bad things
at once: it counts criterion 4 twice in ``n_met``, and it computes the same quantity twice,
so a criterion and its own gate can end up contradicting each other in one artefact.

What is asserted:

* a ``claim_gate_component`` criterion reads its verdict from ``evaluate_claim_gate``
  rather than recomputing it — including when the gate itself has no verdict;
* ids listed in ``claim_gate.satisfies_criteria`` are still *reported* in ``criteria`` but
  excluded from ``n_met``, and both readings are visible;
* the ``e7_3k``/``e7_3v`` pair folds through ``criterion_3_combine`` instead of counting
  design §18 criterion 3 twice;
* ``go_no_go_combine: all`` is honoured, and its absence reproduces today's at-least-one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "crosslingual_semantics"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import aggregate_crosslingual as ag  # noqa: E402


def _contrasts(languages=6, sizes=2, effect=0.30):
    return pd.DataFrame([
        {"id": contrast_id, "status": "ok", "absolute_difference": effect,
         "n_languages_same_direction": languages,
         "n_model_sizes_same_direction": sizes}
        for contrast_id in ("e7_c1", "e7_c2")])


def _retrieval(top1_over_chance=6.0, delta_points=9.0):
    return pd.DataFrame([{
        "model_tag": "t05", "object": "K0_prerope", "layer": 3, "alignment": "cosine",
        "top1": 0.02, "top1_over_chance": top1_over_chance,
        "position0_minus_mid_top1_points": delta_points}])


def _prereg(*, combine=None, criterion_3_combine=None, satisfies=("e7_4", "e7_5")):
    go_no_go = [
        {"id": "e7_1", "text": "retrieval", "input": "retrieval",
         "metric": "top1_over_chance", "objects": ["K0_prerope"],
         "alignment": "cosine", "threshold": 2.0, "direction": "greater_equal"},
        {"id": "e7_3k", "text": "K0", "input": "patching", "contrast": "e7_c1",
         "metric": "absolute_difference", "threshold": 0.10,
         "direction": "greater_equal"},
        {"id": "e7_3v", "text": "V0", "input": "patching", "contrast": "e7_c2",
         "metric": "absolute_difference", "threshold": 0.10,
         "direction": "greater_equal"},
        {"id": "e7_4", "text": "model sizes", "input": "patching",
         "evaluation": "claim_gate_component", "claim_gate_field": "min_model_sizes"},
        {"id": "e7_5", "text": "languages", "input": "patching",
         "evaluation": "claim_gate_component", "claim_gate_field": "min_languages"},
    ]
    prereg = {
        "version": "t", "contrasts": [], "go_no_go": go_no_go,
        "claim_gate": {"id": "e7_claim_gate", "text": "gate",
                       "contrasts": ["e7_c1", "e7_c2"], "min_model_sizes": 2,
                       "min_languages": 5, "n_languages_total": 7,
                       "satisfies_criteria": list(satisfies)},
        "interpretation_matrix": {"status": "ok", "rows": []},
        "analysis": {},
    }
    if combine:
        prereg["go_no_go_combine"] = combine
    if criterion_3_combine:
        prereg["criterion_3_combine"] = criterion_3_combine
    return prereg


# ── the components read the gate ───────────────────────────────────────────────


def test_a_component_criterion_reports_the_gates_own_numbers():
    decision = ag.build_go_no_go(_prereg(), _contrasts(languages=6, sizes=2),
                                 _retrieval())
    by_id = {c["id"]: c for c in decision["criteria"]}

    assert by_id["e7_4"]["observed"] == 2 and by_id["e7_4"]["threshold"] == 2
    assert by_id["e7_5"]["observed"] == 6 and by_id["e7_5"]["threshold"] == 5
    assert by_id["e7_4"]["met"] is True and by_id["e7_5"]["met"] is True
    components = decision["claim_gate"]["components"]
    assert components["min_model_sizes"]["observed"] == by_id["e7_4"]["observed"]
    assert components["min_languages"]["observed"] == by_id["e7_5"]["observed"]


def test_a_component_cannot_disagree_with_the_gate():
    """One languages count below the bar must fail both, never one of the two."""
    decision = ag.build_go_no_go(_prereg(), _contrasts(languages=3, sizes=2),
                                 _retrieval())
    by_id = {c["id"]: c for c in decision["criteria"]}
    assert by_id["e7_5"]["met"] is False
    assert decision["claim_gate"]["met"] is False
    assert decision["claim_gate"]["components"]["min_languages"]["met"] is False


def test_a_component_with_no_gate_verdict_reports_no_input_rather_than_false():
    empty = pd.DataFrame(columns=["id", "status", "absolute_difference",
                                  "n_languages_same_direction",
                                  "n_model_sizes_same_direction"])
    decision = ag.build_go_no_go(_prereg(), empty, _retrieval())
    by_id = {c["id"]: c for c in decision["criteria"]}
    assert by_id["e7_4"]["met"] is None
    assert by_id["e7_4"]["status"] == "no_input"
    assert decision["decision"] == "incomplete"


def test_an_unknown_component_field_is_refused_not_computed_locally():
    prereg = _prereg()
    prereg["go_no_go"][3]["claim_gate_field"] = "min_planets"
    decision = ag.build_go_no_go(prereg, _contrasts(), _retrieval())
    by_id = {c["id"]: c for c in decision["criteria"]}
    assert by_id["e7_4"]["met"] is None
    assert "exposes no" in by_id["e7_4"]["reason"]


# ── no double counting ─────────────────────────────────────────────────────────


def test_gate_components_are_reported_but_not_counted_in_n_met():
    decision = ag.build_go_no_go(_prereg(criterion_3_combine={"applies_to":
                                                              ["e7_3k", "e7_3v"],
                                                              "combine": "any"}),
                                 _contrasts(), _retrieval())

    assert {c["id"] for c in decision["criteria"]} == {"e7_1", "e7_3k", "e7_3v",
                                                       "e7_4", "e7_5"}
    # counted: e7_1 and the folded criterion 3. e7_4/e7_5 are the gate itself.
    assert decision["n_counted"] == 2
    assert decision["n_met"] == 2
    assert decision["n_met_excluding_gate_components"] == decision["n_met"]
    # Both readings visible: all five criteria met, but only two independent ones.
    assert decision["n_met_all_criteria"] == 5


def test_removing_satisfies_criteria_counts_them_again():
    """The exclusion is read from the YAML, not hard-coded on the id."""
    with_exclusion = ag.build_go_no_go(_prereg(), _contrasts(), _retrieval())
    without = ag.build_go_no_go(_prereg(satisfies=()), _contrasts(), _retrieval())
    assert without["n_counted"] > with_exclusion["n_counted"]


def test_criterion_3s_two_objects_fold_into_one_verdict():
    decision = ag.build_go_no_go(
        _prereg(criterion_3_combine={"applies_to": ["e7_3k", "e7_3v"],
                                     "combine": "any"}),
        _contrasts(), _retrieval())
    composite = decision["criterion_3"]
    assert composite["id"] == "e7_3" and composite["met"] is True
    assert composite["combine"] == "any"
    assert composite["limbs"] == {"e7_3k": True, "e7_3v": True}
    # Both limbs are still reported individually.
    assert {"e7_3k", "e7_3v"} <= {c["id"] for c in decision["criteria"]}


def test_any_and_all_differ_when_the_objects_disagree():
    contrasts = _contrasts()
    contrasts.loc[contrasts["id"] == "e7_c2", "absolute_difference"] = 0.01

    any_ = ag.build_go_no_go(_prereg(criterion_3_combine={
        "applies_to": ["e7_3k", "e7_3v"], "combine": "any"}), contrasts, _retrieval())
    all_ = ag.build_go_no_go(_prereg(criterion_3_combine={
        "applies_to": ["e7_3k", "e7_3v"], "combine": "all"}), contrasts, _retrieval())

    assert any_["criterion_3"]["met"] is True
    assert all_["criterion_3"]["met"] is False


def test_a_pending_criterion_3_combine_refuses_to_fold_and_blocks_the_decision():
    decision = ag.build_go_no_go(_prereg(criterion_3_combine={
        "status": "PENDING_DECISION_COMBINE", "applies_to": ["e7_3k", "e7_3v"],
        "reason": "D6 is undecided"}), _contrasts(), _retrieval())

    assert decision["criterion_3"]["met"] is None
    assert decision["decision"] == "incomplete"
    assert decision["supported"] is None
    assert "criterion_3_combine" in decision["pending_decisions_blocking"]


# ── go_no_go_combine ───────────────────────────────────────────────────────────


def test_combine_all_requires_every_counted_criterion():
    contrasts = _contrasts()
    contrasts["absolute_difference"] = 0.01          # criterion 3 fails, e7_1 passes
    fold = {"applies_to": ["e7_3k", "e7_3v"], "combine": "any"}

    any_ = ag.build_go_no_go(_prereg(criterion_3_combine=fold), contrasts, _retrieval())
    all_ = ag.build_go_no_go(_prereg(combine="all", criterion_3_combine=fold),
                             contrasts, _retrieval())

    assert any_["decision"] == "continue", "at-least-one is today's default"
    assert all_["decision"] == "stop"
    assert all_["combine"] == "all" and any_["combine"] == "any"


def test_combine_all_continues_when_everything_is_met():
    fold = {"applies_to": ["e7_3k", "e7_3v"], "combine": "any"}
    decision = ag.build_go_no_go(_prereg(combine="all", criterion_3_combine=fold),
                                 _contrasts(), _retrieval())
    assert decision["decision"] == "continue"
    assert decision["supported"] is True


def test_the_gate_still_vetoes_a_continue():
    fold = {"applies_to": ["e7_3k", "e7_3v"], "combine": "any"}
    decision = ag.build_go_no_go(_prereg(criterion_3_combine=fold),
                                 _contrasts(languages=1, sizes=1), _retrieval())
    assert decision["claim_gate"]["met"] is False
    assert decision["decision"] == "stop" and decision["supported"] is False
