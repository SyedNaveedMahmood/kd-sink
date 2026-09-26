# -*- coding: utf-8 -*-
"""test_combine_refusals.py — a half-read decision rule is refused, not defaulted.

The defect this pins down was live in both aggregators. ``criterion_3_combine`` decides how
design §18 criterion 3's two objects fold, and the fold read it as::

    combine = str(spec.get("combine") or spec.get("value") or "any").lower()
    ...
    if combine == "all": ... else: <any>

Two ways that goes wrong, both silent:

* **A missing key on a resolved-looking entry.** Deleting ``status: PENDING_DECISION_COMBINE``
  without adding ``combine:`` — the natural half-finished edit — produced ``status: "ok"``,
  ``combine: "any"`` and a verdict. Decision D6 was resolved *in code*, which `05` §6
  forbids.
* **A value the code does not honour.** ``combine: ALL_`` fell through the ``else`` to
  ``any`` while the artefact recorded ``combine: "all_"``. That is worse than crashing: the
  row looks like an audit trail and is not one.

``go_no_go_combine`` (both aggregators) and ``e6b_go_no_go.combine`` shared the second half.
An *absent* ``go_no_go_combine`` still defaults to ``any`` — that is `08` §5.2's specified
default and today's behaviour — so the absent and the unrecognised cases are asserted apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "crosslingual_semantics",
              REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import aggregate_crosslingual as ag7  # noqa: E402
import aggregate_transformation as ag6  # noqa: E402

CORPUS = "tinystories_validation_sink_300"


# ── E7: criterion_3_combine ────────────────────────────────────────────────────


def _limbs(k=True, v=False):
    return [{"id": "e7_3k", "met": k}, {"id": "e7_3v", "met": v}]


def _spec(**extra):
    return {"criterion_3_combine": {"applies_to": ["e7_3k", "e7_3v"], **extra}}


@pytest.mark.parametrize("combine,expected", [("any", True), ("all", False)])
def test_an_explicit_rule_is_honoured(combine, expected):
    out = ag7._combine_criterion_3(_spec(combine=combine), _limbs())
    assert out["status"] == "ok"
    assert out["combine"] == combine
    assert out["met"] is expected


def test_deleting_the_status_without_adding_combine_is_refused():
    """The half-finished edit. Before the fix this returned met=True, status='ok'."""
    out = ag7._combine_criterion_3(_spec(options=["any", "all"]), _limbs())
    assert out["met"] is None
    assert out["combine"] is None
    assert out["status"] == "unresolved_combine"
    assert "D6" in out["reason"]
    assert "combine:" in out["reason"]


@pytest.mark.parametrize("bad", ["ALL_", "enny", "", "  ", "either", "1", "None"])
def test_an_unrecognised_rule_is_refused_rather_than_defaulted(bad):
    out = ag7._combine_criterion_3(_spec(combine=bad), _limbs())
    assert out["met"] is None, f"{bad!r} silently resolved to a verdict"
    assert out["combine"] is None
    assert out["declared_combine"] == bad, (
        "the refused value must still be reported, or the operator cannot see the typo")


@pytest.mark.parametrize("messy,expected", [("  All  ", "all"), ("ANY", "any"),
                                            ("Any\n", "any")])
def test_case_and_surrounding_whitespace_are_tolerated(messy, expected):
    """Tolerating a YAML formatting artefact is not the same as guessing a rule."""
    out = ag7._combine_criterion_3(_spec(combine=messy), _limbs())
    assert out["status"] == "ok" and out["combine"] == expected


def test_a_pending_status_still_short_circuits_ahead_of_the_refusal():
    out = ag7._combine_criterion_3(
        _spec(status="PENDING_DECISION_COMBINE", reason="D6 undecided"), _limbs())
    assert out["met"] is None
    assert out["status"] == "PENDING_DECISION_COMBINE"
    assert out["reason"] == "D6 undecided"


def test_the_refusal_reaches_the_decision(tmp_path):
    """An unresolved fold must force ``incomplete``, not merely be reported."""
    contrasts = pd.DataFrame([
        {"id": cid, "status": "ok", "absolute_difference": 0.30,
         "n_languages_same_direction": 6, "n_model_sizes_same_direction": 2}
        for cid in ("e7_c1", "e7_c2")])
    prereg = {
        "version": "t", "contrasts": [], "analysis": {},
        "go_no_go": [
            {"id": "e7_3k", "input": "patching", "contrast": "e7_c1",
             "metric": "absolute_difference", "threshold": 0.10,
             "direction": "greater_equal"},
            {"id": "e7_3v", "input": "patching", "contrast": "e7_c2",
             "metric": "absolute_difference", "threshold": 0.10,
             "direction": "greater_equal"}],
        "claim_gate": {"contrasts": ["e7_c1", "e7_c2"], "min_model_sizes": 2,
                       "min_languages": 5},
        "interpretation_matrix": {"status": "ok", "rows": []},
        "criterion_3_combine": {"applies_to": ["e7_3k", "e7_3v"]},   # no `combine:`
    }
    decision = ag7.build_go_no_go(prereg, contrasts, None)
    assert decision["criterion_3"]["met"] is None
    assert decision["decision"] == "incomplete"
    assert decision["supported"] is None


# ── E7: go_no_go_combine ───────────────────────────────────────────────────────


def _minimal_prereg(**extra):
    return {"version": "t", "contrasts": [], "analysis": {},
            "go_no_go": [{"id": "c", "input": "patching", "contrast": "e7_c1",
                          "metric": "absolute_difference", "threshold": 0.10,
                          "direction": "greater_equal"}],
            "claim_gate": {"contrasts": ["e7_c1"], "min_model_sizes": 1,
                           "min_languages": 1},
            "interpretation_matrix": {"status": "ok", "rows": []}, **extra}


def _one_contrast():
    return pd.DataFrame([{"id": "e7_c1", "status": "ok", "absolute_difference": 0.30,
                          "n_languages_same_direction": 6,
                          "n_model_sizes_same_direction": 2}])


def test_an_absent_go_no_go_combine_still_defaults_to_any():
    """`08` §5.2's specified default, and today's behaviour. Must not have moved."""
    decision = ag7.build_go_no_go(_minimal_prereg(), _one_contrast(), None)
    assert decision["combine"] == "any"
    assert decision["combine_error"] == ""
    assert decision["decision"] == "continue"


@pytest.mark.parametrize("bad", ["al", "ALL!", "both", "yes"])
def test_an_unrecognised_go_no_go_combine_forces_incomplete(bad):
    decision = ag7.build_go_no_go(_minimal_prereg(go_no_go_combine=bad),
                                  _one_contrast(), None)
    assert decision["decision"] == "incomplete"
    assert decision["supported"] is None
    assert bad in decision["combine_error"]


# ── E6: go_no_go_combine and e6b_go_no_go.combine ──────────────────────────────


def _e6_frame(cos_d2=0.80, cos_d0=0.40):
    rows = []
    for seed in (0, 1, 2):
        for condition, cos in (("D0", cos_d0), ("D2", cos_d2)):
            rows.append({"experiment_id": "e6a", "run_id": f"r{condition}{seed}",
                         "condition": condition, "seed": seed, "checkpoint_step": 1000,
                         "corpus_id": CORPUS, "status": "ok", "n_items": 100,
                         "n_failed": 0, "intervention_registry_version": "v1",
                         "validation_ce": 2.5, "fingerprint_cosine_to_teacher": cos,
                         "n_keys_used": 9})
    return pd.DataFrame(rows)


def _e6_prereg(**extra):
    return {"version": "t", "contrasts": [], "analysis": {}, "e6b": {},
            "go_no_go": [{"id": "c", "text": "t",
                          "metric": "fingerprint_cosine_to_teacher",
                          "condition_a": "D2", "condition_b": "D0",
                          "corpus_id": CORPUS, "threshold": 0.15,
                          "direction": "greater_equal"}], **extra}


def test_e6_absent_combine_defaults_to_any():
    decision = ag6.build_go_no_go(_e6_frame(), _e6_prereg(), {})
    assert decision["combine"] == "any"
    assert decision["combine_error"] == ""
    assert decision["decision"] == "continue"


@pytest.mark.parametrize("bad", ["al", "every", "ALL?"])
def test_e6_unrecognised_combine_forces_incomplete(bad):
    decision = ag6.build_go_no_go(_e6_frame(), _e6_prereg(go_no_go_combine=bad), {})
    assert decision["decision"] == "incomplete"
    assert bad in decision["combine_error"]


def test_the_e6b_block_can_override_the_rule_and_is_validated_too():
    frame = _e6_frame()
    good = ag6.build_go_no_go(frame, {"version": "t", "contrasts": [], "analysis": {},
                                      "e6b": {}, "go_no_go": [],
                                      "e6b_go_no_go": {"combine": "all",
                                                       "criteria": []}},
                              {}, experiment="e6b", criteria_key="e6b_go_no_go")
    assert good["combine"] == "all" and good["combine_error"] == ""

    bad = ag6.build_go_no_go(frame, {"version": "t", "contrasts": [], "analysis": {},
                                     "e6b": {}, "go_no_go": [],
                                     "e6b_go_no_go": {"combine": "alll",
                                                      "criteria": []}},
                             {}, experiment="e6b", criteria_key="e6b_go_no_go")
    assert bad["decision"] == "incomplete"
    assert "alll" in bad["combine_error"]


def test_the_shipped_preregistrations_still_parse_cleanly():
    """Neither shipped YAML may be tripped up by the new validation."""
    e6 = ag6.build_go_no_go(_e6_frame(), ag6.load_preregistration(), {})
    assert e6["combine_error"] == "", e6["combine_error"]
    assert e6["decision"] == "incomplete"      # e6a_3 / e6a_4 are still undecided

    e7 = ag7.build_go_no_go(ag7.load_preregistration(), _one_contrast(), None)
    assert e7["combine"] == "all", "the shipped E7 file declares a conjunction"
    assert e7["combine_error"] == ""
    assert e7["decision"] == "incomplete"      # criterion_3_combine is still undecided
