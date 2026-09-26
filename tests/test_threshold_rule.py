# -*- coding: utf-8 -*-
"""test_threshold_rule.py — calibrated thresholds (WP12, ``08`` §7).

Framing (B) of decisions D2/D3/D4 pre-registers a **rule** rather than a number: "high"
means within *k* null spreads of an anchor, where the spread is measured across seeds
*within* one condition. The rule is the pre-registration; the number it yields is a result,
so it is written into ``go_no_go.json`` and never back into the YAML.

Two guards carry the weight, and both are refusals rather than fallbacks:

* fewer than three seeds in the null condition — a spread from two points is a gap, and the
  threshold would be an artefact of which two seeds happened to run;
* the null condition absent from a frame that *does* carry the treatment arm — returning
  "no threshold" there is indistinguishable from a criterion that never had one.

These tests assert the mechanism, never that a calibrated threshold takes a value.
"""

from __future__ import annotations

import json
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


def _row(condition, seed, value, step=1000):
    return {"experiment_id": "e6a", "run_id": f"e6a_{condition}_seed{seed}",
            "condition": condition, "seed": seed, "checkpoint_step": step,
            "corpus_id": CORPUS, "status": "ok", "n_items": 100, "n_failed": 0,
            "intervention_registry_version": "v1", "validation_ce": 2.5,
            "functional_cosine_to_teacher": value, "n_keys_used": 9}


def _frame(null_values=(0.90, 0.94, 0.98), treatment=0.95):
    rows = [_row("D0", seed, value) for seed, value in enumerate(null_values)]
    rows += [_row("D1", seed, treatment) for seed in range(len(null_values))]
    return pd.DataFrame(rows)


RULE = {"kind": "null_seed_spread", "metric": "functional_cosine_to_teacher",
        "null_condition": "D0", "statistic": "range", "k": 2.0, "anchor": 1.0}


def _entry(rule=RULE, direction="greater_equal"):
    entry = {"id": "e6a_3", "text": "functional inheritance remains high",
             "metric": "functional_cosine_to_teacher", "condition_a": "D1",
             "corpus_id": CORPUS, "direction": direction, "family": "e6a_go_no_go"}
    if rule:
        entry["threshold_rule"] = rule
    return entry


# ── the rule produces a number, and the number is recorded ─────────────────────


def test_the_range_rule_derives_a_threshold_from_the_null_conditions_spread():
    out = ag.evaluate_criterion(_frame(null_values=(0.90, 0.94, 0.98)), _entry(), {})
    rule = out["threshold_rule"]

    assert rule["spread"] == pytest.approx(0.08)                # 0.98 - 0.90
    assert rule["threshold"] == pytest.approx(1.0 - 2.0 * 0.08)  # anchor - k * spread
    assert out["threshold"] == pytest.approx(rule["threshold"])
    assert rule["n_seeds"] == 3
    assert rule["null_seed_values"] == pytest.approx([0.90, 0.94, 0.98])


def test_the_std_statistic_is_available_and_differs_from_the_range():
    values = (0.90, 0.94, 0.98)
    by_range = ag.evaluate_criterion(_frame(values), _entry(), {})["threshold_rule"]
    by_std = ag.evaluate_criterion(
        _frame(values), _entry({**RULE, "statistic": "std"}), {})["threshold_rule"]
    assert by_std["spread"] != pytest.approx(by_range["spread"])
    assert by_std["statistic"] == "std"


def test_k_and_anchor_come_from_the_yaml_not_the_code():
    frame = _frame()
    loose = ag.evaluate_criterion(frame, _entry({**RULE, "k": 2.0}), {})
    tight = ag.evaluate_criterion(frame, _entry({**RULE, "k": 0.1}), {})
    assert tight["threshold"] > loose["threshold"]
    # A tighter bar can flip the verdict on identical data — the proof it is not hard-coded.
    assert loose["met"] is True and tight["met"] is False


def test_the_realised_threshold_reaches_go_no_go_json(tmp_path):
    """`08` §7: the rule is the pre-registration, the number is written beside it."""
    path = tmp_path / "e6a" / "all" / "seed0" / "checkpoint_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame().to_csv(path, index=False)

    prereg_path = tmp_path / "prereg.yaml"
    import yaml
    prereg_path.write_text(yaml.safe_dump(
        {"version": "t", "contrasts": [], "go_no_go": [_entry()], "analysis": {},
         "e6b": {}}, sort_keys=False), encoding="utf-8")

    ag.aggregate(tmp_path, prereg_path=prereg_path, figures=False, progress=False)
    decision = json.loads((tmp_path / "aggregate" / "go_no_go.json")
                          .read_text(encoding="utf-8"))
    criterion = decision["criteria"][0]
    assert criterion["threshold_rule"]["kind"] == "null_seed_spread"
    assert criterion["threshold"] == pytest.approx(criterion["threshold_rule"]
                                                   ["threshold"])
    assert "RULE" in criterion["threshold_rule"]["note"].upper()


# ── which side of the anchor the boundary sits on ──────────────────────────────


def test_omitting_anchor_side_reproduces_the_pre_extension_threshold():
    """An additive field must change nothing when it is absent.

    ``anchor_side`` was added for D3/D4; every rule written before it exists must resolve
    to exactly the number it resolved to then, or the extension has quietly re-registered
    criteria that were already decided.
    """
    absent = ag.evaluate_criterion(_frame(), _entry(), {})["threshold_rule"]
    explicit = ag.evaluate_criterion(
        _frame(), _entry({**RULE, "anchor_side": "below"}), {})["threshold_rule"]
    assert absent["anchor_side"] == "below"
    assert absent["threshold"] == pytest.approx(explicit["threshold"])
    assert absent["threshold"] == pytest.approx(1.0 - 2.0 * 0.08)


def test_anchor_side_above_puts_the_boundary_above_the_anchor():
    """``e6a_4`` and ``e6b_2`` compare *absolute differences* anchored at 0.

    With the default ``below`` those resolve to ``0 - k*spread``, a negative bar: a
    ``less_equal`` limb ("similar function") could never be satisfied and a
    ``greater_equal`` limb ("measurably different") always would. Two criteria that
    silently cannot fail is worse than two that cannot be evaluated.
    """
    rule = {**RULE, "anchor": 0.0, "anchor_side": "above"}
    out = ag.evaluate_criterion(_frame(), _entry(rule), {})["threshold_rule"]
    assert out["spread"] == pytest.approx(0.08)
    assert out["threshold"] == pytest.approx(2.0 * 0.08)
    assert out["threshold"] > 0.0, "an absolute-difference bar must be positive"

    below = ag.evaluate_criterion(
        _frame(), _entry({**RULE, "anchor": 0.0}), {})["threshold_rule"]
    assert below["threshold"] < 0.0, (
        "this is the defect the field exists to prevent: the default side gives a "
        "negative bar for an anchor at 0")


def test_an_unknown_anchor_side_is_refused_rather_than_defaulted():
    """A present-but-unrecognised value is always refused (CLAUDE.md trap 13).

    Folding ``anchor_side: ABOVE`` into ``below`` would resolve the criterion *and* record
    a side it did not honour — an artefact that looks like an audit trail and is not one.
    """
    for bad in ("ABOVE", "above ", "over", True):
        out = ag.evaluate_criterion(_frame(), _entry({**RULE, "anchor_side": bad}), {})
        assert out["met"] is None, bad
        assert out["threshold"] is None, bad
        assert "unknown threshold_rule anchor_side" in out["threshold_rule"]["reason"]


def test_anchor_side_can_flip_a_verdict_on_identical_data():
    """The proof the field is read rather than merely recorded."""
    frame = _frame(null_values=(0.90, 0.94, 0.98), treatment=0.10)
    entry = _entry({**RULE, "anchor": 0.0, "anchor_side": "above"},
                   direction="less_equal")
    above = ag.evaluate_criterion(frame, entry, {})
    below = ag.evaluate_criterion(
        frame, _entry({**RULE, "anchor": 0.0}, direction="less_equal"), {})
    assert above["threshold"] == pytest.approx(0.16) and above["met"] is True
    assert below["threshold"] == pytest.approx(-0.16) and below["met"] is False


def test_the_calibration_uses_the_same_corpus_as_the_observed_value():
    """``--corpus X`` must move the threshold too, not only the number it is compared to.

    The rule scoped itself with a hard-coded ``None`` corpus override while the contrast
    beside it honoured the caller's. On a run using ``--corpus`` the criterion then
    compared a value measured on one corpus against a spread calibrated on another —
    a threshold that *looks* calibrated to the number it judges and is not.
    """
    other = "e1_100x40"
    rows = [_row("D0", seed, value) for seed, value in enumerate((0.90, 0.94, 0.98))]
    rows += [_row("D1", seed, 0.95) for seed in range(3)]
    # Same conditions and seeds on a second corpus, with a deliberately different spread.
    for seed, value in enumerate((0.10, 0.50, 0.90)):
        rows.append({**_row("D0", seed, value), "corpus_id": other})
    for seed in range(3):
        rows.append({**_row("D1", seed, 0.95), "corpus_id": other})
    frame = pd.DataFrame(rows)

    default = ag.evaluate_criterion(frame, _entry(), {})["threshold_rule"]
    overridden = ag.evaluate_criterion(frame, _entry(), {},
                                       corpus_override=other)["threshold_rule"]

    assert default["spread"] == pytest.approx(0.08)        # 0.98 - 0.90
    assert overridden["spread"] == pytest.approx(0.80)     # 0.90 - 0.10
    assert overridden["corpus_id"] == other
    assert default["threshold"] != pytest.approx(overridden["threshold"])


# ── the two guards ─────────────────────────────────────────────────────────────


def test_two_seeds_are_refused_because_a_spread_needs_three():
    out = ag.evaluate_criterion(_frame(null_values=(0.90, 0.98)), _entry(), {})
    assert out["met"] is None
    assert out["threshold"] is None
    assert "at least three" in out["threshold_rule"]["reason"]
    assert out["threshold_rule"]["n_seeds"] == 2


def test_a_missing_null_arm_beside_a_present_treatment_arm_is_refused_loudly():
    frame = _frame()
    frame = frame[frame["condition"] != "D0"]          # treatment present, null gone
    out = ag.evaluate_criterion(frame, _entry(), {})
    assert out["met"] is None
    reason = out["threshold_rule"]["reason"]
    assert "no usable" in reason
    assert "treatment arm" in reason, (
        "an absent null condition must say the calibration was refused, not merely that "
        "there was no threshold (08 §7)")


def test_an_unknown_rule_kind_is_refused_rather_than_guessed():
    out = ag.evaluate_criterion(_frame(), _entry({**RULE, "kind": "vibes"}), {})
    assert out["met"] is None
    assert "refuses to guess" in out["threshold_rule"]["reason"]


def test_an_unknown_statistic_is_refused():
    out = ag.evaluate_criterion(_frame(),
                                _entry({**RULE, "statistic": "interquartile"}), {})
    assert out["met"] is None
    assert "unknown threshold_rule statistic" in out["threshold_rule"]["reason"]


# ── an explicit threshold still wins, and absence changes nothing ──────────────


def test_an_explicit_threshold_takes_precedence_over_the_rule():
    entry = {**_entry(), "threshold": 0.50}
    out = ag.evaluate_criterion(_frame(), entry, {})
    assert out["threshold"] == 0.50
    assert "threshold_rule" not in out, "the rule must not be evaluated when unused"


def test_a_criterion_without_a_rule_or_threshold_is_unresolved_not_calibrated():
    out = ag.evaluate_criterion(_frame(), _entry(rule=None), {})
    assert out["met"] is None
    assert out["threshold"] is None
    assert "no threshold" in out["reason"]
