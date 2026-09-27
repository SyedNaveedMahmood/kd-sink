# -*- coding: utf-8 -*-
"""test_e7_interpretation_exclusivity.py — the §10.12 matrix is disjoint (``08`` §10).

``build_interpretation_matrix`` refuses to pick a row when several match, and that refusal
is correct: two matching rows means the matrix is defective, not that the result is
ambiguous. But the consequence is a **permanent ``matched_row: null`` that looks exactly
like a null result** — the artefact would say "matches none of the pre-registered rows"
forever and nothing would point at the matrix as the cause.

So exclusivity is proved rather than argued. This walks every one of the ``2**n`` truth
assignments over the distinct condition keys the shipped rows use, constructs observed
values just above and just below each key's threshold, and asserts no assignment matches
more than one row.

`08` §10 is explicit that **exhaustiveness must not be asserted**: §10.12 has no residual
category and ``04`` §6.3 permits "matches none of them", which is the honest outcome for
patterns like ``K & !V & !Ku``.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "crosslingual_semantics"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import aggregate_crosslingual as ag  # noqa: E402

SHIPPED_PREREG = REPO / "crosslingual_semantics" / "configs" / "e7_preregistration.yaml"

#: Comfortably larger than any gap between a threshold and its neighbours in the matrix,
#: yet small enough that "just above" and "just below" stay on their intended sides.
EPSILON = 1e-6


def _matrix():
    return ag.load_preregistration(SHIPPED_PREREG)["interpretation_matrix"]


def _rows():
    return list(_matrix().get("rows") or [])


def _thresholds():
    """``key -> threshold``. One threshold per key, which the proof assumes."""
    per_key: dict = {}
    for row in _rows():
        for key, condition in (row.get("conditions") or {}).items():
            per_key.setdefault(key, set()).add(float(condition["value"]))
    return per_key


def _matching_rows(observed):
    matched = []
    for row in _rows():
        if all(ag._condition_holds(observed.get(key), condition)
               for key, condition in (row.get("conditions") or {}).items()):
            matched.append(row["id"])
    return matched


# ── the shape the proof depends on ─────────────────────────────────────────────


def test_the_shipped_matrix_has_rows_to_check():
    assert len(_rows()) >= 6, "the six §10.12 rows must be transcribed"


def test_each_condition_key_uses_exactly_one_threshold():
    """`08` §10: "the disjointness proof assumes one threshold per driver".

    Editing rows individually is how exclusivity silently breaks, so the assumption is
    asserted rather than trusted.
    """
    offenders = {key: sorted(values) for key, values in _thresholds().items()
                 if len(values) > 1}
    assert not offenders, (
        f"these keys carry more than one threshold across the rows: {offenders}. "
        "Replace the value consistently in all six rows (08 §10).")


def test_the_declared_thresholds_match_the_literals_the_rows_use():
    """The ``thresholds:`` block is a declaration; the rows are what is evaluated.

    ``_condition_holds`` reads ``row["conditions"][key]["value"]`` and never looks at
    ``interpretation_matrix.thresholds``, so replacing a value in that block alone changes
    nothing — silently. Decision D7 is exactly the edit that would hit this: every test
    would pass while the matrix kept comparing against the placeholder 0.10.
    """
    report = ag.check_threshold_declaration(_matrix())
    assert report["consistent"] is True, report["reason"]


def test_a_declaration_that_drifts_from_the_rows_is_caught():
    """Both drift directions, since either edit can be the one that was forgotten."""
    block_only = {"thresholds": {"semantic_sensitivity": 0.05, "retrieval_high": 2.0},
                  "rows": [{"id": "r", "conditions": {
                      "k0_effect": {"op": "greater_equal", "value": 0.10},
                      "retrieval_top1_over_chance": {"op": "greater_equal",
                                                     "value": 2.0}}}]}
    report = ag.check_threshold_declaration(block_only)
    assert report["consistent"] is False
    assert "editing the `thresholds:` block alone changes nothing" in report["reason"]
    assert report["declared"] == [0.05, 2.0] and report["used"] == [0.1, 2.0]

    rows_only = {"thresholds": {"semantic_sensitivity": 0.10, "retrieval_high": 2.0},
                 "rows": [{"id": "r", "conditions": {
                     "k0_effect": {"op": "greater_equal", "value": 0.05},
                     "retrieval_top1_over_chance": {"op": "greater_equal",
                                                    "value": 2.0}}}]}
    assert ag.check_threshold_declaration(rows_only)["consistent"] is False


def test_the_consistency_verdict_reaches_the_artefact():
    """Per trap 11: ship the reader as well as the test, or the guard rots."""
    prereg = ag.load_preregistration(SHIPPED_PREREG)
    out = ag.build_interpretation_matrix(prereg, pd.DataFrame())
    assert out["threshold_declaration_consistent"] is True
    assert out["threshold_declaration"]["declared"]


def test_every_operator_is_one_condition_holds_accepts():
    """v2 shipped ``>=`` and ``<``; ``_condition_holds`` accepts word operators only."""
    for row in _rows():
        for key, condition in (row.get("conditions") or {}).items():
            assert ag._condition_holds(0.0, condition) in (True, False), (
                f"{row['id']}.{key} uses operator {condition.get('op')!r}")
            assert isinstance(float(condition["value"]), float)


# ── the exhaustive sweep ───────────────────────────────────────────────────────


def test_no_truth_assignment_matches_more_than_one_row():
    thresholds = {key: next(iter(values)) for key, values in _thresholds().items()}
    keys = sorted(thresholds)
    overlaps = []
    for assignment in itertools.product((False, True), repeat=len(keys)):
        observed = {key: (thresholds[key] + EPSILON if above
                          else thresholds[key] - EPSILON)
                    for key, above in zip(keys, assignment)}
        matched = _matching_rows(observed)
        if len(matched) > 1:
            overlaps.append((dict(zip(keys, assignment)), matched))

    assert not overlaps, (
        f"{len(overlaps)} of {2 ** len(keys)} assignments match more than one row; the "
        f"matrix is not mutually exclusive. First: {overlaps[0]}")


def test_the_sweep_actually_covered_every_key_and_assignment():
    """Guard against a vacuous pass — an empty key set would make the sweep trivial."""
    keys = sorted(_thresholds())
    assert len(keys) >= 6, keys
    assert 2 ** len(keys) >= 64


def test_exhaustiveness_is_not_asserted():
    """`08` §10 and ``04`` §6.3: "matches none of them" is a legitimate outcome.

    Recorded as a test so a later well-meaning edit does not add a catch-all row on the
    grounds that some assignment matched nothing.
    """
    thresholds = {key: next(iter(values)) for key, values in _thresholds().items()}
    keys = sorted(thresholds)
    unmatched = 0
    for assignment in itertools.product((False, True), repeat=len(keys)):
        observed = {key: (thresholds[key] + EPSILON if above
                          else thresholds[key] - EPSILON)
                    for key, above in zip(keys, assignment)}
        if not _matching_rows(observed):
            unmatched += 1
    assert unmatched > 0, (
        "every assignment matching a row would mean the matrix has a residual category; "
        "§10.12 has none")


# ── the aggregator honours it end to end ───────────────────────────────────────


def test_the_aggregator_refuses_to_pick_a_row_when_two_match():
    """The behaviour the exclusivity proof protects, asserted directly."""
    prereg = {"interpretation_matrix": {"status": "ok", "rows": [
        {"id": "a", "conditions": {"k0_effect": {"op": "greater_equal",
                                                 "value": -99.0}}},
        {"id": "b", "conditions": {"k0_effect": {"op": "less", "value": 99.0}}}]},
        "go_no_go": []}
    frame = pd.DataFrame([{"id": "e7_c1", "status": "ok", "absolute_difference": 0.5,
                           "n_languages_same_direction": 3,
                           "n_model_sizes_same_direction": 2}])
    out = ag.build_interpretation_matrix(prereg, frame)
    assert out["n_matched"] == 2 and out["matched_row"] is None
    assert "mutually exclusive" in out["reason"]


def test_every_observed_key_the_rows_use_is_one_observed_pattern_emits():
    """A row conditioning on a key nothing computes is a permanent ``matched_row: null``."""
    emitted = set(ag.observed_pattern(pd.DataFrame(), prereg={}))
    used = set(_thresholds())
    assert used <= emitted, f"rows condition on keys nothing computes: {used - emitted}"


def test_the_matrixs_declared_observed_keys_agree_with_what_it_uses():
    declared = set(_matrix().get("observed_keys") or [])
    used = set(_thresholds())
    assert used <= declared, f"used but undeclared: {used - declared}"
