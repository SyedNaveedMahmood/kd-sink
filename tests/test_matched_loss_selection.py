# -*- coding: utf-8 -*-
"""test_matched_loss_selection.py — E6's matched-loss comparison (WP10).

``03_MODULE_SPEC_e6_transformation.md`` §6.4. Equal-step comparison confounds
"inherits the teacher's mechanism" with "is further along in training", so the design also
requires a matched-loss comparison. Checkpoints are discrete, so the matched checkpoint
almost never lands on the target CE exactly, and the failure mode this guards is specific:
a comparison quietly made across a 0.4-nat gap looks identical in the output table to one
made across 0.01 nats.

Asserted here: the selection rule (closest CE, earliest step on a tie), that the realised
gap is always recorded, and that a gap above 0.05 nats is refused rather than made.
No test asserts which condition wins — that is the experiment (``06`` §5).
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

MAX_GAP = 0.05


def _frame(rows):
    """Minimal metric rows: (condition, seed, step, validation_ce)."""
    return pd.DataFrame([
        {"experiment_id": "e6a", "run_id": f"e6a_{c}_seed{s}", "condition": c,
         "seed": s, "checkpoint_step": step, "corpus_id": "tinystories_x",
         "validation_ce": ce, "status": "ok", "n_items": 100, "n_failed": 0,
         "intervention_registry_version": "v1",
         "fingerprint_cosine_to_teacher": cos}
        for c, s, step, ce, cos in rows])


# ── the selection rule ──────────────────────────────────────────────────────────


def test_selects_the_closest_ce_checkpoint_and_records_the_gap():
    frame = _frame([("D2", 0, 0, 4.00, 0.1), ("D2", 0, 250, 3.20, 0.2),
                    ("D2", 0, 500, 2.55, 0.3), ("D2", 0, 1000, 2.10, 0.4)])
    selection = ag.matched_loss_checkpoint(frame, target_ce=2.52, max_gap=MAX_GAP)
    assert selection["step"] == 500
    assert selection["realised_gap"] == pytest.approx(0.03)
    assert selection["comparable"] is True
    assert selection["target_ce"] == pytest.approx(2.52)


def test_ties_break_to_the_earliest_step():
    """`03` §6.4 says *earliest*; two equally-close checkpoints must not pick the later."""
    frame = _frame([("D2", 0, 250, 2.50, 0.1), ("D2", 0, 500, 2.50, 0.2),
                    ("D2", 0, 1000, 2.90, 0.3)])
    assert ag.matched_loss_checkpoint(frame, 2.50, max_gap=MAX_GAP)["step"] == 250


def test_a_gap_above_the_threshold_is_refused_not_silently_compared():
    frame = _frame([("D2", 0, 250, 3.00, 0.1), ("D2", 0, 500, 2.90, 0.2)])
    selection = ag.matched_loss_checkpoint(frame, target_ce=2.00, max_gap=MAX_GAP)
    assert selection["comparable"] is False
    assert selection["realised_gap"] == pytest.approx(0.90)
    assert "refused" in selection["reason"]
    # the gap is still reported — refusing is not the same as hiding the number
    assert selection["step"] == 500


def test_the_threshold_boundary_is_inclusive_at_0_05():
    frame = _frame([("D2", 0, 500, 2.55, 0.1)])
    assert ag.matched_loss_checkpoint(frame, 2.50, max_gap=MAX_GAP)["comparable"] is True
    frame = _frame([("D2", 0, 500, 2.56, 0.1)])
    assert ag.matched_loss_checkpoint(frame, 2.50, max_gap=MAX_GAP)["comparable"] is False


def test_no_validation_ce_is_reported_not_guessed():
    frame = _frame([("D2", 0, 500, None, 0.1)])
    selection = ag.matched_loss_checkpoint(frame, 2.5, max_gap=MAX_GAP)
    assert selection["step"] is None
    assert selection["comparable"] is False
    assert "validation_ce" in selection["reason"]


# ── the table ───────────────────────────────────────────────────────────────────


def test_matched_loss_table_targets_the_reference_condition_final_ce():
    frame = _frame([
        ("D0", 0, 0, 4.00, 0.1), ("D0", 0, 1000, 2.60, 0.2),      # D0 final CE = 2.60
        ("D1", 0, 0, 4.00, 0.1), ("D1", 0, 500, 2.62, 0.5),
        ("D1", 0, 1000, 2.20, 0.6),
    ])
    table = ag.build_matched_loss(frame, reference_condition="D0", max_gap=MAX_GAP)
    d1 = table[table["condition"] == "D1"].iloc[0]
    assert d1["reference_final_ce"] == pytest.approx(2.60)
    assert d1["matched_step"] == 500                  # 2.62 is closest to 2.60
    assert d1["equal_step"] == 1000                   # equal-step is still reported
    assert d1["comparable"]
    assert d1["realised_gap"] == pytest.approx(0.02)


def test_matched_loss_contrast_skips_seeds_whose_comparison_was_refused():
    """A refused seed must drop out of the contrast, not contribute a mismatched pair."""
    frame = _frame([
        # seed 0: D1 has a checkpoint near D0's final CE -> comparable
        ("D0", 0, 0, 4.0, 0.10), ("D0", 0, 1000, 2.60, 0.20),
        ("D1", 0, 500, 2.62, 0.50), ("D1", 0, 1000, 2.20, 0.60),
        # seed 1: D1 never gets near 2.60 -> refused
        ("D0", 1, 0, 4.0, 0.10), ("D0", 1, 1000, 2.60, 0.20),
        ("D1", 1, 500, 3.90, 0.50), ("D1", 1, 1000, 3.80, 0.60),
    ])
    table = ag.build_matched_loss(frame, reference_condition="D0", max_gap=MAX_GAP)
    steps = ag._matched_steps(table)
    assert steps[(0, "D1")] == 500
    assert steps[(1, "D1")] is None

    entry = {"id": "x", "metric": "fingerprint_cosine_to_teacher", "condition_a": "D1",
             "condition_b": "D0", "selection": "matched_loss", "family": "f"}
    contrast = ag.evaluate_contrast(frame, entry, steps)
    assert contrast["n_seeds"] == 1
    per_seed = json.loads(contrast["per_seed_json"])
    assert [row.get("seed") for row in per_seed] == [0, 1]
    assert "skipped" not in per_seed[0] and "refused" in per_seed[1]["skipped"]


def test_equal_step_contrast_requires_a_common_final_step():
    """Different final steps are not comparable at 'equal step'; the seed is skipped."""
    frame = _frame([("D2", 0, 1000, 2.0, 0.9), ("D0", 0, 500, 2.0, 0.4)])
    entry = {"id": "x", "metric": "fingerprint_cosine_to_teacher", "condition_a": "D2",
             "condition_b": "D0", "selection": "equal_step", "family": "f"}
    contrast = ag.evaluate_contrast(frame, entry, {})
    assert contrast["n_seeds"] == 0
    assert contrast["status"] == "no_data"


def test_paired_contrast_carries_the_min_attainable_p_note():
    """Three seeds cannot reach p < 0.25; the table must say so (02 §4.6)."""
    frame = _frame([
        ("D2", s, 1000, 2.0, 0.80 + 0.01 * s) for s in (0, 1, 2)
    ] + [("D0", s, 1000, 2.0, 0.40 + 0.01 * s) for s in (0, 1, 2)])
    entry = {"id": "x", "metric": "fingerprint_cosine_to_teacher", "condition_a": "D2",
             "condition_b": "D0", "selection": "equal_step", "family": "f"}
    contrast = ag.evaluate_contrast(frame, entry, {})
    assert contrast["n_seeds"] == 3
    assert contrast["mean_diff"] == pytest.approx(0.40)
    assert contrast["min_attainable_p"] == pytest.approx(0.25)
    assert contrast["note"] == "descriptive, not inferential"
    assert contrast["ci_uncertainty_kind"] == "sampling_over_seeds"
