# -*- coding: utf-8 -*-
"""test_pair_specific_matched_loss.py — ``matched_loss_reference`` (WP12, ``08`` §3).

``build_matched_loss`` matches every condition against one global reference (D0's final
CE). Design §8.7 contrast 4 needs something else: D1's checkpoint nearest **P8M's** CE.
For a D1-vs-D0 contrast the two coincide, which is why the omission went unnoticed; for
``e6a_c4a``/``e6a_c4b`` they do not, and using the global reference would silently compare
at the wrong checkpoint.

``tests/test_matched_loss_selection.py`` covers the global path and must keep passing
unchanged; this file covers only what ``matched_loss_reference`` adds.
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


def _row(condition, seed, step, ce, cos):
    return {"experiment_id": "e6a", "run_id": f"e6a_{condition}_seed{seed}",
            "condition": condition, "seed": seed, "checkpoint_step": step,
            "corpus_id": CORPUS, "status": "ok", "n_items": 100, "n_failed": 0,
            "intervention_registry_version": "v1", "validation_ce": ce,
            "fingerprint_cosine_to_teacher": cos, "n_keys_used": 9}


def _frame():
    """Two reference conditions whose final CEs differ, and a D1 that passes both.

    * ``D0`` runs to step 1000 and finishes at CE **2.00**;
    * ``D2`` stops at step 100 and therefore finishes at CE **3.00**;
    * ``D1`` is at CE 3.00 (cos 0.55) at step 100 and CE 2.00 (cos 0.75) at step 1000.

    So ``matched_loss_reference: D0`` must select D1's step 1000 and
    ``matched_loss_reference: D2`` must select step 100 — two checkpoints with
    deliberately different fingerprint values, so the wrong selection cannot pass by
    coincidence. ``P8M`` is the seedless public reference (``08`` §4): one row, no training
    step, so no checkpoint has to be selected for it at all.
    """
    rows = []
    for seed in (0, 1, 2):
        rows += [_row("D0", seed, 100, 3.00, 0.10),
                 _row("D0", seed, 1000, 2.00, 0.20),
                 _row("D2", seed, 100, 3.00, 0.30),
                 _row("D1", seed, 100, 3.00, 0.55),
                 _row("D1", seed, 1000, 2.00, 0.75)]
    rows.append(_row("P8M", ag.REFERENCE_STEP, ag.REFERENCE_STEP, 2.50, 0.90))
    return pd.DataFrame(rows)


def _entry(reference=None):
    entry = {"id": "c", "metric": "fingerprint_cosine_to_teacher",
             "condition_a": "P8M", "condition_b": "D1", "selection": "matched_loss",
             "corpus_id": CORPUS, "family": "f"}
    if reference:
        entry["matched_loss_reference"] = reference
    return entry


# ── the reference really changes the selected checkpoint ───────────────────────


def test_the_global_reference_selects_d0s_final_ce():
    frame = _frame()
    matched = ag.build_matched_loss(frame, reference_condition="D0", max_gap=0.05)
    steps = ag._matched_steps(matched)
    row = ag.evaluate_contrast(frame, _entry(), steps)
    # D0's final CE is 2.00, so D1 is taken at step 1000 (cos 0.75).
    assert json.loads(row["per_seed_json"])[0]["step_b"] == 1000
    assert row["mean_b"] == pytest.approx(0.75)


def test_a_pair_specific_reference_selects_a_different_checkpoint():
    frame = _frame()
    matched = ag.build_matched_loss(frame, reference_condition="D0", max_gap=0.05)
    steps = ag._matched_steps(matched)
    row = ag.evaluate_contrast(frame, _entry(reference="D2"), steps)
    # D2's final CE is 3.00, so D1 is taken at step 100 (cos 0.55) instead.
    assert json.loads(row["per_seed_json"])[0]["step_b"] == 100
    assert row["mean_b"] == pytest.approx(0.55)
    assert row["matched_loss_reference"] == "D2"


def test_the_two_references_disagree_on_the_same_data():
    """The whole point: same frame, same metric, different reference, different answer."""
    frame = _frame()
    steps = ag._matched_steps(ag.build_matched_loss(frame, reference_condition="D0"))
    default = ag.evaluate_contrast(frame, _entry(), steps)
    specific = ag.evaluate_contrast(frame, _entry(reference="D2"), steps)
    assert default["mean_diff"] != pytest.approx(specific["mean_diff"])


# ── the 0.05-nat refusal is untouched ──────────────────────────────────────────


def test_a_gap_above_the_threshold_is_still_refused_under_a_pair_specific_reference():
    frame = _frame()
    # Move D2's CE far from anything D1 ever reaches.
    frame.loc[frame["condition"] == "D2", "validation_ce"] = 9.99
    steps = ag._matched_steps(ag.build_matched_loss(frame, reference_condition="D0"))
    row = ag.evaluate_contrast(frame, _entry(reference="D2"), steps)

    assert row["status"] == "no_data"
    units = json.loads(row["per_seed_json"])
    assert units and all("matched-loss comparison refused" in u["skipped"] for u in units)


def test_the_realised_gap_is_reported_on_the_contrast_not_only_in_the_table():
    """`08` §3: a reader of one contrast row must see how far from matched it was."""
    frame = _frame()
    frame.loc[frame["condition"] == "D2", "validation_ce"] = 3.02   # 0.02 from step 100
    steps = ag._matched_steps(ag.build_matched_loss(frame, reference_condition="D0"))
    row = ag.evaluate_contrast(frame, _entry(reference="D2"), steps)

    assert row["status"] == "ok"
    assert row["max_realised_gap"] == pytest.approx(0.02)
    assert json.loads(row["per_seed_json"])[0]["realised_gap_b"] == pytest.approx(0.02)


def test_the_default_matched_loss_table_also_reports_its_gap_when_supplied():
    frame = _frame()
    matched = ag.build_matched_loss(frame, reference_condition="D0", max_gap=0.05)
    row = ag.evaluate_contrast(frame, _entry(), ag._matched_steps(matched),
                               matched_table=matched)
    assert row["max_realised_gap"] == pytest.approx(0.0)


# ── an entry without the field is unchanged ────────────────────────────────────


def test_an_entry_without_the_field_records_no_reference():
    frame = _frame()
    steps = ag._matched_steps(ag.build_matched_loss(frame, reference_condition="D0"))
    row = ag.evaluate_contrast(frame, _entry(), steps)
    assert "matched_loss_reference" not in row or row["matched_loss_reference"] is None


def test_the_reference_table_is_computed_once_per_reference():
    """Two contrasts naming the same reference must not rebuild the table twice."""
    frame = _frame()
    steps = ag._matched_steps(ag.build_matched_loss(frame, reference_condition="D0"))
    cache: dict = {}
    ag.evaluate_contrast(frame, _entry(reference="D2"), steps, matched_cache=cache)
    n_after_first = len(cache)
    ag.evaluate_contrast(frame, {**_entry(reference="D2"), "id": "c2"}, steps,
                         matched_cache=cache)
    assert len(cache) == n_after_first == 1
