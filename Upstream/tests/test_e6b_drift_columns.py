# -*- coding: utf-8 -*-
"""test_e6b_drift_columns.py — the E6B evaluator link (WP6).

``05`` §2 has carried ``task_accuracy``, ``task_nll``, ``ece_10bin`` and the three
``*_drift_from_base`` columns since WP10, and nothing populated any of them. The aggregator
reads them straight off the frame, so ``e6b_drift.csv`` would have stayed header-only
forever and design §18's E6B criteria 2 and 3 could never evaluate — computable on paper,
permanently ``no_data`` in practice, which is trap 12's shape exactly.

Two seams, asserted separately:

* **task metrics are read, not recomputed.** The trainer measured accuracy, NLL and ECE
  with the model's own scorer; a second implementation in the evaluator would be a second
  source of truth for the numbers criterion ``e6b_1`` compares.
* **drift dispatches to** :mod:`inheritance_metrics`. And with no base record the columns
  stay ``None`` — never ``0.0``, because zero drift is a *measurement* and a run that
  simply had no comparand must not be readable as one that did not move.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import evaluate_transformation as ev  # noqa: E402
import inheritance_metrics as im  # noqa: E402
from test_evaluate_transformation import _record  # noqa: E402


class _Corpus:
    corpus_id = "sst2_validation_prompted"
    manifest_sha256 = "d" * 64


KEYS = ["int_a", "int_c", "int_g"]


def _run(condition="F1"):
    return ev.RunInfo(run_dir=Path("."), run_id=f"e6b_{condition}_seed0",
                      experiment_id="e6b", condition=condition, seed=0, teacher=None,
                      teacher_revision=None, tokenizer_name="t", smoke=True,
                      raw={"base_model": "distilbert/distilgpt2"})


def _build(record, base_record, task_metrics=None):
    return ev.build_row(_run(), 100, record, None, KEYS, validation_ce=None,
                        validation_ce_n_blocks=None, delta_ce_ci=None,
                        delta_ce_n_blocks=None, checkpoint_sha256=None,
                        base_record=base_record, task_metrics=task_metrics)


# ── drift is computed, and by the frozen metrics module ────────────────────────


def test_drift_columns_are_populated_when_a_base_record_is_supplied():
    record = _record(_Corpus(), value=0.5)
    base = _record(_Corpus(), value=0.2)
    row = _build(record, base)

    for column in ("fingerprint_drift_from_base", "topology_drift_from_base",
                   "carrier_drift_from_base"):
        assert row[column] is not None, column
        assert np.isfinite(row[column]), column


def test_drift_dispatches_to_inheritance_metrics_rather_than_reimplementing():
    """`02` §4 owns these distances; the evaluator may only call them."""
    record = _record(_Corpus(), value=0.5)
    base = _record(_Corpus(), value=0.2)
    row = _build(record, base)

    assert row["fingerprint_drift_from_base"] == pytest.approx(
        im.fingerprint_drift(record, base, KEYS))
    assert row["topology_drift_from_base"] == pytest.approx(
        im.topology_drift(record.depth_profile_16, base.depth_profile_16))
    assert row["carrier_drift_from_base"] == pytest.approx(
        im.carrier_drift(np.asarray(record.per_head_sink),
                         np.asarray(base.per_head_sink)))


def test_a_model_identical_to_its_base_shows_zero_drift():
    """The measurement that *is* zero, so the None-vs-0.0 distinction below has teeth."""
    record = _record(_Corpus(), value=0.4)
    row = _build(record, _record(_Corpus(), value=0.4))
    assert row["fingerprint_drift_from_base"] == pytest.approx(0.0, abs=1e-9)
    assert row["topology_drift_from_base"] == pytest.approx(0.0, abs=1e-9)


# ── absent base is empty, never zero ───────────────────────────────────────────


def test_no_base_record_leaves_the_columns_empty_not_zero():
    row = _build(_record(_Corpus(), value=0.5), None)
    for column in ("fingerprint_drift_from_base", "topology_drift_from_base",
                   "carrier_drift_from_base"):
        assert row[column] is None, (
            f"{column} was written as {row[column]!r}; a missing comparand must not be "
            "readable as 'measured, and it did not move'")


def test_unequal_head_counts_are_recorded_as_a_warning_not_coerced():
    """`02` §4.3 — carrier overlap needs equal head counts and refuses otherwise."""
    record = _record(_Corpus(), num_heads=4, value=0.5)
    base = _record(_Corpus(), num_heads=8, value=0.2)
    row = _build(record, base)

    assert row["carrier_drift_from_base"] is None
    assert "carrier_drift" in row["warning"]
    # The other two do not depend on head count and must still be measured.
    assert row["fingerprint_drift_from_base"] is not None
    assert row["topology_drift_from_base"] is not None


# ── task metrics are read from the trainer's eval_log ──────────────────────────


def test_task_metrics_are_carried_through_from_the_eval_log():
    metrics = {"step": 100, "task_accuracy": 0.91, "task_nll": 0.27, "ece_10bin": 0.043}
    row = _build(_record(_Corpus(), value=0.5), None, task_metrics=metrics)
    assert row["task_accuracy"] == pytest.approx(0.91)
    assert row["task_nll"] == pytest.approx(0.27)
    assert row["ece_10bin"] == pytest.approx(0.043)


def test_absent_task_metrics_stay_empty():
    row = _build(_record(_Corpus(), value=0.5), None, task_metrics={"step": 100})
    assert row["task_accuracy"] is None
    assert row["task_nll"] is None
    assert row["ece_10bin"] is None


def test_a_non_finite_task_metric_becomes_an_empty_cell():
    """`_finite` keeps the text 'nan' out of the CSV — an empty cell reads as absent."""
    row = _build(_record(_Corpus(), value=0.5), None,
                 task_metrics={"task_accuracy": float("nan"), "task_nll": 0.3,
                               "ece_10bin": float("inf")})
    assert row["task_accuracy"] is None and row["ece_10bin"] is None
    assert row["task_nll"] == pytest.approx(0.3)


def test_an_e6a_row_is_unaffected_by_the_e6b_link():
    """Additive: no base, no task metrics, byte-identical to the pre-WP6 row."""
    record = _record(_Corpus(), value=0.5)
    with_link = _build(record, None, task_metrics={})
    plain = ev.build_row(_run("D2"), 100, record, None, KEYS, validation_ce=None,
                         validation_ce_n_blocks=None, delta_ce_ci=None,
                         delta_ce_n_blocks=None, checkpoint_sha256=None)
    ignore = {"run_id", "condition", "measured_utc"}
    assert {k: v for k, v in with_link.items() if k not in ignore} == \
           {k: v for k, v in plain.items() if k not in ignore}


# ── the merged checkpoint is what gets fingerprinted ───────────────────────────


def test_fingerprint_dir_prefers_the_merged_copy(tmp_path):
    """A LoRA step holds ``adapter/`` and ``merged/``; only the merged one is loadable."""
    step = tmp_path / "step_100"
    (step / "adapter").mkdir(parents=True)
    (step / "merged").mkdir(parents=True)
    (step / "merged" / "config.json").write_text(
        '{"architectures": ["GPT2LMHeadModel"]}', encoding="utf-8")

    assert ev.fingerprint_dir(step) == step / "merged"
    assert ev.arch_of_checkpoint(step) == "gpt2"


def test_fingerprint_dir_falls_back_to_the_step_root_for_full_fine_tuning(tmp_path):
    step = tmp_path / "step_100"
    step.mkdir(parents=True)
    (step / "config.json").write_text('{"architectures": ["GPT2LMHeadModel"]}',
                                      encoding="utf-8")
    assert ev.fingerprint_dir(step) == step
    assert ev.arch_of_checkpoint(step) == "gpt2"


def test_the_resolution_is_by_weights_present_not_by_condition_name(tmp_path):
    """Renaming a condition must not send the evaluator at the PEFT adapter."""
    step = tmp_path / "step_100"
    (step / "adapter").mkdir(parents=True)
    (step / "merged").mkdir(parents=True)
    (step / "merged" / "config.json").write_text(
        '{"architectures": ["GPT2LMHeadModel"]}', encoding="utf-8")
    assert ev.fingerprint_dir(step).name == "merged"
