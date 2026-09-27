# -*- coding: utf-8 -*-
"""test_public_reference_rows.py — the seedless P8M condition (WP12, ``08`` §4).

Design §8.7 contrast 4 compares the independently trained public 8M against D1 and D2. The
8M has **one** measurement and **no** training seed, while the students have three seeds
each, so it cannot be an arm of a paired-seed contrast without misrepresenting where the
uncertainty comes from.

What these tests hold in place:

* the row's shape — ``condition``, ``checkpoint_step == -1``, the seed sentinel — because
  ``is_reference_condition`` keys off the step, not the condition name, so a rename cannot
  turn a fixed reference into a third replicate;
* the instrument is shared, asserted through ``manifest_sha256``: the reference must be
  measured on the *same corpus objects* as the students, or contrast 4 measures the
  instrument rather than the model;
* the contrast against it is labelled ``reference_vs_seeds`` /
  ``descriptive_reference_contrast`` and never reuses the paired-seed permutation label.

The model is stubbed, as in ``test_evaluate_transformation.py``; the real-weights run is a
separate GPU step recorded in ``NEXT_STEPS.md``.
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
import evaluate_transformation as ev  # noqa: E402
import fingerprint_runner as fr  # noqa: E402
from test_evaluate_transformation import _record, _run_dir, _tokenizer  # noqa: E402

CORPUS = "tinystories_validation_sink_300"


@pytest.fixture
def stub_reference(monkeypatch):
    """Stub every model seam and count fingerprint calls, keeping the corpora real."""
    calls = []

    def _compute(handle, corpus, **kwargs):
        calls.append({"corpus_id": corpus.corpus_id,
                      "manifest_sha256": corpus.manifest_sha256,
                      "condition": kwargs.get("condition"),
                      "seed": kwargs.get("seed")})
        return _record(corpus)

    monkeypatch.setattr(ev.fr, "compute_fingerprint", _compute)
    monkeypatch.setattr(ev.fr, "load_handle",
                        lambda *a, **k: type("H", (), {
                            "nn_engine": type("E", (), {"num_layers": 8})()})())
    monkeypatch.setattr(ev.fr, "available_interventions",
                        lambda handle: ["int_a", "int_c", "int_g"])
    monkeypatch.setattr(ev.fr, "mutual_interventions",
                        lambda *handles: ["int_a", "int_c", "int_g"])
    monkeypatch.setattr(ev, "arch_of_model_id", lambda *a, **k: "neo")
    return calls


# ── the row's shape ────────────────────────────────────────────────────────────


def test_the_reference_row_carries_the_sentinel_step_and_seed(tmp_path, stub_reference):
    tokenizer_dir = tmp_path / "tok"
    _tokenizer(tokenizer_dir)
    summary = ev.evaluate_public_reference(
        "roneneldan/TinyStories-8M", tmp_path / "reference", teacher=None,
        tokenizer_source=str(tokenizer_dir), smoke=True, progress=False)

    frame = pd.read_csv(summary["metrics_csv"])
    assert (frame["condition"] == "P8M").all()
    assert (frame["checkpoint_step"] == ev.REFERENCE_STEP).all()
    assert (frame["seed"] == ev.REFERENCE_SEED).all()
    assert (frame["status"] == "ok").all()
    assert frame["baseline_sink"].notna().all()
    # -1 can never collide with a real training step, which starts at 0 (03 §1.6).
    assert ev.REFERENCE_STEP < 0 and ev.REFERENCE_SEED < 0


def test_the_condition_id_is_configurable_but_the_step_is_not(tmp_path, stub_reference):
    tokenizer_dir = tmp_path / "tok"
    _tokenizer(tokenizer_dir)
    summary = ev.evaluate_public_reference(
        "some/other-8M", tmp_path / "reference", teacher=None,
        tokenizer_source=str(tokenizer_dir), condition="PUBLIC", smoke=True,
        progress=False)
    frame = pd.read_csv(summary["metrics_csv"])
    assert (frame["condition"] == "PUBLIC").all()
    assert (frame["checkpoint_step"] == ev.REFERENCE_STEP).all()


def test_the_summary_records_the_instrument_it_used(tmp_path, stub_reference):
    tokenizer_dir = tmp_path / "tok"
    _tokenizer(tokenizer_dir)
    summary = ev.evaluate_public_reference(
        "roneneldan/TinyStories-8M", tmp_path / "reference", teacher=None,
        tokenizer_source=str(tokenizer_dir), revision="reference-pin",
        teacher_revision="teacher-pin", smoke=True, progress=False)

    assert summary["public_reference"] == "roneneldan/TinyStories-8M"
    assert summary["reference_revision"] == "reference-pin"
    assert summary["teacher_revision"] == "teacher-pin"
    assert "tokenizer_revision" in summary
    assert summary["intervention_registry_version"] == fr.INTERVENTION_REGISTRY_VERSION
    assert summary["band"] == [2, 8]
    assert summary["manifest_sha256"]
    written = json.loads((tmp_path / "reference" / "reference_summary.json").read_text(
        encoding="utf-8"))
    assert written["teacher_revision"] == "teacher-pin"


# ── the shared instrument ──────────────────────────────────────────────────────


def test_the_reference_reuses_the_runs_corpora_rather_than_rebuilding_them(
        tmp_path, stub_reference):
    """`08` §4: "it must reuse the same corpora objects, not rebuild them"."""
    run_dir = _run_dir(tmp_path)
    ev.evaluate_run(run_dir, smoke=True, progress=False,
                    public_reference="roneneldan/TinyStories-8M",
                    reference_out=tmp_path / "reference")

    student = [c for c in stub_reference if c["condition"] == "D2"]
    reference = [c for c in stub_reference if c["condition"] == "P8M"]
    assert student and reference
    # Identical digests are the guarantee; identical corpus ids alone would not be.
    assert ({c["manifest_sha256"] for c in reference}
            == {c["manifest_sha256"] for c in student})
    assert ({c["corpus_id"] for c in reference} == {c["corpus_id"] for c in student})


def test_the_reference_lands_in_its_own_directory(tmp_path, stub_reference):
    """A P8M row must never sit inside a student run's checkpoint_metrics.csv."""
    run_dir = _run_dir(tmp_path)
    summary = ev.evaluate_run(run_dir, smoke=True, progress=False,
                              public_reference="roneneldan/TinyStories-8M",
                              reference_out=tmp_path / "reference")

    student_rows = pd.read_csv(summary["metrics_csv"])
    assert "P8M" not in set(student_rows["condition"])
    reference_rows = pd.read_csv(summary["public_reference"]["metrics_csv"])
    assert set(reference_rows["condition"]) == {"P8M"}
    # ...but the aggregator's rglob still finds both.
    assert len(ag.load_all_metrics(tmp_path)) == len(student_rows) + len(reference_rows)


def test_a_run_without_the_flag_produces_no_reference(tmp_path, stub_reference):
    summary = ev.evaluate_run(_run_dir(tmp_path), smoke=True, progress=False)
    assert summary["public_reference"] is None
    assert not any(c["condition"] == "P8M" for c in stub_reference)


# ── how the aggregator pairs it ────────────────────────────────────────────────


def _row(condition, seed, step, cos):
    return {"experiment_id": "e6a", "run_id": f"e6a_{condition}_seed{seed}",
            "condition": condition, "seed": seed, "checkpoint_step": step,
            "corpus_id": CORPUS, "status": "ok", "n_items": 100, "n_failed": 0,
            "intervention_registry_version": "v1", "validation_ce": 2.5,
            "fingerprint_cosine_to_teacher": cos, "n_keys_used": 9}


def _mixed_frame():
    rows = [_row("D1", seed, 1000, 0.50 + 0.01 * seed) for seed in (0, 1, 2)]
    rows.append(_row("P8M", ev.REFERENCE_SEED, ev.REFERENCE_STEP, 0.30))
    return pd.DataFrame(rows)


def test_a_reference_condition_is_detected_by_its_step_not_its_name():
    frame = _mixed_frame()
    assert ag.is_reference_condition(frame, "P8M") is True
    assert ag.is_reference_condition(frame, "D1") is False
    # Rename it and the classification is unchanged — that is the point.
    renamed = frame.replace({"condition": {"P8M": "whatever"}})
    assert ag.is_reference_condition(renamed, "whatever") is True


def test_the_reference_seed_is_not_treated_as_a_fourth_replicate():
    assert ag.student_seeds(_mixed_frame()) == [0, 1, 2]


def test_the_contrast_is_labelled_a_descriptive_reference_contrast():
    entry = {"id": "e6a_c4a", "metric": "fingerprint_cosine_to_teacher",
             "condition_a": "P8M", "condition_b": "D1", "corpus_id": CORPUS,
             "family": "e6a_primary"}
    row = ag.evaluate_contrast(_mixed_frame(), entry, {})

    assert row["status"] == "ok"
    assert row["pairing"] == "reference_vs_seeds"
    assert row["test"] == "descriptive_reference_contrast", (
        "reusing the paired-seed permutation label would claim replication the public "
        "checkpoint cannot supply (08 §4)")
    assert row["n_seeds"] == 3
    assert "student seeds only" in row["note"]
    # One fixed value minus each student seed.
    assert row["mean_a"] == pytest.approx(0.30)
    assert row["mean_diff"] == pytest.approx(0.30 - 0.51)


def test_every_seed_is_differenced_against_the_same_reference_value():
    entry = {"id": "c", "metric": "fingerprint_cosine_to_teacher",
             "condition_a": "P8M", "condition_b": "D1", "corpus_id": CORPUS}
    units = json.loads(ag.evaluate_contrast(_mixed_frame(), entry, {})["per_seed_json"])
    assert [unit["value_a"] for unit in units] == [0.30, 0.30, 0.30]
    assert [unit["seed"] for unit in units] == [0, 1, 2]
