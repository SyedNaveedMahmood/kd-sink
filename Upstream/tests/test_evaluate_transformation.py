# -*- coding: utf-8 -*-
"""test_evaluate_transformation.py — the E6 evaluator's bookkeeping (WP10).

``06_TEST_PLAN.md`` assigns no tests to WP10 — its inventory table has no WP10 row — so
these follow §5's rule instead: assert invariants, determinism and refusals, never that a
scientific quantity takes a particular value. Nothing here asserts what a sink *is*.

What would go wrong without each test:

* a resumed run that recomputes finished units burns the 1.5–3 GPU-day evaluation budget
  twice, and one that recomputes *without* replacing the old row double-counts that
  checkpoint in every downstream mean;
* a resumed run that skips a unit whose instrument changed silently mixes two registries
  into one trajectory;
* a checkpoint that fails and raises loses every later checkpoint, and one that fails
  silently is indistinguishable from one that measured zero (``05`` §7.2).

The model is stubbed: this file tests the evaluator's control flow, not NNsight. The real
end-to-end lives in ``tests/nnsight_e6_eval_smoke.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import evaluate_transformation as ev  # noqa: E402
import fingerprint_runner as fr  # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────────


def _tokenizer(path: Path, vocab_size: int = 64):
    """The repo's standard offline word-level tokenizer (nnsight_smoke_utils pattern)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}
    vocab.update({f"t{i}": i + 3 for i in range(vocab_size - 3)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                        pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.save_pretrained(str(path))
    return tokenizer


def _run_dir(root: Path, *, steps=(0, 5), condition="D2", seed=0) -> Path:
    """A run directory shaped exactly like train_distillation.py writes one."""
    run_dir = root / "e6a" / condition / f"seed{seed}"
    (run_dir / "_smoke").mkdir(parents=True, exist_ok=True)
    _tokenizer(run_dir / "_smoke" / "tokenizer")
    ev.prov.write_json(run_dir / "run_config.json", {
        "experiment_id": "e6a", "condition_id": condition,
        "run_id": f"e6a_{condition}_seed{seed}", "training_seed": seed,
        "tokenizer_name": "smoke_tokenizer", "smoke": True, "teacher": None})
    for step in steps:
        ckpt = run_dir / "checkpoints" / f"step_{step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "config.json").write_text(
            json.dumps({"architectures": ["GPTNeoForCausalLM"], "num_layers": 8}),
            encoding="utf-8")
        (ckpt / "checkpoint_sha256.txt").write_text(f"sha_{step}\n", encoding="utf-8")
    rows = [{"step": step, "validation_ce": 4.0 - 0.1 * step} for step in steps]
    (run_dir / "eval_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return run_dir


def _record(corpus, *, num_layers=8, num_heads=4, value=0.5) -> fr.FingerprintRecord:
    keys = ["int_a", "int_c", "int_g"]
    return fr.FingerprintRecord(
        run_id="stub", experiment_id="e6a", condition="D2", seed=0,
        model_name="stub", model_revision=None, checkpoint_step=0, arch="neo",
        num_layers=num_layers, num_heads=num_heads, hidden_size=32, param_count=1,
        corpus_id=corpus.corpus_id, manifest_sha256=corpus.manifest_sha256,
        band=(2, 8), band_depth=(0.2857, 1.0), band_version="depth_band_v1",
        engine="nnsight", dtype="float32", device="cpu",
        intervention_registry_version=fr.INTERVENTION_REGISTRY_VERSION,
        available_interventions=keys,
        baseline_sink=value, per_layer_sink=[value] * num_layers,
        per_head_sink=[[value] * num_heads for _ in range(num_layers)],
        depth_profile_16=[value] * 16, frac_cells_above_0_2=0.5,
        carrier_concentration=0.3,
        top_carrier_heads=[(0.0, 0, value)],
        fingerprint={"int_a": 1.0, "int_c": value, "int_g": 0.8},
        raw_sink_by_intervention={"int_a": 1.0, "int_c": value, "int_g": 0.8},
        delta_ce={"int_c": 0.1, "int_g": 0.2}, baseline_ce=3.0,
        massive_coords=[1, 2], massive_coord_scope="per_model",
        n_items=4, n_failed=0, failures=[], wallclock_s=0.1,
        provenance={"delta_ce_per_item": {"int_c": [0.1, 0.2, 0.05, 0.15]}})


class _Counter:
    """Stubs ``compute_fingerprint`` and counts how many units were actually computed."""

    def __init__(self):
        self.calls = 0

    def __call__(self, handle, corpus, **kwargs):
        self.calls += 1
        return _record(corpus)


@pytest.fixture
def stubbed(monkeypatch):
    """Replace every model-loading seam; the evaluator's control flow is what is tested."""
    counter = _Counter()
    monkeypatch.setattr(ev.fr, "compute_fingerprint", counter)
    monkeypatch.setattr(ev.fr, "load_handle",
                        lambda *a, **k: type("H", (), {
                            "nn_engine": type("E", (), {"num_layers": 8})()})())
    monkeypatch.setattr(ev.fr, "available_interventions",
                        lambda handle: ["int_a", "int_c", "int_g"])
    monkeypatch.setattr(ev.fr, "mutual_interventions",
                        lambda *handles: ["int_a", "int_c", "int_g"])
    return counter


# ── resumability ────────────────────────────────────────────────────────────────


def test_resume_skips_completed_units_and_computes_nothing(tmp_path, stubbed):
    run_dir = _run_dir(tmp_path)
    first = ev.evaluate_run(run_dir, smoke=True, progress=False)
    assert first["n_written"] > 0
    computed = stubbed.calls

    second = ev.evaluate_run(run_dir, smoke=True, progress=False)
    assert second["n_written"] == 0
    assert second["n_skipped"] == first["n_written"]
    # The point of resumability: no forward pass is repeated, not merely no row rewritten.
    assert stubbed.calls == computed


def test_force_recomputes_and_replaces_rows_in_place(tmp_path, stubbed):
    import pandas as pd

    run_dir = _run_dir(tmp_path)
    ev.evaluate_run(run_dir, smoke=True, progress=False)
    before = pd.read_csv(run_dir / "checkpoint_metrics.csv")
    computed = stubbed.calls

    ev.evaluate_run(run_dir, smoke=True, resume=False, progress=False)
    after = pd.read_csv(run_dir / "checkpoint_metrics.csv")

    assert stubbed.calls > computed                      # it really recomputed
    assert len(after) == len(before)                     # upsert, not append
    keys = ["run_id", "checkpoint_step", "corpus_id"]
    assert not after.duplicated(subset=keys).any()


def test_a_changed_instrument_invalidates_the_unit(tmp_path, stubbed, monkeypatch):
    """A unit measured with a different intervention set must not be skipped."""
    run_dir = _run_dir(tmp_path)
    ev.evaluate_run(run_dir, smoke=True, progress=False)
    computed = stubbed.calls

    monkeypatch.setattr(ev.fr, "available_interventions",
                        lambda handle: ["int_a", "int_c"])   # registry changed
    summary = ev.evaluate_run(run_dir, smoke=True, progress=False)
    assert summary["n_skipped"] == 0
    assert stubbed.calls > computed


def test_unit_is_complete_requires_every_field_to_match():
    expected = {"interventions": ["int_a"], "manifest_sha256": "abc",
                "registry_version": "v1", "band": [2, 8], "checkpoint_sha256": "s"}
    ledger = {"k": {"unit": "k", "status": "ok", **expected}}
    assert ev.unit_is_complete(ledger, "k", expected)
    for field in expected:
        drifted = dict(expected)
        drifted[field] = "different"
        assert not ev.unit_is_complete(ledger, "k", drifted), field
    ledger["k"]["status"] = "fingerprint_failed"
    assert not ev.unit_is_complete(ledger, "k", expected)


# ── failures are data ───────────────────────────────────────────────────────────


def test_a_failed_checkpoint_writes_rows_and_keeps_going(tmp_path, stubbed, monkeypatch):
    import pandas as pd

    run_dir = _run_dir(tmp_path, steps=(0, 5))
    real_arch = ev.arch_of_checkpoint

    def _fail_on_step_0(ckpt):
        if Path(ckpt).name == "step_0":
            raise RuntimeError("simulated corrupt checkpoint")
        return real_arch(ckpt)

    monkeypatch.setattr(ev, "arch_of_checkpoint", _fail_on_step_0)
    summary = ev.evaluate_run(run_dir, smoke=True, progress=False)

    frame = pd.read_csv(run_dir / "checkpoint_metrics.csv")
    failed = frame[frame["checkpoint_step"] == 0]
    assert len(failed) and (failed["status"] == "checkpoint_load_failed").all()
    assert failed["baseline_sink"].isna().all()        # sentinel, never a number
    assert failed["warning"].str.contains("simulated corrupt").all()
    # and the later checkpoint still measured
    assert (frame[frame["checkpoint_step"] == 5]["status"] == "ok").all()
    assert summary["n_failed"] and summary["n_written"]


def test_blank_row_fills_provenance_and_blanks_every_metric():
    run = ev.RunInfo(run_dir=Path("."), run_id="r", experiment_id="e6a", condition="D2",
                     seed=0, teacher=None, teacher_revision=None, tokenizer_name=None,
                     smoke=True)
    row = ev.blank_row(run, 100, "c", status="oom", warning="w", engine="nnsight",
                       dtype="float32", checkpoint_sha256="s")
    assert row["status"] == "oom" and row["git_sha"]
    assert row["checkpoint_step"] == 100 and row["corpus_id"] == "c"
    for column in ("baseline_sink", "fingerprint_json", "fingerprint_cosine_to_teacher",
                   "n_items", "validation_ce"):
        assert row[column] is None, column


# ── comparison guards (05 §7.1) ─────────────────────────────────────────────────


def test_manifest_mismatch_raises(tmp_path):
    import corpus_providers as cp

    tok = _tokenizer(tmp_path / "tok")
    a = cp.synthetic_corpus(tok, "random_uniform", 2, 0, cut_length=8)
    b = cp.synthetic_corpus(tok, "random_zipf", 2, 1, cut_length=8)
    with pytest.raises(ValueError, match="manifest_sha256 mismatch"):
        ev.guard_comparison(_record(a), _record(b), allow_band_mismatch=False)


def test_registry_mismatch_raises(tmp_path):
    import corpus_providers as cp

    tok = _tokenizer(tmp_path / "tok")
    corpus = cp.synthetic_corpus(tok, "random_uniform", 2, 0, cut_length=8)
    student, teacher = _record(corpus), _record(corpus)
    teacher.intervention_registry_version = "some-other-registry-v9"
    with pytest.raises(ValueError, match="intervention_registry_version mismatch"):
        ev.guard_comparison(student, teacher, allow_band_mismatch=False)


def test_band_mismatch_raises_unless_allowed(tmp_path):
    import corpus_providers as cp

    tok = _tokenizer(tmp_path / "tok")
    corpus = cp.synthetic_corpus(tok, "random_uniform", 2, 0, cut_length=8)
    student = _record(corpus, num_layers=8)
    teacher = _record(corpus, num_layers=2)      # depth interval far from the student's
    with pytest.raises(ValueError, match="depth bands disagree"):
        ev.guard_comparison(student, teacher, allow_band_mismatch=False)
    ev.guard_comparison(student, teacher, allow_band_mismatch=True)   # recorded, allowed


def test_the_real_4l_teacher_8l_student_pair_is_within_tolerance(tmp_path):
    """CLAUDE.md trap 1: the pair the normalised band exists for must pass the guard."""
    import corpus_providers as cp

    tok = _tokenizer(tmp_path / "tok")
    corpus = cp.synthetic_corpus(tok, "random_uniform", 2, 0, cut_length=8)
    ev.guard_comparison(_record(corpus, num_layers=8), _record(corpus, num_layers=4),
                        allow_band_mismatch=False)


# ── schema and derived values ───────────────────────────────────────────────────


def test_row_carries_every_schema_column_and_the_mutual_key_count(tmp_path):
    import corpus_providers as cp

    tok = _tokenizer(tmp_path / "tok")
    corpus = cp.synthetic_corpus(tok, "random_uniform", 2, 0, cut_length=8)
    run = ev.RunInfo(run_dir=tmp_path, run_id="r", experiment_id="e6a", condition="D2",
                     seed=0, teacher="t", teacher_revision=None, tokenizer_name=None,
                     smoke=True)
    keys = ["int_a", "int_c", "int_g"]
    row = ev.build_row(run, 5, _record(corpus), _record(corpus), keys,
                       validation_ce=3.5, validation_ce_n_blocks=300,
                       delta_ce_ci=None, delta_ce_n_blocks=300,
                       checkpoint_sha256="s")
    assert set(row) == set(ev.METRIC_COLUMNS)
    assert row["n_keys_used"] == len(keys)
    assert row["status"] == "ok"
    # identical records: cosine 1, Wasserstein 0, Jaccard 1 (mirrors WP4's identity test)
    assert row["fingerprint_cosine_to_teacher"] == pytest.approx(1.0)
    assert row["topology_wasserstein_to_teacher"] == pytest.approx(0.0, abs=1e-12)
    assert row["carrier_jaccard_to_teacher"] == pytest.approx(1.0)


def test_the_coarse_and_surgical_key_sets_partition_the_mutual_keys(tmp_path):
    """Exploratory decomposition of the functional vector: disjoint and exhaustive.

    ``int_g`` (zero every MLP) and ``int_h`` (zero all positional embeddings) ablate whole
    sub-systems and their ΔCE is one to two orders of magnitude larger than any other
    intervention's, so they dominate the norm and the all-key cosine is high for any two
    models that are both damaged by them. The split is *added*; the pre-registered primary
    metric is untouched.
    """
    import corpus_providers as cp

    tok = _tokenizer(tmp_path / "tok")
    corpus = cp.synthetic_corpus(tok, "random_uniform", 2, 0, cut_length=8)
    run = ev.RunInfo(run_dir=tmp_path, run_id="r", experiment_id="e6a", condition="D2",
                     seed=0, teacher="t", teacher_revision=None, tokenizer_name=None,
                     smoke=True)
    keys = ["int_a", "int_c", "int_f", "int_g", "int_h", "int_i"]
    func_keys = [k for k in keys if k != "int_a"]
    coarse = [k for k in func_keys if k in ev.COARSE_INTERVENTIONS]
    surgical = [k for k in func_keys if k not in ev.COARSE_INTERVENTIONS]
    assert set(coarse) | set(surgical) == set(func_keys)
    assert not set(coarse) & set(surgical)
    assert set(coarse) == {"int_g", "int_h"}

    # The pilot's shape, built explicitly: the two coarse ablations agree closely while the
    # surgical ones point in an unrelated direction. The all-key cosine must stay high and
    # the surgical one must not.
    teacher = _record(corpus)
    student = _record(corpus)
    teacher.delta_ce = {"int_c": 0.03, "int_f": 0.05, "int_g": 6.2, "int_h": 3.2,
                        "int_i": 0.71}
    student.delta_ce = {"int_c": 0.004, "int_f": 0.04, "int_g": 4.8, "int_h": 1.2,
                        "int_i": 0.002}

    row = ev.build_row(run, 5, student, teacher, keys,
                       validation_ce=3.5, validation_ce_n_blocks=300,
                       delta_ce_ci=None, delta_ce_n_blocks=300, checkpoint_sha256="s")
    assert set(row) == set(ev.METRIC_COLUMNS)
    assert row["functional_cosine_to_teacher"] > 0.9
    assert row["functional_cosine_coarse_to_teacher"] > 0.9
    assert row["functional_cosine_surgical_to_teacher"] < 0.5, (
        "the surgical subset is not being computed over the surgical keys")


def test_the_functional_split_is_empty_not_zero_when_a_partition_has_no_key(tmp_path):
    """A cosine over an empty key set is not a similarity of zero."""
    import corpus_providers as cp

    tok = _tokenizer(tmp_path / "tok")
    corpus = cp.synthetic_corpus(tok, "random_uniform", 2, 0, cut_length=8)
    run = ev.RunInfo(run_dir=tmp_path, run_id="r", experiment_id="e6a", condition="D2",
                     seed=0, teacher="t", teacher_revision=None, tokenizer_name=None,
                     smoke=True)
    teacher, student = _record(corpus), _record(corpus)
    teacher.delta_ce = {"int_g": 6.2}
    student.delta_ce = {"int_g": 4.8}

    row = ev.build_row(run, 5, student, teacher, ["int_a", "int_g"],
                       validation_ce=3.5, validation_ce_n_blocks=300,
                       delta_ce_ci=None, delta_ce_n_blocks=300, checkpoint_sha256="s")
    assert row["functional_cosine_coarse_to_teacher"] is not None
    assert row["functional_cosine_surgical_to_teacher"] is None


def test_delta_ce_bootstrap_is_labelled_as_sampling_uncertainty(tmp_path):
    import corpus_providers as cp

    tok = _tokenizer(tmp_path / "tok")
    corpus = cp.synthetic_corpus(tok, "random_uniform", 2, 0, cut_length=8)
    ci = ev.bootstrap_delta_ce(_record(corpus), n_boot=200, alpha=0.05, seed=0)
    assert ci["uncertainty_kind"] == "sampling_over_corpus_items"
    entry = ci["per_intervention"]["int_c"]
    assert entry["ci_lo"] <= entry["mean"] <= entry["ci_hi"]
    assert entry["n_items"] == 4


def test_delta_ce_corpus_follows_design_delta_d3():
    """300-block subset on the trajectory; the 2,000-block set only at the endpoints."""
    trajectory = type("C", (), {"items": [0] * 300})()
    endpoint = type("C", (), {"items": [0] * 2000})()
    corpora = ev.CorpusSet(sink={}, ce_trajectory=trajectory, ce_endpoint=endpoint)
    assert ev._ce_corpus_for(corpora, 0, 10000, True)[1] == 2000
    assert ev._ce_corpus_for(corpora, 10000, 10000, True)[1] == 2000
    assert ev._ce_corpus_for(corpora, 2000, 10000, True)[1] == 300
    assert ev._ce_corpus_for(corpora, 2000, 10000, False) == (None, None)


# ── discovery ───────────────────────────────────────────────────────────────────


def test_arch_is_read_from_the_checkpoint_not_the_run_config(tmp_path):
    ckpt = tmp_path / "step_0"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"architectures": ["GPT2LMHeadModel"]}),
                                      encoding="utf-8")
    assert ev.arch_of_checkpoint(ckpt) == "gpt2"

    (ckpt / "config.json").write_text(json.dumps({"architectures": ["MambaForCausalLM"]}),
                                      encoding="utf-8")
    with pytest.raises(ValueError, match="map to no known arch key"):
        ev.arch_of_checkpoint(ckpt)


def test_select_steps_rejects_a_step_that_was_never_checkpointed(tmp_path):
    run_dir = _run_dir(tmp_path, steps=(0, 5))
    available = ev.discover_checkpoints(run_dir)
    assert [s for s, _ in available] == [0, 5]
    assert [s for s, _ in ev.select_steps(available, "5")] == [5]
    with pytest.raises(SystemExit, match="no such checkpoint"):
        ev.select_steps(available, "0,999")


def test_checkpoints_are_ordered_numerically_not_lexically(tmp_path):
    """``step_1000`` must not sort before ``step_250``; a mis-order breaks trajectories."""
    run_dir = _run_dir(tmp_path, steps=(0, 250, 1000))
    assert [s for s, _ in ev.discover_checkpoints(run_dir)] == [0, 250, 1000]
