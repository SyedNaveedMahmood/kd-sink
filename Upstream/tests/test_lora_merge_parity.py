# -*- coding: utf-8 -*-
"""test_lora_merge_parity.py — merged LoRA reproduces the adapter (WP6, design-delta D5).

Only the **merged** checkpoint is ever fingerprinted: the frozen GPT-2 harness resolves
module paths like ``transformer.h.0.attn.c_attn`` and a PEFT wrapper renames them. So every
E6B sink number is taken from a model that was never the one trained, and the merge is the
join between them. If it does not reproduce the adapter exactly, every E6B fingerprint
describes a different model than the one whose accuracy is reported beside it.

``06_TEST_PLAN.md`` §1: merged vs unmerged logits agree < 1e-5 **in fp32 on CPU**, on five
parity sentences.

**The adapter is perturbed before the check.** A freshly initialised LoRA has ``B = 0``, so
it contributes exactly nothing and merged/unmerged agree at 0.0 for a reason that has
nothing to do with merging being correct. A test that passed on that would pass equally
well against a broken merge, so the B matrices are filled first and the test asserts the
adapter is actually doing something before asserting the two agree.
"""

from __future__ import annotations

import gc
import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_sentiment_adaptation as ts  # noqa: E402


def _assets(tmp: Path):
    tokenizer, _tokenizer_dir, base_dir = ts.build_smoke_assets(tmp / "_smoke")
    return tokenizer, base_dir


def _peft_model(base_dir, tokenizer, *, perturb: bool = True):
    """A LoRA-wrapped tiny GPT-2 whose adapter actually changes the output."""
    from peft import get_peft_model

    model = ts.load_base_model(str(base_dir), dtype="float32", device="cpu",
                               local_files_only=True)
    peft_model = get_peft_model(model, ts.build_lora_config(
        {"lora": {"r": 4, "alpha": 8, "dropout": 0.0,
                  "target_modules": ["c_attn", "c_proj"]}}))
    if perturb:
        # lora_B is zero-initialised by design, so an untouched adapter is a no-op and the
        # parity check would be vacuous. Filling it makes the merge do real work.
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if "lora_B" in name:
                    param.copy_(torch.randn(param.shape, generator=generator) * 0.05)
    return peft_model


def _logits(model, tokenizer, sentence):
    ids = tokenizer(sentence, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        return model(**ids).logits


# ── the check is not vacuous ───────────────────────────────────────────────────


def test_a_perturbed_adapter_actually_changes_the_logits():
    """Guard against a parity test that would pass against a broken merge."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
        tmp = Path(temp)
        tokenizer, base_dir = _assets(tmp)
        base = ts.load_base_model(str(base_dir), dtype="float32", device="cpu",
                                  local_files_only=True).eval()
        adapted = _peft_model(base_dir, tokenizer).eval()

        sentence = ts.PARITY_SENTENCES[0]
        delta = float((_logits(base, tokenizer, sentence)
                       - _logits(adapted, tokenizer, sentence)).abs().max())
        assert delta > ts.MERGE_PARITY_TOL * 100, (
            f"the adapter moved the logits by only {delta:.3e}; the parity check below "
            "would be measuring nothing")
        del base, adapted
        gc.collect()


# ── the parity check itself ────────────────────────────────────────────────────


def test_merged_and_unmerged_agree_within_tolerance():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
        tmp = Path(temp)
        tokenizer, base_dir = _assets(tmp)
        peft_model = _peft_model(base_dir, tokenizer)

        report = ts.merge_and_verify(peft_model, tmp / "step_0", tokenizer=tokenizer,
                                     training_dtype="bfloat16", device="cpu")

        assert report["passed"] is True
        assert report["max_abs_logit_delta"] < ts.MERGE_PARITY_TOL
        assert report["n_sentences"] == 5 == len(ts.PARITY_SENTENCES)
        assert len(report["per_sentence"]) == 5
        gc.collect()


def test_the_report_says_what_precision_gated_and_is_not_a_bf16_claim():
    """`03` §4.5 — a reader must not mistake 1e-5 next to bfloat16 for a bf16 tolerance."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
        tmp = Path(temp)
        tokenizer, base_dir = _assets(tmp)
        report = ts.merge_and_verify(_peft_model(base_dir, tokenizer), tmp / "step_0",
                                     tokenizer=tokenizer, training_dtype="bfloat16",
                                     device="cpu")

        assert report["check_dtype"] == ts.MERGE_PARITY_DTYPE_NAME
        assert report["check_device"] == "cpu"
        assert report["training_dtype"] == "bfloat16"
        assert "not a bf16 round-trip claim" in report["note"]
        gc.collect()


def test_the_tolerance_is_the_specs_own_number_and_the_deviation_is_recorded():
    """The precision was raised; the bar was **not** moved.

    `03` §4.5 says fp32 and 1e-5. The 1e-5 stands unchanged. The working precision is
    float64 because §4.5's bar is *absolute* and real distilgpt2 logits reach ~104, which
    puts 1e-5 below float32's own epsilon — no implementation could pass in fp32, so an
    fp32 gate would fail on correct code and say nothing about correctness. Recorded in the
    artefact, not just in a commit message.
    """
    assert ts.MERGE_PARITY_TOL == 1e-5, "the spec's tolerance must not be loosened"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
        tmp = Path(temp)
        tokenizer, base_dir = _assets(tmp)
        report = ts.merge_and_verify(_peft_model(base_dir, tokenizer), tmp / "step_0",
                                     tokenizer=tokenizer, training_dtype="bfloat16",
                                     device="cpu")

        assert report["tolerance"] == 1e-5
        assert report["spec_check_dtype"] == "float32"
        assert "TOLERANCE is unchanged" in report["deviation_from_spec"]
        assert report["max_relative_logit_delta"] is not None
        gc.collect()


def test_both_copies_are_kept_and_only_the_merged_one_is_fingerprinted():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
        tmp = Path(temp)
        tokenizer, base_dir = _assets(tmp)
        out = tmp / "step_0"
        report = ts.merge_and_verify(_peft_model(base_dir, tokenizer), out,
                                     tokenizer=tokenizer, training_dtype="float32",
                                     device="cpu")

        assert (out / "adapter").is_dir(), "the adapter must be kept (03 §4.5)"
        assert (out / "merged" / "config.json").exists()
        assert "merged only" in report["fingerprinted"]
        # The evaluator must resolve the merged copy, never the PEFT adapter.
        import evaluate_transformation as ev
        assert ev.fingerprint_dir(out) == out / "merged"
        gc.collect()


def test_the_parity_json_is_written_beside_the_checkpoint():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
        tmp = Path(temp)
        tokenizer, base_dir = _assets(tmp)
        out = tmp / "step_0"
        ts.merge_and_verify(_peft_model(base_dir, tokenizer), out, tokenizer=tokenizer,
                            training_dtype="float32", device="cpu")

        payload = json.loads((out / "merge_parity.json").read_text(encoding="utf-8"))
        assert payload["passed"] is True
        assert payload["git_sha"], "provenance is mandatory on every artefact (05 §7)"
        gc.collect()


def test_a_failing_merge_raises_rather_than_recording_a_pass(monkeypatch):
    """A merge that does not reproduce the adapter must stop the run, not be logged."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
        tmp = Path(temp)
        tokenizer, base_dir = _assets(tmp)
        monkeypatch.setattr(ts, "MERGE_PARITY_TOL", 1e-30)
        with pytest.raises(AssertionError, match="merge parity failed"):
            ts.merge_and_verify(_peft_model(base_dir, tokenizer), tmp / "step_0",
                                tokenizer=tokenizer, training_dtype="float32",
                                device="cpu")
        gc.collect()


def test_the_parity_sentences_are_fixed_and_not_drawn_from_sst2():
    """Five fixed sentences, so the check can never run on data the adapter was fitted to."""
    assert len(ts.PARITY_SENTENCES) == 5
    assert len(set(ts.PARITY_SENTENCES)) == 5
    assert all(isinstance(s, str) and s.strip() for s in ts.PARITY_SENTENCES)
