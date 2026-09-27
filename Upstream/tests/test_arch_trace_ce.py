"""WP3: ``run_arch_trace`` cross-entropy and capture surface (06 test_arch_trace_ce.py).

Proves, on random gpt2 / neo / qwen / opt models (CPU, fp32, offline):

* ``run_arch_trace(capture_token_ce=True)`` with the identity plan (int_a) reproduces a
  direct HF forward + the frozen ``token_cross_entropy`` to 1e-6 — i.e. the generic trace
  reaches the real LM head and computes CE with the *same* instrument E5 uses (G3);
* ``capture_logits`` returns finite ``[1, S, V]`` logits matching the HF forward;
* the three ``capture_attention`` modes return the documented shapes (or None);
* an actual intervention (int_g, zero layer-0..N MLP) still yields finite CE, so the
  ΔCE-under-intervention path Neo needs is live.

Windows-safe teardown: each model is released and gc'd inside its TemporaryDirectory.
"""

from __future__ import annotations

import gc
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnsight_engine import EditPlan  # noqa: E402

from golden_fingerprint import build_arch_engine, fixed_inputs  # noqa: E402

CE_ATOL = 1e-6
ARCHS = ("gpt2", "neo", "qwen", "opt")


def _token_cross_entropy():
    try:
        from evaluation_robustness.evaluation_robustness_analysis import (
            token_cross_entropy,
        )
    except ImportError:
        from evaluation_robustness_analysis import token_cross_entropy
    return token_cross_entropy


@pytest.mark.parametrize("arch", ARCHS)
def test_arch_trace_ce_matches_hf_forward(arch) -> None:
    token_cross_entropy = _token_cross_entropy()
    inputs = fixed_inputs(n_sentences=1)[0]
    with tempfile.TemporaryDirectory(
            prefix=f"arch_ce_{arch}_", ignore_cleanup_errors=True) as temp:
        engine, _ = build_arch_engine(arch, Path(temp) / arch)
        try:
            vocab = engine.hf.config.vocab_size
            seq_len = inputs["input_ids"].shape[-1]
            plan_a = engine.plans["int_a"]

            out = engine.run_arch_trace(
                plan_a, inputs, capture_attention="targets",
                target_positions=(0,), capture_logits=True, capture_token_ce=True)

            # Shapes and finiteness.
            heads = engine.hf.config.num_attention_heads
            assert out["logits"].shape == (1, seq_len, vocab)
            assert out["token_ce"].shape == (seq_len - 1,)
            assert out["target_attention"].shape == (engine.num_layers, heads, 1)
            assert torch.isfinite(out["logits"]).all()
            assert torch.isfinite(out["token_ce"]).all()

            # Reference: the *same* payload through a direct HF forward + frozen CE.
            payload = engine._payload(EditPlan("int_a"), inputs)
            with torch.no_grad():
                ref_logits = engine.hf(**payload).logits
            ref_ce = token_cross_entropy(ref_logits.float(), inputs["input_ids"])

            assert torch.allclose(out["logits"], ref_logits.float(), atol=CE_ATOL), (
                arch, float((out["logits"] - ref_logits.float()).abs().max()))
            assert torch.allclose(out["token_ce"], ref_ce, atol=CE_ATOL), (
                arch, float((out["token_ce"] - ref_ce).abs().max()))
        finally:
            del engine
            gc.collect()


@pytest.mark.parametrize("arch", ARCHS)
def test_arch_trace_capture_modes(arch) -> None:
    inputs = fixed_inputs(n_sentences=1)[0]
    seq_len = inputs["input_ids"].shape[-1]
    with tempfile.TemporaryDirectory(
            prefix=f"arch_modes_{arch}_", ignore_cleanup_errors=True) as temp:
        engine, _ = build_arch_engine(arch, Path(temp) / arch)
        try:
            plan_a = engine.plans["int_a"]
            heads = engine.hf.config.num_attention_heads

            full = engine.run_arch_trace(plan_a, inputs, capture_attention="full")
            assert full["target_attention"] is None
            assert len(full["attention"]) == engine.num_layers
            for m in full["attention"]:
                assert m.shape == (heads, seq_len, seq_len)

            tgt = engine.run_arch_trace(
                plan_a, inputs, capture_attention="targets", target_positions=(0, 1))
            assert tgt["attention"] is None
            assert tgt["target_attention"].shape == (engine.num_layers, heads, 2)

            none = engine.run_arch_trace(
                plan_a, inputs, capture_attention="none", capture_token_ce=True)
            assert none["attention"] is None and none["target_attention"] is None
            assert none["token_ce"].shape == (seq_len - 1,)

            # An actual MLP-zeroing intervention still reaches the LM head with finite CE.
            plan_g = engine.plans["int_g"]
            edited = engine.run_arch_trace(
                plan_g, inputs, capture_attention="none", capture_token_ce=True)
            assert torch.isfinite(edited["token_ce"]).all()
        finally:
            del engine
            gc.collect()


def test_capture_position0_not_implemented() -> None:
    inputs = fixed_inputs(n_sentences=1)[0]
    with tempfile.TemporaryDirectory(
            prefix="arch_pos0_", ignore_cleanup_errors=True) as temp:
        engine, _ = build_arch_engine("gpt2", Path(temp) / "gpt2")
        try:
            with pytest.raises(NotImplementedError):
                engine.run_arch_trace(
                    engine.plans["int_a"], inputs, capture_position0={"foo": 1})
        finally:
            del engine
            gc.collect()
