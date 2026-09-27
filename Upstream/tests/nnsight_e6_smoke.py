"""Offline E6 smoke: fingerprint + run_arch_trace + CE on random gpt2/neo/opt.

Mirrors the E3/E4/E5 smoke programs (06 test_plan nnsight_e6_smoke.py): random tiny
checkpoints under a TemporaryDirectory, CPU/fp32, no network. Exercises the WP3 additions
end to end — the shared ``_apply_edits`` fingerprint path and the arch-generic
``run_arch_trace`` reaching the LM head for per-token CE on a non-fused (GPT-Neo) student,
which is the capability E6A's ΔCE-under-intervention analysis depends on (G3).

Run: ``./.venv/Scripts/python.exe -m tests.nnsight_e6_smoke``.

On Windows the process may still exit non-zero at TemporaryDirectory teardown (the known
safetensors mmap-release flake shared with the e4/e5 smokes); all assertions run first.
"""

from __future__ import annotations

import gc
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnsight_engine import INTERVENTION_ORDER  # noqa: E402

from golden_fingerprint import (  # noqa: E402
    build_arch_engine,
    fixed_inputs,
    reference_fingerprint,
)

E6_ARCHS = ("gpt2", "neo", "opt")


def _exercise(arch: str, base: Path) -> None:
    engine, _ = build_arch_engine(arch, base / arch)
    try:
        inputs_list = fixed_inputs(n_sentences=2)

        # 1. Full fingerprint through the refactored run_intervention / _apply_edits.
        records = reference_fingerprint(engine, inputs_list)
        assert len(records) == len(inputs_list)
        for row in records:
            assert set(row) == set(INTERVENTION_ORDER), (arch, sorted(row))
            for key, payload in row.items():
                assert len(payload["attn"]) == engine.num_layers, (arch, key)
                for m in payload["attn"]:
                    assert torch.isfinite(m).all(), (arch, key)

        # 2. Arch-generic trace with CE + target attention, identity and an intervention.
        inputs = inputs_list[0]
        seq_len = inputs["input_ids"].shape[-1]
        heads = engine.hf.config.num_attention_heads
        for key in ("int_a", "int_g", "int_i"):
            out = engine.run_arch_trace(
                engine.plans[key], inputs,
                massive_coords=[4, 17], random_columns=None,
                capture_attention="targets", target_positions=(0,),
                capture_logits=True, capture_token_ce=True)
            assert out["logits"].shape == (1, seq_len, engine.hf.config.vocab_size)
            assert out["token_ce"].shape == (seq_len - 1,)
            assert out["target_attention"].shape == (engine.num_layers, heads, 1)
            assert torch.isfinite(out["logits"]).all(), (arch, key)
            assert torch.isfinite(out["token_ce"]).all(), (arch, key)
        print(f"  [{arch}] fingerprint + run_arch_trace + CE OK "
              f"(L={engine.num_layers}, heads={heads}, seq={seq_len})")
    finally:
        del engine
        gc.collect()


def main() -> None:
    with tempfile.TemporaryDirectory(
            prefix="sinks_e6_nnsight_", ignore_cleanup_errors=True) as temp:
        base = Path(temp)
        for arch in E6_ARCHS:
            _exercise(arch, base)
    print("E6 NNsight offline smoke test passed")


if __name__ == "__main__":
    main()
