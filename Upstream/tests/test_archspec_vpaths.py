"""WP3: ArchSpec value/output capture surface resolves (06 test_archspec_vpaths.py, G5).

For a random model of each architecture, asserts that the additive ``v_path`` / ``o_path``
fields resolve to real ``nn`` modules, and that the K/V projection trailing dimension
equals ``num_kv_heads * head_dim`` — the guard (`01` §4) that makes E7's pre-``_repeat_kv``
extraction/patching shape-safe. On Qwen (GQA, ``num_kv_heads=2 < num_heads=8``) this is a
strictly smaller dimension than Q; a silent mismatch here would produce plausible but wrong
patch results, so it is asserted explicitly.

GPT-2 fuses V inside ``c_attn`` (``v_path is None``); only its ``o_path`` is checked.
"""

from __future__ import annotations

import gc
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnsight_engine import ARCH_SPECS, _resolve  # noqa: E402

from golden_fingerprint import build_arch_engine  # noqa: E402

ARCHS = ("gpt2", "neo", "qwen", "opt")


def _head_dim(cfg) -> int:
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is not None:
        return int(head_dim)
    return int(cfg.hidden_size // cfg.num_attention_heads)


def _num_kv_heads(cfg) -> int:
    return int(getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads)


@pytest.mark.parametrize("arch", ARCHS)
def test_vpaths_resolve_and_kv_dim(arch) -> None:
    spec = ARCH_SPECS[arch]
    with tempfile.TemporaryDirectory(
            prefix=f"vpaths_{arch}_", ignore_cleanup_errors=True) as temp:
        engine, _ = build_arch_engine(arch, Path(temp) / arch)
        try:
            cfg = engine.hf.config
            attn = engine._hf_attn(0)

            # o_path must resolve on every arch.
            assert spec.o_path is not None, arch
            out_proj = _resolve(attn, spec.o_path)
            assert isinstance(out_proj, nn.Module), (arch, spec.o_path)

            if spec.qkv_layout == "fused":
                # GPT-2: V is fused inside c_attn -> no separate v_path to resolve.
                assert spec.v_path is None, arch
                assert spec.kv_grouped is False
                return

            # Separate-projection archs: v/k paths resolve and carry the KV-head width.
            assert spec.v_path is not None and spec.k_path is not None, arch
            v_proj = _resolve(attn, spec.v_path)
            k_proj = _resolve(attn, spec.k_path)
            assert isinstance(v_proj, nn.Module) and isinstance(k_proj, nn.Module)

            expected_kv = _num_kv_heads(cfg) * _head_dim(cfg)
            assert k_proj.weight.shape[0] == expected_kv, (
                arch, "k", k_proj.weight.shape, expected_kv)
            assert v_proj.weight.shape[0] == expected_kv, (
                arch, "v", v_proj.weight.shape, expected_kv)

            # kv_grouped must be declared exactly when the arch actually groups.
            grouped = _num_kv_heads(cfg) < cfg.num_attention_heads
            assert spec.kv_grouped == grouped, (arch, spec.kv_grouped, grouped)
            if arch == "qwen":
                assert grouped is True  # the GQA case the guard exists for
        finally:
            del engine
            gc.collect()
