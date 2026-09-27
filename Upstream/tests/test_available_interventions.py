"""WP2: cross-model intervention availability (06 test_available_interventions.py, D2).

The mutually-defined intervention set is *computed from the model*, not hard-coded:
``int_b`` is excluded for GPT-Neo (no query-projection bias) and ``int_e`` for Qwen (no
additive positional embedding to swap). ``mutual_interventions`` intersects while
preserving ``INTERVENTION_ORDER``. Offline: random gpt2/neo/qwen checkpoints.
"""

from __future__ import annotations

import gc
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fingerprint_runner as fr  # noqa: E402
from nnsight_engine import INTERVENTION_ORDER  # noqa: E402

from golden_fingerprint import build_arch_engine  # noqa: E402


def _handle(arch, temp):
    engine, tok = build_arch_engine(arch, Path(temp) / arch)
    return fr.handle_from_module(arch, engine.hf, tok, model_name=arch)


def test_neo_excludes_int_b_and_qwen_excludes_int_e() -> None:
    with tempfile.TemporaryDirectory(prefix="avail_", ignore_cleanup_errors=True) as temp:
        gpt2 = _handle("gpt2", temp)
        neo = _handle("neo", temp)
        qwen = _handle("qwen", temp)
        try:
            gpt2_av = fr.available_interventions(gpt2)
            neo_av = fr.available_interventions(neo)
            qwen_av = fr.available_interventions(qwen)

            assert "int_b" in gpt2_av        # GPT-2 has a query bias
            assert "int_b" not in neo_av     # GPT-Neo has none -> (b) is a structural no-op
            assert "int_e" not in qwen_av    # Qwen is RoPE -> no additive PE to swap
            assert "int_e" in gpt2_av

            # Order preserved and mutual set is the intersection.
            mutual = fr.mutual_interventions(gpt2, neo, qwen)
            assert mutual == [k for k in INTERVENTION_ORDER
                              if k in set(gpt2_av) & set(neo_av) & set(qwen_av)]
            assert "int_b" not in mutual and "int_e" not in mutual
            assert mutual == sorted(mutual, key=INTERVENTION_ORDER.index)
        finally:
            del gpt2, neo, qwen
            gc.collect()


def test_available_preserves_order_and_subset() -> None:
    with tempfile.TemporaryDirectory(prefix="avail2_", ignore_cleanup_errors=True) as temp:
        gpt2 = _handle("gpt2", temp)
        try:
            av = fr.available_interventions(gpt2)
            assert av == [k for k in INTERVENTION_ORDER if k in set(av)]
            assert set(av).issubset(set(INTERVENTION_ORDER))
        finally:
            del gpt2
            gc.collect()
