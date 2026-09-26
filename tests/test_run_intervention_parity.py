"""WP3 parity gate: ``run_intervention`` is byte-equivalent after the ``_apply_edits`` factoring.

The golden fixture ``tests/fixtures/run_intervention_golden.pt`` was captured from the
pre-refactor ``run_intervention`` (via ``python -m tests.golden_fingerprint``). This test
rebuilds the identical random gpt2 + neo checkpoints (same seed -> same weights), reruns
the *refactored* ``run_intervention`` over the same inputs / massive coords / swap dirs,
and asserts agreement. The BOS metric — the number that reaches the Table-1 CSV — is the
gate at ``METRIC_ATOL/METRIC_RTOL``; the raw ``[heads, seq, seq]`` maps are additionally
checked at the tight attention tolerance, since a pure refactor should not move them.

If this drifts, the ``_apply_edits`` refactor changed observable behaviour and the frozen
E1-E5 instrument no longer measures what E6/E7 measure (`06` §3).

Windows-safe teardown: models are released and gc'd; the fixture build uses a
``TemporaryDirectory(ignore_cleanup_errors=True)``.
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

from nnsight_engine import ATTN_ATOL, ATTN_RTOL, METRIC_ATOL, METRIC_RTOL  # noqa: E402

from golden_fingerprint import (  # noqa: E402
    FIXTURE_PATH,
    PARITY_ARCHS,
    build_arch_engine,
    fixed_inputs,
    reference_fingerprint,
)


@pytest.fixture(scope="module")
def golden():
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"golden fixture missing at {FIXTURE_PATH}; run "
            "`python -m tests.golden_fingerprint` before the _apply_edits refactor")
    return torch.load(FIXTURE_PATH, weights_only=False)


def test_golden_covers_parity_archs(golden) -> None:
    assert set(PARITY_ARCHS).issubset(golden["archs"])
    for arch in PARITY_ARCHS:
        assert arch in golden, arch


@pytest.mark.parametrize("arch", PARITY_ARCHS)
def test_run_intervention_matches_golden(golden, arch) -> None:
    inputs_list = fixed_inputs()
    with tempfile.TemporaryDirectory(
            prefix=f"parity_{arch}_", ignore_cleanup_errors=True) as temp:
        engine, _ = build_arch_engine(arch, Path(temp) / arch)
        try:
            fresh = reference_fingerprint(engine, inputs_list)
        finally:
            del engine
            gc.collect()

    golden_records = golden[arch]["records"]
    assert len(fresh) == len(golden_records)

    for sent_idx, (fresh_row, gold_row) in enumerate(zip(fresh, golden_records)):
        assert set(fresh_row) == set(gold_row), sent_idx
        for key in fresh_row:
            gold_bos = gold_row[key]["bos"]
            fresh_bos = fresh_row[key]["bos"]
            # Gate: the CSV-bound BOS metric.
            assert abs(fresh_bos - gold_bos) <= METRIC_ATOL + METRIC_RTOL * abs(gold_bos), (
                arch, sent_idx, key, fresh_bos, gold_bos)

            # A pure refactor should also leave the raw maps within attention tolerance.
            gold_maps = gold_row[key]["attn"]
            fresh_maps = fresh_row[key]["attn"]
            assert len(fresh_maps) == len(gold_maps), (arch, key)
            for layer, (fm, gm) in enumerate(zip(fresh_maps, gold_maps)):
                assert fm.shape == gm.shape, (arch, key, layer)
                assert torch.allclose(fm, gm, atol=ATTN_ATOL, rtol=ATTN_RTOL), (
                    arch, sent_idx, key, layer,
                    float((fm - gm).abs().max()))
