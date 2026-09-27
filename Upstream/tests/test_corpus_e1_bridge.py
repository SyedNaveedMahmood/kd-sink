"""WP1: frozen E1 corpus bridge (06 test_corpus_e1_bridge.py).

``frozen_e1_corpus`` must reproduce the checked-in E1 ``sample_manifest.csv`` row-for-row
— the regression bridge proving the new corpus path samples exactly what E1 sampled.

This gate needs the real benchmark datasets (SST-2/GSM8K/HumanEval), the real GPT-2
tokenizer, and a checked-in reference manifest. None are available on the smoke-only box,
so the test SKIPS here and runs for real on the compute PC. Point it at a reference with
``SINKS_E1_MANIFEST_CSV=/path/to/sample_manifest.csv`` (and a working HF cache).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))

import corpus_providers as cp  # noqa: E402


def _reference_csv() -> Path | None:
    env = os.environ.get("SINKS_E1_MANIFEST_CSV")
    if env and Path(env).exists():
        return Path(env)
    for candidate in REPO.rglob("sample_manifest.csv"):
        return candidate
    return None


def _reference_corpus_kwargs(reference: Path) -> dict:
    """Reconstruct the corpus with the settings that produced ``reference``.

    Multiseed E1 artefacts intentionally use different sampling seeds.  The companion
    ``run_config.json`` is therefore the source of truth; falling back to provider
    defaults is only valid for older standalone manifests without provenance.
    """
    config_path = reference.with_name("run_config.json")
    if not config_path.exists():
        return {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        key: int(config[key])
        for key in ("sample_size", "cut_length", "seed")
        if config.get(key) is not None
    }


def test_frozen_e1_corpus_matches_reference_manifest() -> None:
    reference = _reference_csv()
    if reference is None:
        pytest.skip("no checked-in E1 sample_manifest.csv; run on the compute PC "
                    "(set SINKS_E1_MANIFEST_CSV)")
    try:
        import pandas as pd
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"GPT-2 tokenizer unavailable offline: {type(exc).__name__}: {exc}")

    try:
        corpus = cp.frozen_e1_corpus(
            tokenizer, **_reference_corpus_kwargs(reference))
    except Exception as exc:  # pragma: no cover - needs dataset downloads
        pytest.skip(f"benchmark datasets unavailable offline: {type(exc).__name__}: {exc}")

    ref = pd.read_csv(reference)
    got = [(it.meta["dataset"], it.meta["source_index"]) for it in corpus.items]
    expected = list(zip(ref["dataset"], ref["source_index"]))
    assert got == expected, "frozen_e1_corpus row order/identity drifted from E1"
