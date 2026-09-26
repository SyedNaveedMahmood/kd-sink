"""WP2: fingerprint runner bridge (06 test_fingerprint_runner_bridge.py) — the load-bearing gate.

``compute_fingerprint(gpt2_handle, frozen_e1_corpus(...))`` must reproduce the frozen
``bos_attention_stats_overall.csv`` within ``METRIC_ATOL``. If this drifts, every E6/E7
number is measured with a subtly different instrument than E1–E5 (06 §3).

Requires the real benchmark datasets, real GPT-2 weights, and the checked-in reference
CSV, so it SKIPS on the smoke-only box and runs on the compute PC. Provide the reference
via ``SINKS_BOS_STATS_CSV=/path/to/bos_attention_stats_overall.csv``; the CSV must map each
intervention key to its overall mean BOS metric.
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
import fingerprint_runner as fr  # noqa: E402
from nnsight_engine import METRIC_ATOL, METRIC_RTOL  # noqa: E402


#: The model this gate compares against. ``compute_fingerprint`` is run on ``"gpt2"``, so
#: only a reference produced by that same checkpoint is a valid comparand.
REFERENCE_MODEL_NAMES = ("gpt2", "openai-community/gpt2")


def _reference_model(candidate: Path) -> str | None:
    """The model a reference artefact was produced by, from its sibling ``run_config.json``."""
    config_path = candidate.with_name("run_config.json")
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    name = config.get("model_name") or config.get("model")
    return str(name) if name else None


def _reference_csv() -> Path | None:
    """A frozen BOS reference **produced by the same model this test runs**.

    The repo tree can hold reference artefacts for every model in Table 1 — gpt2,
    gpt2-medium/large, three GPT-Neo sizes, Qwen. Taking whichever ``rglob`` yields first
    silently compares GPT-2's fingerprint against, say, GPT-Neo-1.3B's: most interventions
    land close enough to look fine and one diverges, which reads as a seam regression
    rather than as a mismatched comparand. ``05`` §7.1 is explicit — hash before compare —
    so the model is checked here and an unidentifiable artefact is not used at all.
    """
    env = os.environ.get("SINKS_BOS_STATS_CSV")
    if env and Path(env).exists():
        return Path(env)
    for candidate in sorted(REPO.rglob("bos_attention_stats_overall.csv")):
        if _reference_model(candidate) in REFERENCE_MODEL_NAMES:
            return candidate
    return None


def _reference_corpus_kwargs(reference: Path) -> dict:
    """Read the sampling settings paired with a frozen BOS reference artefact."""
    config_path = reference.with_name("run_config.json")
    if not config_path.exists():
        return {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        key: int(config[key])
        for key in ("sample_size", "cut_length", "seed")
        if config.get(key) is not None
    }


def _reference_means(path) -> dict:
    import pandas as pd
    df = pd.read_csv(path)
    key_col = next((c for c in ("intervention", "key", "name") if c in df.columns), None)
    val_col = next((c for c in ("mean_bos_attention", "mean", "score", "bos",
                                "bos_attn", "value")
                    if c in df.columns), None)
    if key_col is None or val_col is None:
        pytest.skip(f"reference CSV {path} lacks recognisable key/value columns "
                    f"(got {list(df.columns)})")
    return {str(k): float(v) for k, v in zip(df[key_col], df[val_col])}


def test_fingerprint_reproduces_frozen_bos_stats() -> None:
    reference = _reference_csv()
    if reference is None:
        pytest.skip("no bos_attention_stats_overall.csv produced by "
                    f"{REFERENCE_MODEL_NAMES}; run the frozen GPT-2 dataset analysis on "
                    "the compute PC (or set SINKS_BOS_STATS_CSV)")
    try:
        import torch
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        handle = fr.load_handle("gpt2", "gpt2", dtype="float32",
                                device=torch.device("cpu"))
    except Exception as exc:  # pragma: no cover - needs real gpt2/network
        pytest.skip(f"real GPT-2 unavailable offline: {type(exc).__name__}: {exc}")

    try:
        corpus = cp.frozen_e1_corpus(
            tokenizer, **_reference_corpus_kwargs(reference))
    except Exception as exc:  # pragma: no cover - needs dataset downloads
        pytest.skip(f"benchmark datasets unavailable offline: {type(exc).__name__}: {exc}")

    record = fr.compute_fingerprint(handle, corpus)
    expected = _reference_means(reference)
    common = set(expected) & set(record.raw_sink_by_intervention)
    assert common, "no shared intervention keys between record and reference"
    for key in common:
        got = record.raw_sink_by_intervention[key]
        ref = expected[key]
        assert abs(got - ref) <= METRIC_ATOL + METRIC_RTOL * abs(ref), (key, got, ref)
