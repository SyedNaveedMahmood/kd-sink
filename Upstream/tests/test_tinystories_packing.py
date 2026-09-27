"""WP1: TinyStories block packing (06 test_tinystories_packing.py).

Blocks are exactly ``block_size``; exactly one EOS separates consecutive stories; each
block manifest maps to real story indices; and the sink / ppl corpora are disjoint.

Offline: ``datasets_loader.load_dataset`` is monkeypatched to a tiny in-memory story set,
and the repo's WordLevel tokenizer (``t3..`` -> id 3..) makes token structure exact and
checkable. ``SINK_BLOCKS_RESERVED`` is shrunk so disjointness is testable at small scale.
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus_providers as cp  # noqa: E402
import datasets_loader as dl  # noqa: E402
from nnsight_smoke_utils import _tiny_tokenizer  # noqa: E402

# Stories of differing lengths so packing boundaries are non-trivial.
_STORIES = [
    {"text": "t3 t4 t5"},
    {"text": "t6 t7 t8 t9 t10"},
    {"text": "t11 t12"},
    {"text": "t13 t14 t15 t16"},
    {"text": "t17 t18 t19 t20 t21 t22"},
    {"text": "t23 t24 t25"},
    {"text": "t26 t27 t28 t29"},
    {"text": "t30 t31 t32 t33 t34"},
]


@pytest.fixture
def tokenizer():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        yield _tiny_tokenizer(Path(t) / "tok", 64)


@pytest.fixture(autouse=True)
def _patch_dataset(monkeypatch):
    def fake_load_dataset(path, *args, split=None, **kwargs):
        assert path == cp._dl.TINYSTORIES_HF_PATH
        return list(_STORIES)
    monkeypatch.setattr(dl, "load_dataset", fake_load_dataset)


def _ideal_stream(tokenizer, seed, eos):
    order = list(range(len(_STORIES)))
    random.Random(seed).shuffle(order)
    stream = []
    for idx in order:
        stream.extend(tokenizer(_STORIES[idx]["text"], add_special_tokens=False)["input_ids"])
        stream.append(eos)
    return stream


def test_blocks_exact_size_and_one_eos_between_stories(tokenizer) -> None:
    block_size, seed = 4, 0
    blocks, manifest = dl.load_tinystories_blocks(
        tokenizer, "validation", block_size=block_size, seed=seed, eos_between=True)
    assert blocks, "expected at least one packed block"
    for ids, row in zip(blocks, manifest):
        assert len(ids) == block_size
        assert row["n_tokens"] == block_size

    # Packing is the exact prefix of the ideal one-EOS-per-story stream.
    emitted = [t for ids in blocks for t in ids]
    ideal = _ideal_stream(tokenizer, seed, tokenizer.eos_token_id)
    assert len(emitted) % block_size == 0
    assert emitted == ideal[:len(emitted)]

    # Every manifest story index is a real story.
    for row in manifest:
        for idx in row["source_story_indices"]:
            assert 0 <= idx < len(_STORIES)


def test_sink_and_ppl_disjoint(tokenizer, monkeypatch) -> None:
    # Shrink the reservation so ppl's offset is reachable with the tiny story set.
    monkeypatch.setattr(cp, "SINK_BLOCKS_RESERVED", 2)
    sink = cp.tinystories_corpus(tokenizer, "validation", 2, block_size=3, seed=1,
                                 purpose="sink")
    ppl = cp.tinystories_corpus(tokenizer, "validation", 2, block_size=3, seed=1,
                                purpose="ppl")
    sink_ids = {it.meta["block_index"] for it in sink.items}
    ppl_ids = {it.meta["block_index"] for it in ppl.items}
    assert sink_ids and ppl_ids
    assert sink_ids.isdisjoint(ppl_ids)
    assert all(it.n_tokens == 3 for it in list(sink.items) + list(ppl.items))
