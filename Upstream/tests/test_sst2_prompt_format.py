"""WP1: SST-2 prompt format (06 test_sst2_prompt_format.py).

The sentence begins at position 0 with no special-token prefix; label token counts are
recorded; and the loss/scoring span covers the label tokens only. Offline via a
monkeypatched ``load_dataset`` and the repo WordLevel tokenizer.
"""

from __future__ import annotations

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

_ROWS = [
    {"sentence": "t3 t4 t5", "label": 1},
    {"sentence": "t6 t7 t8 t9", "label": 0},
    {"sentence": "t10 t11", "label": 1},
]


@pytest.fixture
def tokenizer():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        yield _tiny_tokenizer(Path(t) / "tok", 64)


@pytest.fixture(autouse=True)
def _patch_dataset(monkeypatch):
    def fake_load_dataset(path, *args, split=None, **kwargs):
        assert path == "stanfordnlp/sst2"
        return list(_ROWS)
    monkeypatch.setattr(dl, "load_dataset", fake_load_dataset)


def test_sentence_at_position_zero_and_label_span(tokenizer) -> None:
    rows, _manifest, info = dl.load_sst2_prompted(tokenizer, "train", seed=0)
    assert info["sentence_starts_at"] == 0
    assert set(info["label_token_counts"]) == {" positive", " negative"}

    for row in rows:
        sentence_ids = tokenizer(row["sentence"], add_special_tokens=False)["input_ids"]
        # No prefix: input_ids begin with exactly the sentence tokens.
        assert row["input_ids"][:len(sentence_ids)] == list(sentence_ids)
        # Label span covers the label tokens only, and they sit at the end.
        start, end = row["label_span"]
        assert row["input_ids"][start:end] == row["label_token_ids"]
        assert end == len(row["input_ids"])
        assert end - start == row["label_token_count"]
        # Positive example (label==1) maps to the first label string.
        expect = " positive" if row["label"] == 1 else " negative"
        assert row["label_str"] == expect


def test_corpus_carries_label_metadata(tokenizer) -> None:
    corpus = cp.sst2_prompt_corpus(tokenizer, "train", seed=0)
    for item in corpus.items:
        assert item.meta["corrupted"] is False
        start, end = item.meta["label_span"]
        assert item.input_ids[start:end] == item.meta["label_token_ids"]


def test_corruption_flips_label_token(tokenizer) -> None:
    plain = cp.sst2_prompt_corpus(tokenizer, "train", seed=0)
    target = plain.items[0]
    flipped = 1 - target.meta["label"]
    corrupted = cp.sst2_prompt_corpus(
        tokenizer, "train", seed=0,
        corrupted_manifest={target.meta["source_index"]: flipped})
    item = next(it for it in corrupted.items
                if it.meta["source_index"] == target.meta["source_index"])
    assert item.meta["corrupted"] is True
    assert item.meta["label"] == flipped
    assert item.meta["original_label"] == target.meta["label"]
    assert corrupted.manifest_sha256 != plain.manifest_sha256
