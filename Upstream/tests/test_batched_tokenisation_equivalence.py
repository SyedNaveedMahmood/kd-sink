# -*- coding: utf-8 -*-
"""test_batched_tokenisation_equivalence.py — batching the tokenizer packs the same corpus.

``_pack_documents`` tokenised one document per call, which is ~17-30 minutes of
single-threaded work for the E6A-GPT2 training window (400,000 OpenWebText documents), paid
on every run and every resume. ``tokenizer_batch_size`` groups documents into one call so a
HuggingFace *fast* tokenizer can use its Rust thread pool.

That is only safe if it is *proved* to pack identically. ``manifest_sha256`` is the join key
for every cached artefact in the project (``05`` §7.1); a batched path that moved it by one
byte would not crash — it would silently partition results into two incomparable sets
(CLAUDE.md trap 19). So these tests drive the same documents through both paths and assert
byte-identical blocks, manifests and digests, including on the **real gpt2 tokenizer**,
because the stub cannot exercise the property that actually matters: that a fast
tokenizer's per-sequence output does not depend on what else is in its batch.

No downloads: the gpt2 tokenizer comes from the repo's pinned ``.hf_cache``, and the test
skips rather than fails if it is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import datasets_loader as dl  # noqa: E402
import provenance as prov  # noqa: E402


class _Tok:
    """Deterministic whitespace tokenizer that accepts a str *or* a list of str.

    The same recipe as ``test_block_packing_equivalence._Tok`` (no ``hash()``, so it is
    stable across runs), extended with the batch call form the new path uses.
    """

    eos_token_id = 2

    def _one(self, text):
        return [(sum(ord(c) for c in w) % 5000) + 3 for w in text.split()]

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, (list, tuple)):
            return {"input_ids": [self._one(t) for t in text]}
        return {"input_ids": self._one(text)}


# Uneven lengths on purpose: equal-length documents would hide a batching bug that
# mis-associates ids with documents.
DOCUMENTS = [{"text": " ".join(f"w{i}_{j}" for j in range(3 + (i * 7) % 41))}
             for i in range(400)]
# Empty and whitespace-only documents are skipped *before* the tokenizer sees them, so they
# must not shift the alignment between a batch's texts and its returned id lists.
DOCUMENTS[5] = {"text": ""}
DOCUMENTS[6] = {"text": "   "}
DOCUMENTS[311] = {"text": ""}


@pytest.fixture
def documents(monkeypatch):
    monkeypatch.setattr(dl, "_load_openwebtext", lambda split, **kwargs: list(DOCUMENTS))
    return _Tok()


def _pack(tok, batch, **kwargs):
    return dl.load_openwebtext_blocks(
        tok, "train", block_size=16, seed=0, tokenizer_batch_size=batch, **kwargs)


def _assert_identical(left, right, label):
    l_blocks, l_manifest = left
    r_blocks, r_manifest = right
    assert len(l_blocks) == len(r_blocks), f"{label}: block count differs"
    assert np.array_equal(np.asarray(l_blocks), np.asarray(r_blocks)), \
        f"{label}: block CONTENTS differ"
    assert l_manifest == r_manifest, f"{label}: manifest rows differ"
    assert prov.sha256_int_rows(l_blocks) == prov.sha256_int_rows(r_blocks), \
        f"{label}: manifest_sha256 moved"


@pytest.mark.parametrize("batch", [1, 2, 7, 64, 1_000, 100_000])
def test_batched_packing_equals_unbatched_at_every_batch_size(documents, batch):
    """Any grouping, including one bigger than the corpus, packs what no grouping packs."""
    _assert_identical(_pack(documents, None, as_array=True, manifest_input_ids=False),
                      _pack(documents, batch, as_array=True, manifest_input_ids=False),
                      f"batch={batch}")


@pytest.mark.parametrize("n_blocks", [1, 5, 37, None])
def test_the_n_blocks_cap_stops_at_the_same_place(documents, n_blocks):
    """A batch may tokenise past the cap; it must not *emit* past it.

    This is the one place the batched path does measurably more work than the original, so
    it is the one place the output could quietly diverge.
    """
    _assert_identical(
        _pack(documents, None, n_blocks=n_blocks, as_array=True, manifest_input_ids=False),
        _pack(documents, 7, n_blocks=n_blocks, as_array=True, manifest_input_ids=False),
        f"n_blocks={n_blocks}")


def test_the_list_form_and_manifest_ids_agree_too(documents):
    """The other flag combination — list blocks with duplicated manifest ids."""
    _assert_identical(_pack(documents, None), _pack(documents, 7), "list form")


def test_a_zero_or_negative_batch_is_refused(documents):
    """Refused, never folded into the default: a `0` from a config typo must not silently
    become "one document at a time" (CLAUDE.md trap 13)."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="tokenizer_batch_size"):
            _pack(documents, bad)


def test_a_misaligned_batch_is_refused(documents, monkeypatch):
    """If a tokenizer ever returned a different number of sequences than it was given,
    zipping them would pair ids with the wrong documents and produce a plausible corpus
    that is not the registered one. That must raise, not pack."""

    class _Dropping(_Tok):
        def __call__(self, text, add_special_tokens=False):
            out = super().__call__(text, add_special_tokens=add_special_tokens)
            if isinstance(text, (list, tuple)) and len(out["input_ids"]) > 1:
                out["input_ids"] = out["input_ids"][:-1]
            return out

    with pytest.raises(ValueError, match="misaligned"):
        _pack(_Dropping(), 7)


def test_tinystories_loader_does_not_accept_the_flag():
    """Deliberate asymmetry, asserted so it cannot be "tidied up" later.

    The TinyStories arm's packed records are already on disk and joined on by digest; the
    only safe number of ways to produce one of those is one.
    """
    with pytest.raises(TypeError):
        dl.load_tinystories_blocks(_Tok(), "train", tokenizer_batch_size=7)


# ═══════════════════════════════════════════════════════════════════════════════
# The real tokenizer — the stub cannot exercise batch-dependence
# ═══════════════════════════════════════════════════════════════════════════════


REAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "  leading and trailing whitespace  ",
    "Unicode: naïve café — em-dash, ellipsis… and an emoji 🙂",
    "a",
    "Numbers 1234567890 and punctuation !?;:'\"[]{}",
    "A much longer paragraph, of the sort OpenWebText is full of, with several clauses, "
    "subordinate structure, and enough tokens that it spans more than one packed block "
    "once the block size is small. " * 6,
    "tabs\tand\nnewlines\r\nmixed together",
    "Repeated repeated repeated repeated words words words words",
]


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    import os

    os.environ.setdefault("HF_HOME", str(REPO / ".hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception as exc:                      # pragma: no cover - environment-dependent
        pytest.skip(f"gpt2 tokenizer unavailable offline: {type(exc).__name__}: {exc}")


def test_the_real_gpt2_tokenizer_batches_identically(gpt2_tokenizer, monkeypatch):
    """The assertion the stub cannot make.

    A fast tokenizer *could* in principle behave differently in a batch — padding,
    truncation, or a shared state — and if it did, every block boundary after the first
    difference would move. Nothing here requests padding or truncation, and this proves it
    on the actual tokenizer both E6A-GPT2 arms use.
    """
    documents = [{"text": t} for t in REAL_TEXTS] * 12
    monkeypatch.setattr(dl, "_load_openwebtext", lambda split, **kwargs: list(documents))
    for batch in (1, 3, 1_000):
        _assert_identical(
            dl.load_openwebtext_blocks(gpt2_tokenizer, "train", block_size=32, seed=0,
                                       as_array=True, manifest_input_ids=False),
            dl.load_openwebtext_blocks(gpt2_tokenizer, "train", block_size=32, seed=0,
                                       as_array=True, manifest_input_ids=False,
                                       tokenizer_batch_size=batch),
            f"real gpt2, batch={batch}")


def test_the_real_tokenizer_gives_the_same_ids_per_document_in_a_batch(gpt2_tokenizer):
    """The property underneath the one above, isolated so a failure is diagnosable."""
    one_at_a_time = [gpt2_tokenizer(t, add_special_tokens=False)["input_ids"]
                     for t in REAL_TEXTS]
    batched = gpt2_tokenizer(REAL_TEXTS, add_special_tokens=False)["input_ids"]
    assert batched == one_at_a_time
