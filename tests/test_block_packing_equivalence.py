# -*- coding: utf-8 -*-
"""test_block_packing_equivalence.py — the OpenWebText loader packs identically.

The E6A-GPT2 arm needs a second training corpus, and ``_pack_documents`` is a **faithful
copy** of ``load_tinystories_blocks``'s packing loop rather than a refactor of it. The copy
is deliberate: ``manifest_sha256`` is the join key for every cached artefact in the project
(``05`` §7.1), the TinyStories arm's records are already on disk, and an edit inside that
function that moved the digest by one byte would silently partition them into two
incomparable sets (CLAUDE.md trap 19) — a failure that looks like nothing at all.

A copy is only safe if it is *proved* to be one. These tests drive both loaders over the
**same** synthetic documents and assert byte-identical blocks, manifests and digests, so a
divergence between the two implementations fails here rather than in a results table six
GPU-days later. No downloads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import datasets_loader as dl  # noqa: E402
import provenance as prov  # noqa: E402


class _Tok:
    """Deterministic whitespace tokenizer — no ``hash()``, so it is stable across runs."""

    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [(sum(ord(c) for c in w) % 5000) + 3 for w in text.split()]}

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


DOCUMENTS = [{"text": " ".join(f"w{i}_{j}" for j in range(37))} for i in range(400)]


@pytest.fixture
def both_loaders(monkeypatch):
    """Point *both* dataset readers at the same documents, so only the code differs."""
    monkeypatch.setattr(dl, "_load_tinystories", lambda split: list(DOCUMENTS))
    monkeypatch.setattr(dl, "_load_openwebtext",
                        lambda split, **kwargs: list(DOCUMENTS))
    return _Tok()


def _pair(tok, **kwargs):
    left = dl.load_tinystories_blocks(tok, "train", block_size=16, seed=0, **kwargs)
    right = dl.load_openwebtext_blocks(tok, "train", block_size=16, seed=0, **kwargs)
    return left, right


def test_the_two_loaders_pack_the_same_tokens_in_the_same_order(both_loaders):
    (blocks_a, manifest_a), (blocks_b, manifest_b) = _pair(both_loaders)
    assert len(blocks_a) > 10, "fixture must produce enough blocks to be meaningful"
    assert blocks_a == blocks_b
    assert manifest_a == manifest_b


def test_the_digest_is_identical(both_loaders):
    """The property that actually matters: `manifest_sha256` must not depend on the loader."""
    (blocks_a, _), (blocks_b, _) = _pair(both_loaders)
    assert prov.sha256_json(blocks_a) == prov.sha256_json(blocks_b)
    assert prov.sha256_int_rows(blocks_a) == prov.sha256_int_rows(blocks_b)


@pytest.mark.parametrize("kwargs", [
    {"as_array": True},
    {"manifest_input_ids": False},
    {"as_array": True, "manifest_input_ids": False},
    {"eos_between": False},
    {"n_blocks": 7},
])
def test_every_flag_behaves_identically_in_both(both_loaders, kwargs):
    """The memory flags and the cap are part of the contract, not incidental options."""
    (blocks_a, manifest_a), (blocks_b, manifest_b) = _pair(both_loaders, **kwargs)
    if isinstance(blocks_a, np.ndarray):
        assert isinstance(blocks_b, np.ndarray)
        assert blocks_a.dtype == blocks_b.dtype == np.int32
        assert np.array_equal(blocks_a, blocks_b)
    else:
        assert blocks_a == blocks_b
    assert manifest_a == manifest_b


def test_the_array_fold_size_changes_nothing_observable(both_loaders, monkeypatch):
    """Folding is a memory strategy; it must not touch a token or a manifest row."""
    monkeypatch.setattr(dl, "_ARRAY_FOLD_ROWS", 3)
    folded_a, folded_b = _pair(both_loaders, as_array=True)
    monkeypatch.setattr(dl, "_ARRAY_FOLD_ROWS", 50_000)
    whole_a, whole_b = _pair(both_loaders, as_array=True)

    assert np.array_equal(folded_a[0], whole_a[0])
    assert np.array_equal(folded_b[0], whole_b[0])
    assert np.array_equal(folded_a[0], folded_b[0])


def test_the_seed_orders_documents_before_packing(both_loaders):
    """Design §1.4: the shuffle is on document order and happens *before* packing."""
    seed_0 = dl.load_openwebtext_blocks(both_loaders, "train", block_size=16, seed=0)[0]
    seed_1 = dl.load_openwebtext_blocks(both_loaders, "train", block_size=16, seed=1)[0]
    assert seed_0 != seed_1, "a different seed must produce a different block sequence"

    again = dl.load_openwebtext_blocks(both_loaders, "train", block_size=16, seed=0)[0]
    assert again == seed_0, "the same seed must reproduce the same block sequence"


def test_a_block_size_below_two_is_refused(both_loaders):
    with pytest.raises(ValueError, match="block_size"):
        dl.load_openwebtext_blocks(both_loaders, "train", block_size=1)


def test_eos_between_requires_an_eos_token(both_loaders):
    class _NoEos(_Tok):
        eos_token_id = None

    with pytest.raises(ValueError, match="eos_token_id"):
        dl.load_openwebtext_blocks(_NoEos(), "train", block_size=16, eos_between=True)
