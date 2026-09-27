# -*- coding: utf-8 -*-
"""test_block_corpus_memory.py — the corpus fits in RAM, and no number moved.

The E6A trainer used to hold the packed TinyStories corpus twice as Python objects (once
in ``blocks``, once again as ``manifest[i]["input_ids"]``) and hash it through
``prov.sha256_json``. Measured at 5,630 bytes per block, the train split's ~3.57M blocks
came to ~20 GB resident, with an ~8.8 GB spike and ~11.7 minutes for the digest on top —
peak ~29 GB against a 16 GB host. Only the 5-step ``--smoke`` path had ever run, so the
suite was green and E6A had never actually started.

The fix is a **representation** change. These tests exist to prove it is *only* that: the
same tokens, in the same order, with the same ``manifest_sha256`` and the same parquet.
A memory optimisation that quietly repacked the corpus would invalidate every downstream
comparison while looking like a speedup.

Uses a synthetic tokenizer and a stub dataset — no downloads.
"""

from __future__ import annotations

import json
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
import train_distillation as td  # noqa: E402


class _Tok:
    """Whitespace tokenizer with a stable id per token; enough to pack blocks."""

    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [(abs(hash(w)) % 5000) + 3 for w in text.split()]}


@pytest.fixture
def stub_split(monkeypatch):
    """A deterministic stand-in for the TinyStories split."""
    stories = [{"text": " ".join(f"w{i}_{j}" for j in range(37))} for i in range(400)]
    monkeypatch.setattr(dl, "_load_tinystories", lambda split: stories)
    return stories


def _load(**kwargs):
    return dl.load_tinystories_blocks(_Tok(), "train", block_size=16, seed=0, **kwargs)


def test_array_form_holds_exactly_the_same_tokens(stub_split):
    lists, manifest_a = _load()
    array, manifest_b = _load(as_array=True)

    assert isinstance(array, np.ndarray) and array.dtype == np.int32
    assert array.shape == (len(lists), 16)
    assert array.tolist() == lists, "the array form must be the list form, element for element"
    assert [r["block_index"] for r in manifest_a] == [r["block_index"] for r in manifest_b]


def test_manifest_sha256_is_unchanged_by_the_representation(stub_split):
    """The digest is an identity, not an implementation detail (`05` §7.1)."""
    lists, _ = _load()
    array, _ = _load(as_array=True)

    old_way = prov.sha256_json([[int(t) for t in b] for b in lists])
    new_way = prov.sha256_int_rows(array)
    assert new_way == old_way


def test_dropping_the_duplicate_ids_keeps_every_other_manifest_field(stub_split):
    with_ids, without_ids = _load()[1], _load(manifest_input_ids=False)[1]
    assert len(with_ids) == len(without_ids)
    for a, b in zip(with_ids, without_ids):
        assert "input_ids" in a and "input_ids" not in b
        for key in ("block_index", "source_story_indices", "n_tokens", "n_eos"):
            assert a[key] == b[key], key


def test_defaults_reproduce_the_original_behaviour_exactly(stub_split):
    """Additive-only discipline: an unflagged call must be byte-identical to before."""
    blocks, manifest = _load()
    assert isinstance(blocks, list) and isinstance(blocks[0], list)
    assert all("input_ids" in row for row in manifest)
    assert all(row["input_ids"] == blocks[row["block_index"]] for row in manifest)


def test_the_manifest_rows_do_not_alias_the_blocks(stub_split):
    """Mutating a block must not rewrite the record of what was packed."""
    blocks, manifest = _load()
    blocks[0][0] = 99999
    assert manifest[0]["input_ids"][0] != 99999


def test_folding_happens_during_packing_not_at_the_end(stub_split, monkeypatch):
    """Converting to int32 *after* building the list would save nothing.

    The peak this fix exists to remove IS the list-of-lists. A first attempt at the fix
    accumulated Python lists and called ``np.asarray`` at the end, which measured 10.4 GB
    on the real split — no better than before. Blocks are now folded into int32 chunks as
    they are produced, so this pins that the fold is incremental: with a fold size of 3 the
    result must still be correct, which it cannot be if indices or ordering depend on when
    the fold happens.
    """
    monkeypatch.setattr(dl, "_ARRAY_FOLD_ROWS", 3)
    folded, manifest = _load(as_array=True)
    lists, _ = _load()

    assert folded.tolist() == lists
    assert [row["block_index"] for row in manifest] == list(range(len(lists))), (
        "block_index must count every block, not just the ones still pending a fold")


@pytest.mark.parametrize("fold", [1, 3, 7, 10**6])
def test_the_fold_size_changes_nothing_observable(stub_split, monkeypatch, fold):
    monkeypatch.setattr(dl, "_ARRAY_FOLD_ROWS", fold)
    blocks, manifest = _load(as_array=True, manifest_input_ids=False)
    reference, _ = _load()
    assert blocks.tolist() == reference
    assert prov.sha256_int_rows(blocks) == prov.sha256_json(reference)
    assert [r["block_index"] for r in manifest] == list(range(len(reference)))


@pytest.mark.parametrize("cap", [1, 5, 12])
def test_n_blocks_caps_identically_in_both_representations(stub_split, monkeypatch, cap):
    """The cap interacts with folding — an off-by-one here silently shortens the corpus."""
    monkeypatch.setattr(dl, "_ARRAY_FOLD_ROWS", 3)
    array, manifest_a = _load(as_array=True, n_blocks=cap)
    lists, manifest_b = _load(n_blocks=cap)
    assert len(array) == len(lists) == cap
    assert array.tolist() == lists
    assert len(manifest_a) == len(manifest_b) == cap


def test_build_block_dataset_uses_the_array_path_and_batches_correctly(stub_split):
    dataset = td.build_block_dataset(_Tok(), "train", block_size=16, seed=0)
    assert isinstance(dataset.blocks, np.ndarray)
    assert not any("input_ids" in row for row in dataset.manifest)

    lists, _ = _load()
    assert dataset.manifest_sha256 == prov.sha256_json([[int(t) for t in b] for b in lists])

    import torch

    batch = dataset.batch([3, 1, 0], device="cpu")
    assert batch.dtype == torch.long and batch.shape == (3, 16)
    assert batch.tolist() == [lists[3], lists[1], lists[0]]


def test_the_written_manifest_still_carries_input_ids(stub_split, tmp_path):
    """`05` §4's schema is unchanged — only the in-memory duplication went away."""
    import pandas as pd

    dataset = td.build_block_dataset(_Tok(), "train", block_size=16, seed=0)
    written = td.write_block_manifest(dataset, tmp_path / "block_manifest.parquet")

    frame = (pd.read_parquet(written) if written.suffix == ".parquet"
             else pd.read_csv(written))
    assert list(frame.columns) == ["block_index", "input_ids", "source_story_indices",
                                   "n_tokens", "n_eos"]
    assert len(frame) == len(dataset.blocks)
    lists, _ = _load()
    assert json.loads(frame.iloc[0]["input_ids"]) == lists[0]
    assert json.loads(frame.iloc[-1]["input_ids"]) == lists[len(lists) - 1]


def test_the_manifest_is_written_in_chunks(stub_split, tmp_path, monkeypatch):
    """Chunking must not change the file — assert equality against a single-chunk write."""
    import pandas as pd

    dataset = td.build_block_dataset(_Tok(), "train", block_size=16, seed=0)
    assert len(dataset.blocks) > 3, "fixture must exceed the chunk size below"

    one_shot = td.write_block_manifest(dataset, tmp_path / "one.parquet")
    monkeypatch.setattr(td, "MANIFEST_CHUNK_ROWS", 3)
    chunked = td.write_block_manifest(dataset, tmp_path / "many.parquet")

    left = pd.read_parquet(one_shot) if one_shot.suffix == ".parquet" \
        else pd.read_csv(one_shot)
    right = pd.read_parquet(chunked) if chunked.suffix == ".parquet" \
        else pd.read_csv(chunked)
    pd.testing.assert_frame_equal(left, right)
