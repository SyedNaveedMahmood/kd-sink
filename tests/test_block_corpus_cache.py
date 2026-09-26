# -*- coding: utf-8 -*-
"""test_block_corpus_cache.py — a cache hit is the cold pack, or it refuses.

``block_corpus_cache`` exists because packing a training corpus is deterministic and
expensive: 400,000 OpenWebText documents for the E6A-GPT2 arm, and reaching its *validation*
window means streaming past all 400,000 training documents first — a cost paid again on
every run, every resume and every evaluation.

A corpus cache is also the most dangerous kind of optimisation in this project, because its
failure mode is silence: `manifest_sha256` is the join key for every cached artefact
(``05`` §7.1), so serving the wrong or a truncated corpus would not crash, it would produce
a second incomparable set of results (CLAUDE.md trap 19). These tests therefore pin the
refusals as hard as the hits.

No downloads: the loader is monkeypatched onto a synthetic document window.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import block_corpus_cache as bcc  # noqa: E402
import corpus_providers as cp  # noqa: E402
import datasets_loader as dl  # noqa: E402


class _Tok:
    eos_token_id = 2
    vocab_size = 5003
    name_or_path = "stub"

    def __len__(self):
        return 5003

    def _one(self, text):
        return [(sum(ord(c) for c in w) % 5000) + 3 for w in text.split()]

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, (list, tuple)):
            return {"input_ids": [self._one(t) for t in text]}
        return {"input_ids": self._one(text)}

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


DOCUMENTS = [{"text": " ".join(f"w{i}_{j}" for j in range(3 + (i * 7) % 41))}
             for i in range(1200)]


@pytest.fixture
def documents(monkeypatch):
    monkeypatch.setattr(dl, "_load_openwebtext", lambda split, **kwargs: list(DOCUMENTS))
    return _Tok()


@pytest.fixture
def cache_root():
    with tempfile.TemporaryDirectory(prefix="corpus_cache_",
                                     ignore_cleanup_errors=True) as tmp:
        yield Path(tmp)


def _pack(tok, cache_root, n_blocks=None, **kwargs):
    return bcc.packed_blocks(
        dl.load_openwebtext_blocks, tok, "train", dataset="Skylion007/openwebtext",
        block_size=16, seed=0, eos_between=True, n_blocks=n_blocks,
        cache_root=cache_root, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Hits
# ═══════════════════════════════════════════════════════════════════════════════


def test_a_warm_hit_reproduces_the_cold_pack(documents, cache_root):
    cold_blocks, cold_manifest, cold_digest, cold_info = _pack(documents, cache_root)
    warm_blocks, warm_manifest, warm_digest, warm_info = _pack(documents, cache_root)

    assert cold_info["cache_hit"] is False
    assert warm_info["cache_hit"] is True
    assert warm_info["cache_key"] == cold_info["cache_key"]
    assert np.array_equal(np.asarray(cold_blocks), np.asarray(warm_blocks))
    assert warm_manifest == cold_manifest
    assert warm_digest == cold_digest


def test_the_cache_is_bypassed_entirely_without_a_root(documents):
    """`cache_root=None` is what every caller did before this module existed."""
    blocks, manifest, digest, info = _pack(documents, None)
    direct_blocks, direct_manifest = dl.load_openwebtext_blocks(
        documents, "train", block_size=16, seed=0, eos_between=True,
        as_array=True, manifest_input_ids=False)
    assert info == {"cache_hit": False, "cache_key": None, "cache_path": None,
                    "cache_root": None}
    assert np.array_equal(np.asarray(blocks), np.asarray(direct_blocks))
    assert manifest == direct_manifest


def test_a_smaller_request_is_served_as_a_prefix_of_a_larger_pack(documents, cache_root):
    """`n_blocks` is deliberately NOT in the key: packing is a prefix operation, so one
    full pack serves every smaller request. If it were in the key, the sink (300) and ppl
    (2,300) corpora would each re-stream the whole document window."""
    full_blocks, full_manifest, _digest, _info = _pack(documents, cache_root)
    assert len(full_blocks) > 10

    sliced_blocks, sliced_manifest, sliced_digest, info = _pack(documents, cache_root,
                                                               n_blocks=10)
    assert info["cache_hit"] is True
    assert np.array_equal(np.asarray(sliced_blocks), np.asarray(full_blocks[:10]))
    assert sliced_manifest == full_manifest[:10]

    # ...and the prefix is what a direct capped pack produces, digest included.
    direct_blocks, direct_manifest, direct_digest, _ = _pack(documents, None, n_blocks=10)
    assert np.array_equal(np.asarray(sliced_blocks), np.asarray(direct_blocks))
    assert sliced_manifest == direct_manifest
    assert sliced_digest == direct_digest


def test_a_shorter_entry_is_a_miss_not_a_truncated_hit(documents, cache_root):
    """Serving 10 blocks when 40 were asked for would be a silently short corpus."""
    _pack(documents, cache_root, n_blocks=10)
    blocks, _manifest, _digest, info = _pack(documents, cache_root, n_blocks=40)
    assert info["cache_hit"] is False
    assert len(blocks) == 40


# ═══════════════════════════════════════════════════════════════════════════════
# The key
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("field,value", [
    ("dataset", "roneneldan/TinyStories"),
    ("split", "validation"),
    ("block_size", 32),
    ("seed", 1),
    ("eos_between", False),
])
def test_every_key_field_changes_the_key(field, value):
    base = dict(dataset="Skylion007/openwebtext", split="train", block_size=16, seed=0,
                eos_between=True, loader_kwargs={"train_documents": 100},
                tokenizer=_Tok())
    changed = {**base, field: value}
    assert bcc.cache_key(**base) != bcc.cache_key(**changed), \
        f"{field} does not participate in the cache key"


def test_loader_kwargs_participate_and_are_order_insensitive():
    """A different document window is a different corpus; a different dict ordering is not."""
    tok = _Tok()
    common = dict(dataset="Skylion007/openwebtext", split="train", block_size=16, seed=0,
                  eos_between=True, tokenizer=tok)
    a = bcc.cache_key(**common, loader_kwargs={"train_documents": 100,
                                               "validation_documents": 8})
    b = bcc.cache_key(**common, loader_kwargs={"validation_documents": 8,
                                               "train_documents": 100})
    c = bcc.cache_key(**common, loader_kwargs={"train_documents": 200,
                                               "validation_documents": 8})
    assert a == b
    assert a != c


def test_a_performance_only_kwarg_does_NOT_change_the_key():
    """`tokenizer_batch_size` changes how fast packing runs and nothing about its output
    (proved in test_batched_tokenisation_equivalence.py). If it were in the key, the
    trainer — which batches because it packs ~3.1M blocks — and a corpus provider — which
    does not, because it packs ~2,300 — would each build their own entry, and each would
    re-stream the whole 400,000-document window to do it."""
    common = dict(dataset="Skylion007/openwebtext", split="train", block_size=16, seed=0,
                  eos_between=True, tokenizer=_Tok())
    plain = bcc.cache_key(**common, loader_kwargs={"train_documents": 100})
    batched = bcc.cache_key(**common, loader_kwargs={"train_documents": 100,
                                                     "tokenizer_batch_size": 1000})
    other_batch = bcc.cache_key(**common, loader_kwargs={"train_documents": 100,
                                                         "tokenizer_batch_size": 7})
    assert plain == batched == other_batch


def test_a_batched_pack_and_an_unbatched_pack_share_one_entry(documents, cache_root):
    """The end-to-end consequence: the trainer writes it, a provider reads it."""
    _b, _m, cold_digest, cold = _pack(documents, cache_root,
                                      loader_kwargs={"tokenizer_batch_size": 7})
    _b2, _m2, warm_digest, warm = _pack(documents, cache_root)
    assert cold["cache_hit"] is False
    assert warm["cache_hit"] is True
    assert warm["cache_key"] == cold["cache_key"]
    assert warm_digest == cold_digest


def test_a_different_tokenizer_changes_the_key():
    class _Other(_Tok):
        name_or_path = "other"

    common = dict(dataset="Skylion007/openwebtext", split="train", block_size=16, seed=0,
                  eos_between=True, loader_kwargs={})
    assert bcc.cache_key(**common, tokenizer=_Tok()) != \
        bcc.cache_key(**common, tokenizer=_Other())


# ═══════════════════════════════════════════════════════════════════════════════
# Refusals — the reason the digest is re-derived on every read
# ═══════════════════════════════════════════════════════════════════════════════


def test_a_corrupted_block_file_raises_rather_than_being_served(documents, cache_root):
    """The whole point. A cache that repacks silently over a digest mismatch would hide
    exactly the failure the verification exists to catch."""
    _blocks, _manifest, _digest, info = _pack(documents, cache_root)
    entry = Path(info["cache_path"])

    corrupted = np.load(entry / "blocks.npy")
    corrupted[0, 0] = int(corrupted[0, 0]) + 1
    np.save(entry / "blocks.npy", corrupted)

    with pytest.raises(bcc.CorpusCacheError, match="hash to"):
        _pack(documents, cache_root)


def test_a_missing_manifest_beside_present_blocks_raises(documents, cache_root):
    _b, _m, _d, info = _pack(documents, cache_root)
    entry = Path(info["cache_path"])
    for stale in list(entry.glob("manifest.*")):
        stale.unlink()
    with pytest.raises(bcc.CorpusCacheError, match="manifest"):
        _pack(documents, cache_root)


def test_a_stale_format_version_is_a_miss_not_an_error(documents, cache_root):
    """An old layout must repack, not raise: the entry is not corrupt, only obsolete."""
    _b, _m, _d, info = _pack(documents, cache_root)
    meta_path = Path(info["cache_path"]) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["format"] = "block_corpus_cache_v0"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    _b2, _m2, _d2, info2 = _pack(documents, cache_root)
    assert info2["cache_hit"] is False


def test_a_partial_write_is_never_visible_as_a_hit(documents, cache_root):
    """Entries are staged and renamed, so an interrupted write leaves a `.partial`
    directory that no reader looks at."""
    _b, _m, _d, info = _pack(documents, cache_root)
    entry = Path(info["cache_path"])
    staging = entry.with_name(entry.name + ".partial")
    staging.mkdir()
    (staging / "blocks.npy").write_bytes(b"truncated garbage")
    _b2, _m2, _d2, info2 = _pack(documents, cache_root)
    assert info2["cache_hit"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# The provider seam — the corpus digest must not depend on the cache
# ═══════════════════════════════════════════════════════════════════════════════


def test_openwebtext_corpus_is_identical_with_and_without_the_cache(documents, cache_root):
    """`Corpus.manifest_sha256` is the project's proof that two runs saw the same inputs.
    A cache hit returns int32 numpy rows where a cold pack returns Python lists, so this
    pins that the normalisation in the provider makes the two indistinguishable."""
    uncached = cp.openwebtext_corpus(documents, "train", 12, block_size=16, seed=0,
                                     purpose="sink")
    cold = cp.openwebtext_corpus(documents, "train", 12, block_size=16, seed=0,
                                 purpose="sink", cache_root=cache_root)
    warm = cp.openwebtext_corpus(documents, "train", 12, block_size=16, seed=0,
                                 purpose="sink", cache_root=cache_root)

    assert cold.manifest_sha256 == uncached.manifest_sha256
    assert warm.manifest_sha256 == uncached.manifest_sha256
    assert [i.input_ids for i in warm.items] == [i.input_ids for i in uncached.items]
    assert [i.text for i in warm.items] == [i.text for i in uncached.items]
    assert [i.item_id for i in warm.items] == [i.item_id for i in uncached.items]
    assert [i.meta for i in warm.items] == [i.meta for i in uncached.items]


def test_the_sink_and_ppl_windows_stay_disjoint_through_the_cache(documents, cache_root):
    """The disjointness in `02` §2.3 is a slicing property of one seeded block stream, and
    prefix-serving must not disturb it."""
    sink = cp.openwebtext_corpus(documents, "train", 20, block_size=16, seed=0,
                                 purpose="sink", cache_root=cache_root)
    ppl = cp.openwebtext_corpus(documents, "train", 20, block_size=16, seed=0,
                                purpose="ppl", cache_root=cache_root)
    sink_indices = {i.meta["block_index"] for i in sink.items}
    ppl_indices = {i.meta["block_index"] for i in ppl.items}
    assert sink_indices.isdisjoint(ppl_indices)
    assert min(ppl_indices) >= cp.SINK_BLOCKS_RESERVED
