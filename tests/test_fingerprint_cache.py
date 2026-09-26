"""WP2: fingerprint cache behaviour (06 test_fingerprint_cache.py).

Cache hit on an identical key; recompute when any key field changes; cached and fresh
records are equal. This is what makes ``evaluate_transformation.py`` resumable at
(run_id, step) granularity (master plan §4.1). Offline: random gpt2 + synthetic corpus.
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

import corpus_providers as cp  # noqa: E402
import fingerprint_runner as fr  # noqa: E402

from golden_fingerprint import build_arch_engine  # noqa: E402


@pytest.fixture
def gpt2_handle_and_corpus():
    with tempfile.TemporaryDirectory(prefix="fpcache_", ignore_cleanup_errors=True) as temp:
        engine, tok = build_arch_engine("gpt2", Path(temp) / "gpt2")
        handle = fr.handle_from_module("gpt2", engine.hf, tok, model_name="tiny",
                                       checkpoint_step=0, checkpoint_sha256="deadbeef")
        corpus = cp.synthetic_corpus(tok, "random_uniform", 4, seed=2, cut_length=12)
        try:
            yield handle, corpus, Path(temp)
        finally:
            del engine, handle
            gc.collect()


def test_cache_hit_and_equal(gpt2_handle_and_corpus) -> None:
    handle, corpus, temp = gpt2_handle_and_corpus
    cache = temp / "cache"
    first = fr.compute_fingerprint(handle, corpus, cache_dir=cache)
    files = list(cache.rglob("fingerprint.json"))
    assert len(files) == 1

    mtime = files[0].stat().st_mtime_ns
    second = fr.compute_fingerprint(handle, corpus, cache_dir=cache)
    # A hit does not rewrite the file, and returns an equal record.
    assert files[0].stat().st_mtime_ns == mtime
    assert second.manifest_sha256 == first.manifest_sha256
    assert second.fingerprint == first.fingerprint
    assert second.raw_sink_by_intervention == first.raw_sink_by_intervention
    assert second.per_layer_sink == first.per_layer_sink


def test_key_change_recomputes(gpt2_handle_and_corpus) -> None:
    handle, corpus, temp = gpt2_handle_and_corpus
    cache = temp / "cache2"
    base = fr.compute_fingerprint(handle, corpus, cache_dir=cache)

    # A different intervention subset is a different cache key -> recompute, not the hit.
    subset = fr.compute_fingerprint(handle, corpus, cache_dir=cache,
                                    interventions=["int_a", "int_g"])
    assert set(subset.raw_sink_by_intervention) == {"int_a", "int_g"}
    assert set(base.raw_sink_by_intervention) != set(subset.raw_sink_by_intervention)

    # A different band is also a different key.
    banded = fr.compute_fingerprint(handle, corpus, cache_dir=cache, band=(0, 2))
    assert tuple(banded.band) == (0, 2)
    assert tuple(base.band) != (0, 2)


def test_two_corpora_do_not_overwrite_each_other(gpt2_handle_and_corpus) -> None:
    """One checkpoint, two corpora: two artefacts, and each still readable.

    The regression: the cache path was ``<run>/step_<n>/fingerprint.json``, with no corpus in
    it, while ``evaluate_transformation.py`` loops corpora *inside* one step. The second
    corpus therefore overwrote the first, and the whole first E6A pilot archive ended up
    holding one corpus' record for every checkpoint — including the teacher's, on the corpus
    every criterion is scored on. The ``_cache_key`` guard means the survivor was never *read*
    as the wrong corpus, so no number was wrong; the audit trail was what was destroyed
    (CLAUDE.md trap 24).
    """
    handle, corpus_a, temp = gpt2_handle_and_corpus
    tok = handle.tokenizer
    corpus_b = cp.synthetic_corpus(tok, "random_uniform", 6, seed=2, cut_length=12)
    assert corpus_a.corpus_id != corpus_b.corpus_id

    cache = temp / "cache_two_corpora"
    record_a = fr.compute_fingerprint(handle, corpus_a, cache_dir=cache)
    record_b = fr.compute_fingerprint(handle, corpus_b, cache_dir=cache)
    assert len(list(cache.rglob("fingerprint.json"))) == 2

    # The first corpus' record survived corpus B's write, and is still a hit.
    files = {path.parent.name: path for path in cache.rglob("fingerprint.json")}
    assert set(files) == {corpus_a.corpus_id, corpus_b.corpus_id}
    mtime = files[corpus_a.corpus_id].stat().st_mtime_ns
    again = fr.compute_fingerprint(handle, corpus_a, cache_dir=cache)
    assert files[corpus_a.corpus_id].stat().st_mtime_ns == mtime, "corpus A recomputed"
    assert again.corpus_id == corpus_a.corpus_id
    assert again.manifest_sha256 == record_a.manifest_sha256
    assert record_b.manifest_sha256 != record_a.manifest_sha256


def test_same_corpus_id_different_content_recomputes_rather_than_misreads(
        gpt2_handle_and_corpus) -> None:
    """The residual collision is a recompute, never a wrong record.

    ``corpus_id`` is the path component, so two corpora that share an id but not their
    content still share a file — ``synthetic_corpus`` names itself from kind and size, not
    from its seed. That is deliberate: the path stays readable, and the ``_cache_key`` check
    (which carries ``manifest_sha256``) makes the collision cost a recomputation instead of
    handing one corpus' fingerprint back for another's. Pinned so the limitation is recorded
    rather than rediscovered.
    """
    handle, corpus_a, temp = gpt2_handle_and_corpus
    other_seed = cp.synthetic_corpus(handle.tokenizer, "random_uniform", 4, seed=99,
                                     cut_length=12)
    assert other_seed.corpus_id == corpus_a.corpus_id
    assert other_seed.manifest_sha256 != corpus_a.manifest_sha256

    cache = temp / "cache_collision"
    first = fr.compute_fingerprint(handle, corpus_a, cache_dir=cache)
    second = fr.compute_fingerprint(handle, other_seed, cache_dir=cache)
    assert len(list(cache.rglob("fingerprint.json"))) == 1
    assert second.manifest_sha256 == other_seed.manifest_sha256, (
        "the cached record for a different corpus was returned; the manifest_sha256 guard "
        "in _cache_key is not being honoured")
    assert first.manifest_sha256 == corpus_a.manifest_sha256


def test_a_legacy_cache_file_is_still_read(gpt2_handle_and_corpus) -> None:
    """A pre-fix cache whose key matches is a valid record and must still hit.

    Read, never written: a re-run migrates the layout instead of duplicating it.
    """
    import json

    handle, corpus, temp = gpt2_handle_and_corpus
    cache = temp / "cache_legacy"
    fr.compute_fingerprint(handle, corpus, cache_dir=cache)

    new_path = next(cache.rglob("fingerprint.json"))
    legacy_path = new_path.parent.parent / "fingerprint.json"
    payload = json.loads(new_path.read_text(encoding="utf-8"))
    legacy_path.write_text(json.dumps(payload), encoding="utf-8")
    new_path.unlink()

    hit = fr.compute_fingerprint(handle, corpus, cache_dir=cache)
    assert hit.manifest_sha256 == payload["record"]["manifest_sha256"]
    assert not new_path.exists(), "a legacy hit must not rewrite the record"


def test_round_trip_serialisation(gpt2_handle_and_corpus) -> None:
    handle, corpus, temp = gpt2_handle_and_corpus
    rec = fr.compute_fingerprint(handle, corpus)
    restored = fr.FingerprintRecord.from_json(rec.to_json())
    assert restored.fingerprint == rec.fingerprint
    assert tuple(restored.band) == tuple(rec.band)
    assert restored.top_carrier_heads == rec.top_carrier_heads
