# -*- coding: utf-8 -*-
"""test_parallel_manifest_alignment.py — the cross-language join is not positional (WP8).

``06_TEST_PLAN.md`` row 33 and ``02_MODULE_SPEC_common.md`` §5.2. This is the test that
catches CLAUDE.md trap 2: **XNLI must be joined on ``promptID``, never on row index.** A
positional join produces a manifest that looks entirely healthy and silently destroys every
cross-language claim, so the fixture deliberately emits each language's shard in a
different order and the test asserts gold labels agree across all seven languages for every
semantic id.

FLORES is different and the test says so explicitly: ``devtest`` really is line-aligned
across configs, so a source-index join is correct there — and the loader asserts the row
counts match before relying on it.

Offline: ``datasets_loader.load_dataset`` is monkeypatched, matching the pattern in
``test_tinystories_packing.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus_providers as cp  # noqa: E402
import datasets_loader as dl  # noqa: E402
import paired_manifests as pm  # noqa: E402

import wp8_fake_data as fake  # noqa: E402


@pytest.fixture()
def tokenizer():
    return fake.tiny_tokenizer(None)


def _patch(monkeypatch, **kwargs):
    monkeypatch.setattr(dl, "load_dataset", fake.fake_load_dataset(**kwargs))


# ═══════════════════════════════════════════════════════════════════════════════
# XNLI — the join key
# ═══════════════════════════════════════════════════════════════════════════════


def test_xnli_joins_on_prompt_id_not_row_index(monkeypatch, tokenizer):
    """Gold labels must agree across all seven languages for every semantic id."""
    _patch(monkeypatch, xnli=fake.make_xnli())
    manifest = pm.build_xnli_manifest(tokenizer, n=12, seed=42, max_tokens=200)

    assert manifest.provenance["loader"]["join_strategy"] == "promptID"
    assert manifest.provenance["loader"]["join_key_column"] == "promptID"
    assert len(manifest.semantic_ids) > 0

    for sid in manifest.semantic_ids:
        labels = {manifest.rows[sid][lang]["gold_label"] for lang in manifest.languages}
        assert len(labels) == 1, (sid, labels)

    # verify_manifest is the mechanical form of the same assertion.
    report = pm.verify_manifest(manifest)
    assert report["n_semantic_ids"] == len(manifest.semantic_ids)
    assert set(report["languages"]) == set(fake.XNLI_LANGS)


def test_a_positional_join_would_have_been_caught(monkeypatch, tokenizer):
    """The fixture is genuinely order-scrambled, so the test above has teeth.

    Without this, a fixture that happened to be in the same order everywhere would make
    ``test_xnli_joins_on_prompt_id_not_row_index`` pass for a positionally-joining
    implementation too.
    """
    shards = fake.make_xnli()
    reference = [row["promptID"] for row in shards["en"]]
    differing = [lang for lang in fake.XNLI_LANGS
                 if [row["promptID"] for row in shards[lang]] != reference]
    assert len(differing) >= len(fake.XNLI_LANGS) - 1, differing

    # And a positional read really does mismatch the gold labels.
    positional_disagreements = 0
    for i in range(len(shards["en"])):
        labels = {shards[lang][i]["label"] for lang in fake.XNLI_LANGS}
        if len(labels) > 1:
            positional_disagreements += 1
    assert positional_disagreements > 0


def test_xnli_falls_back_to_structural_alignment_never_to_position(monkeypatch, tokenizer):
    """Without an id column, the structurally aligned ``all_languages`` config is used."""
    _patch(monkeypatch,
           xnli=fake.make_xnli(with_prompt_id=False),
           xnli_all=fake.make_xnli_all_languages())
    manifest = pm.build_xnli_manifest(tokenizer, n=12, seed=42, max_tokens=200)

    assert manifest.provenance["loader"]["join_strategy"] == "all_languages_structural"
    for sid in manifest.semantic_ids:
        labels = {manifest.rows[sid][lang]["gold_label"] for lang in manifest.languages}
        assert len(labels) == 1, (sid, labels)


def test_xnli_raises_when_no_alignment_is_available(monkeypatch, tokenizer):
    """No id column and no ``all_languages`` config ⇒ raise. There is no positional path."""
    _patch(monkeypatch, xnli=fake.make_xnli(with_prompt_id=False), xnli_all=None)
    with pytest.raises(RuntimeError, match="positional join"):
        pm.build_xnli_manifest(tokenizer, n=12, seed=42, max_tokens=200)


def test_xnli_prompt_has_the_premise_at_position_zero(monkeypatch, tokenizer):
    """Design §16.2: no instruction prefix, or position 0 becomes a shared anchor."""
    _patch(monkeypatch, xnli=fake.make_xnli())
    manifest = pm.build_xnli_manifest(tokenizer, n=9, seed=42, max_tokens=200)
    sid = manifest.semantic_ids[0]
    for lang in manifest.languages:
        row = manifest.rows[sid][lang]
        assert row["text"].startswith(row["premise"]), lang
        premise_ids = tokenizer(row["premise"], add_special_tokens=False)["input_ids"]
        assert row["input_ids"][0] == premise_ids[0], lang


# ═══════════════════════════════════════════════════════════════════════════════
# FLORES — a source-index join is legitimate, and guarded
# ═══════════════════════════════════════════════════════════════════════════════


def test_flores_rows_share_a_source_index(monkeypatch, tokenizer):
    _patch(monkeypatch, flores=fake.make_flores())
    manifest = pm.build_flores_manifest(tokenizer, n=20, seed=42)

    assert manifest.provenance["loader"]["join_strategy"] == "flores_source_index"
    for sid in manifest.semantic_ids:
        indices = {manifest.rows[sid][lang]["source_index"]
                   for lang in manifest.languages}
        assert len(indices) == 1, (sid, indices)
        assert sid.endswith(f"{indices.pop():05d}")
    pm.verify_manifest(manifest)


def test_flores_keeps_a_row_only_if_every_language_passes_the_length_filter(
        monkeypatch, tokenizer):
    """Row 0 is too short in one language and row 1 too long in another; both are dropped."""
    _patch(monkeypatch, flores=fake.make_flores(short_rows=(0,), long_rows=(1,)))
    manifest = pm.build_flores_manifest(tokenizer, n=100, seed=42,
                                        min_tokens=12, max_tokens=160)
    kept = {int(sid.split("_")[-1]) for sid in manifest.semantic_ids}
    assert 0 not in kept
    assert 1 not in kept
    assert len(kept) > 0
    for lang in manifest.languages:
        for count in manifest.token_counts(lang):
            assert 12 <= count <= 160


def test_flores_marks_a_thin_manifest_invalid(monkeypatch, tokenizer):
    """``04`` §1: fewer than 200 surviving rows ⇒ ``provenance['valid'] = False``."""
    _patch(monkeypatch, flores=fake.make_flores(n_rows=30))
    manifest = pm.build_flores_manifest(tokenizer, n=300, seed=42)
    assert manifest.provenance["valid"] is False
    assert manifest.provenance["realised_n"] < 200


def test_flores_refuses_a_ragged_join(monkeypatch, tokenizer):
    """Unequal config lengths invalidate the source-index join, so it raises."""
    shards = fake.make_flores(n_rows=20)
    shards["deu_Latn"] = type(shards["deu_Latn"])(shards["deu_Latn"][:15])
    _patch(monkeypatch, flores=shards)
    with pytest.raises(ValueError, match="line-aligned"):
        pm.build_flores_manifest(tokenizer, n=10, seed=42)


# ═══════════════════════════════════════════════════════════════════════════════
# Determinism, hashing, and the corpus projection
# ═══════════════════════════════════════════════════════════════════════════════


def test_manifest_hash_is_deterministic_and_content_sensitive(monkeypatch, tokenizer):
    _patch(monkeypatch, flores=fake.make_flores())
    a = pm.build_flores_manifest(tokenizer, n=20, seed=42)
    b = pm.build_flores_manifest(tokenizer, n=20, seed=42)
    c = pm.build_flores_manifest(tokenizer, n=20, seed=7)
    assert a.sha256 == b.sha256
    assert a.sha256 != c.sha256

    # Partitions are part of the identity.
    parts = pm.grouped_partition(a, seed=1)
    assert pm.with_partitions(a, parts).sha256 != a.sha256


def test_manifest_round_trips_through_disk(monkeypatch, tokenizer, tmp_path):
    _patch(monkeypatch, flores=fake.make_flores())
    manifest = pm.build_flores_manifest(tokenizer, n=20, seed=42)
    path = manifest.save(tmp_path / "flores.json")
    reloaded = pm.ParallelManifest.load(path)
    assert reloaded.sha256 == manifest.sha256
    assert reloaded.semantic_ids == manifest.semantic_ids

    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["semantic_ids"] = list(payload["semantic_ids"])[:-1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        pm.ParallelManifest.load(path)


def test_corpus_projection_preserves_semantic_id_order(monkeypatch, tokenizer):
    """Row *i* must be the same sentence in every language (``02`` §2.3)."""
    _patch(monkeypatch, flores=fake.make_flores())
    manifest = pm.build_flores_manifest(tokenizer, n=20, seed=42)
    ids = list(manifest.semantic_ids)[:8][::-1]      # a non-default order

    corpora = {lang: cp.flores_corpus(tokenizer, lang, ids, manifest=manifest)
               for lang in manifest.languages}
    for lang, corpus in corpora.items():
        assert [item.meta["semantic_id"] for item in corpus.items] == ids, lang
        assert corpus.provenance["manifest_sha256"] == manifest.sha256
        for item in corpus.items:
            assert item.n_tokens == len(item.input_ids)

    # The same row index is the same semantic id in every language.
    for row in range(len(ids)):
        semantics = {corpus.items[row].meta["semantic_id"] for corpus in corpora.values()}
        assert len(semantics) == 1


def test_corpus_projection_refuses_a_missing_manifest_and_a_foreign_tokenizer(
        monkeypatch, tokenizer):
    _patch(monkeypatch, flores=fake.make_flores(), xnli=fake.make_xnli())
    manifest = pm.build_flores_manifest(tokenizer, n=20, seed=42)

    with pytest.raises(ValueError, match="ParallelManifest"):
        cp.flores_corpus(tokenizer, "eng_Latn", None)
    with pytest.raises(ValueError, match="not in the manifest"):
        cp.flores_corpus(tokenizer, "xx_Xxxx", None, manifest=manifest)

    xnli = pm.build_xnli_manifest(tokenizer, n=9, seed=42, max_tokens=200)
    with pytest.raises(ValueError, match="expected a FLORES manifest"):
        cp.flores_corpus(tokenizer, "eng_Latn", None, manifest=xnli)

    other = fake.tiny_tokenizer(None)
    other.name_or_path = "some_other_tokenizer"
    with pytest.raises(ValueError, match="tokenizer-specific"):
        cp.flores_corpus(other, "eng_Latn", None, manifest=manifest)


def test_xnli_corpus_carries_scoring_metadata(monkeypatch, tokenizer):
    _patch(monkeypatch, xnli=fake.make_xnli())
    manifest = pm.build_xnli_manifest(tokenizer, n=9, seed=42, max_tokens=200)
    corpus = cp.xnli_prompt_corpus(tokenizer, "de", None, manifest=manifest)
    for item in corpus.items:
        assert item.meta["gold_label"] in (0, 1, 2)
        assert len(item.meta["label_candidate_ids"]) == 3
        assert all(len(ids) >= 1 for ids in item.meta["label_candidate_ids"])
        assert item.meta["premise"] in item.text


def test_length_matched_subset_bounds_the_relative_difference(monkeypatch, tokenizer):
    """Rows whose best length match exceeds the bound are dropped, and the bound holds."""
    _patch(monkeypatch, flores=fake.make_flores(n_rows=40, skew_every=4))
    manifest = pm.build_flores_manifest(tokenizer, n=30, seed=42)
    subset = pm.build_length_matched_subset(manifest, reference_lang="eng_Latn",
                                            max_rel_diff=0.20)
    info = subset.provenance["length_matched"]
    assert set(subset.semantic_ids) <= set(manifest.semantic_ids)
    assert info["n_after"] < info["n_before"], "the skewed rows should have been dropped"
    for lang, realised in info["realised_max_rel_diff"].items():
        assert realised <= 0.20 + 1e-9, (lang, realised)
    # The row set really changed, so the content hash must change with it. (When a subset
    # happens to keep every row it is the same input set and an equal hash is correct —
    # the hash identifies content, not intent.)
    assert subset.sha256 != manifest.sha256


def test_length_matched_subset_that_drops_nothing_keeps_its_hash(monkeypatch, tokenizer):
    """An unchanged row set is the same input set, so ``05`` §7.1 must see the same hash."""
    _patch(monkeypatch, flores=fake.make_flores(n_rows=40))
    manifest = pm.build_flores_manifest(tokenizer, n=30, seed=42)
    subset = pm.build_length_matched_subset(manifest, reference_lang="eng_Latn",
                                            max_rel_diff=0.20)
    if subset.provenance["length_matched"]["n_after"] == \
            subset.provenance["length_matched"]["n_before"]:
        assert subset.sha256 == manifest.sha256
        assert subset.manifest_id != manifest.manifest_id
