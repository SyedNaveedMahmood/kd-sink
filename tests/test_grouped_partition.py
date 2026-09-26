# -*- coding: utf-8 -*-
"""test_grouped_partition.py — partitions split by semantic id, never by row (WP8).

``06_TEST_PLAN.md`` row 35 and ``02_MODULE_SPEC_common.md`` §5.1. E7 selects a patch layer
on 200 dev examples and reports on 400 disjoint test examples (``04`` §5.3), and fits
Procrustes on 100 held-out ids evaluated on the disjoint 200 (``04`` §4). Both splits come
from this one function.

Splitting by row rather than by semantic id is the leak that matters here: the same
sentence in another language would land on the other side of the split, and the "held-out"
set would not be held out at all.

Offline: monkeypatched ``load_dataset``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets_loader as dl  # noqa: E402
import paired_manifests as pm  # noqa: E402

import wp8_fake_data as fake  # noqa: E402


def _xnli_manifest(monkeypatch, n, n_rows=None):
    n_rows = n_rows if n_rows is not None else n * 2
    monkeypatch.setattr(dl, "load_dataset",
                        fake.fake_load_dataset(xnli=fake.make_xnli(n_rows=n_rows)))
    return pm.build_xnli_manifest(fake.tiny_tokenizer(None), n=n, seed=42, max_tokens=200)


def test_dev_and_test_are_disjoint_and_exhaustive(monkeypatch):
    manifest = _xnli_manifest(monkeypatch, n=60)
    parts = pm.grouped_partition(manifest, fractions=(0.0, 1 / 3, 1.0),
                                 names=("dev", "test"), seed=42)
    dev, test = set(parts["dev"]), set(parts["test"])

    assert dev.isdisjoint(test)
    assert dev | test == set(manifest.semantic_ids)
    assert len(dev) + len(test) == len(manifest.semantic_ids)
    # 1/3 of 60 is 20, so dev/test is 20/40 — the 200/400 shape at fixture scale.
    assert len(dev) == len(manifest.semantic_ids) // 3
    assert len(test) == len(manifest.semantic_ids) - len(dev)


def test_the_split_is_by_semantic_id_in_every_language(monkeypatch):
    """No sentence appears in two partitions in *any* language."""
    manifest = _xnli_manifest(monkeypatch, n=60)
    parts = pm.grouped_partition(manifest, seed=42)
    manifest = pm.with_partitions(manifest, parts)

    for lang in manifest.languages:
        dev_texts = {manifest.rows[sid][lang]["text"] for sid in parts["dev"]}
        test_texts = {manifest.rows[sid][lang]["text"] for sid in parts["test"]}
        assert dev_texts.isdisjoint(test_texts), lang

    # verify_manifest enforces the same property structurally.
    report = pm.verify_manifest(manifest)
    assert report["partitions"] == {"dev": len(parts["dev"]), "test": len(parts["test"])}


def test_a_row_level_split_would_have_been_caught(monkeypatch):
    """The fixture has the same semantic id present in all seven languages.

    That is what makes the disjointness check above meaningful: a row-level split of the
    projected corpora would put one language's copy in dev and another's in test.
    """
    manifest = _xnli_manifest(monkeypatch, n=30)
    for sid in manifest.semantic_ids:
        assert set(manifest.rows[sid]) == set(manifest.languages)


def test_fixture_sentences_are_unique(monkeypatch):
    """Guards the text-as-identity proxy used by the disjointness test above.

    Colliding fixture sentences would make that test fail (or pass) for reasons that have
    nothing to do with the partition logic, so uniqueness is asserted rather than assumed.
    """
    manifest = _xnli_manifest(monkeypatch, n=60)
    texts = [manifest.rows[sid][lang]["text"]
             for sid in manifest.semantic_ids for lang in manifest.languages]
    assert len(set(texts)) == len(texts)


def test_procrustes_split_sizes(monkeypatch):
    """``04`` §4: fit on 100 held-out ids, evaluate on the disjoint 200."""
    monkeypatch.setattr(dl, "load_dataset",
                        fake.fake_load_dataset(flores=fake.make_flores(n_rows=400)))
    manifest = pm.build_flores_manifest(fake.tiny_tokenizer(None), n=300, seed=42)
    assert len(manifest.semantic_ids) == 300

    parts = pm.grouped_partition(manifest, fractions=(0.0, 1 / 3, 1.0),
                                 names=("procrustes_train", "procrustes_test"), seed=42)
    assert len(parts["procrustes_train"]) == 100
    assert len(parts["procrustes_test"]) == 200
    assert set(parts["procrustes_train"]).isdisjoint(parts["procrustes_test"])


def test_dev_test_split_sizes_at_full_scale(monkeypatch):
    """``04`` §5.3: 200 dev / 400 test out of 600."""
    monkeypatch.setattr(dl, "load_dataset",
                        fake.fake_load_dataset(xnli=fake.make_xnli(n_rows=900)))
    manifest = pm.build_xnli_manifest(fake.tiny_tokenizer(None), n=600, seed=42,
                                      max_tokens=200)
    assert len(manifest.semantic_ids) == 600
    parts = pm.grouped_partition(manifest, seed=42)
    assert len(parts["dev"]) == 200
    assert len(parts["test"]) == 400


def test_partition_is_deterministic_and_seed_sensitive(monkeypatch):
    manifest = _xnli_manifest(monkeypatch, n=60)
    a = pm.grouped_partition(manifest, seed=42)
    b = pm.grouped_partition(manifest, seed=42)
    c = pm.grouped_partition(manifest, seed=7)
    assert a == b
    assert a != c
    # Still a valid partition under a different seed.
    assert set(c["dev"]).isdisjoint(c["test"])
    assert set(c["dev"]) | set(c["test"]) == set(manifest.semantic_ids)


def test_three_way_partition(monkeypatch):
    manifest = _xnli_manifest(monkeypatch, n=60)
    parts = pm.grouped_partition(manifest, fractions=(0.0, 0.25, 0.5, 1.0),
                                 names=("train", "dev", "test"), seed=42)
    assert [len(parts[k]) for k in ("train", "dev", "test")] == [15, 15, 30]
    seen = set()
    for members in parts.values():
        assert seen.isdisjoint(members)
        seen |= set(members)
    assert seen == set(manifest.semantic_ids)


def test_malformed_fraction_specifications_raise(monkeypatch):
    manifest = _xnli_manifest(monkeypatch, n=30)
    with pytest.raises(ValueError, match="cut points"):
        pm.grouped_partition(manifest, fractions=(0.0, 1.0), names=("dev", "test"))
    with pytest.raises(ValueError, match="non-decreasing"):
        pm.grouped_partition(manifest, fractions=(0.0, 0.9, 0.5, 1.0),
                             names=("a", "b", "c"))
    with pytest.raises(ValueError, match="start at 0.0"):
        pm.grouped_partition(manifest, fractions=(0.1, 0.5, 1.0), names=("a", "b"))


def test_verify_manifest_rejects_an_overlapping_partition(monkeypatch):
    manifest = _xnli_manifest(monkeypatch, n=30)
    ids = list(manifest.semantic_ids)
    bad = pm.with_partitions(manifest, {"dev": tuple(ids[:10]),
                                        "test": tuple(ids[8:])})
    with pytest.raises(ValueError, match="is in both"):
        pm.verify_manifest(bad)

    unknown = pm.with_partitions(manifest, {"dev": ("not_a_real_id",)})
    with pytest.raises(ValueError, match="unknown id"):
        pm.verify_manifest(unknown)
