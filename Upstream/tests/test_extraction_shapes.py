# -*- coding: utf-8 -*-
"""test_extraction_shapes.py — WP9 extraction contracts (06_TEST_PLAN.md §1).

Per-object shapes and layer count, fp32 on disk, a ``semantic_id -> row`` index that really
maps, and — the assertion the test plan singles out — **the output directory stays under a
size bound**. That bound is what catches an accidental full-hidden-state capture: a run that
persisted ``[seq, hidden]`` per example instead of a reduction would still produce
well-formed arrays, correct-looking indexes and no exception. Only the size gives it away.

The fixture is a random tiny Qwen2 plus a ``wp8_fake_data``-backed FLORES manifest, so the
whole path runs offline. Both builders emit the same WordLevel vocabulary (``t{i} -> i+3``),
so the manifest's token ids are meaningful to the model.
"""

from __future__ import annotations

import gc
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "crosslingual_semantics"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cross_example_patching as cep  # noqa: E402
import datasets_loader as dl  # noqa: E402
import extract_sink_representations as ex  # noqa: E402
import paired_manifests as pm  # noqa: E402
import wp8_fake_data as fake  # noqa: E402
from fingerprint_runner import load_handle  # noqa: E402
from nnsight_smoke_utils import create_tiny_qwen_checkpoint  # noqa: E402

LANGS = ("eng_Latn", "deu_Latn")
N_IDS = 8
N_LAYERS = 4


class _Fixture:
    """A tiny Qwen2 plus a manifest built with that model's own tokenizer."""

    def __init__(self, tmp: Path, monkeypatch=None):
        self.root = tmp
        self.ckpt = tmp / "qwen"
        create_tiny_qwen_checkpoint(self.ckpt, seed=4003,
                                    vocab_size=fake.FAKE_VOCAB_SIZE,
                                    n_positions=256, n_layer=N_LAYERS)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(self.ckpt))
        original = dl.load_dataset
        dl.load_dataset = fake.fake_load_dataset(flores=fake.make_flores(n_rows=14))
        try:
            manifest = pm.build_flores_manifest(tokenizer, n=N_IDS, seed=42)
        finally:
            dl.load_dataset = original
        self.manifest = pm.with_partitions(manifest, pm.grouped_partition(
            manifest, (0.0, 1 / 3, 1.0), ("procrustes_train", "procrustes_test"), seed=42))
        self.handle = load_handle("qwen", str(self.ckpt), dtype="float32", device="cpu",
                                  local_files_only=True)
        self.config = ex.smoke_config(str(self.ckpt), tag="tiny")

    def extract(self, out=None, **kwargs):
        out = out if out is not None else self.root / "reps"
        return ex.extract(self.config, self.manifest, out, handle=self.handle,
                          langs=list(LANGS), progress=False, **kwargs)

    def close(self) -> None:
        del self.handle
        gc.collect()


@pytest.fixture
def fixture():
    with tempfile.TemporaryDirectory(prefix="wp9_ext_", ignore_cleanup_errors=True) as t:
        item = _Fixture(Path(t))
        try:
            yield item
        finally:
            item.close()


# ── shapes ────────────────────────────────────────────────────────────────────


def test_every_object_gets_one_array_of_the_declared_shape(fixture):
    summary = fixture.extract()
    assert summary["n_failed"] == 0
    assert summary["layers"] == list(range(N_LAYERS))

    for lang in LANGS:
        lang_dir = fixture.root / "reps" / "tiny" / lang
        for obj in fixture.config.objects:
            array = np.load(lang_dir / f"{obj}.npy", mmap_mode="r")
            assert array.dtype == np.float32, obj
            assert array.shape[0] == N_IDS, obj
            assert array.shape[1] == N_LAYERS, obj
            assert array.shape[2] == ex.object_dim(fixture.handle, obj), obj
            assert np.isfinite(np.asarray(array)).all(), obj


def test_kv_and_query_objects_keep_their_structured_shapes(fixture):
    """``index.json`` records the pre-flattening shape so the analysis never guesses."""
    fixture.extract()
    index = json.loads((fixture.root / "reps" / "tiny" / "eng_Latn" /
                        "index.json").read_text(encoding="utf-8"))
    geom = cep.geometry(fixture.handle)

    assert index["object_shapes"]["R0"] == [geom["hidden"]]
    assert index["object_shapes"]["V0"] == [geom["num_kv_heads"], geom["head_dim"]]
    assert index["object_shapes"]["Qmean"] == [geom["num_heads"], geom["head_dim"]]
    for obj, shape in index["object_shapes"].items():
        assert int(np.prod(shape)) == index["object_dims"][obj], obj


def test_storage_is_fp32_on_cpu_regardless_of_the_model_dtype(fixture):
    """``04`` §3.2: representations are fp32 on CPU; the model dtype is recorded separately."""
    fixture.extract()
    index = json.loads((fixture.root / "reps" / "tiny" / "eng_Latn" /
                        "index.json").read_text(encoding="utf-8"))
    assert index["storage_dtype"] == "float32"
    assert index["model_dtype"] == fixture.config.dtype


# ── the index really maps ─────────────────────────────────────────────────────


def test_index_maps_semantic_id_to_the_right_row(fixture):
    """Row *i* must be the sentence the index says it is — checked by recapturing it."""
    fixture.extract()
    index = json.loads((fixture.root / "reps" / "tiny" / "eng_Latn" /
                        "index.json").read_text(encoding="utf-8"))
    assert sorted(index["row_of"]) == sorted(index["semantic_ids"])
    assert sorted(index["row_of"].values()) == list(range(N_IDS))
    assert all(sid in fixture.manifest.rows for sid in index["semantic_ids"])

    import corpus_providers as cp
    from dataclasses import replace

    corpus = cp.flores_corpus(fixture.handle.tokenizer, "eng_Latn",
                              fixture.manifest.semantic_ids, split="devtest",
                              manifest=fixture.manifest)
    chosen = corpus.items[N_IDS - 1]
    recaptured = cep.capture_sources(
        fixture.handle, replace(corpus, items=(chosen,)),
        [cep.PatchSite("R0", 2)])[chosen.item_id].tensors["R0@L2"]

    array = np.load(fixture.root / "reps" / "tiny" / "eng_Latn" / "R0.npy", mmap_mode="r")
    row = index["row_of"][chosen.meta["semantic_id"]]
    assert np.allclose(array[row, 2, :], recaptured.numpy(), atol=1e-6)
    # The index key is the semantic id, not the language-qualified corpus item id.
    assert index["item_id_of"][chosen.meta["semantic_id"]] == chosen.item_id
    assert chosen.item_id != chosen.meta["semantic_id"]


def test_the_same_row_is_the_same_sentence_in_every_language(fixture):
    """The property the whole experiment rests on: row *i* is one semantic id everywhere."""
    fixture.extract()
    rows = {}
    for lang in LANGS:
        index = json.loads((fixture.root / "reps" / "tiny" / lang /
                            "index.json").read_text(encoding="utf-8"))
        rows[lang] = index["row_of"]
    assert rows["eng_Latn"] == rows["deu_Latn"]


# ── the size bound (06 §1) ────────────────────────────────────────────────────


def test_output_stays_far_below_a_full_hidden_state_capture(fixture):
    """The assertion that catches accidental full-state persistence.

    A derived-tensor capture costs ``n_items x n_layers x dim`` floats per object. Storing a
    full ``[seq, hidden]`` state instead would multiply the residual objects by the sequence
    length — roughly 20x here, and hundreds of times over on a real run. The bound is set at
    **3x** the derived size, which is loose enough to absorb the index/JSON overhead and far
    too tight to hide a full-state capture.
    """
    fixture.extract()
    reps = fixture.root / "reps" / "tiny"

    expected_floats = sum(
        N_IDS * N_LAYERS * ex.object_dim(fixture.handle, obj)
        for obj in fixture.config.objects) * len(LANGS)
    expected_bytes = expected_floats * 4
    actual = sum(f.stat().st_size for f in reps.rglob("*") if f.is_file())

    assert actual < 3 * expected_bytes, (
        f"extraction wrote {actual} bytes against a derived-tensor budget of "
        f"{expected_bytes}; something is persisting more than a per-layer reduction")

    seq_lens = json.loads((reps / "eng_Latn" / "index.json").read_text(
        encoding="utf-8"))["token_counts"]
    assert min(seq_lens.values()) > 3, "the bound is only meaningful for seq_len >> 1"


# ── refusals and resumability ─────────────────────────────────────────────────


def test_a_manifest_from_a_different_tokenizer_is_refused(fixture):
    from dataclasses import replace

    other = replace(fixture.manifest,
                    provenance={**fixture.manifest.provenance,
                                "tokenizer": "some/other-tokenizer"})
    with pytest.raises(ValueError, match="tokenizer"):
        ex.check_tokenizer(other, fixture.handle.tokenizer)


def test_resume_skips_a_completed_language_and_recomputes_nothing(fixture):
    first = fixture.extract()
    assert first["n_skipped"] == 0

    second = fixture.extract()
    assert second["n_skipped"] == len(LANGS)
    assert all(row["status"] != "failed" for row in second["languages"])


def test_a_changed_instrument_invalidates_the_unit(fixture):
    """Dropping an object from the request must not silently reuse the old arrays."""
    fixture.extract()
    fixture.config.objects = tuple(o for o in fixture.config.objects if o != "Qmean")
    again = fixture.extract()
    assert again["n_skipped"] == 0
    assert [row["status"] for row in again["languages"]] == ["ok"] * len(LANGS)


def test_force_recomputes_even_a_complete_unit(fixture):
    fixture.extract()
    forced = fixture.extract(resume=False)
    assert forced["n_skipped"] == 0


def test_an_unknown_extraction_object_is_refused():
    with pytest.raises(ValueError, match="unknown extraction objects"):
        ex.config_from_dict({"model": "x", "extraction": {"objects": ["R0", "NotAThing"]}})


def test_arch_is_read_from_a_local_checkpoint_config(fixture):
    assert ex.arch_of(str(fixture.ckpt)) == "qwen"
    assert ex.arch_of("Qwen/Qwen2.5-0.5B") == "qwen"
    with pytest.raises(ValueError, match="cannot infer an ArchSpec"):
        ex.arch_of("some/unknown-model")
