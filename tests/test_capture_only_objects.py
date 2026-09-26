# -*- coding: utf-8 -*-
"""test_capture_only_objects.py — the WP9 additive extension to WP7's capture path.

``04`` §3.1 asks extraction for three objects that are *reductions over a span* rather than
single-position tensors: ``Qmean`` (mean query over the second-half positions), ``Rlast``
and ``Rmean``. They were added to ``cross_example_patching`` so extraction runs through one
trace path, and they must be **capture-only**: a mean over a span has no single site to
write back to, so accepting one as a patch target would produce a plausible effect size for
a patch that never happened.

These tests prove three things:

1. the reductions are *exactly* what they claim (compared against per-position captures);
2. the writer and ``PatchSpec`` both refuse them;
3. the streaming (``on_item``/``keep=False``) form returns identical tensors to the
   accumulating one — that is the form WP9 extraction relies on for its memory budget.

Windows teardown: the safetensors mmap must be released before the TemporaryDirectory is
removed, hence ``del handle; gc.collect()`` in every ``finally``.
"""

from __future__ import annotations

import gc
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cross_example_patching as cep  # noqa: E402
from corpus_providers import Corpus, CorpusItem, compute_manifest_sha256  # noqa: E402
from fingerprint_runner import load_handle  # noqa: E402
from nnsight_smoke_utils import create_tiny_qwen_checkpoint  # noqa: E402

TOKENS = list(range(3, 19))          # seq_len 16
LAYER = 1


def _item(item_id: str, ids) -> CorpusItem:
    return CorpusItem(item_id=item_id, text="", input_ids=list(ids),
                      n_tokens=len(ids), meta={})


def _corpus(items) -> Corpus:
    items = tuple(items)
    return Corpus(corpus_id="capture_only", items=items, tokenizer_name="tiny",
                  tokenizer_revision=None, add_special_tokens=False, cut_length=None,
                  seed=0, manifest_sha256=compute_manifest_sha256(items), provenance={})


class _Model:
    """RAII holder, matching ``tests/test_cross_example_patching.py``."""

    def __init__(self, tmp: Path, *, seed: int = 4003):
        path = tmp / "qwen"
        create_tiny_qwen_checkpoint(path, seed=seed, n_positions=64)
        self.handle = load_handle("qwen", str(path), dtype="float32", device="cpu",
                                  local_files_only=True)

    def close(self) -> None:
        del self.handle
        gc.collect()


# ── taxonomy ──────────────────────────────────────────────────────────────────


def test_capture_only_objects_are_not_patch_objects():
    for obj in cep.CAPTURE_ONLY_OBJECTS:
        assert obj not in cep.PATCH_OBJECTS
        assert obj in cep.CAPTURE_OBJECTS
    assert set(cep.CAPTURE_OBJECTS) == set(cep.PATCH_OBJECTS) | set(cep.CAPTURE_ONLY_OBJECTS)


@pytest.mark.parametrize("obj,expected", [
    ("Rmean", (0, 16, "mean")),
    ("Qmean", (8, 16, "mean")),
    ("Rlast", (15, None, "none")),
    ("R0", (0, None, "none")),
    ("Rmid", (8, None, "none")),
])
def test_capture_span_resolves_per_example(obj, expected):
    assert cep.capture_span(obj, 16, rope_inside=True) == expected


def test_capture_span_resolves_against_this_example_not_a_fixed_length():
    assert cep.capture_span("Rlast", 5, rope_inside=True)[0] == 4
    assert cep.capture_span("Rlast", 40, rope_inside=True)[0] == 39
    assert cep.capture_span("Qmean", 40, rope_inside=True) == (20, 40, "mean")


# ── the refusals ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("obj", list(cep.CAPTURE_ONLY_OBJECTS))
def test_patch_spec_refuses_a_capture_only_object(obj):
    with pytest.raises(ValueError, match="capture-only"):
        cep.PatchSpec(cep.PatchSite(obj, LAYER), "identity")


@pytest.mark.parametrize("obj", list(cep.CAPTURE_ONLY_OBJECTS))
def test_the_writer_refuses_a_capture_only_object(obj):
    """Second guard: a caller that reaches ``_make_write`` another way still cannot write."""
    with tempfile.TemporaryDirectory(prefix="cep_conly_w_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            handle = model.handle
            site = cep.PatchSite(obj, LAYER)
            surface, _kind, _variant = cep._object_parts(obj, rope_inside=True)
            with pytest.raises(ValueError, match="capture-only"):
                cep._make_write(handle, site, surface, 0, torch.zeros(1),
                                torch.device("cpu"), torch.float32)
        finally:
            model.close()


# ── the reductions are exact ──────────────────────────────────────────────────


def test_reductions_match_per_position_captures_exactly():
    with tempfile.TemporaryDirectory(prefix="cep_conly_r_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            handle = model.handle
            corpus = _corpus([_item("a", TOKENS)])
            sites = [cep.PatchSite(obj, LAYER)
                     for obj in ("Rmean", "Rlast", "R0", "Rmid")]
            cache = cep.capture_sources(handle, corpus, sites)["a"]

            per_position = []
            for position in range(len(TOKENS)):
                one = cep.capture_sources(
                    handle, corpus, [cep.PatchSite("R0", LAYER, position=position)])
                per_position.append(one["a"].tensors[f"R0@L{LAYER}"])
            stacked = torch.stack(per_position)

            assert torch.allclose(cache.tensors[f"Rmean@L{LAYER}"],
                                  stacked.mean(dim=0), atol=1e-6)
            assert torch.equal(cache.tensors[f"Rlast@L{LAYER}"], stacked[-1])
            assert torch.equal(cache.tensors[f"R0@L{LAYER}"], stacked[0])
            assert torch.equal(cache.tensors[f"Rmid@L{LAYER}"],
                               stacked[len(TOKENS) // 2])
        finally:
            model.close()


def test_qmean_has_the_query_head_shape_not_the_kv_head_shape():
    """``q_proj`` emits one row per *attention* head; ``k_proj``/``v_proj`` per KV head.

    The tiny model is deliberately 8 query heads / 2 KV heads, so reshaping ``Qmean``
    against ``num_kv_heads`` would produce the wrong tensor without any error.
    """
    with tempfile.TemporaryDirectory(prefix="cep_conly_q_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            handle = model.handle
            geom = cep.geometry(handle)
            assert geom["num_heads"] == 8 and geom["num_kv_heads"] == 2

            cache = cep.capture_sources(
                handle, _corpus([_item("a", TOKENS)]),
                [cep.PatchSite("Qmean", LAYER), cep.PatchSite("V0", LAYER)])["a"]
            assert tuple(cache.tensors[f"Qmean@L{LAYER}"].shape) == \
                (geom["num_heads"], geom["head_dim"])
            assert tuple(cache.tensors[f"V0@L{LAYER}"].shape) == \
                (geom["num_kv_heads"], geom["head_dim"])
            assert cache.tensors[f"Qmean@L{LAYER}"].dtype is torch.float32
            assert cache.tensors[f"Qmean@L{LAYER}"].device.type == "cpu"
        finally:
            model.close()


# ── streaming form ────────────────────────────────────────────────────────────


def test_streaming_capture_matches_the_accumulating_form_and_keeps_nothing():
    with tempfile.TemporaryDirectory(prefix="cep_conly_s_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            handle = model.handle
            corpus = _corpus([_item("a", TOKENS), _item("b", TOKENS[:12])])
            sites = [cep.PatchSite(obj, LAYER) for obj in cep.CAPTURE_OBJECTS]

            accumulated = cep.capture_sources(handle, corpus, sites)
            streamed = {}
            returned = cep.capture_sources(
                handle, corpus, sites,
                on_item=lambda item_id, cache: streamed.__setitem__(item_id, cache),
                keep=False)

            assert returned == {}, "keep=False must not accumulate"
            assert sorted(streamed) == sorted(accumulated) == ["a", "b"]
            for item_id, cache in accumulated.items():
                for key, tensor in cache.tensors.items():
                    assert torch.equal(tensor, streamed[item_id].tensors[key]), key
                assert streamed[item_id].resolved_positions == cache.resolved_positions
                assert streamed[item_id].seq_len == cache.seq_len
        finally:
            model.close()


def test_every_capture_object_resolves_on_a_real_model():
    """Shapes and positions for the whole ``CAPTURE_OBJECTS`` set, in one pass."""
    with tempfile.TemporaryDirectory(prefix="cep_conly_a_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            handle = model.handle
            geom = cep.geometry(handle)
            cache = cep.capture_sources(
                handle, _corpus([_item("a", TOKENS)]),
                [cep.PatchSite(obj, LAYER) for obj in cep.CAPTURE_OBJECTS])["a"]

            assert len(cache.tensors) == len(cep.CAPTURE_OBJECTS)
            for obj in cep.CAPTURE_OBJECTS:
                key = f"{obj}@L{LAYER}"
                surface, _kind, _variant = cep._object_parts(obj, rope_inside=True)
                shape = tuple(cache.tensors[key].shape)
                if surface == "r":
                    assert shape == (geom["hidden"],), obj
                elif surface == "q":
                    assert shape == (geom["num_heads"], geom["head_dim"]), obj
                else:
                    assert shape == (geom["num_kv_heads"], geom["head_dim"]), obj
                assert torch.isfinite(cache.tensors[key]).all(), obj
        finally:
            model.close()
