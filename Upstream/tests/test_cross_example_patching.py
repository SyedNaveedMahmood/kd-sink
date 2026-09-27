# -*- coding: utf-8 -*-
"""test_cross_example_patching.py — the nine invariants of ``02_MODULE_SPEC_common.md`` §6.2.

WP7. Every causal claim in E7 rests on ``common/cross_example_patching.py`` being correct,
and a patching implementation that gets these wrong does not crash — it produces smooth,
plausible, entirely fictitious effect sizes. Invariants 1 and 2 (identity and self-source
patches are exact no-ops) are the load-bearing pair; ``06_TEST_PLAN.md`` §3 singles them
out.

All models are random tiny Qwen2 checkpoints built offline by ``nnsight_smoke_utils``, with
``num_attention_heads=8, num_key_value_heads=2`` so the pre-``_repeat_kv`` invariant is
meaningful (each key/value head serves exactly four query heads).

Windows-safe teardown: every model is released and gc'd inside its TemporaryDirectory.
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

EXACT = cep.EXACTNESS_TOLERANCE          # 1e-6, the spec's fp32 no-op tolerance
TARGET_IDS = list(range(3, 23))          # 20 tokens
SOURCE_IDS = list(range(11, 23))         # 12 tokens — deliberately a different length


def _corpus(items):
    """A minimal in-memory Corpus; the providers' own hashing convention is reused."""
    items = tuple(items)
    return Corpus(corpus_id="wp7_test", items=items, tokenizer_name="tiny",
                  tokenizer_revision=None, add_special_tokens=False, cut_length=None,
                  seed=0, manifest_sha256=compute_manifest_sha256(items),
                  provenance={"schema": "corpus_v1", "source": "test"})


def _item(item_id, ids):
    return CorpusItem(item_id=item_id, text="", input_ids=list(ids),
                      n_tokens=len(ids), meta={})


class _Model:
    """Build a random tiny Qwen2 and hold it for the duration of a test."""

    def __init__(self, tmp: Path, *, seed: int = 4003, tag: str = "qwen"):
        path = tmp / tag
        create_tiny_qwen_checkpoint(path, seed=seed)
        self.handle = load_handle("qwen", str(path), dtype="float32", device="cpu",
                                  local_files_only=True)

    def close(self):
        del self.handle
        gc.collect()


def _run(handle, target, specs, sources, **kwargs):
    return cep.run_patched(handle, target, specs, sources, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Invariants 1 and 2 — exact no-ops
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("obj", ["K0_prerope", "V0", "R0", "K0_postrope",
                                 "Kmid_prerope", "Vmid", "Rmid"])
def test_identity_patch_is_a_no_op(obj):
    """Invariant 1: writing the target's own captured tensor back changes nothing."""
    with tempfile.TemporaryDirectory(prefix="cep_ident_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            handle = model.handle
            target = _item("target", TARGET_IDS)
            site = cep.PatchSite(obj, layer=2)
            spec = cep.PatchSpec(site=site, norm_condition="identity", seed=0)
            out = _run(handle, target, [spec], {}, capture_sink=True,
                       score_candidates=[[5], [6]], gold_index=0)

            row = out["patches"][0]
            assert row["status"] == "ok", row["warning"]
            assert abs(row["margin_delta"]) < EXACT, (obj, row["margin_delta"])
            assert abs(row["patched_sink"] - row["baseline_sink"]) < EXACT
            assert row["jsd_output"] < EXACT
        finally:
            model.close()


@pytest.mark.parametrize("obj", ["K0_prerope", "V0", "R0"])
def test_self_source_patch_is_a_no_op(obj):
    """Invariant 2: patching with ``source_item_id == target_item_id`` changes nothing.

    Restricted to the pre-RoPE objects because they are bit-exact round trips; the
    post-RoPE variant round-trips through a rotation and its inverse, which is exercised
    (with the float error it necessarily has) in
    :func:`test_postrope_self_source_round_trip`.
    """
    with tempfile.TemporaryDirectory(prefix="cep_self_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            handle = model.handle
            target = _item("target", TARGET_IDS)
            corpus = _corpus([target])
            site = cep.PatchSite(obj, layer=2)
            sources = cep.capture_sources(handle, corpus, [site])

            spec = cep.PatchSpec(site=site, norm_condition="direct",
                                 source_item_id="target", seed=0)
            out = _run(handle, target, [spec], sources, capture_sink=True,
                       score_candidates=[[5], [6]], gold_index=0)
            row = out["patches"][0]
            assert row["status"] == "ok", row["warning"]
            assert abs(row["margin_delta"]) < EXACT, (obj, row["margin_delta"])
            assert row["jsd_output"] < EXACT
        finally:
            model.close()


def test_postrope_self_source_round_trip():
    """The forward/inverse rotation pair is a round trip at any position.

    ``02`` §6.4 leaves ``K*_postrope`` ambiguous unless the inverse rotation is exact; this
    checks the algebra directly and then end-to-end through ``run_patched``.
    """
    with tempfile.TemporaryDirectory(prefix="cep_rope_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            handle = model.handle
            geom = cep.geometry(handle)
            k = torch.randn(geom["num_kv_heads"], geom["head_dim"],
                            generator=torch.Generator().manual_seed(7))
            for position in (0, 1, 9):
                rotated = cep.apply_rope_forward(k, position, head_dim=geom["head_dim"],
                                                 rope_theta=geom["rope_theta"])
                back = cep.apply_rope_inverse(rotated, position,
                                              head_dim=geom["head_dim"],
                                              rope_theta=geom["rope_theta"])
                assert torch.allclose(back, k, atol=EXACT), position

            target = _item("target", TARGET_IDS)
            corpus = _corpus([target])
            site = cep.PatchSite("Kmid_postrope", layer=1)
            sources = cep.capture_sources(handle, corpus, [site])
            spec = cep.PatchSpec(site=site, norm_condition="direct",
                                 source_item_id="target", seed=0)
            out = _run(handle, target, [spec], sources, capture_sink=False,
                       score_candidates=[[5], [6]], gold_index=0)
            row = out["patches"][0]
            assert row["status"] == "ok", row["warning"]
            assert abs(row["margin_delta"]) < 1e-5, row["margin_delta"]
        finally:
            model.close()


def test_rope_position0_is_the_identity_numerically():
    """``02`` §6.4: verify the position-0 pre/post-RoPE coincidence, never assume it."""
    with tempfile.TemporaryDirectory(prefix="cep_rope0_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            geom = cep.geometry(model.handle)
            k = torch.randn(geom["num_kv_heads"], geom["head_dim"],
                            generator=torch.Generator().manual_seed(11))
            deviation = cep.rope_position0_deviation(model.handle, k)
            assert deviation == 0.0 or deviation < EXACT, deviation
        finally:
            model.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Invariant 3 — locality
# ═══════════════════════════════════════════════════════════════════════════════


def test_patch_is_local_to_its_layer_and_above():
    """A ``K0@L2`` patch changes layer 2 and leaves every layer below it untouched."""
    with tempfile.TemporaryDirectory(prefix="cep_local_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            handle = model.handle
            target = _item("target", TARGET_IDS)
            source = _item("source", TARGET_IDS[::-1])
            corpus = _corpus([source])
            site = cep.PatchSite("K0_prerope", layer=2)
            sources = cep.capture_sources(handle, corpus, [site])

            spec = cep.PatchSpec(site=site, norm_condition="direct",
                                 source_item_id="source", seed=0)
            out = _run(handle, target, [spec], sources, capture_sink=False,
                       capture_block_outputs=(0, 1, 2, 3))
            row = out["patches"][0]
            assert row["status"] == "ok", row["warning"]

            before = out["baseline"]["block_outputs"]
            after = row["block_outputs"]
            for layer in (0, 1):
                assert torch.equal(before[layer], after[layer]), layer
            assert not torch.equal(before[2], after[2])
            assert not torch.equal(before[3], after[3])
        finally:
            model.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Invariant 4 — V is patched before _repeat_kv
# ═══════════════════════════════════════════════════════════════════════════════


def test_v0_patch_is_applied_before_repeat_kv():
    """Patching one of two key/value heads must move exactly four of eight query heads.

    ``attn.output[0]`` is ``context @ W_o^T + b``, and ``W_o`` is square for this config, so
    the per-head context delta is recovered exactly by ``Δcontext = Δout @ (W_o^T)^-1``.
    That tests the ``_repeat_kv`` *mapping* (kv head ``j`` serves query heads
    ``j*n_rep … j*n_rep+n_rep-1``), not merely the count of affected heads.
    """
    with tempfile.TemporaryDirectory(prefix="cep_repeat_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            handle = model.handle
            geom = cep.geometry(handle)
            n_rep = geom["num_heads"] // geom["num_kv_heads"]
            assert (geom["num_heads"], geom["num_kv_heads"]) == (8, 2), geom

            layer = 2
            patched_kv_head = 0
            target = _item("target", TARGET_IDS)
            site = cep.PatchSite("V0", layer=layer, kv_head=patched_kv_head)
            # A norm-matched random draw is a real perturbation of that head alone.
            spec = cep.PatchSpec(site=site, norm_condition="random", seed=3)
            out = _run(handle, target, [spec], {}, capture_sink=False,
                       capture_attn_outputs=(layer,))
            row = out["patches"][0]
            assert row["status"] == "ok", row["warning"]

            delta = (row["attn_outputs"][layer] - out["baseline"]["attn_outputs"][layer])[0]
            attn_module = handle.nn_engine._hf_attn(layer)
            w_o = attn_module.o_proj.weight.detach().float()          # [hidden, n_h*d]
            # attn_out = context @ W^T (+b), so delta = dc @ W^T. Solving W X = delta^T
            # gives X = dc^T, i.e. dc = delta @ (W^T)^-1.
            delta_context = torch.linalg.solve(w_o, delta.T).T        # [seq, n_h*d]
            per_head = delta_context.view(-1, geom["num_heads"], geom["head_dim"])
            magnitudes = per_head.abs().amax(dim=(0, 2))

            expected = set(range(patched_kv_head * n_rep, (patched_kv_head + 1) * n_rep))
            scale = float(magnitudes.max())
            assert scale > 1e-4, "the patch did not perturb anything"
            # The two populations are separated by ~5 orders of magnitude. The floor is not
            # zero because recovering dc inverts a 64x64 fp32 system, whose residual is
            # ~1e-6 in absolute terms; the threshold sits above that residual and four
            # orders below the real signal, so it cannot mask a genuine leak into an
            # unmapped head.
            for head in range(geom["num_heads"]):
                if head in expected:
                    assert magnitudes[head] > scale * 1e-2, (head, magnitudes.tolist())
                else:
                    assert magnitudes[head] < scale * 1e-4, (head, magnitudes.tolist())
        finally:
            model.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Invariant 5 — norm matching
# ═══════════════════════════════════════════════════════════════════════════════


def test_norm_conditions_match_the_target_norm():
    """``rescaled`` and ``random`` both land on ``||target||`` to 1e-6; ``direct`` does not."""
    generator = torch.Generator().manual_seed(19)
    target = torch.randn(2, 8, generator=generator)
    source = torch.randn(2, 8, generator=generator) * 4.0
    site = cep.PatchSite("V0", layer=0)

    target_norm = float(torch.linalg.vector_norm(target.reshape(-1)))
    for condition in ("rescaled", "random"):
        spec = cep.PatchSpec(site=site, norm_condition=condition,
                             source_item_id="s", seed=5)
        value, norms, warning = cep.build_patch_value(spec, target, source)
        assert warning == ""
        assert abs(norms["patched_norm"] - target_norm) < EXACT, condition
        assert abs(float(torch.linalg.vector_norm(value.reshape(-1))) - target_norm) < EXACT

    direct = cep.PatchSpec(site=site, norm_condition="direct", source_item_id="s")
    value, norms, _ = cep.build_patch_value(direct, target, source)
    assert torch.equal(value, source)
    assert abs(norms["patched_norm"] - norms["source_norm"]) < EXACT

    identity = cep.PatchSpec(site=site, norm_condition="identity")
    value, norms, _ = cep.build_patch_value(identity, target, None)
    assert torch.equal(value, target)

    # `random` is reproducible from its seed and differs across seeds.
    a = cep.build_patch_value(cep.PatchSpec(site=site, norm_condition="random",
                                            source_item_id="s", seed=5), target, source)[0]
    b = cep.build_patch_value(cep.PatchSpec(site=site, norm_condition="random",
                                            source_item_id="s", seed=5), target, source)[0]
    c = cep.build_patch_value(cep.PatchSpec(site=site, norm_condition="random",
                                            source_item_id="s", seed=6), target, source)[0]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# ═══════════════════════════════════════════════════════════════════════════════
# Invariant 6 — length independence
# ═══════════════════════════════════════════════════════════════════════════════


def test_source_and_target_may_differ_in_length():
    """``*0`` sites are unaffected by length; ``*mid`` resolves per example and is recorded."""
    with tempfile.TemporaryDirectory(prefix="cep_len_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            handle = model.handle
            target = _item("target", TARGET_IDS)          # 20 tokens
            source = _item("source", SOURCE_IDS)          # 12 tokens
            corpus = _corpus([source])
            sites = [cep.PatchSite("K0_prerope", layer=1),
                     cep.PatchSite("Kmid_prerope", layer=1)]
            sources = cep.capture_sources(handle, corpus, sites)

            assert sources["source"].seq_len == len(SOURCE_IDS)
            assert sources["source"].resolved_positions["K0_prerope@L1"] == 0
            assert sources["source"].resolved_positions["Kmid_prerope@L1"] == \
                len(SOURCE_IDS) // 2

            specs = [cep.PatchSpec(site=site, norm_condition="direct",
                                   source_item_id="source", seed=0) for site in sites]
            out = _run(handle, target, specs, sources, capture_sink=False)
            by_object = {row["patch_object"]: row for row in out["patches"]}
            for row in out["patches"]:
                assert row["status"] == "ok", row["warning"]

            assert by_object["K0_prerope"]["patch_position"] == 0
            assert by_object["K0_prerope"]["source_patch_position"] == 0
            assert by_object["Kmid_prerope"]["patch_position"] == len(TARGET_IDS) // 2
            assert by_object["Kmid_prerope"]["source_patch_position"] == \
                len(SOURCE_IDS) // 2
            assert by_object["Kmid_prerope"]["target_seq_len"] == len(TARGET_IDS)
            assert by_object["Kmid_prerope"]["source_seq_len"] == len(SOURCE_IDS)
        finally:
            model.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Invariant 7 — no batch leakage
# ═══════════════════════════════════════════════════════════════════════════════


def test_batched_patch_does_not_leak_across_items():
    """Patching item 0 in a two-item batch leaves item 1 bit-identical to its own forward."""
    with tempfile.TemporaryDirectory(prefix="cep_batch_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            handle = model.handle
            geom = cep.geometry(handle)
            ids_a = TARGET_IDS
            ids_b = list(reversed(TARGET_IDS))

            clean = cep.run_batched_logits(handle, [ids_a, ids_b], {})
            value = torch.randn(geom["num_kv_heads"], geom["head_dim"],
                                generator=torch.Generator().manual_seed(23))
            site = cep.PatchSite("V0", layer=1)
            patched = cep.run_batched_logits(handle, [ids_a, ids_b], {0: [(site, value)]})

            assert torch.equal(clean[1], patched[1]), "item 1 was contaminated"
            assert not torch.equal(clean[0], patched[0]), "item 0 was not patched"

            with pytest.raises(ValueError, match="equal-length"):
                cep.run_batched_logits(handle, [ids_a, ids_a[:-1]], {})
        finally:
            model.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Invariant 8 — model fingerprint must match
# ═══════════════════════════════════════════════════════════════════════════════


def test_cross_model_patching_raises():
    """A cache captured from a different model is refused, not silently applied."""
    with tempfile.TemporaryDirectory(prefix="cep_fp_", ignore_cleanup_errors=True) as t:
        first = _Model(Path(t), seed=4003, tag="qwen_a")
        second = _Model(Path(t), seed=9999, tag="qwen_b")
        try:
            site = cep.PatchSite("V0", layer=1)
            source = _item("source", TARGET_IDS)
            caches = cep.capture_sources(second.handle, _corpus([source]), [site])

            assert cep.model_fingerprint(first.handle) != \
                cep.model_fingerprint(second.handle)
            assert cep.model_fingerprint(first.handle) == \
                cep.model_fingerprint(first.handle)

            spec = cep.PatchSpec(site=site, norm_condition="direct",
                                 source_item_id="source", seed=0)
            with pytest.raises(ValueError, match="different model"):
                _run(first.handle, _item("target", TARGET_IDS), [spec], caches)
        finally:
            second.close()
            first.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Invariant 9 — shape and dtype at write time
# ═══════════════════════════════════════════════════════════════════════════════


def test_write_asserts_shape_and_casts_dtype():
    """Wrong-shaped values raise; correct ones are cast to the model's real dtype."""
    with tempfile.TemporaryDirectory(prefix="cep_shape_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            handle = model.handle
            geom = cep.geometry(handle)
            device = torch.device("cpu")
            model_dtype = next(handle.model.parameters()).dtype

            good = torch.zeros(geom["num_kv_heads"], geom["head_dim"],
                               dtype=torch.float64)
            write = cep._make_write(handle, cep.PatchSite("V0", layer=0), "v", 0,
                                    good, device, model_dtype)
            assert write.value.dtype == model_dtype
            assert write.value.shape == (geom["num_kv_heads"] * geom["head_dim"],)

            bad = torch.zeros(geom["num_kv_heads"] + 1, geom["head_dim"])
            with pytest.raises(ValueError, match="expected"):
                cep._make_write(handle, cep.PatchSite("V0", layer=0), "v", 0,
                                bad, device, model_dtype)

            bad_r = torch.zeros(geom["hidden"] + 1)
            with pytest.raises(ValueError, match="expected"):
                cep._make_write(handle, cep.PatchSite("R0", layer=0), "r", 0,
                                bad_r, device, model_dtype)

            # A single kv head writes only its own column slice.
            single = cep._make_write(handle, cep.PatchSite("V0", layer=0, kv_head=1),
                                     "v", 0, good, device, model_dtype)
            assert (single.col_start, single.col_stop) == (geom["head_dim"],
                                                           2 * geom["head_dim"])
            assert single.value.shape == (geom["head_dim"],)
        finally:
            model.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Object taxonomy and failure handling
# ═══════════════════════════════════════════════════════════════════════════════


def test_bare_k_objects_are_refused_on_rope_architectures():
    """``K0``/``Kmid`` are ambiguous on Qwen and must name a RoPE variant (``02`` §6.4)."""
    with pytest.raises(ValueError, match="prerope"):
        cep._object_parts("K0", rope_inside=True)
    # …but are unambiguous where RoPE is not applied inside the attention module.
    assert cep._object_parts("K0", rope_inside=False) == ("k", "zero", "prerope")
    with pytest.raises(ValueError, match="Unknown patch object"):
        cep._object_parts("Q0", rope_inside=False)


def test_missing_source_is_recorded_not_raised():
    """A data-level failure writes a row with ``status``/``warning`` (``02`` §6.5)."""
    with tempfile.TemporaryDirectory(prefix="cep_fail_", ignore_cleanup_errors=True) as t:
        model = _Model(Path(t))
        try:
            spec = cep.PatchSpec(site=cep.PatchSite("V0", layer=1),
                                 norm_condition="direct", source_item_id="absent", seed=0)
            out = _run(model.handle, _item("target", TARGET_IDS), [spec], {},
                       capture_sink=False)
            row = out["patches"][0]
            assert row["status"] == "failed"
            assert "absent" in row["warning"]
        finally:
            model.close()


def test_patch_spec_requires_a_source_unless_identity():
    with pytest.raises(ValueError, match="source_item_id"):
        cep.PatchSpec(site=cep.PatchSite("V0", layer=0), norm_condition="direct")
    with pytest.raises(ValueError, match="norm condition"):
        cep.PatchSpec(site=cep.PatchSite("V0", layer=0), norm_condition="scaled",
                      source_item_id="s")
    # identity legitimately has no source.
    cep.PatchSpec(site=cep.PatchSite("V0", layer=0), norm_condition="identity")
