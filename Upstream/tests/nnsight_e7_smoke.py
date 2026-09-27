# -*- coding: utf-8 -*-
"""nnsight_e7_smoke.py — offline end-to-end smoke for the E7 patching machinery (WP7/WP9).

Random tiny Qwen2 checkpoint with ``num_attention_heads=8, num_key_value_heads=2``
(``06_TEST_PLAN.md`` §1), CPU, fp32, no downloads. Exercises every patch object crossed
with every norm condition through ``capture_sources`` + ``run_patched``, plus the
continuous identity control that ``04`` §5.4 requires during production runs.

Run:

    ./.venv/Scripts/python.exe tests/nnsight_e7_smoke.py

Known Windows flake, shared with the other smoke scripts: the process can exit non-zero at
TemporaryDirectory teardown while safetensors still holds an mmap. Every assertion runs
before that point, and the success line is printed first.
"""

from __future__ import annotations

import gc
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cross_example_patching as cep  # noqa: E402
from corpus_providers import Corpus, CorpusItem, compute_manifest_sha256  # noqa: E402
from depth_band import normalised_depth_band  # noqa: E402
from fingerprint_runner import load_handle  # noqa: E402

from nnsight_smoke_utils import create_tiny_qwen_checkpoint  # noqa: E402

TARGET_IDS = list(range(3, 25))
SOURCE_IDS = list(range(9, 27))
CANDIDATES = [[5], [6, 7], [8]]          # deliberately mixed lengths


def _corpus(items):
    items = tuple(items)
    return Corpus(corpus_id="e7_smoke", items=items, tokenizer_name="tiny",
                  tokenizer_revision=None, add_special_tokens=False, cut_length=None,
                  seed=0, manifest_sha256=compute_manifest_sha256(items),
                  provenance={"schema": "corpus_v1", "source": "nnsight_e7_smoke"})


def _item(item_id, ids):
    return CorpusItem(item_id=item_id, text="", input_ids=list(ids),
                      n_tokens=len(ids), meta={})


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="e7_smoke_", ignore_cleanup_errors=True) as t:
        path = Path(t) / "qwen"
        create_tiny_qwen_checkpoint(path, seed=4003)
        handle = load_handle("qwen", str(path), dtype="float32", device="cpu",
                             local_files_only=True)
        try:
            geom = cep.geometry(handle)
            assert (geom["num_heads"], geom["num_kv_heads"]) == (8, 2), geom
            num_layers = geom["num_layers"]
            band_start, band_end, _meta = normalised_depth_band(num_layers)

            layer = num_layers // 2
            objects = list(cep.PATCH_OBJECTS)
            sites = [cep.PatchSite(obj, layer=layer) for obj in objects]

            target = _item("target", TARGET_IDS)
            source = _item("source", SOURCE_IDS)
            sources = cep.capture_sources(handle, _corpus([source]), sites)

            assert set(sources) == {"source"}
            cache = sources["source"]
            assert cache.seq_len == len(SOURCE_IDS)
            for site in sites:
                tensor = cache.tensors[site.key()]
                assert tensor.dtype == torch.float32, site.key()
                assert tensor.device.type == "cpu", site.key()
                surface, _kind, _variant = cep._object_parts(
                    site.object, rope_inside=True)
                expected = ((geom["hidden"],) if surface == "r"
                            else (geom["num_kv_heads"], geom["head_dim"]))
                assert tuple(tensor.shape) == expected, (site.key(), tensor.shape)
                position = cache.resolved_positions[site.key()]
                assert position in (0, len(SOURCE_IDS) // 2), (site.key(), position)

            # Every object x every norm condition, in one call.
            specs = []
            for site in sites:
                for condition in cep.NORM_CONDITIONS:
                    specs.append(cep.PatchSpec(
                        site=site, norm_condition=condition,
                        source_item_id=None if condition == "identity" else "source",
                        seed=17))

            carrier_heads = [(layer, 0), (layer, geom["num_heads"] - 1)]
            result = cep.run_patched(
                handle, target, specs, sources,
                score_candidates=CANDIDATES, gold_index=1,
                capture_sink=True, band=(band_start, band_end),
                carrier_heads=carrier_heads)

            assert result["target_seq_len"] == len(TARGET_IDS)
            assert len(result["patches"]) == len(objects) * len(cep.NORM_CONDITIONS)
            baseline = result["baseline"]
            assert 0.0 <= baseline["sink"] <= 1.0, baseline["sink"]
            assert len(baseline["candidate_logprobs"]) == len(CANDIDATES)
            assert baseline["prediction"] in range(len(CANDIDATES))

            n_identity = 0
            n_moved = 0
            for row in result["patches"]:
                assert row["status"] == "ok", (row["patch_object"],
                                               row["norm_condition"], row["warning"])
                assert 0.0 <= row["patched_sink"] <= 1.0
                assert row["jsd_output"] >= 0.0
                assert row["patch_position"] in (0, len(TARGET_IDS) // 2)
                assert row["patch_registry_version"] == cep.PATCH_REGISTRY_VERSION

                if row["norm_condition"] == "identity":
                    # 04 §5.4: the continuous correctness monitor.
                    n_identity += 1
                    assert abs(row["margin_delta"]) < cep.IDENTITY_TOLERANCE, row
                    assert row["jsd_output"] < cep.EXACTNESS_TOLERANCE, row
                else:
                    if abs(row["margin_delta"]) > cep.EXACTNESS_TOLERANCE:
                        n_moved += 1
                    # rescaled/random land on the target's norm; direct does not have to.
                    if row["norm_condition"] in ("rescaled", "random"):
                        assert abs(row["patched_norm"] - row["target_norm"]) < 1e-5, row

            assert n_identity == len(objects), n_identity
            assert n_moved > 0, "no non-identity patch changed anything — writer inert?"

            # Cross-model patching must fail loudly rather than run.
            other = Path(t) / "qwen_other"
            create_tiny_qwen_checkpoint(other, seed=8123)
            other_handle = load_handle("qwen", str(other), dtype="float32", device="cpu",
                                       local_files_only=True)
            try:
                assert cep.model_fingerprint(other_handle) != \
                    cep.model_fingerprint(handle)
                try:
                    cep.run_patched(other_handle, target, specs[:1], sources,
                                    capture_sink=False)
                except ValueError as exc:
                    assert "different model" in str(exc)
                else:  # pragma: no cover - the raise is the point of the check
                    raise AssertionError("cross-model patching was not refused")
            finally:
                del other_handle
                gc.collect()

            print(f"  objects={len(objects)} norm_conditions={len(cep.NORM_CONDITIONS)} "
                  f"rows={len(result['patches'])} identity_rows={n_identity} "
                  f"moved_rows={n_moved} band=[{band_start},{band_end})")
            print("E7 NNsight offline smoke test passed")
        finally:
            del handle
            gc.collect()


if __name__ == "__main__":
    main()
