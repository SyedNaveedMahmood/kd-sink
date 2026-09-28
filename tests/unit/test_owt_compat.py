"""Independent fixture parity for the audited historical OWT recipe."""

import hashlib
import json
import random

import pytest

from sinklab.data import save_manifest
from sinklab.order import UpdateOrder
from sinklab.owt_compat import (OWTError, prepare_owt_corpus, prepare_owt_panels,
                                upstream_normalize, validate_calibration_blocks_export,
                                validate_owt_corpus,
                                validate_owt_panels)
from sinklab.provenance import canonical_json_bytes, seal_payload
from sinklab.training_entry import _require_s1_data_lock


WINDOWS = {"training": (0, 4), "evaluation": (4, 8), "calibration": (8, 12)}


class Tokenizer:
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) % 37 + 1 for c in text]}


class BatchedTokenizer(Tokenizer):
    is_fast = True

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [super(BatchedTokenizer, self).__call__(
                item, add_special_tokens=add_special_tokens)["input_ids"] for item in text]}
        return super().__call__(text, add_special_tokens=add_special_tokens)


def _fixture():
    rows = [{"text": text} for text in (
        " ab  cd ", "ef\n gh", "", "ij", "kl", "mn\t op", "qr", "st",
        "uv", "wx", "yz", " 123  45 ")]
    document = prepare_owt_corpus(rows, Tokenizer(), seed=7,
        dataset_revision="a" * 40, tokenizer_revision="b" * 40,
        tokenizer_sha256="c" * 64, windows=WINDOWS, block_size=4)
    return rows, document


def test_upstream_document_windows_normalization_shuffle_eos_blocks_and_hash(tmp_path):
    rows, document = _fixture()
    payload, digest = validate_owt_corpus(document, tokenizer_sha256="c" * 64,
                                         expected_windows=WINDOWS)
    assert digest == document["sha256"]
    assert upstream_normalize("  a\n\tb  c  ") == "a b c"
    assert upstream_normalize(None) == ""
    for split, (start, stop) in WINDOWS.items():
        subset = rows[start:stop]
        order = list(range(len(subset)))
        random.Random(7).shuffle(order)  # independent historical reference
        ids, owners, kept = [], [], []
        for local in order:
            text = " ".join(str(subset[local]["text"]).strip().split())
            if not text:
                continue
            tokens = [ord(c) % 37 + 1 for c in text]
            kept.append(start + local)
            ids += tokens + [99]
            owners += [start + local] * (len(tokens) + 1)
        part = payload["partitions"][split]
        assert [d["source_index"] for d in part["documents"]] == kept
        assert [b["token_ids"] for b in part["blocks"]] == [
            ids[i:i + 4] for i in range(0, len(ids) - 3, 4)]
        assert [b["source_indices"] for b in part["blocks"]] == [
            sorted(set(owners[i:i + 4])) for i in range(0, len(ids) - 3, 4)]
        assert part["tail_token_ids"] == ids[len(part["blocks"]) * 4:]
        assert all(d["normalized_text_sha256"] == hashlib.sha256(
            upstream_normalize(rows[d["source_index"]]["text"]).encode()).hexdigest()
            for d in part["documents"])
    path = save_manifest(document, tmp_path, "owt-corpus")
    assert json.loads(path.read_text())["sha256"] == digest
    altered = json.loads(path.read_text())["payload"]
    altered["partitions"]["training"]["blocks"][0]["token_ids"][0] += 1
    with pytest.raises(OWTError, match="block IDs"):
        validate_owt_corpus(seal_payload(altered), tokenizer_sha256="c" * 64,
                            expected_windows=WINDOWS)


def test_batched_tokenization_preserves_exact_pack_and_manifest():
    rows, one_at_a_time = _fixture()
    batched = prepare_owt_corpus(rows, BatchedTokenizer(), seed=7,
        dataset_revision="a" * 40, tokenizer_revision="b" * 40,
        tokenizer_sha256="c" * 64, windows=WINDOWS, block_size=4)
    assert batched == one_at_a_time


def test_upstream_epoch_order_matches_reference_and_resume_preserves_suffix():
    blocks = [str(i) for i in range(5)]
    order = UpdateOrder(blocks, seed=7, scheme="upstream-owt-epoch-v1")
    first = list(blocks)
    random.Random((7 + 1) * 100003).shuffle(first)
    assert order.permutation == first
    observed = [order._next()["block_id"] for _ in range(7)]
    expected_next = list(blocks)
    random.Random((7 + 1) * 100003 + 1).shuffle(expected_next)
    assert observed == first + expected_next[:2]
    resumed = UpdateOrder.resume(blocks, order.snapshot(), expected_scheme="upstream-owt-epoch-v1")
    assert [order._next() for _ in range(12)] == [resumed._next() for _ in range(12)]
    with pytest.raises(ValueError, match="scheme changed"):
        UpdateOrder.resume(blocks, order.snapshot(), expected_scheme="v2-horizon-independent")


def test_historical_sink_and_following_ppl_regions_are_disjoint():
    class LongTokenizer:
        eos_token_id = 99
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1] * ({"train": 128, "eval": 2300 * 128,
                                          "cal": 1024 * 128}[text])}
    windows = {"training": (0, 1), "evaluation": (1, 2), "calibration": (2, 3)}
    corpus = prepare_owt_corpus([{"text": x} for x in ("train", "eval", "cal")],
        LongTokenizer(), seed=0, dataset_revision="a" * 40,
        tokenizer_revision="b" * 40, tokenizer_sha256="c" * 64,
        windows=windows)
    panels = prepare_owt_panels(corpus, expected_windows=windows)
    validate_owt_panels(panels, corpus, expected_windows=windows)
    p = panels["payload"]
    assert p["owt_dense64"] == p["owt_full300"][:64]
    assert len(p["owt_full300"]) == 300 and len(p["owt_lm2000"]) == 2000
    assert set(p["owt_full300"]).isdisjoint(p["owt_lm2000"])
    assert len(p["calibration16x64"]) == 16


def test_production_s1_data_lock_binds_source_tokenizer_corpus_and_panels():
    _, document = _fixture()
    corpus = document["payload"]
    data = {"recipe": "owt-upstream-gpt2-pack-v1",
            "dataset_revision": corpus["dataset"]["revision"],
            "tokenizer_revision": corpus["tokenizer"]["revision"],
            "tokenizer_files_sha256": corpus["tokenizer"]["files_sha256"],
            "production_corpus_sha256": document["sha256"],
            "frozen_panels_sha256": "d" * 64}
    _require_s1_data_lock({"data": data}, corpus, document["sha256"], "d" * 64)
    for key in ("dataset_revision", "tokenizer_revision", "tokenizer_files_sha256",
                "production_corpus_sha256", "frozen_panels_sha256"):
        bad = dict(data)
        bad[key] = "e" * len(bad[key])
        with pytest.raises(ValueError, match="differ"):
            _require_s1_data_lock({"data": bad}, corpus, document["sha256"], "d" * 64)


def test_calibration_export_requires_exact_16x64_128token_masks_and_window():
    batches = []
    for batch_index in range(16):
        batch = []
        for within_batch in range(64):
            index = batch_index * 64 + within_batch
            ids, sources = [index % 50257] * 128, [408000 + index]
            ident = hashlib.sha256(canonical_json_bytes({
                "split": "calibration", "block_index": index,
                "token_ids": ids, "source_indices": sources})).hexdigest()
            batch.append({"id": ident, "block_index": index,
                          "token_ids": ids, "source_indices": sources})
        batches.append(batch)
    payload = {"kind": "s1-calibration16x64-blocks-v1",
               "corpus_sha256": "a" * 64, "panels_sha256": "b" * 64,
               "dataset_revision": "c" * 40, "tokenizer_files_sha256": "d" * 64,
               "source_window": [408000, 416000], "block_size": 128,
               "attention_mask_rule": "all_ones_128_no_padding", "batches": batches}
    validate_calibration_blocks_export(seal_payload(payload))
    payload["batches"][0][0]["source_indices"] = [407999]
    with pytest.raises(OWTError, match="ownership"):
        validate_calibration_blocks_export(seal_payload(payload))
