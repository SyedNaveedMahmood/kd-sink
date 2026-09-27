import copy
import hashlib

import pytest

from sinklab.data import (DataError, document_hash, document_split, load_corpus, load_manifest,
                          normalized_text, prepare_corpus, save_manifest, validate_corpus)
from sinklab.provenance import seal_payload


class TinyTokenizer:
    eos_token_id = 999

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) for char in text]


def row_for(split, start=0, length=140):
    for n in range(start, start + 200000):
        text = str(n).zfill(length)
        if document_split(document_hash(text)) == split:
            return text
    raise AssertionError("could not find synthetic split fixture")


def prepared():
    train = row_for("training")
    cal = row_for("calibration")
    evaluation = row_for("evaluation")
    rows = [
        {"source_index": 3, "document_id": "train-duplicate", "text": train},
        {"source_index": 2, "document_id": "evaluation", "text": evaluation},
        {"source_index": 0, "document_id": "train", "text": train},
        {"source_index": 1, "document_id": "calibration", "text": cal},
    ]
    return prepare_corpus(rows, TinyTokenizer(), dataset_id="synthetic", dataset_revision="v1",
        license_id="fixture-only", tokenizer_id="tiny", tokenizer_revision="v1",
        tokenizer_sha256="a" * 64)


def test_normalization_hash_split_and_exact_dedup():
    assert normalized_text("  e\u0301\r\n") == "é"
    assert document_hash("  e\u0301\r\n") == hashlib.sha256("é".encode()).hexdigest()
    with pytest.raises(DataError, match="empty"):
        normalized_text(" \r\n")
    manifest = prepared()
    assert manifest["sha256"] == prepared()["sha256"]
    payload, digest = validate_corpus(manifest, tokenizer_sha256="a" * 64)
    assert len(digest) == 64
    all_docs = [d for part in payload["partitions"].values() for d in part["documents"]]
    assert len(all_docs) == len({d["sha256"] for d in all_docs}) == 3
    assert [d["document_id"] for d in payload["partitions"]["training"]["documents"]] == ["train"]
    assert all(document_split(d["sha256"]) == split
               for split, part in payload["partitions"].items() for d in part["documents"])


def test_packing_tail_and_integrity(tmp_path):
    manifest = prepared()
    part = manifest["payload"]["partitions"]["training"]
    assert len(part["blocks"]) == 1 and part["dropped_tail_tokens"] == 12
    assert part["blocks"][0]["token_ids"] == [ord("0")] * 128
    path = save_manifest(manifest, tmp_path, "corpus")
    assert load_manifest(path, stem="corpus")[1] == manifest["sha256"]
    assert load_corpus(path, tokenizer_sha256="a" * 64)[1] == manifest["sha256"]
    with pytest.raises(DataError, match="tokenizer"):
        load_corpus(path, tokenizer_sha256="b" * 64)
    with pytest.raises(DataError, match="tokenizer"):
        validate_corpus(manifest, tokenizer_sha256="b" * 64)
    corrupt = copy.deepcopy(manifest)
    corrupt["payload"]["partitions"]["training"]["blocks"][0]["token_ids"][0] = 7
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_corpus(corrupt)
    bad_semantics = seal_payload({**manifest["payload"], "partitions": copy.deepcopy(manifest["payload"]["partitions"])})
    bad_semantics["payload"]["partitions"]["training"]["documents"][0]["token_count"] = 1
    with pytest.raises(DataError, match="boundary"):
        validate_corpus(seal_payload(bad_semantics["payload"]))


def test_exactly_one_eos_between_documents():
    a = row_for("training", length=130)
    b = row_for("training", start=300000, length=130)
    manifest = prepare_corpus([
        {"source_index": 0, "document_id": "a", "text": a},
        {"source_index": 1, "document_id": "b", "text": b}], TinyTokenizer(),
        dataset_id="synthetic", dataset_revision="v1", license_id="fixture-only",
        tokenizer_id="tiny", tokenizer_revision="v1", tokenizer_sha256="a" * 64)
    docs = manifest["payload"]["partitions"]["training"]["documents"]
    part = manifest["payload"]["partitions"]["training"]
    stream = [x for block in part["blocks"] for x in block["token_ids"]] + part["tail_token_ids"]
    assert docs[1]["start"] == docs[0]["end"] + 1
    assert stream[docs[1]["start"] - 1] == 999
    validate_corpus(manifest)
