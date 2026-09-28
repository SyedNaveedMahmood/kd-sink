"""Pinned S1 OpenWebText windows and historical GPT-2 block packing.

The algorithm is independently implemented from the audited historical recipe.
This module never imports or reads the reference tree at runtime.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Iterable

from .data import load_manifest
from .provenance import COMMIT_PATTERN, SHA256_PATTERN, canonical_json_bytes, seal_payload, verify_envelope


class OWTError(ValueError):
    pass


WINDOWS = {"training": (0, 400_000), "evaluation": (400_000, 408_000),
           "calibration": (408_000, 416_000)}
RECIPE = "owt-upstream-gpt2-pack-v1"


class FastLocalGPT2Tokenizer:
    """Batch encode the pinned local GPT-2 tokenizer.json with the Rust backend."""

    is_fast = True

    def __init__(self, directory: str | Path):
        from tokenizers import Tokenizer

        path = Path(directory) / "tokenizer.json"
        if not path.is_file():
            raise OWTError("pinned local tokenizer.json is required")
        self.backend = Tokenizer.from_file(str(path))
        self.eos_token_id = self.backend.token_to_id("<|endoftext|>")
        if self.eos_token_id != 50256:
            raise OWTError("local GPT-2 EOS ID differs from the pinned token set")

    def __call__(self, texts, *, add_special_tokens: bool):
        if add_special_tokens is not False:
            raise OWTError("OWT packing forbids added special tokens")
        if isinstance(texts, str):
            return {"input_ids": self.backend.encode(texts, add_special_tokens=False).ids}
        return {"input_ids": [row.ids for row in self.backend.encode_batch(
            texts, add_special_tokens=False)]}


def upstream_normalize(value) -> str:
    """Historical whitespace collapse; no NFC or case conversion."""
    return "" if value is None else " ".join(str(value).strip().split())


def _digest(value: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def prepare_owt_corpus(rows: Iterable[dict], tokenizer, *, seed: int,
                       dataset_revision: str, tokenizer_revision: str,
                       tokenizer_sha256: str, windows: dict | None = None,
                       block_size: int = 128) -> dict:
    """Consume one pinned source stream; shuffle each disjoint document window."""
    windows = dict(WINDOWS if windows is None else windows)
    if (set(windows) != set(WINDOWS) or any(type(v) not in (tuple, list) or len(v) != 2
            or any(type(x) is not int for x in v) or v[0] < 0 or v[1] <= v[0]
            for v in windows.values()) or
            not (windows["training"][1] == windows["evaluation"][0] and
                 windows["evaluation"][1] == windows["calibration"][0])):
        raise OWTError("three ordered, disjoint training/evaluation/calibration windows required")
    if type(seed) is not int or seed < 0 or block_size < 2:
        raise OWTError("explicit nonnegative seed and block size >=2 required")
    if (not COMMIT_PATTERN.fullmatch(dataset_revision) or
            not COMMIT_PATTERN.fullmatch(tokenizer_revision) or
            not SHA256_PATTERN.fullmatch(tokenizer_sha256)):
        raise OWTError("immutable dataset/tokenizer revisions and tokenizer file SHA-256 required")
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise OWTError("GPT-2 tokenizer EOS required")
    selected = {split: [] for split in WINDOWS}
    end = windows["calibration"][1]
    count = 0
    for index, row in enumerate(rows):
        if index >= end:
            break
        count += 1
        for split, (start, stop) in windows.items():
            if start <= index < stop:
                if not isinstance(row, dict) or "text" not in row:
                    raise OWTError("source row requires text")
                selected[split].append(upstream_normalize(row["text"]))
                break
    if count < end or any(len(selected[s]) != b - a for s, (a, b) in windows.items()):
        raise OWTError("source train stream ended before the frozen document windows")
    partitions = {}
    for split, texts in selected.items():
        start, stop = windows[split]
        order = list(range(len(texts)))
        random.Random(seed).shuffle(order)
        stream, owners, docs, skipped = [], [], [], []
        for offset in range(0, len(order), 1000):
            group = [(local_index, texts[local_index])
                     for local_index in order[offset:offset + 1000] if texts[local_index]]
            if group and getattr(tokenizer, "is_fast", False):
                encoded = tokenizer([text for _, text in group],
                                    add_special_tokens=False)["input_ids"]
            else:
                encoded = [tokenizer(text, add_special_tokens=False)["input_ids"]
                           for _, text in group]
            if len(encoded) != len(group):
                raise OWTError("batched tokenizer output is not aligned to source documents")
            tokenized = dict(zip((index for index, _ in group), encoded))
            for local_index in order[offset:offset + 1000]:
                text = texts[local_index]
                if not text:
                    skipped.append(start + local_index)
                    continue
                ids = tokenized[local_index]
                if not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids):
                    raise OWTError("tokenizer returned invalid IDs")
                if not ids:
                    skipped.append(start + local_index)
                    continue
                source_index = start + local_index
                docs.append({"source_index": source_index,
                             "normalized_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                             "token_count": len(ids)})
                stream.extend(ids)
                stream.append(eos)  # Historical loop appends EOS after every nonempty document.
                owners.extend([source_index] * (len(ids) + 1))
        blocks = []
        for block_index, offset in enumerate(range(0, len(stream) - block_size + 1, block_size)):
            ids = stream[offset:offset + block_size]
            sources = sorted(set(owners[offset:offset + block_size]))
            blocks.append({"id": _digest({"split": split, "block_index": block_index,
                                           "token_ids": ids, "source_indices": sources}),
                           "block_index": block_index, "token_ids": ids,
                           "source_indices": sources, "n_eos": sum(t == eos for t in ids)})
        source_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        partitions[split] = {"source_window": [start, stop],
            "source_text_sha256": source_hashes,
            "source_window_sha256": _digest({"normalized_text_sha256": source_hashes}),
            "documents": docs, "skipped_source_indices": skipped,
            "blocks": blocks, "tail_token_ids": stream[len(blocks) * block_size:],
            "tail_source_indices": owners[len(blocks) * block_size:],
            "stream_sha256": _digest({"token_ids": stream, "source_indices": owners})}
    return seal_payload({"kind": RECIPE, "dataset": {"id": "Skylion007/openwebtext",
        "revision": dataset_revision, "split": "train"}, "tokenizer": {
        "id": "openai-community/gpt2", "revision": tokenizer_revision,
        "files_sha256": tokenizer_sha256, "eos_token_id": eos,
        "add_special_tokens": False}, "seed": seed, "block_size": block_size,
        "windows": {k: list(v) for k, v in windows.items()}, "partitions": partitions})


def validate_owt_corpus(document: dict, *, tokenizer_sha256: str,
                        expected_windows: dict | None = None) -> tuple[dict, str]:
    payload, digest = verify_envelope(document)
    if (payload.get("kind") != RECIPE or payload.get("dataset", {}).get("id") != "Skylion007/openwebtext"
            or payload["dataset"].get("split") != "train" or
            payload.get("tokenizer", {}).get("id") != "openai-community/gpt2" or
            payload["tokenizer"].get("files_sha256") != tokenizer_sha256 or
            payload["tokenizer"].get("add_special_tokens") is not False or
            set(payload.get("partitions", {})) != set(WINDOWS)):
        raise OWTError("pinned OWT source, tokenizer or partition schema mismatch")
    if (not COMMIT_PATTERN.fullmatch(payload["dataset"].get("revision", "")) or
            not COMMIT_PATTERN.fullmatch(payload["tokenizer"].get("revision", "")) or
            type(payload.get("seed")) is not int or payload["seed"] < 0 or
            type(payload.get("block_size")) is not int or payload["block_size"] < 2 or
            type(payload["tokenizer"].get("eos_token_id")) is not int):
        raise OWTError("immutable revisions, seed, block size and EOS required")
    windows = {k: list(v) for k, v in (WINDOWS if expected_windows is None else expected_windows).items()}
    if payload["windows"] != windows:
        raise OWTError("production document windows differ from approved recipe")
    seed, size, eos = payload["seed"], payload["block_size"], payload["tokenizer"]["eos_token_id"]
    for split, (start, stop) in windows.items():
        part = payload["partitions"][split]
        if part["source_window"] != [start, stop]:
            raise OWTError("source window mismatch")
        docs, skipped, blocks = part["documents"], part["skipped_source_indices"], part["blocks"]
        source_hashes = part["source_text_sha256"]
        if (len(source_hashes) != stop - start or
                part["source_window_sha256"] != _digest({"normalized_text_sha256": source_hashes}) or
                any(type(d["token_count"]) is not int or d["token_count"] < 1 for d in docs) or
                any(d["normalized_text_sha256"] != source_hashes[d["source_index"] - start]
                    for d in docs)):
            raise OWTError("source document hash or window hash mismatch")
        doc_indices = [d["source_index"] for d in docs]
        if (len(doc_indices) + len(skipped) != stop - start or
                len(set(doc_indices + skipped)) != stop - start or
                any(not start <= x < stop for x in doc_indices + skipped)):
            raise OWTError("document ownership does not cover the exact source window")
        order = list(range(start, stop))
        random.Random(seed).shuffle(order)
        skipped_set = set(skipped)
        if doc_indices != [i for i in order if i not in skipped_set]:
            raise OWTError("document shuffle order mismatch")
        owners = [i for d in docs for i in [d["source_index"]] * (d["token_count"] + 1)]
        tokens = [t for b in blocks for t in b["token_ids"]] + part["tail_token_ids"]
        if len(tokens) != len(owners) or any(type(t) is not int or t < 0 for t in tokens):
            raise OWTError("stream token count or ID invalid")
        if part["tail_source_indices"] != owners[len(blocks) * size:]:
            raise OWTError("tail ownership mismatch")
        cursor = 0
        for d in docs:
            cursor += d["token_count"]
            if tokens[cursor] != eos:
                raise OWTError("missing post-document EOS")
            cursor += 1
        if len(blocks) != len(tokens) // size or len(part["tail_token_ids"]) != len(tokens) % size:
            raise OWTError("greedy block boundaries mismatch")
        for index, block in enumerate(blocks):
            ids = block["token_ids"]
            sources = sorted(set(owners[index * size:(index + 1) * size]))
            if (block["block_index"] != index or len(ids) != size or
                    block["source_indices"] != sources or block["n_eos"] != sum(t == eos for t in ids) or
                    block["id"] != _digest({"split": split, "block_index": index,
                                             "token_ids": ids, "source_indices": sources})):
                raise OWTError("block IDs, ownership, EOS or boundaries mismatch")
        if part["stream_sha256"] != _digest({"token_ids": tokens, "source_indices": owners}):
            raise OWTError("stream content hash mismatch")
    return payload, digest


def load_owt_corpus(path: Path, *, tokenizer_sha256: str) -> tuple[dict, str]:
    payload, digest = load_manifest(path, stem="owt-corpus")
    validate_owt_corpus(seal_payload(payload), tokenizer_sha256=tokenizer_sha256)
    return payload, digest


def prepare_owt_panels(corpus: dict, *, expected_windows: dict | None = None) -> dict:
    """Historical first-300 and following-2000 validation regions; v2 dense64."""
    payload, digest = validate_owt_corpus(
        corpus, tokenizer_sha256=corpus["payload"]["tokenizer"]["files_sha256"],
        expected_windows=expected_windows)
    return panels_from_validated_corpus(payload, digest)


def panels_from_validated_corpus(payload: dict, digest: str) -> dict:
    """Freeze panels after the caller has verified the complete corpus once."""
    evaluation = [b["id"] for b in payload["partitions"]["evaluation"]["blocks"]]
    calibration = [b["id"] for b in payload["partitions"]["calibration"]["blocks"]]
    if len(evaluation) < 2300 or len(calibration) < 1024:
        raise OWTError("insufficient frozen evaluation or calibration blocks")
    return seal_payload({"kind": "owt-upstream-panels-v1", "corpus_sha256": digest,
        "owt_dense64": evaluation[:64], "owt_full300": evaluation[:300],
        "owt_lm2000": evaluation[300:2300],
        "calibration16x64": [calibration[i:i + 64] for i in range(0, 1024, 64)]})


def calibration_blocks_export(corpus_payload: dict, corpus_digest: str,
                              panel_document: dict) -> dict:
    """Export just the frozen 16x64 training-only blocks for the 3090 handoff."""
    panel, panel_digest = verify_envelope(panel_document)
    expected = panels_from_validated_corpus(corpus_payload, corpus_digest)["payload"]
    if panel != expected:
        raise OWTError("calibration export requires the exact validated frozen panels")
    blocks = {block["id"]: block for block in corpus_payload["partitions"]["calibration"]["blocks"]}
    batches = []
    for ids in panel["calibration16x64"]:
        batch = []
        for ident in ids:
            block = blocks[ident]
            batch.append({"id": ident, "block_index": block["block_index"],
                          "token_ids": block["token_ids"],
                          "source_indices": block["source_indices"]})
        batches.append(batch)
    result = seal_payload({"kind": "s1-calibration16x64-blocks-v1",
        "corpus_sha256": corpus_digest, "panels_sha256": panel_digest,
        "dataset_revision": corpus_payload["dataset"]["revision"],
        "tokenizer_files_sha256": corpus_payload["tokenizer"]["files_sha256"],
        "source_window": list(WINDOWS["calibration"]),
        "block_size": corpus_payload["block_size"],
        "attention_mask_rule": "all_ones_128_no_padding", "batches": batches})
    validate_calibration_blocks_export(result)
    return result


def validate_calibration_blocks_export(document: dict) -> tuple[dict, str]:
    payload, digest = verify_envelope(document)
    if (payload.get("kind") != "s1-calibration16x64-blocks-v1" or
            payload.get("source_window") != [408_000, 416_000] or
            payload.get("block_size") != 128 or
            not SHA256_PATTERN.fullmatch(payload.get("corpus_sha256", "")) or
            not SHA256_PATTERN.fullmatch(payload.get("panels_sha256", "")) or
            not SHA256_PATTERN.fullmatch(payload.get("tokenizer_files_sha256", "")) or
            not COMMIT_PATTERN.fullmatch(payload.get("dataset_revision", "")) or
            payload.get("attention_mask_rule") != "all_ones_128_no_padding"):
        raise OWTError("calibration block export has invalid source or artifact identity")
    batches = payload.get("batches")
    if not isinstance(batches, list) or len(batches) != 16 or any(
            not isinstance(batch, list) or len(batch) != 64 for batch in batches):
        raise OWTError("calibration export requires exactly 16 effective batches of 64 blocks")
    seen = set()
    for batch_index, batch in enumerate(batches):
        for within_batch, block in enumerate(batch):
            ids, sources, ident = block["token_ids"], block["source_indices"], block["id"]
            if (not isinstance(ids, list) or len(ids) != 128 or
                    any(type(token) is not int or not 0 <= token <= 50256 for token in ids) or
                    not isinstance(sources, list) or not sources or
                    any(type(index) is not int or not 408_000 <= index < 416_000 for index in sources) or
                    sources != sorted(set(sources)) or ident in seen or
                    type(block.get("block_index")) is not int or
                    block["block_index"] != batch_index * 64 + within_batch):
                raise OWTError("calibration export block, mask or source ownership invalid")
            seen.add(ident)
            if ident != _digest({"split": "calibration", "block_index": block["block_index"],
                                 "token_ids": ids, "source_indices": sources}):
                raise OWTError("calibration block identity mismatch")
    return payload, digest


def validate_owt_panels(document: dict, corpus: dict, *, expected_windows: dict | None = None) -> str:
    payload, digest = verify_envelope(document)
    if payload != prepare_owt_panels(corpus, expected_windows=expected_windows)["payload"]:
        raise OWTError("OWT panel membership or source corpus mismatch")
    return digest
