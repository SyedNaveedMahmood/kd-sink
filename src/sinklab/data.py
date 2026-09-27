"""Offline, deterministic document preparation and verified token blocks."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from .provenance import LockError, _no_duplicate_keys, canonical_json_bytes, seal_payload, verify_envelope


class DataError(ValueError):
    """Prepared data violates its declared contract."""


def normalized_text(value: str) -> str:
    if not isinstance(value, str):
        raise DataError("document text must be a string")
    result = unicodedata.normalize("NFC", value.replace("\r\n", "\n")).strip()
    if not result:
        raise DataError("empty normalized document")
    return result


def document_hash(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def document_split(digest: str) -> str:
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise DataError("invalid document SHA-256")
    bucket = int(digest[:16], 16) % 1000
    return "evaluation" if bucket < 10 else "calibration" if bucket < 20 else "training"


def local_tokenizer(path: str | Path):
    """Load only operator-provided tokenizer files; never resolve a Hub name."""
    path = Path(path)
    if not path.is_dir() or not any(path.iterdir()):
        raise DataError("tokenizer must be a populated local directory")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise DataError("tokenizer has no EOS token")
    return tokenizer


def tokenizer_files_hash(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise DataError("tokenizer path is not a directory")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files or any(p.is_symlink() for p in files):
        raise DataError("tokenizer files missing or symlinked")
    h = hashlib.sha256()
    for file in files:
        name = file.relative_to(root).as_posix().encode("utf-8")
        contents = file.read_bytes()
        h.update(len(name).to_bytes(8, "big") + name)
        h.update(len(contents).to_bytes(8, "big") + contents)
    return h.hexdigest()


def _ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
        raise DataError("tokenizer returned invalid or empty token IDs")
    return ids


def prepare_corpus(rows: Iterable[dict[str, Any]], tokenizer: Any, *,
                   dataset_id: str, dataset_revision: str, license_id: str,
                   tokenizer_id: str, tokenizer_revision: str,
                   tokenizer_sha256: str, block_size: int = 128) -> dict[str, Any]:
    if block_size != 128:
        raise DataError("Stage01 requires 128-token blocks")
    for value in (dataset_id, dataset_revision, license_id, tokenizer_id, tokenizer_revision):
        if not isinstance(value, str) or not value.strip():
            raise DataError("dataset/tokenizer identity and license fields are required")
    if len(tokenizer_sha256) != 64:
        raise DataError("tokenizer file SHA-256 is required")
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise DataError("tokenizer EOS ID is invalid")
    ordered = []
    source_keys = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"source_index", "document_id", "text"}:
            raise DataError("row requires exactly source_index, document_id, text")
        index, ident = row["source_index"], row["document_id"]
        if type(index) is not int or index < 0 or not isinstance(ident, str) or not ident:
            raise DataError("invalid source index or document ID")
        if index in source_keys:
            raise DataError("duplicate source index")
        source_keys.add(index)
        norm = normalized_text(row["text"])
        ordered.append((index, document_hash(norm), ident, norm))
    ordered.sort(key=lambda item: (item[0], item[1]))
    partitions: dict[str, list[tuple[int, str, str, str]]] = {
        "training": [], "calibration": [], "evaluation": []}
    seen = set()
    for item in ordered:
        if item[1] not in seen:
            seen.add(item[1])
            partitions[document_split(item[1])].append(item)
    result: dict[str, Any] = {"kind": "corpus-v1", "preprocessing": "nfc-crlf-strip-sha64mod1000-eos-between-v1",
        "dataset": {"id": dataset_id, "revision": dataset_revision, "license": license_id},
        "tokenizer": {"id": tokenizer_id, "revision": tokenizer_revision,
                      "files_sha256": tokenizer_sha256, "eos_token_id": eos},
        "block_size": block_size, "partitions": {}}
    for split, items in partitions.items():
        stream: list[int] = []
        docs = []
        for index, digest, ident, norm in items:
            if stream:
                stream.append(eos)
            start = len(stream)
            tokens = _ids(tokenizer, norm)
            stream.extend(tokens)
            docs.append({"source_index": index, "document_id": ident, "sha256": digest,
                         "start": start, "end": len(stream), "token_count": len(tokens)})
        blocks = []
        for start in range(0, len(stream) - block_size + 1, block_size):
            ids = stream[start:start + block_size]
            block_id = hashlib.sha256(canonical_json_bytes({"split": split,
                "offset": start, "token_ids": ids})).hexdigest()
            blocks.append({"id": block_id, "offset": start, "token_ids": ids})
        result["partitions"][split] = {"documents": docs, "blocks": blocks,
            "stream_token_count": len(stream), "dropped_tail_tokens": len(stream) % block_size,
            "tail_token_ids": stream[len(blocks) * block_size:],
            "stream_sha256": hashlib.sha256(canonical_json_bytes({"token_ids": stream})).hexdigest()}
    return seal_payload(result)


def validate_corpus(document: Any, *, tokenizer_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    payload, digest = verify_envelope(document)
    if payload.get("kind") != "corpus-v1" or payload.get("block_size") != 128:
        raise DataError("unsupported corpus manifest")
    if tokenizer_sha256 is not None and payload["tokenizer"]["files_sha256"] != tokenizer_sha256:
        raise DataError("tokenizer file hash mismatch")
    seen = set()
    eos = payload["tokenizer"]["eos_token_id"]
    if set(payload["partitions"]) != {"training", "calibration", "evaluation"}:
        raise DataError("missing corpus partition")
    for split, part in payload["partitions"].items():
        docs, blocks = part["documents"], part["blocks"]
        if [d["source_index"] for d in docs] != sorted(d["source_index"] for d in docs):
            raise DataError("document source order mismatch")
        position = 0
        for i, doc in enumerate(docs):
            digest_doc = doc["sha256"]
            if digest_doc in seen or document_split(digest_doc) != split:
                raise DataError("duplicate or mispartitioned document")
            seen.add(digest_doc)
            if i:
                position += 1
            if doc["start"] != position or doc["end"] != position + doc["token_count"] or doc["token_count"] < 1:
                raise DataError("invalid document boundary")
            position = doc["end"]
        if position != part["stream_token_count"] or part["dropped_tail_tokens"] != position % 128:
            raise DataError("invalid stream length")
        if len(blocks) != position // 128:
            raise DataError("invalid block count")
        for i, block in enumerate(blocks):
            ids = block["token_ids"]
            expected = hashlib.sha256(canonical_json_bytes({"split": split,
                "offset": i * 128, "token_ids": ids})).hexdigest()
            if block["offset"] != i * 128 or len(ids) != 128 or block["id"] != expected:
                raise DataError("invalid block content or order")
        stream = [token for block in blocks for token in block["token_ids"]] + part["tail_token_ids"]
        if any(type(token) is not int or token < 0 for token in stream):
            raise DataError("invalid token ID")
        if len(stream) != position or len(part["tail_token_ids"]) != part["dropped_tail_tokens"]:
            raise DataError("invalid stream tail")
        if hashlib.sha256(canonical_json_bytes({"token_ids": stream})).hexdigest() != part["stream_sha256"]:
            raise DataError("stream hash mismatch")
        for doc in docs[1:]:
            eos_offset = doc["start"] - 1
            if stream[eos_offset] != eos:
                raise DataError("missing inter-document EOS")
    return payload, digest


def save_manifest(document: dict[str, Any], directory: str | Path, stem: str) -> Path:
    _, digest = verify_envelope(document)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{stem}-{digest}.json"
    contents = canonical_json_bytes(document) + b"\n"
    if path.exists() and path.read_bytes() != contents:
        raise DataError("content-addressed manifest collision")
    path.write_bytes(contents)
    return path


def load_manifest(path: str | Path, *, stem: str, expected_digest: str | None = None) -> tuple[dict[str, Any], str]:
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
        payload, digest = verify_envelope(document)
    except (OSError, UnicodeError, json.JSONDecodeError, LockError) as exc:
        raise DataError(f"cannot verify manifest: {exc}") from exc
    if path.name != f"{stem}-{digest}.json" or (expected_digest is not None and digest != expected_digest):
        raise DataError("manifest filename or expected digest mismatch")
    return payload, digest


def load_corpus(path: str | Path, *, tokenizer_sha256: str,
                expected_digest: str | None = None) -> tuple[dict[str, Any], str]:
    """Verify envelope, content address, tokenizer identity and corpus invariants."""
    payload, digest = load_manifest(path, stem="corpus", expected_digest=expected_digest)
    validate_corpus(seal_payload(payload), tokenizer_sha256=tokenizer_sha256)
    return payload, digest
