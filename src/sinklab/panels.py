"""Frozen offline evaluation, calibration and domain panels."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from .data import DataError, _ids, document_hash, normalized_text, validate_corpus
from .provenance import seal_payload, verify_envelope


SALT = "e6a-v2-panels"
DOMAIN_FIELDS = {"sst2": ("validation", "sentence"),
                 "gsm8k": ("test", "question"),
                 "humaneval": ("test", "prompt")}


def _rank(identifier: str, label: str) -> str:
    return hashlib.sha256(f"{SALT}:{label}:{identifier}".encode("utf-8")).hexdigest()


def prepare_owt_panels(corpus: dict[str, Any]) -> dict[str, Any]:
    payload, corpus_digest = validate_corpus(corpus)
    evaluation = [b["id"] for b in payload["partitions"]["evaluation"]["blocks"]]
    calibration = [b["id"] for b in payload["partitions"]["calibration"]["blocks"]]
    if len(evaluation) < 2000 or len(calibration) < 1024:
        raise DataError("insufficient held-out blocks for frozen panels")
    ranked = sorted(evaluation, key=lambda ident: (_rank(ident, "owt"), ident))
    cal_ranked = sorted(calibration, key=lambda ident: (_rank(ident, "calibration"), ident))
    return seal_payload({"kind": "owt-panels-v1", "corpus_sha256": corpus_digest,
        "salt": SALT, "owt_dense64": ranked[:64], "owt_full300": ranked[:300],
        "owt_lm2000": ranked[:2000],
        "calibration16x64": [cal_ranked[i:i + 64] for i in range(0, 1024, 64)]})


def validate_owt_panels(document: dict[str, Any], corpus: dict[str, Any]) -> str:
    payload, digest = verify_envelope(document)
    expected = prepare_owt_panels(corpus)["payload"]
    if payload != expected:
        raise DataError("frozen panel membership or order mismatch")
    return digest


def prepare_domain_panels(sources: dict[str, Iterable[dict[str, Any]]], tokenizer: Any,
                          *, tokenizer_sha256: str, revisions: dict[str, str]) -> dict[str, Any]:
    if set(sources) != set(DOMAIN_FIELDS) or set(revisions) != set(DOMAIN_FIELDS):
        raise DataError("all three pinned domain sources/revisions are required")
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0 or len(tokenizer_sha256) != 64:
        raise DataError("valid tokenizer EOS and file hash required")
    domains = {}
    for domain, (expected_split, field) in DOMAIN_FIELDS.items():
        if not isinstance(revisions[domain], str) or not revisions[domain].strip():
            raise DataError("domain revision must be pinned")
        candidates = {}
        for row in sources[domain]:
            if not isinstance(row, dict) or row.get("split") != expected_split or field not in row:
                raise DataError(f"{domain} requires {expected_split}.{field}")
            text = normalized_text(row[field])
            ids = _ids(tokenizer, text)
            if len(ids) < 2:
                continue
            digest = document_hash(text)
            candidates.setdefault(digest, ids)
        ranked = sorted(candidates, key=lambda digest: (_rank(digest, domain), digest))
        if len(ranked) < 100:
            raise DataError(f"insufficient {domain} eligible rows")
        items = []
        for digest in ranked[:100]:
            ids = candidates[digest]
            renderings = {}
            for context in (128, 40):
                real = ids[:context]
                renderings[str(context)] = {"input_ids": real + [eos] * (context - len(real)),
                    "attention_mask": [1] * len(real) + [0] * (context - len(real)),
                    "real_token_count": len(real)}
            items.append({"document_sha256": digest, "source_field": field,
                          "source_split": expected_split, "renderings": renderings})
        domains[domain] = {"revision": revisions[domain], "items": items}
    return seal_payload({"kind": "xdomain300-v1", "salt": SALT,
        "tokenizer_sha256": tokenizer_sha256, "pad_token_id": eos, "domains": domains})


def validate_domain_panels(document: dict[str, Any], *, tokenizer_sha256: str) -> str:
    payload, digest = verify_envelope(document)
    if payload.get("kind") != "xdomain300-v1" or payload.get("tokenizer_sha256") != tokenizer_sha256:
        raise DataError("domain tokenizer or schema mismatch")
    if set(payload["domains"]) != set(DOMAIN_FIELDS):
        raise DataError("domain panel missing source")
    for domain, (split, field) in DOMAIN_FIELDS.items():
        items = payload["domains"][domain]["items"]
        if len(items) != 100 or len({item["document_sha256"] for item in items}) != 100:
            raise DataError("domain panel count or dedup mismatch")
        for item in items:
            if item["source_field"] != field or item["source_split"] != split:
                raise DataError("domain source field mismatch")
            if set(item) != {"document_sha256", "source_field", "source_split", "renderings"}:
                raise DataError("unexpected domain field")
            for context in (128, 40):
                rendered = item["renderings"][str(context)]
                ids, mask, count = rendered["input_ids"], rendered["attention_mask"], rendered["real_token_count"]
                if (len(ids) != context or len(mask) != context or not 2 <= count <= context or
                        mask != [1] * count + [0] * (context - count) or
                        ids[count:] != [payload["pad_token_id"]] * (context - count)):
                    raise DataError("invalid right padding or attention mask")
            if item["renderings"]["40"]["input_ids"][:40] != item["renderings"]["128"]["input_ids"][:40]:
                raise DataError("max40 does not share source prefix")
    return digest
