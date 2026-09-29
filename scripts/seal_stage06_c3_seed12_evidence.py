"""Verify two real seed-specific OWT lineages and seal the D21 decision."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

from transformers import GPT2Config

from sinklab.initialization import load_initialization
from sinklab.order import UpdateOrder
from sinklab.owt_compat import load_owt_corpus, panels_from_validated_corpus
from sinklab.provenance import canonical_json_bytes, seal_payload, verify_envelope
from sinklab.stage06_readiness import (
    D20_PROTOCOL_ROOT, D21_EVIDENCE_PATH, D21_PATH,
)


SOURCE_SHA = "d36784aaf0521f6e96d40d603e0361744397b23df90dc505e29b0b9cf18360eb"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, document: dict) -> None:
    content = canonical_json_bytes(document) + b"\n"
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"sealed evidence already differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--replication-root", type=Path, required=True)
    parser.add_argument("--approved-on-utc-date", required=True)
    args = parser.parse_args()
    repo, source_root, root = (path.resolve() for path in
                               (args.repo, args.source_root, args.replication_root))
    protocol = _read(repo / "protocols/protocol.lock.json")
    if protocol["sha256"] != D20_PROTOCOL_ROOT:
        parser.error("D21 evidence must start from the exact D20 root")
    data = protocol["payload"]["protocol"]["data"]
    source = source_root / "source/openwebtext-train-first416000.jsonl"
    if _sha(source) != SOURCE_SHA:
        parser.error("pinned OWT source bytes differ")
    config = GPT2Config(**_read(source_root / "student-config/config.json"))
    rows = {}
    for seed in (1, 2):
        init = next((root / f"initialization/seed{seed}").glob(
            f"init-seed{seed}-*.json"))
        init_payload, _ = verify_envelope(_read(init))
        if init_payload["seed"] != seed:
            parser.error(f"seed {seed} initialization metadata mismatch")
        weights = init.with_suffix(".safetensors")
        load_initialization(init, config=config, seed=seed)
        init_hash = init_payload["tensor_content_sha256"]
        del init_payload
        gc.collect()
        candidates = list((root / f"corpus/seed{seed}").glob("owt-corpus-*.json"))
        if len(candidates) != 1:
            parser.error(f"seed {seed} requires one corpus")
        corpus_path = candidates[0]
        corpus, corpus_hash = load_owt_corpus(
            corpus_path, tokenizer_sha256=data["tokenizer_files_sha256"])
        if corpus["seed"] != seed:
            parser.error(f"seed {seed} corpus absent")
        if (corpus["dataset"]["revision"] != data["dataset_revision"] or
                corpus["tokenizer"]["revision"] != data["tokenizer_revision"] or
                corpus["windows"] != data["source_windows"]):
            parser.error(f"seed {seed} corpus source or document windows differ")
        expected_panel = panels_from_validated_corpus(corpus, corpus_hash)
        panel_hash = expected_panel["sha256"]
        panel_path = root / f"panels/seed{seed}" / f"owt-panels-{panel_hash}.json"
        if _read(panel_path) != expected_panel:
            parser.error(f"seed {seed} panels differ from packed corpus")
        ids = [block["id"] for block in corpus["partitions"]["training"]["blocks"]]
        expected_order = UpdateOrder(ids, seed=seed, scheme="upstream-owt-epoch-v1").snapshot()
        order_hash = expected_order["sha256"]
        order_path = root / f"order/seed{seed}" / f"update-order-{order_hash}.json"
        if _read(order_path) != expected_order:
            parser.error(f"seed {seed} order differs from packed corpus")
        rows[str(seed)] = {
            "corpus_sha256": corpus_hash, "corpus_file_sha256": _sha(corpus_path),
            "panels_sha256": panel_hash, "panels_file_sha256": _sha(panel_path),
            "order_sha256": order_hash, "order_file_sha256": _sha(order_path),
            "initialization_sha256": init_hash,
            "initialization_metadata_file_sha256": _sha(init),
            "initialization_weights_file_sha256": _sha(weights),
        }
        del corpus, expected_panel, expected_order, ids
        gc.collect()
    evidence = seal_payload({
        "kind": "s1-c3-seed12-repacked-artifacts-v1",
        "predecessor_root_sha256": D20_PROTOCOL_ROOT,
        "source_sha256": SOURCE_SHA,
        "dataset_revision": data["dataset_revision"],
        "tokenizer_revision": data["tokenizer_revision"],
        "tokenizer_sha256": data["tokenizer_files_sha256"],
        "seeds": rows})
    decision = seal_payload({
        "kind": "s1-d21-c3-only-seed12-replication-v1",
        "decision_id": "D21",
        "status": "approved_prospective_optional_replication",
        "approved_on_utc_date": args.approved_on_utc_date,
        "predecessor_production_root": D20_PROTOCOL_ROOT,
        "replication_evidence_sha256": evidence["sha256"],
        "condition": "C3", "seeds": [1, 2], "device_role": "rtx4080super",
        "document_packing": "repack_training_evaluation_calibration_per_seed",
        "evaluation_panels": "separate_per_seed_not_directly_paired",
        "common_schedule": {"microbatch": 4, "accumulation": 16,
                            "effective_sequences": 64, "optimizer_updates": 10000},
        "s_MSE": 68.00580071126464,
        "prospective_only": True,
        "preserve_d20_seed0_run": True,
        "calibration_changed": False})
    _write_once(repo / D21_EVIDENCE_PATH, evidence)
    _write_once(repo / D21_PATH, decision)
    print(json.dumps({"evidence_sha256": evidence["sha256"],
                      "decision_sha256": decision["sha256"],
                      "seed_artifacts": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
