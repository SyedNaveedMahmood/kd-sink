"""Verify and summarize real, external S1 production inputs without embedding them."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

from transformers import GPT2Config

from .data import load_manifest, tokenizer_files_hash
from .initialization import load_initialization
from .order import UpdateOrder
from .owt_compat import (load_owt_corpus, panels_from_validated_corpus,
                         validate_calibration_blocks_export)
from .provenance import canonical_json_bytes, payload_digest, seal_payload, verify_envelope


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one(directory: Path, pattern: str) -> Path:
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected exactly one {pattern} in {directory}")
    return paths[0]


def _init_record(metadata: Path, config: GPT2Config, seed: int) -> dict:
    document = json.loads(metadata.read_text(encoding="utf-8"))
    payload, _ = verify_envelope(document)
    tensors = load_initialization(metadata, config=config, seed=seed)
    del tensors
    gc.collect()
    weights = metadata.with_suffix(".safetensors")
    return {"seed": seed, "tensor_content_sha256": payload["tensor_content_sha256"],
            "metadata_sha256": _sha(metadata), "weights_file_sha256": _sha(weights),
            "weights_bytes": weights.stat().st_size}


def approved_artifact_plan_digest(repo: Path, approval: dict) -> str:
    """Bind artifacts to the plan approved when they were prepared."""
    repo = Path(repo)
    approved_plan = repo / "configs/superseded/d19/s1_jobs_seed0_reviewed.json"
    if not approved_plan.exists():
        approved_plan = repo / "configs/s1_jobs_seed0_reviewed.json"
    plan = json.loads(approved_plan.read_text(encoding="utf-8"))
    digest = payload_digest(plan)
    if approval["reviewed_seed0_plan_sha256"] != digest:
        raise ValueError("approved artifact plan differs from researcher decision approval")
    return digest


def build_inventory(root: Path, repo: Path) -> dict:
    root, repo = Path(root), Path(repo)
    partial, partial_digest = verify_envelope(json.loads((
        repo / "protocols/s1_artifact_partial_v1.json").read_text(encoding="utf-8")))
    approval, approval_digest = verify_envelope(json.loads((
        repo / "protocols/s1_researcher_approval_20260928.json").read_text(encoding="utf-8")))
    tokenizer_digest = tokenizer_files_hash(root / "tokenizer")
    if tokenizer_digest != partial["tokenizer"]["files_sha256"]:
        raise ValueError("pinned tokenizer file set changed")
    source_record, source_digest = verify_envelope(json.loads((
        root / "source/openwebtext-source.json").read_text(encoding="utf-8")))
    source_file = root / "source/openwebtext-train-first416000.jsonl"
    if (_sha(source_file) != source_record["source_jsonl_sha256"] or
            source_file.stat().st_size != source_record["source_jsonl_bytes"] or
            source_record["row_count"] != 416_000 or
            source_record["revision"] != partial["dataset"]["revision"] or
            source_record["license_card"] != ["cc0-1.0"]):
        raise ValueError("pinned OpenWebText source record or raw prefix changed")
    corpus_file = _one(root / "corpus", "owt-corpus-*.json")
    corpus, corpus_digest = load_owt_corpus(corpus_file, tokenizer_sha256=tokenizer_digest)
    if (corpus["dataset"]["revision"] != source_record["revision"] or
            corpus["seed"] != 0 or corpus["block_size"] != 128 or
            corpus["tokenizer"]["revision"] != partial["tokenizer"]["revision"]):
        raise ValueError("production corpus differs from pinned source or seed")
    panels_file = _one(root / "panels", "owt-panels-*.json")
    panels, panels_digest = load_manifest(panels_file, stem="owt-panels")
    if panels != panels_from_validated_corpus(corpus, corpus_digest)["payload"]:
        raise ValueError("frozen evaluation and calibration panels differ from corpus")
    export_file = _one(root / "calibration-panel", "calibration16x64-blocks-*.json")
    export, export_digest = validate_calibration_blocks_export(json.loads(
        export_file.read_text(encoding="utf-8")))
    if (export["corpus_sha256"] != corpus_digest or export["panels_sha256"] != panels_digest or
            [[block["id"] for block in batch] for batch in export["batches"]] !=
            panels["calibration16x64"]):
        raise ValueError("training-only calibration export differs from frozen panels")
    ids = [block["id"] for block in corpus["partitions"]["training"]["blocks"]]
    order_file = _one(root / "order", "owt-update-order-*.json")
    order, order_digest = load_manifest(order_file, stem="owt-update-order")
    resumed = UpdateOrder.resume(ids, seal_payload(order), expected_scheme="upstream-owt-epoch-v1")
    if resumed.seed != 0 or resumed.presentations != 0 or resumed.epoch != 0 or resumed.cursor != 0:
        raise ValueError("seed-0 production order is not an initial upstream epoch cursor")
    config_file = root / "student-config/config.json"
    teacher_dir = root / "teacher"
    if (_sha(config_file) != partial["student_config"]["config_sha256"] or
            _sha(teacher_dir / "config.json") != partial["teacher"]["config_sha256"] or
            _sha(teacher_dir / "model.safetensors") != partial["teacher"]["safetensors_sha256"]):
        raise ValueError("pinned model files changed")
    config = GPT2Config(**json.loads(config_file.read_text(encoding="utf-8")))
    init0 = _init_record(_one(root / "initialization/seed0", "init-seed0-*.json"), config, 0)
    init1729 = _init_record(_one(root / "initialization/seed1729", "init-seed1729-*.json"),
                            config, 1729)
    split_counts = {}
    for name, part in corpus["partitions"].items():
        source_hashes = part["source_text_sha256"]
        split_counts[name] = {"source_documents": len(source_hashes),
            "tokenized_documents": len(part["documents"]),
            "empty_or_untokenizable_documents": len(part["skipped_source_indices"]),
            "duplicate_normalized_text_hashes": len(source_hashes) - len(set(source_hashes)),
            "packed_blocks": len(part["blocks"]), "dropped_tail_tokens": len(part["tail_token_ids"]),
            "source_window_sha256": part["source_window_sha256"],
            "stream_sha256": part["stream_sha256"]}
    approved_plan_sha256 = approved_artifact_plan_digest(repo, approval)
    return seal_payload({"kind": "stage06-production-artifact-inventory-v1",
        "status": "real_artifacts_verified_pending_3090_calibration_and_final_locks",
        "observed_external_root": str(root),
        "researcher_approval_sha256": approval_digest,
        "artifact_partial_sha256": partial_digest,
        "reviewed_plan_sha256": approved_plan_sha256,
        "dataset": {"id": source_record["dataset_id"], "revision": source_record["revision"],
            "configuration": source_record["configuration"], "split": "train",
            "license": "cc0-1.0", "windows": source_record["windows"],
            "preprocessing": "collapse_whitespace_seed0_document_shuffle_no_special_tokens_post_document_eos_greedy128"},
        "source": {"record_sha256": source_digest,
                   "raw_jsonl_sha256": source_record["source_jsonl_sha256"],
                   "raw_jsonl_bytes": source_file.stat().st_size},
        "tokenizer": {"id": partial["tokenizer"]["id"],
            "revision": partial["tokenizer"]["revision"], "license": "mit",
            "files_sha256": tokenizer_digest, "file_sha256": partial["tokenizer"]["file_sha256"]},
        "teacher": {"id": partial["teacher"]["id"], "revision": partial["teacher"]["revision"],
            "license": "mit", "config_sha256": partial["teacher"]["config_sha256"],
            "weights_sha256": partial["teacher"]["safetensors_sha256"],
            "weights_bytes": (teacher_dir / "model.safetensors").stat().st_size},
        "student_config": {"id": partial["student_config"]["id"],
            "revision": partial["student_config"]["revision"], "license": "mit",
            "sha256": partial["student_config"]["config_sha256"]},
        "training_initialization": init0, "calibration_initialization": init1729,
        "corpus": {"payload_sha256": corpus_digest, "file_sha256": _sha(corpus_file),
            "bytes": corpus_file.stat().st_size, "splits": split_counts,
            "document_windows_disjoint": True},
        "panels": {"payload_sha256": panels_digest, "file_sha256": _sha(panels_file),
            "bytes": panels_file.stat().st_size, "dense64": 64, "full300": 300,
            "lm2000": 2000},
        "calibration_export": {"payload_sha256": export_digest,
            "file_sha256": _sha(export_file), "bytes": export_file.stat().st_size,
            "effective_batches": 16, "sequences_per_batch": 64, "attention_mask": "all_ones_128_no_padding"},
        "order_seed0": {"payload_sha256": order_digest, "file_sha256": _sha(order_file),
            "bytes": order_file.stat().st_size, "block_set_sha256": order["block_set_sha256"],
            "training_blocks": len(ids), "scheme": order["scheme"], "presentations": 0}})


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify external real S1 artifacts and seal a small inventory")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = build_inventory(args.artifact_root, args.repo_root)
    if args.out.exists():
        raise FileExistsError("inventory already exists; refuse overwrite")
    args.out.write_bytes(canonical_json_bytes(document) + b"\n")
    print(json.dumps({"path": str(args.out), "sha256": document["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
