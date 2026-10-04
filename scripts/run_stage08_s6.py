"""Run the D24/D25 S6 seed0 domain battery on retained S1 checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.prepare_s6_d25_panels import verify_frozen_panels
from scripts.stage08_readiness import read_json
from scripts.stage08_scientific_common import (CONDITIONS, EXPECTED_GPU_UUID,
    current_gpu, failure_status, output_path_is_external, preflight_sources, require_free_space,
    sha256_file, utc_now,
    scientific_source_hashes, write_new, write_or_validate_run_manifest, write_sealed_new)
from sinklab.data import tokenizer_files_hash
from sinklab.evaluate import RecordStore
from sinklab.followup_policy import D24_SHA256
from sinklab.panels import DOMAIN_FIELDS
from sinklab.provenance import canonical_json_bytes, seal_payload, verify_envelope
from sinklab.s6 import evaluate_s6_domains, render_domain_items, validate_s6_domains

STEPS = (0, 500, 2000, 10000)
CONTEXTS = (40, 128)
DENOMINATOR_FLOOR = 1e-8
ARTIFACT_ROOT = Path(r"E:\KD-SINK-stage06-Adrita\artifacts\KD-SINK-stage06-production")
D25_ROOT = Path(r"D:\KD-SINK-central\analysis\S6_D25")
DEFAULT_OUTPUT = Path(r"E:\KD-SINK-stage08-scientific-20261005\S6")


def emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def _write_result(path: Path, payload: dict) -> str:
    document = seal_payload(payload)
    encoded = canonical_json_bytes(document) + b"\n"
    if path.exists():
        existing, digest = verify_envelope(read_json(path))
        if existing != payload:
            raise ValueError(f"existing S6 summary conflicts with rerun: {path}")
        return digest
    write_new(path, encoded)
    return document["sha256"]


def _student_loader(artifact_root: Path, artifact: dict):
    import torch
    from safetensors.torch import load_file
    from transformers import GPT2Config, GPT2LMHeadModel
    from sinklab.models import GPT2Adapter, ModelShape

    config_path = artifact_root / "student-config" / "config.json"
    if sha256_file(config_path) != artifact["student_config"]["sha256"]:
        raise ValueError("local student config differs from the S1 artifact lock")
    config = GPT2Config(**read_json(config_path))
    config._attn_implementation = "eager"
    model_sha = hashlib.sha256(canonical_json_bytes(config.to_dict())).hexdigest()
    shape = ModelShape(24, 16, 1024)

    def load(checkpoint: dict):
        model_path = Path(checkpoint["model_path"])
        before = model_path.stat()
        if (before.st_size != checkpoint["model_stat"]["bytes"] or
                before.st_mtime_ns != checkpoint["model_stat"]["mtime_ns"]):
            raise ValueError("verified S1 checkpoint changed before load")
        model = GPT2LMHeadModel(config)
        tensors = load_file(str(model_path), device="cpu")
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in tensors.values()):
            raise ValueError("S6 FP32 protocol requires original FP32 checkpoint tensors")
        model.load_state_dict(tensors, strict=True)
        del tensors
        after = model_path.stat()
        if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
            raise ValueError("S1 checkpoint changed while loading")
        model.to(device="cuda:0").eval().requires_grad_(False)
        return GPT2Adapter(model, shape), model
    return load, model_sha


def _panel_inputs(*, repo_root: Path, d25_root: Path, artifact_root: Path) -> tuple[dict, str, dict]:
    output_dir = d25_root / "panels"
    source_root = d25_root / "sources"
    tokenizer_dir = artifact_root / "tokenizer"
    preparation = verify_frozen_panels(repo_root=repo_root, source_root=source_root,
        tokenizer_dir=tokenizer_dir, output_dir=output_dir)
    panel_path = output_dir / "S6_D25_DOMAIN_PANELS.json"
    panel_document = read_json(panel_path)
    payload, panel_sha = validate_s6_domains(
        panel_document, tokenizer_sha256=preparation["tokenizer_sha256"])
    selection_path = output_dir / "S6_D25_SOURCE_SELECTION.json"
    manifest_path = output_dir / "S6_D25_PREPARATION_MANIFEST.json"
    preparation_document = read_json(manifest_path)
    preparation_payload, preparation_sha = verify_envelope(preparation_document)
    source_lock_path = repo_root / "protocols" / "s6_external_dataset_sources_d25.json"
    decision_path = repo_root / "protocols" / "s1_researcher_amendment_d25_s6_external_datasets_20261004.json"
    selection, selection_sha = verify_envelope(read_json(selection_path))
    source_lock_document = read_json(source_lock_path)
    _, source_lock_sha = verify_envelope(source_lock_document)
    decision_document = read_json(decision_path)
    _, decision_sha = verify_envelope(decision_document)
    if (sha256_file(panel_path) != preparation["panel_file_sha256"] or
            sha256_file(selection_path) != preparation_payload["source_selection_file_sha256"] or
            sha256_file(manifest_path) != preparation["preparation_manifest_file_sha256"] or
            preparation_sha != preparation["preparation_manifest_sha256"] or
            source_lock_sha != preparation["source_lock_sha256"] or
            decision_sha != preparation["decision_sha256"] or
            selection_sha != preparation["source_selection_sha256"]):
        raise ValueError("D25 frozen panel file/source/decision receipt mismatch")
    if (set(selection.get("selection", {})) != set(DOMAIN_FIELDS) or
            payload["decision_sha256"] != preparation["decision_sha256"] or
            payload["source_lock_sha256"] != preparation["source_lock_sha256"]):
        raise ValueError("D25 panel, selection, source lock and decision identities disagree")
    for domain in DOMAIN_FIELDS:
        for context in CONTEXTS:
            items, panel_hash = render_domain_items(panel_document,
                tokenizer_sha256=preparation["tokenizer_sha256"],
                domain=domain, context=context)
            if (len(items) != 100 or any(row["valid_target_count"] < 1 for row in items) or
                    panel_hash != preparation["paired_context_panel_sha256"][f"{domain}_max{context}"]):
                raise ValueError(f"D25 {domain}/max{context} panel verification failed")
            hashes = {row["document_sha256"] for row in items}
            if len(hashes) != 100:
                raise ValueError(f"D25 {domain}/max{context} has duplicate documents")
    tokenizer_hash = tokenizer_files_hash(tokenizer_dir)
    receipt = {**preparation, "panel_path": str(panel_path),
        "panel_file_sha256": sha256_file(panel_path),
        "source_selection_path": str(selection_path),
        "source_selection_file_sha256": sha256_file(selection_path),
        "source_selection_sha256": selection_sha,
        "preparation_manifest_path": str(manifest_path),
        "source_lock_path": str(source_lock_path),
        "D25_decision_path": str(decision_path),
        "D25_source_metadata": payload["source_metadata"],
        "panel_context_hashes": preparation["paired_context_panel_sha256"],
        "tokenizer_file_bundle_sha256": tokenizer_hash,
        "contexts": list(CONTEXTS), "optional_contexts": {"512": "disabled", "1024": "disabled"}}
    if tokenizer_hash != preparation["tokenizer_sha256"]:
        raise ValueError("S6 local GPT-2 tokenizer bundle hash changed")
    return panel_document, panel_sha, receipt


def _write_sums(root: Path) -> tuple[Path, int, int]:
    lines, total_bytes = [], 0
    excluded = {"SHA256SUMS.txt", "S6_FINAL_AUDIT.json", "S6_FINAL_AUDIT.json.sha256"}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in excluded):
        digest = sha256_file(path)
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
        total_bytes += path.stat().st_size
    sums = root / "SHA256SUMS.txt"
    write_new(sums, ("\n".join(lines) + "\n").encode("ascii"))
    return sums, len(lines), total_bytes


def run(*, output: Path, runs_root: Path, artifact_root: Path,
        d25_root: Path, resume: bool = False) -> dict:
    import torch
    from scripts.stage08_scientific_common import source_run_directories

    output, artifact_root, d25_root = output.resolve(), artifact_root.resolve(), d25_root.resolve()
    source_paths = list(source_run_directories().values())
    output_path_is_external(output, source_paths)
    require_free_space(output, 3 * 1024 ** 3 if not output.exists() or not any(output.iterdir())
                       else 2 * 1024 ** 3)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"nonempty S6 output requires --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    gpu = current_gpu()
    if gpu["uuid"] != EXPECTED_GPU_UUID:
        raise RuntimeError("S6 inference GPU UUID differs from the authorized Adrita-PC device")

    # The panel verifier rechecks exact local Parquet/README snapshots and their frozen receipts.
    panel_document, panel_sha, panel_receipt = _panel_inputs(
        repo_root=REPO, d25_root=d25_root, artifact_root=artifact_root)
    emit({"event": "s6_d25_panels_verified", "panel_sha256": panel_sha,
          "tokenizer_sha256": panel_receipt["tokenizer_sha256"],
          "paired_context_panel_sha256": panel_receipt["panel_context_hashes"],
          "optional_contexts": panel_receipt["optional_contexts"]})

    def progress(event):
        emit(event)
    source_inventory = preflight_sources(steps=STEPS, study="S6",
        runs_root=runs_root, progress=progress)
    checkpoint_count = sum(len(row["checkpoints"]) for row in source_inventory["sources"].values())
    if checkpoint_count != 28:
        raise ValueError(f"S6 requires 28 individually verified checkpoints; found {checkpoint_count}")

    artifact_document = read_json(REPO / "protocols" / "artifact.lock.json")
    artifact, artifact_sha = verify_envelope(artifact_document)
    model_load, expected_model_sha = _student_loader(artifact_root, artifact)
    if any(source["identity"].get("model_hash") != expected_model_sha
           for source in source_inventory["sources"].values()):
        raise ValueError("S6 local student architecture identity differs from an S1 source run")
    d24_path = REPO / "protocols" / "s1_researcher_amendment_d24_seed0_followups_20261004.json"
    d25_lock_path = REPO / "protocols" / "s6_external_dataset_sources_d25.json"
    d25_decision_path = REPO / "protocols" / "s1_researcher_amendment_d25_s6_external_datasets_20261004.json"
    science_code = scientific_source_hashes(("src/sinklab/s6.py", "src/sinklab/evaluate.py",
        "src/sinklab/metrics.py", "src/sinklab/models.py", "src/sinklab/interventions.py",
        "src/sinklab/panels.py", "src/sinklab/data.py", "src/sinklab/provenance.py",
        "src/sinklab/followup_policy.py"))
    run_manifest_payload = {"schema": "stage08-s6-science-run-v1", "status": "in_progress",
        "created_utc": utc_now(), "hostname": socket.gethostname(),
        "repository_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                             capture_output=True, text=True, check=True).stdout.strip(),
        "study": "S6", "training_seed": 0, "conditions": list(CONDITIONS),
        "steps": list(STEPS), "checkpoint_count": checkpoint_count,
        "D24_sha256": D24_SHA256, "D24_file_sha256": sha256_file(d24_path),
        "D25_decision_sha256": panel_receipt["decision_sha256"],
        "D25_decision_file_sha256": sha256_file(d25_decision_path),
        "D25_source_lock_sha256": panel_receipt["source_lock_sha256"],
        "D25_source_lock_file_sha256": sha256_file(d25_lock_path),
        "artifact_lock_sha256": artifact_sha, "artifact_lock_file_sha256": sha256_file(REPO / "protocols" / "artifact.lock.json"),
        "D24_interpretation": source_inventory["D24_interpretation"],
        "source_inventory": source_inventory, "panel": panel_receipt,
        "contexts": list(CONTEXTS), "operations": ["clean", "delete", "relocate"],
        "denominator_floor": DENOMINATOR_FLOOR, "precision": "fp32",
        "scientific_implementation_sha256": science_code,
        "inference_device": gpu, "human_eval_code_execution": False,
        "optional_512_1024": "disabled", "source_runs_modified": False,
        "training_or_s2_or_stage09": False}
    run_manifest_path = output / "S6_RUN_MANIFEST.json"
    run_manifest_sha = write_or_validate_run_manifest(run_manifest_path, run_manifest_payload)
    store = RecordStore(output / "records")
    completed = []
    for condition in CONDITIONS:
        source = source_inventory["sources"][condition]
        for checkpoint in source["checkpoints"]:
            step = checkpoint["step"]
            emit({"event": "s6_state_start", "condition": condition, "step": step,
                  "run_id": source["run_id"], "checkpoint": checkpoint["path"],
                  "domains": list(DOMAIN_FIELDS), "contexts": list(CONTEXTS)})
            adapter, model = model_load(checkpoint)
            torch.cuda.reset_peak_memory_stats(0)
            run_identity = {"study": "S1", "condition": condition, "seed": 0,
                "model_sha256": source["identity"]["model_hash"],
                "corpus_sha256": source["identity"]["data_hash"],
                "protocol_sha256": source["original_protocol_root_sha256"]}
            with torch.autocast(device_type="cuda", enabled=False):
                result = evaluate_s6_domains(adapter=adapter, document=panel_document,
                    tokenizer_sha256=panel_receipt["tokenizer_sha256"],
                    checkpoint_sha256=checkpoint["model_file_sha256"],
                    run_id=source["run_id"], step=step, store=store,
                    run_identity=run_identity, denominator_floor=DENOMINATOR_FLOOR,
                    precision="fp32", contexts=CONTEXTS)
            peak = torch.cuda.max_memory_allocated(0)
            if result.get("status") != "complete":
                raise ValueError(f"S6 checkpoint evaluation incomplete for {condition}/step{step}")
            if (result["decision_sha256"] != panel_receipt["decision_sha256"] or
                    result["source_lock_sha256"] != panel_receipt["source_lock_sha256"] or
                    result["s6_manifest_sha256"] != panel_sha or
                    result["source_revisions"] != {name: panel_receipt["D25_source_metadata"][name]["revision"]
                                                     for name in DOMAIN_FIELDS}):
                raise ValueError(f"S6 result provenance differs from D25 for {condition}/step{step}")
            summary_path = output / "summaries" / condition / f"step-{step:05d}.json"
            summary_payload = {"kind": "s6-domain-battery-summary-v1", "condition": condition,
                "training_seed": 0, "run_id": source["run_id"],
                "original_protocol_root_sha256": source["original_protocol_root_sha256"],
                "D24_sha256": D24_SHA256, "step": step,
                "checkpoint_manifest_sha256": checkpoint["manifest_file_sha256"],
                "checkpoint_model_sha256": checkpoint["model_file_sha256"],
                "D25_panel_manifest_sha256": panel_sha, "contexts": list(CONTEXTS),
                "domains": list(DOMAIN_FIELDS), "precision": "fp32",
                "pooled": result["pooled"], "paired": result["paired"],
                "aggregate_keys": {name: value["key"] for name, value in result["results"].items()},
                "metric_label": result["metric_label"]}
            summary_sha = _write_result(summary_path, summary_payload)
            emit({"event": "s6_state_complete", "condition": condition, "step": step,
                  "status": result["status"], "domain_context_aggregates": len(result["results"]),
                  "item_operation_records": len(DOMAIN_FIELDS) * len(CONTEXTS) * 100 * 3,
                  "peak_cuda_bytes": peak, "summary_sha256": summary_sha})
            completed.append({"condition": condition, "step": step,
                "summary_path": str(summary_path), "summary_sha256": summary_sha,
                "status": result["status"]})
            del adapter, model
            gc.collect()
            torch.cuda.empty_cache()

    if len(completed) != 28:
        raise RuntimeError("S6 condition/checkpoint coverage differs from 28")
    expected_item_records = len(CONDITIONS) * len(STEPS) * len(DOMAIN_FIELDS) * len(CONTEXTS) * 100 * 3
    expected_aggregate_records = len(CONDITIONS) * len(STEPS) * len(DOMAIN_FIELDS) * len(CONTEXTS) * 3
    item_count = aggregate_count = 0
    for path in (output / "records").glob("*.json"):
        document = read_json(path)
        payload, _ = verify_envelope(document)
        key = payload.get("key", {})
        if not key:
            raise ValueError(f"S6 result record is incomplete or malformed: {path}")
        if key.get("kind") == "aggregate":
            aggregate_count += 1
            if any(row.get("status") != "complete" for row in payload.get("operations", {}).values()):
                raise ValueError(f"S6 aggregate coverage incomplete: {path}")
        else:
            if payload.get("status") != "complete" or payload.get("error") is not None:
                raise ValueError(f"S6 item record is incomplete or malformed: {path}")
            item_count += 1
    if item_count != expected_item_records or aggregate_count != expected_aggregate_records:
        raise ValueError(f"S6 stored record counts differ: items={item_count}, aggregates={aggregate_count}")
    sums, file_count, total_bytes = _write_sums(output)
    audit_payload = {"schema": "stage08-s6-final-audit-v1", "status": "COMPLETE",
        "created_utc": utc_now(), "run_manifest_sha256": run_manifest_sha,
        "run_manifest_path": str(run_manifest_path), "training_seed": 0,
        "condition_checkpoint_batteries": len(completed), "expected_condition_checkpoint_batteries": 28,
        "completed_batteries": completed, "contexts": list(CONTEXTS),
        "domains": list(DOMAIN_FIELDS), "operations": ["clean", "delete", "relocate"],
        "item_record_count": item_count, "expected_item_record_count": expected_item_records,
        "aggregate_record_count": aggregate_count, "expected_aggregate_record_count": expected_aggregate_records,
        "D25_panel_manifest_sha256": panel_sha,
        "paired_context_panel_sha256": panel_receipt["panel_context_hashes"],
        "SHA256SUMS_path": str(sums), "SHA256SUMS_file_sha256": sha256_file(sums),
        "manifested_file_count": file_count, "manifested_uncompressed_bytes": total_bytes,
        "metric_label": "causal_language_modeling_not_domain_task_accuracy",
        "HumanEval_code_executed": False, "optional_512_1024": "disabled",
        "source_runs_modified": False, "model_weights_modified": False,
        "training": False, "S2": False, "Stage09": False}
    audit_path = output / "S6_FINAL_AUDIT.json"
    audit_sha = write_sealed_new(audit_path, audit_payload)
    write_new(output / "S6_FINAL_AUDIT.json.sha256",
              f"{sha256_file(audit_path)}  {audit_path.name}\n".encode("ascii"))
    result = {"status": "COMPLETE", "study": "S6", "audit_sha256": audit_sha,
        "item_record_count": item_count, "aggregate_record_count": aggregate_count,
        "result_directory": str(output), "run_manifest_sha256": run_manifest_sha}
    emit({"event": "s6_study_complete", **result})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path(r"D:\KD-SINK-central\runs"))
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--d25-root", type=Path, default=D25_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(output=args.output, runs_root=args.runs_root,
            artifact_root=args.artifact_root, d25_root=args.d25_root,
            resume=args.resume)
        return 0
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {"schema": "stage08-s6-failure-v1", "status": failure_status(exc),
            "created_utc": utc_now(), "hostname": socket.gethostname(),
            "error_type": type(exc).__name__, "error": str(exc),
            "source_runs_modified": False, "HumanEval_code_executed": False,
            "training": False, "Stage09": False}
        path = args.output / f"S6_FAILURE_{int(time.time())}.json"
        write_sealed_new(path, failure)
        emit({"event": "s6_study_failed", **failure, "failure_path": str(path)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
