"""Run the D24 S4 seed0 probe batteries on the retained S1 checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stage08_readiness import read_json
from scripts.stage08_scientific_common import (CONDITIONS, EXPECTED_GPU_UUID,
    current_gpu, failure_status, output_path_is_external, preflight_sources,
    require_free_space, scientific_source_hashes, sha256_file, utc_now, write_new,
    source_run_directories, write_or_validate_run_manifest, write_sealed_new)
from sinklab.evaluate import RecordStore
from sinklab.followup_policy import D24_SHA256
from sinklab.models import GPT2Adapter, ModelShape
from sinklab.owt_compat import load_owt_corpus, panels_from_validated_corpus
from sinklab.probes import PROBE_IDS, evaluate_probe_battery
from sinklab.provenance import canonical_json_bytes, seal_payload, verify_envelope

STEPS = (0, 100, 500, 2000, 10000)
CONTROL_SEED = 20260927
DENOMINATOR_FLOOR = 1e-8
RESPONSIVENESS_FLOOR = 1e-6
ARTIFACT_ROOT = Path(r"E:\KD-SINK-stage06-Adrita\artifacts\KD-SINK-stage06-production")
DEFAULT_OUTPUT = Path(r"E:\KD-SINK-stage08-scientific-20261005\S4")
TEACHER_RUN_ID = "s4-gpt2-large-full300-reference-v1"


def emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def _write_result(path: Path, payload: dict) -> str:
    document = seal_payload(payload)
    encoded = canonical_json_bytes(document) + b"\n"
    if path.exists():
        existing, digest = verify_envelope(read_json(path))
        if existing != payload:
            raise ValueError(f"existing S4 summary conflicts with rerun: {path}")
        return digest
    write_new(path, encoded)
    return document["sha256"]


def _artifact_inputs(artifact_root: Path) -> tuple[dict, list[dict], str, dict]:
    artifact_document = read_json(REPO / "protocols" / "artifact.lock.json")
    artifact, artifact_sha = verify_envelope(artifact_document)
    corpus_path = artifact_root / "corpus" / f"owt-corpus-{artifact['corpus']['payload_sha256']}.json"
    panels_path = artifact_root / "panels" / f"owt-panels-{artifact['panels']['payload_sha256']}.json"
    corpus_file_sha = sha256_file(corpus_path)
    panel_file_sha = sha256_file(panels_path)
    if (corpus_file_sha != artifact["corpus"]["file_sha256"] or
            panel_file_sha != artifact["panels"]["file_sha256"]):
        raise ValueError("locked OWT corpus or S4 panel file SHA-256 mismatch")
    corpus_payload, corpus_sha = load_owt_corpus(
        corpus_path, tokenizer_sha256=artifact["tokenizer"]["files_sha256"])
    panel_document = read_json(panels_path)
    panel_payload, panel_sha = verify_envelope(panel_document)
    expected_panels = panels_from_validated_corpus(corpus_payload, corpus_sha)
    if (corpus_sha != artifact["corpus"]["payload_sha256"] or
            panel_sha != artifact["panels"]["payload_sha256"] or
            panel_payload != expected_panels["payload"] or
            panel_payload.get("corpus_sha256") != corpus_sha):
        raise ValueError("locked OWT corpus and S4 panel envelope mismatch")
    evaluation = {block["id"]: block["token_ids"]
                  for block in corpus_payload["partitions"]["evaluation"]["blocks"]}
    item_ids = panel_payload.get("owt_full300")
    if not isinstance(item_ids, list) or len(item_ids) != 300 or len(set(item_ids)) != 300:
        raise ValueError("S4 requires the complete unique frozen OWT Full300 panel")
    items = []
    for ident in item_ids:
        tokens = evaluation.get(ident)
        if not isinstance(tokens, list) or len(tokens) != 128 or any(type(token) is not int for token in tokens):
            raise ValueError(f"S4 panel block is missing or malformed: {ident}")
        items.append({"id": ident, "input_ids": tokens, "attention_mask": [1] * 128})
    receipt = {"artifact_lock_sha256": artifact_sha,
        "artifact_lock_path": str(REPO / "protocols" / "artifact.lock.json"),
        "corpus_path": str(corpus_path), "corpus_file_sha256": corpus_file_sha,
        "corpus_payload_sha256": corpus_sha, "panel_path": str(panels_path),
        "panel_file_sha256": panel_file_sha, "panel_manifest_sha256": panel_sha,
        "panel_name": "owt_full300", "panel_item_count": len(items),
        "panel_item_ids_sha256": hashlib.sha256(canonical_json_bytes({"item_ids": item_ids})).hexdigest(),
        "tokenizer_files_sha256": artifact["tokenizer"]["files_sha256"],
        "precision": "fp32", "sequence_length": 128}
    del corpus_payload, evaluation, expected_panels
    gc.collect()
    return artifact, items, panel_sha, receipt


def _student_config(artifact_root: Path, artifact: dict):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    path = artifact_root / "student-config" / "config.json"
    config = GPT2Config(**read_json(path))
    config._attn_implementation = "eager"
    model_sha = hashlib.sha256(canonical_json_bytes(config.to_dict())).hexdigest()
    if sha256_file(path) != artifact["student_config"]["sha256"]:
        raise ValueError("local student configuration file differs from the S1 artifact lock")
    shape = ModelShape(24, 16, 1024)
    def load(checkpoint: dict):
        from safetensors.torch import load_file
        model_path = Path(checkpoint["model_path"])
        before = model_path.stat()
        if (before.st_size != checkpoint["model_stat"]["bytes"] or
                before.st_mtime_ns != checkpoint["model_stat"]["mtime_ns"]):
            raise ValueError("verified S1 checkpoint changed before load")
        model = GPT2LMHeadModel(config)
        tensors = load_file(str(model_path), device="cpu")
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in tensors.values()):
            raise ValueError("S4 FP32 protocol requires original FP32 checkpoint tensors")
        model.load_state_dict(tensors, strict=True)
        del tensors
        after = model_path.stat()
        if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
            raise ValueError("S1 checkpoint changed while loading")
        model.to(device="cuda:0").eval().requires_grad_(False)
        return GPT2Adapter(model, shape), model
    return load, model_sha


def _teacher(artifact_root: Path, artifact: dict, gpu: dict):
    import torch
    from transformers import GPT2LMHeadModel

    teacher_meta = artifact["teacher"]
    if teacher_meta["weights_sha256"] != "5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce":
        raise ValueError("teacher weight identity differs from the pinned S1 artifact lock")
    teacher_root = artifact_root / "teacher"
    weight_path = teacher_root / "model.safetensors"
    if sha256_file(weight_path) != teacher_meta["weights_sha256"]:
        raise ValueError("pinned GPT-2-large teacher weights SHA-256 mismatch")
    model = GPT2LMHeadModel.from_pretrained(str(teacher_root), local_files_only=True,
                                             attn_implementation="eager")
    config_sha = hashlib.sha256(canonical_json_bytes(model.config.to_dict())).hexdigest()
    if config_sha != teacher_meta["config_sha256"]:
        raise ValueError("pinned GPT-2-large teacher config identity mismatch")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("S4 FP32 protocol requires an FP32 GPT-2-large teacher")
    model.to(device="cuda:0").eval().requires_grad_(False)
    adapter = GPT2Adapter(model, ModelShape(36, 20, 1280))
    provenance = {"followup_study": "S4", "reference_model": teacher_meta["id"],
        "reference_revision": teacher_meta["revision"],
        "teacher_weights_sha256": teacher_meta["weights_sha256"],
        "teacher_config_sha256": teacher_meta["config_sha256"],
        "device": gpu, "precision": "fp32", "control_seed": CONTROL_SEED,
        "denominator_floor": DENOMINATOR_FLOOR,
        "responsiveness_floor": RESPONSIVENESS_FLOOR, "D24_sha256": D24_SHA256}
    return adapter, model, provenance


def _write_sums(root: Path, *, excluded: set[str]) -> tuple[Path, int, int]:
    lines = []
    total_bytes = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in excluded):
        digest = sha256_file(path)
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
        total_bytes += path.stat().st_size
    sums = root / "SHA256SUMS.txt"
    write_new(sums, ("\n".join(lines) + "\n").encode("ascii"))
    return sums, len(lines), total_bytes


def _verify_record_coverage(record_root: Path, *, items: list[dict], panel_sha: str,
                            source_inventory: dict, artifact: dict) -> tuple[int, int, int]:
    from sinklab.followup_policy import admit_s1_followup

    item_ids = {item["id"] for item in items}
    expected_checkpoints = {
        (condition, row["step"]): row
        for condition, source in source_inventory["sources"].items()
        for row in source["checkpoints"]
    }
    expected = {("teacher", TEACHER_RUN_ID, None, item_id, probe)
                for item_id in item_ids for probe in PROBE_IDS}
    expected.update(("student", source_inventory["sources"][condition]["run_id"], step,
                     item_id, probe)
        for condition, step in expected_checkpoints
        for item_id in item_ids for probe in PROBE_IDS)
    seen, plans = set(), {}
    teacher_count = student_count = 0
    for path in record_root.glob("*.json"):
        payload, _ = verify_envelope(read_json(path))
        if payload.get("status") != "complete" or not isinstance(payload.get("key"), dict):
            raise ValueError(f"S4 item record is failed or malformed: {path}")
        key = payload["key"]
        role, run_id, step = key.get("model_role"), key.get("run_id"), key.get("step")
        composite = (role, run_id, step, key.get("item_id"), key.get("probe_id"))
        if (composite not in expected or composite in seen or
                key.get("study") != "S4" or key.get("panel_sha256") != panel_sha or
                key.get("probe_version") != "s4-gpt2-route-v1" or
                key.get("probe_id") not in PROBE_IDS or
                not isinstance(key.get("probe"), dict) or
                key.get("denominator_floor") != DENOMINATOR_FLOOR or
                key.get("responsiveness_floor") != RESPONSIVENESS_FLOOR):
            raise ValueError(f"S4 record key/coverage mismatch: {path}")
        battery_key = (role, run_id, step, key["probe_id"])
        plan = plans.setdefault(battery_key, key.get("probe"))
        if plan != key.get("probe"):
            raise ValueError(f"S4 probe plan changes across items: {battery_key}")
        if role == "teacher":
            if (key.get("checkpoint_sha256") != artifact["teacher"]["weights_sha256"] or
                    "source_identity" in key or "followup_policy" in key):
                raise ValueError("S4 teacher reference provenance mismatch")
            teacher_count += 1
        else:
            condition = next(c for c in CONDITIONS
                             if source_inventory["sources"][c]["run_id"] == run_id)
            checkpoint = expected_checkpoints[(condition, step)]
            identity = source_inventory["sources"][condition]["identity"]
            expected_followup = admit_s1_followup(identity, study="S4", step=step)
            if (key.get("checkpoint_sha256") != checkpoint["model_file_sha256"] or
                    key.get("source_identity") != identity or
                    key.get("followup_policy") != expected_followup or
                    key.get("provenance", {}).get("D24_sha256") != D24_SHA256 or
                    key.get("provenance", {}).get("precision") != "fp32"):
                raise ValueError(f"S4 student source/D24 identity mismatch: {condition}/step{step}")
            student_count += 1
        seen.add(composite)
    if seen != expected:
        raise ValueError(f"S4 record coverage incomplete; missing={len(expected-seen)} unexpected={len(seen-expected)}")
    return len(seen), teacher_count, student_count


def run(*, output: Path, runs_root: Path, artifact_root: Path, resume: bool = False) -> dict:
    import torch

    output = output.resolve()
    sources_by_condition = source_run_directories()
    output_path_is_external(output, list(sources_by_condition.values()))
    require_free_space(output, 7 * 1024 ** 3 if not output.exists() or not any(output.iterdir())
                       else 2 * 1024 ** 3)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"nonempty S4 output requires --resume: {output}")
    if not output.exists():
        output.mkdir(parents=True)
    gpu = current_gpu()
    if gpu["uuid"] != EXPECTED_GPU_UUID:
        raise RuntimeError("S4 inference GPU UUID differs from the authorized Adrita-PC device")
    artifact, items, panel_sha, panel_receipt = _artifact_inputs(artifact_root.resolve())

    def progress(event):
        emit(event)
    source_inventory = preflight_sources(steps=STEPS, study="S4",
        runs_root=runs_root, progress=progress)
    checkpoint_count = sum(len(v["checkpoints"]) for v in source_inventory["sources"].values())
    if checkpoint_count != 35:
        raise ValueError(f"S4 requires 35 individually verified checkpoints; found {checkpoint_count}")
    d24_decision_path = REPO / "protocols" / "s1_researcher_amendment_d24_seed0_followups_20261004.json"
    science_code = scientific_source_hashes(("src/sinklab/probes.py", "src/sinklab/evaluate.py",
        "src/sinklab/metrics.py", "src/sinklab/models.py", "src/sinklab/interventions.py",
        "src/sinklab/provenance.py", "src/sinklab/followup_policy.py"))
    manifest_payload = {"schema": "stage08-s4-science-run-v1", "status": "in_progress",
        "created_utc": utc_now(), "hostname": socket.gethostname(),
        "repository_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                             capture_output=True, text=True, check=True).stdout.strip(),
        "study": "S4", "training_seed": 0, "conditions": list(CONDITIONS),
        "steps": list(STEPS), "checkpoint_count": checkpoint_count,
        "D24_sha256": D24_SHA256, "D24_file_sha256": sha256_file(d24_decision_path),
        "interpretation": source_inventory["D24_interpretation"],
        "source_inventory": source_inventory, "panel": panel_receipt,
        "probe_ids": list(PROBE_IDS), "control_seed": CONTROL_SEED,
        "denominator_floor": DENOMINATOR_FLOOR,
        "responsiveness_floor": RESPONSIVENESS_FLOOR,
        "precision": "fp32", "inference_device": gpu,
        "scientific_implementation_sha256": science_code,
        "source_runs_modified": False, "training_or_s2_or_stage09": False}
    run_manifest_path = output / "S4_RUN_MANIFEST.json"
    run_manifest_sha = write_or_validate_run_manifest(run_manifest_path, manifest_payload)
    store = RecordStore(output / "records")

    # Characterize the pinned teacher once against the same production Full300 panel.
    teacher_adapter, teacher_model, teacher_provenance = _teacher(artifact_root.resolve(), artifact, gpu)
    teacher_summary_path = output / "summaries" / "teacher_full300.json"
    if not teacher_summary_path.exists():
        emit({"event": "s4_battery_start", "model_role": "teacher", "step": None,
              "panel": "owt_full300", "items": len(items)})
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", enabled=False):
        teacher_result = evaluate_probe_battery(adapter=teacher_adapter, items=items, store=store,
            checkpoint_sha256=artifact["teacher"]["weights_sha256"], panel_sha256=panel_sha,
            run_id=TEACHER_RUN_ID, model_role="teacher", control_seed=CONTROL_SEED,
            denominator_floor=DENOMINATOR_FLOOR, responsiveness_floor=RESPONSIVENESS_FLOOR,
            provenance=teacher_provenance,
            progress_callback=lambda done, total, item_id: emit({"event": "s4_battery_progress",
                "model_role": "teacher", "completed_items": done, "total_items": total,
                "item_id": item_id, "elapsed_seconds": time.perf_counter() - started,
                "eta_seconds": (total - done) * (time.perf_counter() - started) / done})
                if done % 10 == 0 or done == total else None)
    teacher_peak = torch.cuda.max_memory_allocated(0)
    teacher_payload = {"kind": "s4-probe-battery-summary-v1", "model_role": "teacher",
        "checkpoint_sha256": artifact["teacher"]["weights_sha256"], "panel_sha256": panel_sha,
        "step": None, "precision": "fp32",
        "result": teacher_result}
    teacher_summary_sha = _write_result(teacher_summary_path, teacher_payload)
    emit({"event": "s4_battery_complete", "model_role": "teacher", "status": teacher_result["status"],
          "record_count": len(items) * len(PROBE_IDS), "peak_cuda_bytes": teacher_peak,
          "summary_sha256": teacher_summary_sha})
    del teacher_adapter, teacher_model
    gc.collect()
    torch.cuda.empty_cache()
    if teacher_result["status"] != "complete":
        raise RuntimeError("S4 teacher reference battery contains failed or missing probe items")

    student_load, expected_model_sha = _student_config(artifact_root.resolve(), artifact)
    completed = []
    for condition in CONDITIONS:
        source = source_inventory["sources"][condition]
        if source["identity"].get("model_hash") != expected_model_sha:
            raise ValueError(f"S1 student architecture hash differs from checkpoint identity: {condition}")
        for checkpoint in source["checkpoints"]:
            step = checkpoint["step"]
            summary_path = output / "summaries" / condition / f"step-{step:05d}.json"
            emit({"event": "s4_battery_start", "model_role": "student", "condition": condition,
                  "step": step, "run_id": source["run_id"], "checkpoint": checkpoint["path"],
                  "panel": "owt_full300", "items": len(items)})
            adapter, model = student_load(checkpoint)
            torch.cuda.reset_peak_memory_stats(0)
            started = time.perf_counter()
            identity = source["identity"]
            provenance = {"followup_study": "S4", "condition": condition, "seed": 0,
                "run_id": source["run_id"], "protocol_hash": source["original_protocol_root_sha256"],
                "source_checkpoint_manifest_sha256": checkpoint["manifest_file_sha256"],
                "source_checkpoint_step": step, "D24_sha256": D24_SHA256,
                "inference_device": gpu, "precision": "fp32", "panel_sha256": panel_sha,
                "control_seed": CONTROL_SEED, "denominator_floor": DENOMINATOR_FLOOR,
                "responsiveness_floor": RESPONSIVENESS_FLOOR}
            with torch.autocast(device_type="cuda", enabled=False):
                result = evaluate_probe_battery(adapter=adapter, items=items, store=store,
                    checkpoint_sha256=checkpoint["model_file_sha256"], panel_sha256=panel_sha,
                    run_id=source["run_id"], model_role="student", control_seed=CONTROL_SEED,
                    denominator_floor=DENOMINATOR_FLOOR, responsiveness_floor=RESPONSIVENESS_FLOOR,
                    provenance=provenance, source_identity=identity, step=step,
                    progress_callback=lambda done, total, item_id, c=condition, s=step, t=started:
                        emit({"event": "s4_battery_progress", "model_role": "student",
                            "condition": c, "step": s, "completed_items": done, "total_items": total,
                            "item_id": item_id, "elapsed_seconds": time.perf_counter() - t,
                            "eta_seconds": (total - done) * (time.perf_counter() - t) / done})
                        if done % 10 == 0 or done == total else None)
            peak = torch.cuda.max_memory_allocated(0)
            summary_payload = {"kind": "s4-probe-battery-summary-v1", "model_role": "student",
                "condition": condition, "training_seed": 0, "run_id": source["run_id"],
                "original_protocol_root_sha256": source["original_protocol_root_sha256"],
                "D24_sha256": D24_SHA256, "step": step,
                "checkpoint_manifest_sha256": checkpoint["manifest_file_sha256"],
                "checkpoint_model_sha256": checkpoint["model_file_sha256"],
                "panel_sha256": panel_sha, "precision": "fp32",
                "result": result}
            summary_sha = _write_result(summary_path, summary_payload)
            emit({"event": "s4_battery_complete", "model_role": "student", "condition": condition,
                  "step": step, "status": result["status"],
                  "record_count": len(items) * len(PROBE_IDS), "peak_cuda_bytes": peak,
                  "summary_sha256": summary_sha})
            completed.append({"condition": condition, "step": step, "summary_path": str(summary_path),
                              "summary_sha256": summary_sha, "status": result["status"]})
            del adapter, model
            gc.collect()
            torch.cuda.empty_cache()
            if result["status"] != "complete":
                raise RuntimeError(f"S4 battery incomplete for {condition}/step{step}")

    expected_batteries = 1 + len(CONDITIONS) * len(STEPS)
    if len(completed) != len(CONDITIONS) * len(STEPS):
        raise RuntimeError("S4 checkpoint battery coverage mismatch")
    record_count, teacher_record_count, student_record_count = _verify_record_coverage(
        output / "records", items=items, panel_sha=panel_sha,
        source_inventory=source_inventory, artifact=artifact)
    if record_count != 108000 or teacher_record_count != 3000 or student_record_count != 105000:
        raise RuntimeError("S4 persisted teacher/student item record totals differ")
    sums, file_count, total_bytes = _write_sums(output, excluded={"SHA256SUMS.txt", "S4_FINAL_AUDIT.json",
                                                                 "S4_FINAL_AUDIT.json.sha256"})
    audit_payload = {"schema": "stage08-s4-final-audit-v1", "status": "COMPLETE",
        "created_utc": utc_now(), "run_manifest_sha256": run_manifest_sha,
        "run_manifest_path": str(run_manifest_path), "training_seed": 0,
        "condition_checkpoint_batteries": len(completed), "expected_condition_checkpoint_batteries": 35,
        "teacher_reference_batteries": 1, "expected_total_batteries": expected_batteries,
        "teacher_summary_sha256": teacher_summary_sha,
        "completed_student_batteries": completed,
        "probe_ids": list(PROBE_IDS), "full300_items": len(items),
        "expected_record_count": expected_batteries * len(items) * len(PROBE_IDS),
        "verified_record_count": record_count, "teacher_record_count": teacher_record_count,
        "student_record_count": student_record_count,
        "SHA256SUMS_path": str(sums), "SHA256SUMS_file_sha256": sha256_file(sums),
        "manifested_file_count": file_count, "manifested_uncompressed_bytes": total_bytes,
        "source_runs_modified": False, "model_weights_modified": False,
        "training": False, "S2": False, "Stage09": False}
    audit_path = output / "S4_FINAL_AUDIT.json"
    audit_sha = write_sealed_new(audit_path, audit_payload)
    write_new(output / "S4_FINAL_AUDIT.json.sha256",
              f"{sha256_file(audit_path)}  {audit_path.name}\n".encode("ascii"))
    result = {"status": "COMPLETE", "study": "S4", "audit_sha256": audit_sha,
              "record_count": audit_payload["expected_record_count"],
              "result_directory": str(output), "run_manifest_sha256": run_manifest_sha}
    emit({"event": "s4_study_complete", **result})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path(r"D:\KD-SINK-central\runs"))
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(output=args.output, runs_root=args.runs_root,
            artifact_root=args.artifact_root, resume=args.resume)
        return 0
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {"schema": "stage08-s4-failure-v1", "status": failure_status(exc),
            "created_utc": utc_now(), "hostname": socket.gethostname(),
            "error_type": type(exc).__name__, "error": str(exc),
            "source_runs_modified": False, "training": False, "Stage09": False}
        path = args.output / f"S4_FAILURE_{int(time.time())}.json"
        write_sealed_new(path, failure)
        emit({"event": "s4_study_failed", **failure, "failure_path": str(path)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
