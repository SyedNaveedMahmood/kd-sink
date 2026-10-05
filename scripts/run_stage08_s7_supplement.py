"""One explicit S7 clean-attention supplement on a verified seed-0 S1 state."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_stage08_s4 import _artifact_inputs, _student_config, _teacher  # noqa: E402
from scripts.stage08_scientific_common import (EXPECTED_GPU_UUID, current_gpu,
    preflight_sources, sha256_file)  # noqa: E402
from sinklab.followup_policy import D24_SHA256  # noqa: E402
from sinklab.provenance import _no_duplicate_keys, canonical_json_bytes, verify_envelope  # noqa: E402
from sinklab.s5_analysis import CONDITIONS  # noqa: E402
from sinklab.s5_compatibility import D26_PATH  # noqa: E402
from sinklab.s7_utility import (FULL_STEPS, S7Error, evaluate_clean_decomposition,
                                read_s5_bundle, S5_AUDIT_SHA256)  # noqa: E402

APPROVED_LOCK_PATH = REPO / "protocols" / "s7_utility_analysis_approved.json"


def validate_analysis_lock(document: dict) -> dict:
    payload, digest = verify_envelope(document)
    if not APPROVED_LOCK_PATH.is_file():
        raise S7Error("researcher-approved S7 analysis lock is not checked in")
    if json.loads(APPROVED_LOCK_PATH.read_text(encoding="utf-8"),
                  object_pairs_hook=_no_duplicate_keys) != document:
        raise S7Error("S7 analysis lock differs from checked-in approved document")
    if (payload.get("study") != "S7" or
            payload.get("status") != "approved_prospective" or
            payload.get("kind") != "s7-clean-attention-supplement-v1" or
            payload.get("training_seed") != 0 or
            payload.get("conditions") != list(CONDITIONS) or
            payload.get("steps") != list(FULL_STEPS) or
            payload.get("panel") != "owt_full300" or
            payload.get("precision") != "fp32" or
            payload.get("device_uuid") != EXPECTED_GPU_UUID or
            payload.get("D24_sha256") != D24_SHA256):
        raise S7Error("approved prospective S7 supplement lock required")
    return {**payload, "sha256": digest}


def run(*, condition: str, step: int, runs_root: Path, artifact_root: Path,
        s5_bundle: Path, analysis_lock: Path, output: Path) -> dict:
    if condition not in CONDITIONS or type(step) is not int or step not in FULL_STEPS:
        raise S7Error("one explicit S7 condition and retained step required")
    lock = validate_analysis_lock(json.loads(analysis_lock.read_text(encoding="utf-8"),
                                             object_pairs_hook=_no_duplicate_keys))
    output = output.resolve()
    sources = [runs_root.resolve(), artifact_root.resolve(), s5_bundle.resolve()]
    if (output == REPO or REPO in output.parents or
            any(output == source or source in output.parents or output in source.parents
                for source in sources)):
        raise S7Error("S7 supplement output must be outside Git and source roots")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("S7 supplement output directory must be empty")
    panels, provenance = read_s5_bundle(s5_bundle, d26_path=REPO / D26_PATH,
                                       expected_audit_sha256=S5_AUDIT_SHA256)
    joined = next(row for row in panels["owt_full300"]["joined"] if row["step"] == step)
    gpu = current_gpu()
    if gpu["uuid"] != lock["device_uuid"]:
        raise S7Error("S7 supplement GPU differs from approved lock")
    artifact, items, panel_sha, panel_receipt = _artifact_inputs(artifact_root)
    if panel_sha != joined["panel_sha256"] or len(items) != 300:
        raise S7Error("S7 frozen Full300 item identity differs from S5")
    inventory = preflight_sources(steps=(step,), study="S7", runs_root=runs_root)
    source = inventory["sources"][condition]
    checkpoint = source["checkpoints"][0]
    if (source["run_id"] != joined["source_run_ids"][condition] or
            source["original_protocol_root_sha256"] != joined["source_protocol_roots"][condition] or
            source["identity"]["gpu_uuid"] != joined["source_gpu_uuids"][condition] or
            checkpoint["step"] != step):
        raise S7Error("S7 retained state differs from S5 source identity")
    student_load, expected_model_sha = _student_config(artifact_root, artifact)
    if source["identity"]["model_hash"] != expected_model_sha:
        raise S7Error("S7 student architecture differs from original S1 identity")
    teacher_adapter, teacher_model, teacher_receipt = _teacher(artifact_root, artifact, gpu)
    student_adapter, student_model = student_load(checkpoint)
    try:
        import torch
        from sinklab.training_entry import _model_digest, _teacher_map
        from sinklab.s7_utility import FILES
        with (Path(s5_bundle) / FILES[0]).open(encoding="utf-8") as stream:
            matches = [json.loads(line, object_pairs_hook=_no_duplicate_keys)
                       for line in stream if line.strip()]
        source_row = next(row for row in matches if row["condition"] == condition and
                          row["panel"] == "owt_full300" and row["step"] == step)
        if _model_digest(student_model) != source_row["checkpoint_sha256"]:
            raise S7Error("S7 loaded student tensors differ from S5 evaluation tensors")
        started = time.perf_counter()
        def progress(done: int, total: int, item_id: str) -> None:
            if done % 10 == 0 or done == total:
                elapsed = time.perf_counter() - started
                print(json.dumps({"event": "s7_supplement_progress", "condition": condition,
                    "step": step, "run_id": source["run_id"], "gpu_uuid": gpu["uuid"],
                    "completed_items": done, "total_items": total, "item_id": item_id,
                    "elapsed_seconds": elapsed, "eta_seconds": elapsed / done * (total - done),
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(0)},
                    sort_keys=True), flush=True)
        result = evaluate_clean_decomposition(adapter=student_adapter,
            teacher_adapter=teacher_adapter, teacher_map=list(_teacher_map("S1")),
            items=items, precision="fp32", on_item=progress)
    finally:
        del student_adapter, student_model, teacher_adapter, teacher_model
        gc.collect()
    payload = {"schema_version": 1, "study": "S7", "kind": "clean-attention-supplement",
        "status": "complete", "condition": condition, "training_seed": 0,
        "step": step, "panel": "owt_full300", "panel_receipt": panel_receipt,
        "analysis_lock_sha256": lock["sha256"], "D24_sha256": D24_SHA256,
        "s5_source_audit_sha256": provenance["s5_audit_sha256"],
        "run_id": source["run_id"], "original_protocol_root_sha256":
            source["original_protocol_root_sha256"],
        "source_checkpoint_manifest_file_sha256": checkpoint["manifest_file_sha256"],
        "source_checkpoint_model_file_sha256": checkpoint["model_file_sha256"],
        "teacher_receipt": teacher_receipt, "inference_gpu": gpu,
        "source_runs_modified": False, "training": False, "result": result}
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"S7_{condition}_STEP{step:06d}_FULL300.json"
    data = canonical_json_bytes(payload) + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return {"status": "complete", "path": str(path), "sha256": sha256_file(path),
            "condition": condition, "step": step, "item_count": len(items)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("runs-root", "artifact-root", "s5-bundle", "analysis-lock", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(condition=args.condition, step=args.step, runs_root=args.runs_root,
            artifact_root=args.artifact_root, s5_bundle=args.s5_bundle,
            analysis_lock=args.analysis_lock, output=args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"S7: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
