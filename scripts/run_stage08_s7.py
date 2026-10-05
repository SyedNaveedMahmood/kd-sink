"""Verify a completed S5 bundle and derive the S7 descriptive utility analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sinklab.provenance import (_no_duplicate_keys, canonical_json_bytes,
                                verify_envelope)  # noqa: E402
from sinklab.s5_compatibility import D26_PATH  # noqa: E402
from sinklab.s7_utility import (add_extended_s1_records, analyze_s7,
                                add_clean_supplements, add_lm2000_endpoints,
                                read_extended_s1_records, read_lm2000_records,
                                read_s5_bundle, S5_AUDIT_SHA256)  # noqa: E402
from scripts.run_stage08_s7_supplement import (validate_analysis_lock,
    validate_source_roots)  # noqa: E402


def run(*, s5_bundle: Path, output: Path, runs_root: Path | None = None,
        supplements_dir: Path | None = None, analysis_lock: Path | None = None,
        artifact_root: Path | None = None) -> dict:
    source = s5_bundle.resolve(strict=True)
    output = output.resolve()
    if (output == source or source in output.parents or output in source.parents or
            output == REPO or REPO in output.parents):
        raise ValueError("S7 output must be separate from its source and Git tree")
    if runs_root is not None:
        original = runs_root.resolve(strict=True)
        if output == original or original in output.parents or output in original.parents:
            raise ValueError("S7 output must be separate from original S1 runs")
    if artifact_root is not None:
        artifacts = artifact_root.resolve(strict=True)
        if (runs_root is None or output == artifacts or artifacts in output.parents or
                output in artifacts.parents):
            raise ValueError("S7 lm2000 extraction requires original runs and separate artifact/output roots")
    if supplements_dir is not None:
        supplement_root = supplements_dir.resolve(strict=True)
        if (analysis_lock is None or runs_root is None or artifact_root is None or
                output == supplement_root or
                supplement_root in output.parents or output in supplement_root.parents):
            raise ValueError("S7 final analysis requires separate output, approved lock, S1 runs, and artifacts")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("S7 output directory must be empty")
    panels, provenance = read_s5_bundle(source, d26_path=REPO / D26_PATH,
                                       expected_audit_sha256=S5_AUDIT_SHA256)
    result = analyze_s7(panels, provenance)
    if runs_root is not None:
        result = add_extended_s1_records(result, read_extended_s1_records(
            s5_bundle=source, runs_root=original))
    if artifact_root is not None:
        result = add_lm2000_endpoints(result, read_lm2000_records(
            s5_bundle=source, runs_root=original, artifact_root=artifacts,
            repo_root=REPO))
    if supplements_dir is not None:
        approved = validate_analysis_lock(json.loads(analysis_lock.read_text(encoding="utf-8"),
                                                    object_pairs_hook=_no_duplicate_keys))
        validate_source_roots(approved, runs_root=original, s5_bundle=source,
                              artifact_root=artifacts)
        paths = sorted(supplement_root.rglob("S7_C*_STEP*_FULL300.json"))
        if len(paths) != 36:
            raise ValueError("S7 requires exactly 36 clean decomposition supplements")
        supplements = [json.loads(path.read_text(encoding="utf-8"),
                                  object_pairs_hook=_no_duplicate_keys) for path in paths]
        if any(item.get("analysis_lock_sha256") != approved["sha256"] for item in supplements):
            raise ValueError("S7 supplement analysis lock seal mismatch")
        artifact_document = json.loads((REPO / "protocols" / "artifact.lock.json").read_text(
            encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
        artifact, artifact_lock_sha256 = verify_envelope(artifact_document)
        teacher = artifact["teacher"]
        expected_teacher = {
            "id": teacher["id"], "revision": teacher["revision"],
            "weights_sha256": teacher["weights_sha256"],
            "config_sha256": teacher["config_sha256"],
            "device_model": approved["device_model"],
            "device_uuid": approved["device_uuid"],
            "precision": approved["precision"],
            "D24_sha256": approved["D24_sha256"],
        }
        result = add_clean_supplements(result, panels, supplements,
            expected_analysis_lock_sha256=approved["sha256"],
            expected_teacher_identity=expected_teacher,
            expected_checkpoint_tensor_digests=provenance[
                "s5_full300_checkpoint_tensor_sha256"])
        result["provenance"]["artifact_lock_sha256"] = artifact_lock_sha256
        result["provenance"]["supplement_file_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    data = canonical_json_bytes(result) + b"\n"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "S7_ANALYSIS.json"
    with target.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256(data).hexdigest()
    audit = {"schema_version": 1, "study": "S7", "status": result["status"],
             "s7_analysis_sha256": digest, "source": provenance,
             "source_modified": False, "model_loaded": False, "new_inference": False,
             "original_s1_items_reverified": runs_root is not None,
             "training": False, "science_complete": False}
    with (output / "S7_AUDIT.json").open("xb") as stream:
        stream.write(canonical_json_bytes(audit) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s5-bundle", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path,
                        help="reverify original S1 items to recover teacher matching and accuracy")
    parser.add_argument("--supplements-dir", type=Path)
    parser.add_argument("--analysis-lock", type=Path)
    parser.add_argument("--artifact-root", type=Path,
                        help="verify frozen lm2000 panel and original clean endpoint records")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        audit = run(s5_bundle=args.s5_bundle, output=args.output, runs_root=args.runs_root,
                    supplements_dir=args.supplements_dir, analysis_lock=args.analysis_lock,
                    artifact_root=args.artifact_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"S7: {exc}\n")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
