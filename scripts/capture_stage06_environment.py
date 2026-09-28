"""Capture a sealed, read-only inventory of the active Stage 06 environment."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from sinklab.provenance import seal_payload, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_TOOLS = {"pip", "uv"}
APPROVED_DEVICES = {
    "rtx4080super": ("NVIDIA GeForce RTX 4080 SUPER",
                     "GPU-72b4b307-b613-c35e-ea32-53f4431de9ee"),
    "rtx3090": ("NVIDIA GeForce RTX 3090",
                "GPU-a21766e4-bb31-9b79-5e8f-e58021e9708e"),
}
LOCK_SHA256 = "b71f143ff21a0ccdd45e995b006430399134bb1564b1210bdc70fb3c67f4c3d5"


def _run(*args: str) -> dict:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"command": list(args), "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def _name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def verify_gpu_identity(fields: list[str], role: str, expected_model: str | None,
                        expected_uuid: str | None, *,
                        second_4080_qualification: bool = False,
                        class_environment: bool = False) -> None:
    approved_model, approved_uuid = APPROVED_DEVICES[role]
    if class_environment:
        if (role != "rtx4080super" or len(fields) != 4 or
                fields[0] != approved_model or not fields[1] or
                (expected_model is not None and expected_model != fields[0]) or
                (expected_uuid is not None and expected_uuid != fields[1])):
            raise RuntimeError("4080 class environment requires exact model and actual UUID")
        return
    if second_4080_qualification:
        if role != "rtx4080super" or expected_model != approved_model or not expected_uuid or (
                expected_uuid == approved_uuid):
            raise RuntimeError("second 4080 qualification requires its distinct explicit UUID and model")
        if len(fields) != 4 or fields[:2] != [expected_model, expected_uuid]:
            raise RuntimeError(f"unexpected second 4080 GPU identity: {fields}")
        return
    if role == "rtx3090" and (expected_model is None or expected_uuid is None):
        raise RuntimeError("3090 capture requires explicit expected model and UUID")
    if (expected_model is not None and expected_model != approved_model or
            expected_uuid is not None and expected_uuid != approved_uuid):
        raise RuntimeError("requested GPU identity differs from approved device role")
    if len(fields) != 4 or fields[:2] != [approved_model, approved_uuid]:
        raise RuntimeError(f"unexpected GPU identity for {role}: {fields}")


def verify_locked_distribution_equality(current: dict[str, str], reference: dict) -> None:
    payload, _ = verify_envelope(reference)
    if payload["dependency_lock_sha256"] != LOCK_SHA256:
        raise RuntimeError("4080 reference uses a different dependency lock")
    if current != payload["installed_locked_distributions"]:
        missing = sorted(set(payload["installed_locked_distributions"]) - set(current))
        extra = sorted(set(current) - set(payload["installed_locked_distributions"]))
        changed = sorted(name for name in set(current) & set(payload["installed_locked_distributions"])
                         if current[name] != payload["installed_locked_distributions"][name])
        raise RuntimeError(f"3090/4080 lock-managed distribution map differs: "
                           f"missing={missing}, extra={extra}, changed={changed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device-role", choices=tuple(APPROVED_DEVICES), default="rtx4080super")
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-uuid")
    parser.add_argument("--second-4080-qualification", action="store_true",
                        help="capture a distinct 4080 UUID as qualification evidence only")
    parser.add_argument("--4080-class-environment", action="store_true", dest="class_4080_environment",
                        help="capture exact-model class host software; no per-card profile claim")
    args = parser.parse_args()
    if args.second_4080_qualification and args.class_4080_environment:
        parser.error("choose one 4080 capture policy")

    # Git's Windows checkout can convert the committed LF lock to CRLF. The
    # lock authority is the committed file; accept only that line-ending change.
    lock_bytes = (ROOT / "uv.lock").read_bytes().replace(b"\r\n", b"\n")
    committed_lock = subprocess.check_output(["git", "show", "HEAD:uv.lock"], cwd=ROOT)
    if lock_bytes != committed_lock or hashlib.sha256(lock_bytes).hexdigest() != LOCK_SHA256:
        raise RuntimeError("working dependency lock differs from approved committed uv.lock")
    lock = {_name(row["name"]): row["version"]
            for row in tomllib.loads(lock_bytes.decode("utf-8"))["package"]}
    installed = {_name(row.metadata["Name"]): row.version
                 for row in importlib.metadata.distributions()}
    extras = {name: version for name, version in installed.items()
              if name not in lock and name not in BOOTSTRAP_TOOLS}
    mismatches = {name: {"installed": version, "locked": lock[name]}
                  for name, version in installed.items()
                  if name in lock and version != lock[name]}

    dry_run = _run(sys.executable, "-m", "uv", "sync", "--locked", "--dry-run")
    freeze = _run(sys.executable, "-m", "pip", "freeze")
    pip_check = _run(sys.executable, "-m", "pip", "check")
    gpu = _run("nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
               "--format=csv,noheader")
    source = _run("git", "rev-parse", "HEAD")
    for result in (dry_run, freeze, pip_check, gpu, source):
        if result["exit_code"] != 0:
            raise RuntimeError(f"capture command failed: {result['command']}: {result['stderr']}")
    if mismatches:
        raise RuntimeError(f"installed locked-version mismatch: {mismatches}")
    if args.phase == "before" and extras != {
            "build": "1.3.0", "pyproject-hooks": "1.3.3", "wheel": "0.46.3"}:
        raise RuntimeError(f"unexpected before-state extras: {extras}")
    if args.phase == "after" and extras:
        raise RuntimeError(f"unexpected after-state extras: {extras}")
    if args.phase == "after" and "Would make no changes" not in dry_run["stderr"]:
        raise RuntimeError("locked dry-run still has pending changes")
    gpu_lines = [line.strip() for line in gpu["stdout"].splitlines() if line.strip()]
    if len(gpu_lines) != 1:
        raise RuntimeError(f"expected one production GPU, found {gpu_lines}")
    gpu_fields = [field.strip() for field in gpu_lines[0].split(",")]
    verify_gpu_identity(gpu_fields, args.device_role, args.expected_model, args.expected_uuid,
                        second_4080_qualification=args.second_4080_qualification,
                        class_environment=args.class_4080_environment)
    installed_locked = dict(sorted((name, version) for name, version
                                    in installed.items() if name in lock))
    if args.phase == "after" and (args.device_role == "rtx3090" or
                                  args.second_4080_qualification or
                                  args.class_4080_environment):
        reference = json.loads((ROOT / "protocols/s1_environment_4080_exact_v1.json")
                               .read_text(encoding="utf-8"))
        verify_locked_distribution_equality(installed_locked, reference)

    payload = {
        "kind": ("stage06_rtx4080super_class_environment_v1"
                 if args.class_4080_environment else
                 "stage06_second_rtx4080super_exact_environment_qualification_v1"
                 if args.second_4080_qualification else
                 f"stage06_{args.device_role}_exact_environment_evidence"),
        "phase": args.phase,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source["stdout"].strip(),
        "production_ready": False,
        "dependency_lock_path": "uv.lock",
        "dependency_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "locked_package_count_all_platforms": len(lock),
        "installed_distribution_count_including_bootstrap_and_extras": len(installed),
        "installed_locked_distributions": installed_locked,
        "lock_entries_not_installed_on_this_platform": dict(sorted((name, version)
                                                         for name, version in lock.items()
                                                         if name not in installed)),
        "bootstrap_tools_outside_lock": dict(sorted((name, installed[name])
                                                 for name in BOOTSTRAP_TOOLS if name in installed)),
        "unexpected_active_distributions": dict(sorted(extras.items())),
        "locked_version_mismatches": mismatches,
        "active_environment_exact_sync": args.phase == "after",
        "python": sys.version,
        "os": {"platform": platform.platform(), "version": platform.version(),
               "release": platform.release(), "machine": platform.machine()},
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "numerical_policy": {"student_parameters": "fp32", "adam_states": "fp32",
                             "compute": "bf16_autocast",
                             "probability_and_loss_reductions": "fp32"},
        "pip_freeze": freeze,
        "uv_sync_locked_dry_run": dry_run,
        "pip_check": pip_check,
        "gpu_query": gpu,
        "gpu": {"model": gpu_fields[0], "uuid": gpu_fields[1],
                "total_vram_mib": int(gpu_fields[2].split()[0]),
                "nvidia_driver": gpu_fields[3]},
    }
    if args.class_4080_environment:
        payload["hardware_class_transfer_only"] = True
    if args.second_4080_qualification:
        payload["second_4080_qualification_only"] = True
    if args.phase == "after":
        payload["role_status"] = ("rtx4080super_class_software_exact_sync_verified"
                                  if args.class_4080_environment else
                                  "second_rtx4080super_exact_sync_verified"
                                  if args.second_4080_qualification else
                                  f"{args.device_role}_exact_sync_verified")
        if args.device_role == "rtx4080super" and not (
                args.second_4080_qualification or args.class_4080_environment):
            payload["remaining_environment_task"] = (
                "Reproduce exact same uv.lock environment on approved RTX 3090; "
                "verify installed-distribution equality; record 3090 OS, driver, "
                "CUDA, cuDNN and GPU UUID.")
        else:
            payload["lock_managed_distributions_equal_4080"] = True
            payload["reference_4080_environment_sha256"] = reference["sha256"]
    document = {"schema_version": 1, **seal_payload(payload)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"{args.out}: {document['sha256']}")


if __name__ == "__main__":
    main()
