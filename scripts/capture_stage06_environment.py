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

from sinklab.provenance import seal_payload


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_TOOLS = {"pip", "uv"}


def _run(*args: str) -> dict:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"command": list(args), "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def _name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    lock_bytes = (ROOT / "uv.lock").read_bytes()
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
    if len(gpu_fields) != 4 or gpu_fields[0] != "NVIDIA GeForce RTX 4080 SUPER":
        raise RuntimeError(f"unexpected GPU identity: {gpu_fields}")

    payload = {
        "kind": "stage06_rtx4080super_exact_environment_evidence",
        "phase": args.phase,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source["stdout"].strip(),
        "production_ready": False,
        "dependency_lock_path": "uv.lock",
        "dependency_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "locked_package_count_all_platforms": len(lock),
        "installed_distribution_count_including_bootstrap_and_extras": len(installed),
        "installed_locked_distributions": dict(sorted((name, version) for name, version
                                                       in installed.items() if name in lock)),
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
    if args.phase == "after":
        payload["role_status"] = "rtx4080super_exact_sync_verified"
        payload["remaining_environment_task"] = (
            "Reproduce exact same uv.lock environment on approved RTX 3090; "
            "verify installed-distribution equality; record 3090 OS, driver, "
            "CUDA, cuDNN and GPU UUID.")
    document = {"schema_version": 1, **seal_payload(payload)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"{args.out}: {document['sha256']}")


if __name__ == "__main__":
    main()
