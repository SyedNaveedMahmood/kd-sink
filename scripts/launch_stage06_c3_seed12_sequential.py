"""Launch the sealed optional C3 seeds in sequence; stop on any failed gate.

This operator process is one-shot. A host reboot cancels it; an interrupted run
requires a separate, reviewed same-run resume. It never retries or retunes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_stage06_c3_seed12 as queue


def _no_other_trainer() -> None:
    active = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            words = process.cmdline()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if process.pid != os.getpid() and "sinklab" in " ".join(words).lower() and "train" in words:
            active.append(process.pid)
    if active:
        raise RuntimeError(f"another sinklab trainer is active: {active}")


def _require_clean_pushed_successor(repo: Path, expected_root: str,
                                    source_milestone: str) -> None:
    from sinklab.runtime_provenance import validate_runtime_source
    from sinklab.stage06_readiness import validate_final_lock_set

    queue._cmd(["git", "fetch", "origin", "main"], cwd=repo)
    head = queue._cmd(["git", "rev-parse", "HEAD"], cwd=repo).strip()
    origin = queue._cmd(["git", "rev-parse", "origin/main"], cwd=repo).strip()
    if head != origin or queue._cmd(["git", "status", "--porcelain"], cwd=repo).strip():
        raise RuntimeError("D21 checkout is not clean and pushed to origin/main")
    locks = validate_final_lock_set(repo / "protocols")
    protocol = locks["protocol"]
    if (protocol["sha256"] != expected_root or
            protocol["payload"]["production_runtime_source_commit"] != source_milestone):
        raise RuntimeError("D21 root or runtime source differs from the approved queue")
    validate_runtime_source(repo, source_milestone, 1,
                            loaded_package_dir=repo / "src/sinklab")


def _inputs(replication_root: Path, production_root: Path, seed: int) -> dict[str, Path]:
    return {
        "corpus": queue._one_json_path(replication_root / f"corpus/seed{seed}",
                                       "owt-corpus-*.json"),
        "panels": queue._one_json_path(replication_root / f"panels/seed{seed}",
                                       "owt-panels-*.json"),
        "initialization": queue._one_json_path(
            replication_root / f"initialization/seed{seed}", f"init-seed{seed}-*.json"),
        "student_config": production_root / "student-config/config.json",
        "teacher_dir": production_root / "teacher",
    }


def _train_command(python: str, seed: int, run: Path,
                   inputs: dict[str, Path]) -> list[str]:
    if seed not in (1, 2):
        raise ValueError("D21 queue accepts only C3 seeds 1 and 2")
    return [python, "-u", "-m", "sinklab", "train",
            "--config", f"configs/production/s1/c3_seed{seed}_rtx4080super.json",
            "--protocol-lock", "protocols/protocol.lock.json",
            "--hardware-plan", "protocols/hardware.lock.json",
            "--corpus", str(inputs["corpus"]), "--panels", str(inputs["panels"]),
            "--student-config", str(inputs["student_config"]),
            "--initialization", str(inputs["initialization"]),
            "--teacher-dir", str(inputs["teacher_dir"]),
            "--run-dir", str(run), "--seed", str(seed),
            "--stop-after", "10000", "--mse-scale", queue.MSE_SCALE]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "python", "production-root", "replication-root",
                 "seed0-run-dir", "seed0-prefix-log", "seed1-run-dir",
                 "seed2-run-dir", "ops-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-root", required=True)
    parser.add_argument("--source-milestone", required=True)
    parser.add_argument("--check-now", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    python = str(args.python.resolve())
    production = args.production_root.resolve()
    replicas = args.replication_root.resolve()
    ops = args.ops_root.resolve()
    status = ops / "queue-status.json"
    runs = {1: args.seed1_run_dir.resolve(), 2: args.seed2_run_dir.resolve()}
    try:
        _require_clean_pushed_successor(repo, args.expected_root, args.source_milestone)
        seed0 = queue._verify_complete(args.seed0_run_dir.resolve(), seed=0,
                                       root_sha=queue.D20_ROOT,
                                       prefix_log=args.seed0_prefix_log.resolve(),
                                       prefix_step=2500)
        queue._gpu()
        _no_other_trainer()
        for seed, run in runs.items():
            if run.exists() and any(run.iterdir()):
                raise RuntimeError(f"seed {seed} run directory is occupied: {run}")
            run.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(run.parent).free < 35 * 1024**3:
                raise RuntimeError(f"seed {seed} run storage below 35 GiB free")
            _inputs(replicas, production, seed)
        if args.check_now:
            print(json.dumps({"ready": True, "root": args.expected_root,
                              "seed0": seed0, "gpu_uuid": queue.NODI_UUID,
                              "seed1_run_dir": str(runs[1]),
                              "seed2_run_dir": str(runs[2]),
                              "training_started": False}, sort_keys=True))
            return
        env = dict(os.environ, PYTHONPATH=str(repo / "src"))
        queue._status(status, "sealed_queue_armed", root=args.expected_root,
                      seed0=seed0, seeds=[1, 2])
        for seed, run in runs.items():
            if seed == 2:
                queue._verify_complete(runs[1], seed=1, root_sha=args.expected_root)
            if run.exists() and any(run.iterdir()):
                raise RuntimeError(f"seed {seed} run directory became occupied")
            _no_other_trainer()
            queue._gpu()
            inputs = _inputs(replicas, production, seed)
            preflight = [python, "-u", "-m", "sinklab", "preflight-production",
                         "--config", f"configs/production/s1/c3_seed{seed}_rtx4080super.json",
                         "--protocol-lock", "protocols/protocol.lock.json",
                         "--hardware-plan", "protocols/hardware.lock.json",
                         "--artifact-root", str(production),
                         "--replication-root", str(replicas), "--seed", str(seed)]
            queue._status(status, f"seed{seed}_preflight", root=args.expected_root)
            preflight_output = queue._cmd(preflight, cwd=repo, env=env)
            preflight_result = json.loads(preflight_output.strip().splitlines()[-1])
            if (preflight_result.get("training_started") is not False or
                    preflight_result.get("gpu_uuid") != queue.NODI_UUID or
                    preflight_result.get("protocol_sha256") != args.expected_root or
                    preflight_result.get("seed_replication_artifacts_verified") is not True or
                    preflight_result.get("scientific_artifacts_verified") is not True):
                raise RuntimeError(f"seed {seed} production preflight did not approve launch")
            with (ops / "preflights.log").open("a", encoding="utf-8") as stream:
                stream.write(preflight_output)
            command = _train_command(python, seed, run, inputs)
            queue._status(status, f"seed{seed}_running", root=args.expected_root,
                          command=command, run_dir=str(run))
            queue._cmd(command, cwd=repo, env=env, log=ops / f"seed{seed}.launcher.log")
            result = queue._verify_complete(run, seed=seed, root_sha=args.expected_root)
            queue._status(status, f"seed{seed}_complete", root=args.expected_root,
                          result=result)
        queue._status(status, "queue_complete", root=args.expected_root,
                      seed1_run_dir=str(runs[1]), seed2_run_dir=str(runs[2]))
    except Exception as exc:
        queue._status(status, "stopped_failed_gate", error=str(exc), launch_armed=False)
        raise


if __name__ == "__main__":
    main()
