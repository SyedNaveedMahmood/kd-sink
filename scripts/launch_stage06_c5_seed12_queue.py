"""Run only the explicitly requested C5 seed1 and seed2 jobs, in order."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import launch_stage06_c0_c2_seed12_queue as shared


MODEL = "NVIDIA GeForce RTX 4080 SUPER"
PROTOCOL_ROOT = "7b12e5a637c3a8b6ebe66d4bedbd202077fc7960bc98a443babf8bf68de8d8a7"
JOBS = (
    (1, "production/s1/c5_seed1_rtx4080super.json", "s1-c5-seed1-rtx4080super"),
    (2, "production/s1/c5_seed2_rtx4080super.json", "s1-c5-seed2-rtx4080super"),
)


def _preflight(repo: Path, python: str, production: Path, replicas: Path,
               ops: Path, seed: int, config: str, gpu_uuid: str) -> dict:
    command = [python, "-u", "-m", "sinklab", "preflight-production",
               "--config", f"configs/{config}",
               "--protocol-lock", "protocols/protocol.lock.json",
               "--hardware-plan", "protocols/hardware.lock.json",
               "--artifact-root", str(production),
               "--replication-root", str(replicas), "--seed", str(seed)]
    result = subprocess.run(command, cwd=repo, env=dict(os.environ, PYTHONPATH=str(repo / "src")),
                            text=True, capture_output=True, check=False)
    (ops / f"preflight-seed{seed}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (ops / f"preflight-seed{seed}.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"seed{seed} preflight failed ({result.returncode}); see {ops}")
    evidence = json.loads(result.stdout.strip().splitlines()[-1])
    if (evidence.get("condition") != "C5" or
            evidence.get("protocol_sha256") != PROTOCOL_ROOT or
            evidence.get("gpu_name") != MODEL or evidence.get("gpu_uuid") != gpu_uuid or
            evidence.get("scientific_artifacts_verified") is not True or
            evidence.get("seed_replication_artifacts_verified") is not True or
            evidence.get("training_started") is not False):
        raise RuntimeError(f"seed{seed} preflight evidence is incomplete")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "python", "production-root", "replication-root",
                 "run-root", "ops-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--check-only", action="store_true",
                        help="run all gates and preflights without starting training")
    args = parser.parse_args()
    repo, production, replicas, runs, ops = (path.resolve() for path in
        (args.repo, args.production_root, args.replication_root, args.run_root, args.ops_root))
    ops.mkdir(parents=True, exist_ok=True)
    status = ops / "queue-status.json"
    python = str(args.python.resolve())
    sys.path.insert(0, str(repo / "src"))
    shared.NODI_UUID = args.gpu_uuid
    try:
        head = shared._cmd(["git", "rev-parse", "HEAD"], cwd=repo).strip()
        origin = shared._cmd(["git", "rev-parse", "origin/main"], cwd=repo).strip()
        dirty = shared._cmd(["git", "status", "--porcelain"], cwd=repo).strip()
        if head != args.expected_head or origin != head or dirty:
            raise RuntimeError("queue requires the exact clean commit pushed to origin/main")
        from sinklab.runtime_provenance import validate_runtime_source
        from sinklab.stage06_readiness import validate_final_lock_set

        locks = validate_final_lock_set(repo / "protocols")
        protocol = locks["protocol"]["payload"]
        root = locks["protocol"]["sha256"]
        if root != PROTOCOL_ROOT or protocol["study"] != "S1":
            raise RuntimeError("D23 production root differs from this C5 queue")
        validate_runtime_source(repo, protocol["production_runtime_source_commit"],
                                protocol["execution_critical_path_set_version"],
                                loaded_package_dir=repo / "src/sinklab")
        plan = shared._json(repo / "configs/production/s1_seed12_jobs.json")
        selected = [row for row in plan["jobs"]
                    if row["run_id"] in {run_id for _, _, run_id in JOBS}]
        if (plan.get("status") != "approved" or len(selected) != 2 or
                [(row["seed"], row["config"], row["run_id"]) for row in selected] != list(JOBS)):
            raise RuntimeError("approved D23 plan does not contain exactly the requested C5 jobs")
        shared._gpu()
        if shared._active_trainers():
            raise RuntimeError("another sinklab trainer is active")
        runs.mkdir(parents=True, exist_ok=True)
        queue_jobs = [{"seed": seed, "condition": "C5", "run_id": run_id}
                      for seed, _, run_id in JOBS]
        completed = shared._verified_completed_prefix(queue_jobs, runs, root)
        completed_seeds = {row["seed"] for row in completed}
        for seed, config, run_id in JOBS:
            run = runs / run_id
            if seed not in completed_seeds and shutil.disk_usage(runs).free < 35 * 1024**3:
                raise RuntimeError(f"less than 35 GiB free for {run_id}")
            binding = shared._json(repo / "configs" / config)["production_binding"]
            if (binding.get("seed") != seed or
                    binding.get("production_runtime_source_commit") != protocol["production_runtime_source_commit"]):
                raise RuntimeError(f"C5 seed{seed} config identity differs")

        preflights = [_preflight(repo, python, production, replicas, ops, seed, config,
                                 args.gpu_uuid) for seed, config, _ in JOBS]
        phase = ("seed1_complete_seed2_queued" if 1 in completed_seeds
                 else "preflights_passed_training_not_started")
        shared._status(status, phase,
                       protocol_root=root, gpu_model=MODEL, gpu_uuid=args.gpu_uuid,
                       queued=[run_id for _, _, run_id in JOBS], completed=completed,
                       preflights=preflights)
        if args.check_only:
            print(json.dumps({"ready": True, "protocol_root": root,
                              "gpu_uuid": args.gpu_uuid,
                              "queued": [run_id for _, _, run_id in JOBS],
                              "completed": completed,
                              "training_started": False}, sort_keys=True))
            return

        armed_phase = ("seed1_complete_seed2_launching" if 1 in completed_seeds
                       else "queue_armed_seed1_launching_seed2_queued")
        shared._status(status, armed_phase,
                       protocol_root=root, gpu_model=MODEL, gpu_uuid=args.gpu_uuid,
                       queued=[run_id for _, _, run_id in JOBS], completed=completed,
                       queue_pid=os.getpid())
        for seed, config, run_id in JOBS:
            if seed in completed_seeds:
                continue
            run = runs / run_id
            if seed == 2 and 1 not in completed_seeds:
                raise RuntimeError("seed2 requires verified seed1 completion")
            shared._gpu()
            if shared._active_trainers():
                raise RuntimeError("another sinklab trainer appeared")
            if shutil.disk_usage(runs).free < 35 * 1024**3:
                raise RuntimeError(f"less than 35 GiB free before {run_id}")
            corpus = shared._one(replicas / f"corpus/seed{seed}", "owt-corpus-*.json")
            panels = shared._one(replicas / f"panels/seed{seed}", "owt-panels-*.json")
            init = shared._one(replicas / f"initialization/seed{seed}",
                               f"init-seed{seed}-*.json")
            command = [python, "-u", "-m", "sinklab", "train",
                       "--config", f"configs/{config}",
                       "--protocol-lock", "protocols/protocol.lock.json",
                       "--hardware-plan", "protocols/hardware.lock.json",
                       "--corpus", str(corpus), "--panels", str(panels),
                       "--student-config", str(production / "student-config/config.json"),
                       "--initialization", str(init),
                       "--teacher-dir", str(production / "teacher"),
                       "--run-dir", str(run), "--seed", str(seed),
                       "--stop-after", "10000"]
            shared._status(status, "seed1_launching_seed2_queued" if seed == 1 else "seed2_launching",
                           protocol_root=root, gpu_model=MODEL, gpu_uuid=args.gpu_uuid,
                           queued=[row[2] for row in JOBS], active_run=run_id,
                           command=command, completed=completed, queue_pid=os.getpid())
            log = ops / f"{run_id}.launcher.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("ab") as stream:
                stream.write((f"\nCOMMAND {command!r}\n").encode())
                stream.flush()
                trainer = subprocess.Popen(command, cwd=repo,
                    env=dict(os.environ, PYTHONPATH=str(repo / "src")),
                    stdout=stream, stderr=stream)
                shared._status(status, "seed1_running_seed2_queued" if seed == 1 else "seed2_running",
                               protocol_root=root, gpu_model=MODEL, gpu_uuid=args.gpu_uuid,
                               queued=[row[2] for row in JOBS], active_run=run_id,
                               trainer_pid=trainer.pid, queue_pid=os.getpid(), completed=completed)
                return_code = trainer.wait()
            if return_code:
                raise RuntimeError(f"{run_id} trainer exited with status {return_code}; see {log}")
            result = shared._verify_complete(run, condition="C5", seed=seed, root_sha=root)
            completed.append(result)
            completed_seeds.add(seed)
            shared._status(status, "seed1_complete_seed2_queued" if seed == 1 else "seed2_complete",
                           protocol_root=root, gpu_model=MODEL, gpu_uuid=args.gpu_uuid,
                           queued=[row[2] for row in JOBS], completed=completed)
        shared._status(status, "queue_complete", protocol_root=root, gpu_model=MODEL,
                       gpu_uuid=args.gpu_uuid, completed=completed)
    except Exception as exc:
        shared._status(status, "stopped_failed_gate", error=str(exc), launch_armed=False)
        raise


if __name__ == "__main__":
    main()
