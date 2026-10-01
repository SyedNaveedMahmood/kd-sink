"""Run exactly C4 seed1 then seed2, serially, after D23 gates pass."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


MODEL = "NVIDIA GeForce RTX 3090"
REL_SCALE = "0.120179255876581"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _status(path: Path, phase: str, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps({"phase": phase, "updated_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temp, path)


def _one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {pattern} under {root}; found {len(matches)}")
    return matches[0]


def _no_active_trainer() -> None:
    import psutil
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid == os.getpid():
            continue
        try:
            command = " ".join(process.info.get("cmdline") or []).lower()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if "sinklab" in command and " train " in f" {command} ":
            raise RuntimeError(f"another sinklab trainer is active (PID {process.pid})")


def _verify_complete(run: Path, *, seed: int, root_sha: str, gpu_uuid: str) -> dict:
    from sinklab.checkpoint import verify_checkpoint
    final = verify_checkpoint(run / "checkpoints/final-010000")
    identity = final["identity"]
    if (final["step"] != 10000 or final["kind"] != "final" or
            identity.get("study") != "S1" or identity.get("condition") != "C4" or
            identity.get("seed") != seed or identity.get("run_id") != run.name or
            identity.get("protocol_hash") != root_sha or identity.get("gpu_uuid") != gpu_uuid or
            identity.get("microbatch") != 4 or
            not (run / "checkpoints/final-010000/COMPLETE").is_file()):
        raise RuntimeError(f"C4 seed{seed} final checkpoint identity differs")
    with (run / "train.jsonl").open(encoding="utf-8") as stream:
        start = json.loads(next(stream))
    if (start.get("event") != "start" or start.get("study") != "S1" or
            start.get("condition") != "C4" or start.get("seed") != seed or
            start.get("run_id") != run.name or start.get("protocol_hash") != root_sha or
            start.get("gpu_uuid") != gpu_uuid or start.get("microbatch") != 4 or
            start.get("accumulation") != 16 or start.get("effective_batch") != 64):
        raise RuntimeError(f"C4 seed{seed} training start identity differs")
    updates = 0
    with (run / "train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "update":
                updates += 1
                if (row.get("step") != updates or
                        not isinstance(row.get("relation"), (int, float)) or
                        not math.isfinite(row["relation"]) or
                        not isinstance(row.get("loss"), (int, float)) or
                        not math.isfinite(row["loss"])):
                    raise RuntimeError(f"C4 seed{seed} update sequence breaks at {updates}")
    if updates != 10000:
        raise RuntimeError(f"C4 seed{seed} ended with {updates} updates")
    return {"run_id": run.name, "seed": seed, "updates": updates,
            "final_identity_sha256": final["identity_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--replication-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ops-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    repo, production, replicas, runs, ops = (path.resolve() for path in
        (args.repo, args.production_root, args.replication_root, args.run_root, args.ops_root))
    sys.path.insert(0, str(repo / "src"))
    python = str(args.python.resolve())
    state = ops / "queue-status.json"
    try:
        from sinklab.provenance import verify_envelope
        from sinklab.runtime_provenance import validate_runtime_source
        from sinklab.stage06_readiness import validate_final_lock_set

        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        origin = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=repo, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
        if head != args.expected_head or origin != head or dirty:
            raise RuntimeError("queue requires the exact clean pushed protocol/config milestone")
        locks = validate_final_lock_set(repo / "protocols")
        protocol = locks["protocol"]["payload"]
        root_sha = locks["protocol"]["sha256"]
        if protocol["protocol"].get("production_config_binding_schema") != 6:
            raise RuntimeError("D23 all-condition protocol root is not active")
        validate_runtime_source(repo, protocol["production_runtime_source_commit"],
                                protocol["execution_critical_path_set_version"],
                                loaded_package_dir=repo / "src/sinklab")
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid",
            "--format=csv,noheader"], text=True).strip().splitlines()
        if len(gpu) != 1 or gpu[0].split(",", 1)[0].strip() != MODEL:
            raise RuntimeError("one exact-model RTX3090 is required")
        gpu_uuid = gpu[0].split(",", 1)[1].strip()
        _no_active_trainer()

        jobs = []
        for seed in (1, 2):
            run_id = f"s1-c4-seed{seed}-rtx3090"
            run = runs / run_id
            if run.exists() and any(run.iterdir()):
                raise RuntimeError(f"run directory is occupied; will not overwrite: {run}")
            config = repo / f"configs/production/s1/c4_seed{seed}_rtx3090.json"
            raw = _json(config)
            binding = raw.get("production_binding", {})
            if (raw.get("condition") != "C4" or raw.get("variant") != "causal_qq_kk_vv_v1" or
                    raw.get("device_role") != "rtx3090" or raw.get("protocol_digest") != root_sha or
                    binding.get("seed") != seed or
                    binding.get("production_runtime_source_commit") != protocol["production_runtime_source_commit"]):
                raise RuntimeError(f"C4 seed{seed} config is not bound to current root/source")
            preflight = subprocess.run([python, "-m", "sinklab", "preflight-production",
                "--config", str(config), "--protocol-lock", str(repo / "protocols/protocol.lock.json"),
                "--hardware-plan", str(repo / "protocols/hardware.lock.json"),
                "--artifact-root", str(production), "--replication-root", str(replicas),
                "--seed", str(seed)], cwd=repo, text=True, capture_output=True, check=False)
            (ops / f"preflight-seed{seed}.stdout.log").write_text(preflight.stdout, encoding="utf-8")
            (ops / f"preflight-seed{seed}.stderr.log").write_text(preflight.stderr, encoding="utf-8")
            if preflight.returncode:
                raise RuntimeError(f"seed{seed} preflight failed ({preflight.returncode})")
            evidence = json.loads(preflight.stdout)
            if (evidence.get("training_started") is not False or
                    evidence.get("protocol_sha256") != root_sha or
                    evidence.get("gpu_name") != MODEL or evidence.get("gpu_uuid") != gpu_uuid or
                    evidence.get("seed_replication_artifacts_verified") is not True):
                raise RuntimeError(f"seed{seed} preflight evidence is incomplete")
            corpus = _one(replicas / f"corpus/seed{seed}", "owt-corpus-*.json")
            panels = _one(replicas / f"panels/seed{seed}", "owt-panels-*.json")
            init = _one(replicas / f"initialization/seed{seed}", f"init-seed{seed}-*.json")
            jobs.append((seed, run, config, corpus, panels, init))

        _status(state, "preflights_passed_training_not_started", protocol_root=root_sha,
                gpu_name=MODEL, gpu_uuid=gpu_uuid, jobs=[str(job[1]) for job in jobs])
        completed = []
        for seed, run, config, corpus, panels, init in jobs:
            if shutil.disk_usage(runs).free < 35 * 1024**3:
                raise RuntimeError("less than 35 GiB free in run volume")
            _no_active_trainer()
            command = [python, "-u", "-m", "sinklab", "train",
                "--config", str(config), "--protocol-lock", str(repo / "protocols/protocol.lock.json"),
                "--hardware-plan", str(repo / "protocols/hardware.lock.json"),
                "--corpus", str(corpus), "--panels", str(panels),
                "--student-config", str(production / "student-config/config.json"),
                "--initialization", str(init), "--teacher-dir", str(production / "teacher"),
                "--run-dir", str(run), "--seed", str(seed), "--stop-after", "10000",
                "--rel-scale", REL_SCALE]
            stdout = ops / f"{run.name}.stdout.log"
            stderr = ops / f"{run.name}.stderr.log"
            with stdout.open("ab") as out, stderr.open("ab") as err:
                process = subprocess.Popen(command, cwd=repo, stdout=out, stderr=err)
                _status(state, "job_running", protocol_root=root_sha, gpu_uuid=gpu_uuid,
                        job_index=seed, seed=seed, pid=process.pid, command=command,
                        run_dir=str(run), stdout=str(stdout), stderr=str(stderr),
                        completed=completed)
                return_code = process.wait()
            if return_code:
                raise RuntimeError(f"C4 seed{seed} trainer exited with status {return_code}")
            completed.append(_verify_complete(run, seed=seed, root_sha=root_sha, gpu_uuid=gpu_uuid))
            _status(state, "job_complete", protocol_root=root_sha, gpu_uuid=gpu_uuid,
                    job_index=seed, seed=seed, completed=completed)
        _status(state, "queue_complete", protocol_root=root_sha, gpu_uuid=gpu_uuid,
                completed=completed)
    except Exception as exc:
        _status(state, "stopped_failed_gate", error=str(exc), launch_armed=False)
        raise


if __name__ == "__main__":
    main()
