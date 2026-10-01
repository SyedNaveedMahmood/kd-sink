"""Run the four explicitly approved D22 C0/C2 seed replications serially."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil


NODI_MODEL = "NVIDIA GeForce RTX 4080 SUPER"
NODI_UUID = "GPU-72b4b307-b613-c35e-ea32-53f4431de9ee"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _status(path: Path, phase: str, **details) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"phase": phase,
              "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              **details}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cmd(argv: list[str], *, cwd: Path, env: dict | None = None,
         log: Path | None = None) -> str:
    if log is None:
        result = subprocess.run(argv, cwd=cwd, env=env, text=True,
                                capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(f"command failed ({result.returncode}): {argv!r}\n"
                               f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
        return result.stdout + result.stderr
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        stream.write((f"\nCOMMAND {argv!r}\n").encode())
        stream.flush()
        result = subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=stream,
                                check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {argv!r}; see {log}")
    return ""


def _one(root: Path, pattern: str) -> Path:
    rows = list(root.glob(pattern))
    if len(rows) != 1:
        raise ValueError(f"expected one {pattern} under {root}; found {len(rows)}")
    return rows[0]


def _gpu() -> None:
    output = _cmd(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
                  cwd=Path.cwd()).strip().splitlines()
    fields = [value.strip() for value in output[0].split(",")] if len(output) == 1 else []
    if fields != [NODI_MODEL, NODI_UUID]:
        raise RuntimeError("NodiPC RTX4080 SUPER identity differs")


def _active_trainers() -> list[int]:
    active = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = " ".join(proc.cmdline()).lower()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if proc.pid != os.getpid() and "sinklab" in command and " train " in command:
            active.append(proc.pid)
    return active


def _verify_complete(run: Path, *, condition: str, seed: int, root_sha: str) -> dict:
    from sinklab.checkpoint import verify_checkpoint
    from sinklab.evaluate import RETAINED_FULL
    from sinklab.provenance import verify_envelope

    final = verify_checkpoint(run / "checkpoints/final-010000")
    identity = final["identity"]
    if (final["step"] != 10000 or final["kind"] != "final" or
            identity.get("study") != "S1" or identity.get("condition") != condition or
            identity.get("seed") != seed or identity.get("run_id") != run.name or
            identity.get("protocol_hash") != root_sha or
            identity.get("gpu_uuid") != NODI_UUID or identity.get("microbatch") != 4):
        raise ValueError(f"{condition} seed {seed} final checkpoint identity differs")
    found = 0
    with (run / "train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") != "update":
                continue
            found += 1
            if (row["step"] != found or
                    not all(isinstance(row.get(field), (int, float)) and math.isfinite(row[field])
                            for field in ("loss", "ce", "kd", "attention", "grad_norm", "lr")) or
                    row["input_tokens"] != found * 8192 or
                    row["target_tokens"] != found * 8128):
                raise ValueError(f"{condition} seed {seed} update record differs at {found}")
    if found != 10000:
        raise ValueError(f"{condition} seed {seed} has {found} updates")
    expected = {(step, "owt_dense64", role)
                for step in range(0, 10001, 100) for role in ("student", "teacher")}
    expected |= {(step, "owt_full300", role)
                 for step in RETAINED_FULL for role in ("student", "teacher")}
    expected |= {(step, "owt_lm2000", "student") for step in (0, 10000)}
    complete = set()
    for path in (run / "evaluation").glob("aggregate-*.json"):
        payload, _ = verify_envelope(_json(path))
        key, operations = payload["key"], payload["operations"]
        if (key.get("run_id") != run.name or
                key.get("run_identity", {}).get("protocol_sha256") != root_sha or
                key.get("run_identity", {}).get("seed") != seed or
                key.get("run_identity", {}).get("corpus_sha256") != identity.get("data_hash")):
            continue
        row = (key["step"], key["panel"], key["model_role"])
        if row in expected:
            if any(op["status"] != "complete" or op["missing_item_ids"] or op["failed_item_ids"]
                   for op in operations.values()):
                raise ValueError(f"incomplete evaluation aggregate: {row}")
            complete.add(row)
    if complete != expected:
        raise ValueError(f"{condition} seed {seed} missing {len(expected - complete)} aggregates")
    return {"condition": condition, "seed": seed, "updates": found,
            "aggregate_count": len(complete),
            "final_manifest_sha256": _hash(run / "checkpoints/final-010000/manifest.json")}


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
    repo, production, replicas, run_root, ops = (path.resolve() for path in
        (args.repo, args.production_root, args.replication_root, args.run_root, args.ops_root))
    sys.path.insert(0, str(repo / "src"))
    env = dict(os.environ, PYTHONPATH=str(repo / "src"))
    python = str(args.python.resolve())
    status_path = ops / "queue-status.json"
    try:
        head = _cmd(["git", "rev-parse", "HEAD"], cwd=repo).strip()
        origin = _cmd(["git", "rev-parse", "origin/main"], cwd=repo).strip()
        if head != args.expected_head or origin != head or _cmd(["git", "status", "--porcelain"], cwd=repo).strip():
            raise RuntimeError("queue requires the exact clean pushed seal commit")
        from sinklab.stage06_readiness import validate_final_lock_set
        from sinklab.runtime_provenance import validate_runtime_source
        locks = validate_final_lock_set(repo / "protocols")
        root = locks["protocol"]["sha256"]
        payload = locks["protocol"]["payload"]
        validate_runtime_source(repo, payload["production_runtime_source_commit"],
                                payload["execution_critical_path_set_version"],
                                loaded_package_dir=repo / "src/sinklab")
        queue = _json(repo / "configs/production/s1_c0_c2_seed12_queue.json")
        if queue["protocol_root_sha256"] != root or len(queue["jobs"]) != 4:
            raise RuntimeError("D22 queue manifest differs from current root")
        _gpu()
        if _active_trainers():
            raise RuntimeError("another sinklab trainer is active")
        for row in queue["jobs"]:
            run = run_root / row["run_id"]
            if run.exists() and any(run.iterdir()):
                raise RuntimeError(f"fresh run directory is occupied: {run}")
            _cmd([python, "-m", "sinklab", "preflight-production",
                  "--config", f"configs/{row['config']}",
                  "--protocol-lock", "protocols/protocol.lock.json",
                  "--hardware-plan", "protocols/hardware.lock.json",
                  "--artifact-root", str(production), "--replication-root", str(replicas),
                  "--seed", str(row["seed"])], cwd=repo, env=env,
                 log=ops / "preflight.log")
        _status(status_path, "queue_armed", protocol_root=root, jobs=queue["jobs"])
        results = []
        for index, row in enumerate(queue["jobs"], 1):
            condition, seed = row["condition"], row["seed"]
            run = run_root / row["run_id"]
            if shutil.disk_usage(run_root).free < 35 * 1024**3:
                raise RuntimeError(f"insufficient free storage before {row['run_id']}")
            _gpu()
            if _active_trainers():
                raise RuntimeError("another sinklab trainer appeared")
            corpus = _one(replicas / f"corpus/seed{seed}", "owt-corpus-*.json")
            panels = _one(replicas / f"panels/seed{seed}", "owt-panels-*.json")
            init = _one(replicas / f"initialization/seed{seed}", f"init-seed{seed}-*.json")
            command = [python, "-u", "-m", "sinklab", "train",
                       "--config", f"configs/{row['config']}",
                       "--protocol-lock", "protocols/protocol.lock.json",
                       "--hardware-plan", "protocols/hardware.lock.json",
                       "--corpus", str(corpus), "--panels", str(panels),
                       "--student-config", str(production / "student-config/config.json"),
                       "--initialization", str(init),
                       "--teacher-dir", str(production / "teacher"),
                       "--run-dir", str(run), "--seed", str(seed),
                       "--stop-after", "10000"]
            _status(status_path, "job_running", protocol_root=root, job_index=index,
                    job=row, command=command, completed=results,
                    free_bytes=shutil.disk_usage(run_root).free)
            _cmd(command, cwd=repo, env=env, log=ops / f"{row['run_id']}.launcher.log")
            results.append(_verify_complete(run, condition=condition, seed=seed, root_sha=root))
            _status(status_path, "job_complete", protocol_root=root, job_index=index,
                    job=row, completed=results)
        _status(status_path, "queue_complete", protocol_root=root, completed=results)
    except Exception as exc:
        _status(status_path, "stopped_failed_gate", error=str(exc), launch_armed=False)
        raise


if __name__ == "__main__":
    main()
