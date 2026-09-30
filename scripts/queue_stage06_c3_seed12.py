"""One-shot, fail-closed C3 seed-1/2 queue after verified D20 seed-0 completion.

This is an operator launcher, not a training implementation. Each invocation of
sinklab train still names exactly one C3 seed. A failed gate cancels the queue.
"""

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


D20_ROOT = "fccf4c14bc691e550c6304f4955b72037efb8d042b3e51afb70367920fff0552"
NODI_UUID = "GPU-72b4b307-b613-c35e-ea32-53f4431de9ee"
NODI_MODEL = "NVIDIA GeForce RTX 4080 SUPER"
MSE_SCALE = "68.00580071126464"
SOURCE_SHA = "d36784aaf0521f6e96d40d603e0361744397b23df90dc505e29b0b9cf18360eb"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DATASET_REVISION = "79d93d786212f7344586290adb811d4ae6a1762c"
TOKENIZER_SHA = "68bfbc36e7d352017e23168f34c98b3b92a18bdcb658c163dea499d8d690c580"


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
    record = {"phase": phase, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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


def _worker_running(pid: int, started: float) -> bool:
    try:
        proc = psutil.Process(pid)
        return abs(proc.create_time() - started) < 1 and proc.is_running()
    except psutil.NoSuchProcess:
        return False


def _verify_complete(run: Path, *, seed: int, root_sha: str,
                     prefix_log: Path | None = None, prefix_step: int = 0) -> dict:
    from sinklab.checkpoint import verify_checkpoint
    from sinklab.evaluate import RETAINED_FULL
    from sinklab.provenance import verify_envelope

    final = verify_checkpoint(run / "checkpoints/final-010000")
    identity = final["identity"]
    if (final["step"] != 10000 or final["kind"] != "final" or
            identity.get("study") != "S1" or identity.get("condition") != "C3" or
            identity.get("seed") != seed or identity.get("run_id") != run.name or
            identity.get("protocol_hash") != root_sha or
            identity.get("gpu_uuid") != NODI_UUID or
            identity.get("microbatch") != 4):
        raise ValueError(f"seed {seed} final checkpoint identity differs")
    seen = 0
    def audit(path: Path, first_step: int, last_step: int) -> int:
        nonlocal seen
        found = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("event") != "update":
                    continue
                step = row["step"]
                if step < first_step or step > last_step:
                    continue
                if step != first_step + found:
                    raise ValueError(f"seed {seed} training update sequence broken at {step}")
                if (not all(isinstance(row.get(field), (int, float)) and
                            math.isfinite(row[field]) for field in
                            ("loss", "ce", "kd", "attention", "grad_norm", "lr")) or
                        row["input_tokens"] != step * 8192 or
                        row["target_tokens"] != step * 8128):
                    raise ValueError(f"seed {seed} nonfinite loss or wrong counters at {step}")
                found += 1
        if found != last_step - first_step + 1:
            raise ValueError(f"seed {seed} missing training updates")
        seen += found
        return found
    if prefix_log is not None:
        audit(prefix_log, 1, prefix_step)
        audit(run / "train.jsonl", prefix_step + 1, 10000)
    else:
        audit(run / "train.jsonl", 1, 10000)
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
        if row not in expected:
            continue
        if (set(operations) != ({"clean"} if key["panel"] == "owt_lm2000"
                               else {"clean", "delete", "relocate"}) or
                any(op["status"] != "complete" or op["missing_item_ids"] or
                op["failed_item_ids"] or len(set(op["complete_item_ids"])) !=
                (2000 if key["panel"] == "owt_lm2000"
                 else 300 if key["panel"] == "owt_full300" else 64)
                for op in operations.values())):
            raise ValueError(f"seed {seed} incomplete evaluation aggregate: {row}")
        complete.add(row)
    if complete != expected:
        raise ValueError(f"seed {seed} missing {len(expected - complete)} evaluation aggregates")
    return {"seed": seed, "updates": seen, "aggregate_count": len(complete),
            "final_manifest_sha256": _hash(run / "checkpoints/final-010000/manifest.json")}


def _gpu() -> tuple[str, str]:
    output = _cmd(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
                  cwd=Path.cwd()).strip().splitlines()
    if len(output) != 1:
        raise RuntimeError("one GPU is required")
    fields = [value.strip() for value in output[0].split(",")]
    if len(fields) != 2 or tuple(fields) != (NODI_MODEL, NODI_UUID):
        raise RuntimeError("NodiPC RTX4080 SUPER identity differs")
    return fields[0], fields[1]


def _one_json_path(root: Path, pattern: str) -> Path:
    files = list(root.glob(pattern))
    if len(files) != 1:
        raise ValueError(f"expected one {pattern} under {root}; found {len(files)}")
    return files[0]


def _record_seal(repo: Path, source: str, root: str, seed0: dict) -> None:
    from sinklab.provenance import canonical_json_bytes, seal_payload

    evidence = _json(repo / "reports/stage06_c3_seed12_artifacts.json")
    decision = _json(repo / "protocols/s1_researcher_amendment_d21_c3_seed12_20260929.json")
    queue = _json(repo / "configs/production/s1_c3_seed12_queue.json")
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary = seal_payload({
        "kind": "stage06-d21-c3-seed12-prospective-queue-v1",
        "sealed_at_utc": timestamp,
        "predecessor_d20_root_sha256": D20_ROOT,
        "successor_root_sha256": root,
        "runtime_source_commit": source,
        "replication_evidence_sha256": evidence["sha256"],
        "researcher_amendment_sha256": decision["sha256"],
        "seed0_final": seed0,
        "optional_jobs": queue["jobs"],
        "status": "sealed_pending_sequential_launch",
        "scientific_coverage": "C3_seed0_verified_only_by_this_queue",
        "evaluation_panel_comparison": "seed_specific_panels_not_directly_paired"})
    path = repo / "reports/stage06_d21_queue.json"
    path.write_bytes(canonical_json_bytes(summary) + b"\n")
    report_path = repo / "reports/stage06.json"
    report = _json(report_path)
    report["status"] = "production_ready_under_d21_c3_seed12_successor"
    if "d21_c3_seed12_preparation" in report:
        report["d21_c3_seed12_preparation"].update(
            status="seed0_complete_artifacts_verified",
            production_launch_armed=True)
    report["d21_optional_c3_seed_replications"] = {
        "status": "sealed_pending_sequential_launch",
        "predecessor_d20_root_sha256": D20_ROOT,
        "successor_root_sha256": root,
        "runtime_source_commit": source,
        "amendment_sha256": decision["sha256"],
        "evidence_sha256": evidence["sha256"],
        "seed0_final": seed0,
        "seeds": [1, 2],
        "condition": "C3",
        "separate_seed_packing_and_panels": True,
        "automatic_condition_sweep": False,
        "new_training_started_at_seal": False}
    report["scientific_coverage"] = (
        "C3 seed0 final-010000 and evaluation coverage verified by D21 queue; "
        "other S1 run coverage is not asserted here")
    report_path.write_bytes((json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    note = (
        f"\n## {timestamp} - Codex (GPT-6) - Prospective D21 C3-only seed queue sealed\n\n"
        f"- D20 C3 seed0 completed: verified 10,000 sequential finite updates, "
        f"{seed0['aggregate_count']} complete evaluation aggregates and checksum-verified "
        f"final checkpoint manifest SHA {seed0['final_manifest_sha256']}. Its D20 root and "
        "run identity are unchanged. Host reboot cause remains unresolved.\n"
        f"- New D21 runtime source {source}; predecessor D20 root {D20_ROOT}; "
        f"successor root {root}; amendment SHA {decision['sha256']}; "
        f"replication evidence SHA {evidence['sha256']}. The original artifact, "
        "environment, hardware and calibration locks, C3 objective/scale, 4x16 schedule "
        "and nine seed0 jobs remain unchanged. Seed1/2 have separately packed OWT "
        "corpora, panels, orders and CPU-FP32 initializations, verified against their "
        "sealed files. Separate panels prevent direct per-item cross-seed pairing.\n"
        "- Full CPU unit/integration and standalone clean-wheel tests, pip check, "
        "active uv locked dry-run, diff check, source guard, real RTX4080 preflights "
        "for seeds1/2 passed before seal commit. Exact commands and output are in "
        "the external D21 queue command log. No seed1/2 run had started at this entry. "
        "Next: push the seal, then launch seed1; launch seed2 only after seed1 final "
        "checkpoint and evaluation coverage verify.\n")
    for name in ("IMPLEMENTATION_NOTES_BY_CLAUDE_CORE.md",
                 "IMPLEMENTATION_NOTES_BY_CLAUDE_S1.md"):
        with (repo / "implementation_notes" / name).open("a", encoding="utf-8") as stream:
            stream.write(note)
    next_path = repo / "NEXT_STEPS.md"
    with next_path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n## D21 optional C3-only replications ({timestamp})\n"
            f"D20 C3 seed0 final coverage verified. D21 successor root {root} seals "
            "separately packed seed1/2 C3 artifacts, unchanged C3 science and two "
            "explicit single-seed jobs. The external one-shot queue launches seed1 "
            "only after seal push; seed2 requires verified seed1 completion. "
            "No other S1 condition is queued. Scientific completion remains per run.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--replication-root", type=Path, required=True)
    parser.add_argument("--seed0-run-dir", type=Path, required=True)
    parser.add_argument("--seed0-prefix-log", type=Path, required=True)
    parser.add_argument("--seed0-worker-pid", type=int, required=True)
    parser.add_argument("--seed0-worker-created", type=float, required=True)
    parser.add_argument("--seed1-run-dir", type=Path, required=True)
    parser.add_argument("--seed2-run-dir", type=Path, required=True)
    parser.add_argument("--source-milestone", required=True)
    parser.add_argument("--expected-origin-main", required=True)
    parser.add_argument("--ops-root", type=Path, required=True)
    parser.add_argument("--check-now", action="store_true")
    args = parser.parse_args()
    repo, production, replicas, ops = (
        path.resolve() for path in
        (args.repo, args.production_root, args.replication_root, args.ops_root))
    sys.path.insert(0, str(repo / "src"))
    state_path, command_log = ops / "queue-status.json", ops / "commands.log"
    status = {"seed0_worker_running": _worker_running(
        args.seed0_worker_pid, args.seed0_worker_created),
        "seed0_final_exists": (args.seed0_run_dir / "checkpoints/final-010000").is_dir(),
        "source_milestone": args.source_milestone,
        "expected_origin_main": args.expected_origin_main,
        "launch_armed": not args.check_now}
    if args.check_now:
        print(json.dumps(status, sort_keys=True))
        return
    _status(state_path, "waiting_for_seed0", **status)
    try:
        env = dict(os.environ, PYTHONPATH=str(repo / "src"))
        python = str(args.python.resolve())
        source = production / "source/openwebtext-train-first416000.jsonl"
        tokenizer = production / "tokenizer"
        from sinklab.runtime_provenance import validate_runtime_source
        if _hash(source) != SOURCE_SHA or _cmd(["git", "status", "--porcelain"], cwd=repo).strip():
            raise RuntimeError("queue source tree or pinned OWT source differs")
        validate_runtime_source(repo, args.source_milestone, 1,
                                loaded_package_dir=repo / "src/sinklab")
        for path in (args.seed1_run_dir, args.seed2_run_dir):
            if path.exists() and any(path.iterdir()):
                raise RuntimeError(f"optional run directory is occupied: {path}")
        while _worker_running(args.seed0_worker_pid, args.seed0_worker_created):
            time.sleep(30)
        _status(state_path, "verifying_seed0")
        seed0 = _verify_complete(args.seed0_run_dir, seed=0, root_sha=D20_ROOT,
                                 prefix_log=args.seed0_prefix_log, prefix_step=2500)
        _gpu()
        _status(state_path, "preparing_seed_data", seed0=seed0)
        for seed in (1, 2):
            if psutil.virtual_memory().available < 32 * 1024**3:
                raise RuntimeError("insufficient free host RAM for OWT repacking")
            corpus_dir = replicas / f"corpus/seed{seed}"
            panels_dir = replicas / f"panels/seed{seed}"
            order_dir = replicas / f"order/seed{seed}"
            if not list(corpus_dir.glob("owt-corpus-*.json")):
                _cmd([python, "-u", "-m", "sinklab", "prepare-owt-compat",
                      "--input-jsonl", str(source), "--tokenizer-dir", str(tokenizer),
                      "--dataset-revision", DATASET_REVISION,
                      "--tokenizer-revision", TOKENIZER_REVISION,
                      "--seed", str(seed), "--out-dir", str(corpus_dir)],
                     cwd=repo, env=env, log=command_log)
            corpus = _one_json_path(corpus_dir, "owt-corpus-*.json")
            if not list(panels_dir.glob("owt-panels-*.json")):
                _cmd([python, "-u", "-m", "sinklab", "prepare-owt-compat-panels",
                      "--corpus", str(corpus), "--tokenizer-sha256", TOKENIZER_SHA,
                      "--out-dir", str(panels_dir)],
                     cwd=repo, env=env, log=command_log)
            if not list(order_dir.glob("update-order-*.json")):
                _cmd([python, "-u", "-m", "sinklab", "prepare-owt-compat-order",
                      "--corpus", str(corpus), "--tokenizer-sha256", TOKENIZER_SHA,
                      "--seed", str(seed), "--updates", "0", "--out-dir", str(order_dir)],
                     cwd=repo, env=env, log=command_log)
        _cmd([python, "scripts/seal_stage06_c3_seed12_evidence.py",
              "--source-root", str(production), "--replication-root", str(replicas),
              "--approved-on-utc-date", "2026-09-29"], cwd=repo, env=env, log=command_log)
        _status(state_path, "sealing_d21", seed0=seed0)
        _cmd([python, "scripts/seal_stage06_c3_seed12_successor.py",
              "--production-runtime-source-commit", args.source_milestone,
              "--seed1-run-dir", str(args.seed1_run_dir),
              "--seed2-run-dir", str(args.seed2_run_dir)],
             cwd=repo, env=env, log=command_log)
        root = _json(repo / "protocols/protocol.lock.json")["sha256"]
        _cmd([python, "-m", "pytest", "-q", "tests/unit", "tests/integration",
              "-m", "not gpu and not network"], cwd=repo, env=env, log=command_log)
        _cmd([python, "-m", "pytest", "-q", "tests/integration/test_clean_wheel.py"],
             cwd=repo, env=env, log=command_log)
        _cmd([python, "-m", "pip", "check"], cwd=repo, env=env, log=command_log)
        environment_project = args.python.resolve().parents[2]
        if _hash(repo / "uv.lock") != _hash(environment_project / "uv.lock"):
            raise RuntimeError("active environment uv.lock differs from D21")
        dry_run = _cmd([python, "-m", "uv", "sync", "--locked", "--dry-run"],
                       cwd=environment_project)
        if "Would make no changes" not in dry_run:
            raise RuntimeError("active production environment is not exact-sync")
        _cmd(["git", "diff", "--check"], cwd=repo, log=command_log)
        if _cmd(["git", "diff", "--name-only", args.expected_origin_main,
                 "--", "Upstream", "upstream"], cwd=repo).strip():
            raise RuntimeError("Upstream tree changed")
        for seed in (1, 2):
            _cmd([python, "-m", "sinklab", "preflight-production",
                  "--config", f"configs/production/s1/c3_seed{seed}_rtx4080super.json",
                  "--protocol-lock", "protocols/protocol.lock.json",
                  "--hardware-plan", "protocols/hardware.lock.json",
                  "--artifact-root", str(production),
                  "--replication-root", str(replicas), "--seed", str(seed)],
                 cwd=repo, env=env, log=command_log)
        _record_seal(repo, args.source_milestone, root, seed0)
        _cmd(["git", "diff", "--check"], cwd=repo, log=command_log)
        _cmd(["git", "fetch", "origin", "main"], cwd=repo, log=command_log)
        if _cmd(["git", "rev-parse", "origin/main"], cwd=repo).strip() != args.expected_origin_main:
            raise RuntimeError("origin/main advanced; queue must be reviewed before push")
        if (not _cmd(["git", "config", "user.name"], cwd=repo).strip() or
                not _cmd(["git", "config", "user.email"], cwd=repo).strip()):
            raise RuntimeError("Git commit identity is missing")
        _cmd(["git", "add", "protocols/protocol.lock.json",
              "protocols/s1_researcher_amendment_d21_c3_seed12_20260929.json",
              "protocols/superseded", "reports/stage06_c3_seed12_artifacts.json",
              "reports/stage06_d21_queue.json", "reports/stage06.json",
              "configs/production", "configs/superseded/d20",
              "NEXT_STEPS.md", "implementation_notes/IMPLEMENTATION_NOTES_BY_CLAUDE_CORE.md",
              "implementation_notes/IMPLEMENTATION_NOTES_BY_CLAUDE_S1.md"],
             cwd=repo, log=command_log)
        _cmd(["git", "commit", "-m", "feat(stage06): seal C3-only seed replications"],
             cwd=repo, log=command_log)
        _cmd(["git", "push", "origin", "HEAD:main"], cwd=repo, log=command_log)
        _cmd(["git", "fetch", "origin", "main"], cwd=repo, log=command_log)
        if (_cmd(["git", "rev-parse", "HEAD"], cwd=repo).strip() !=
                _cmd(["git", "rev-parse", "origin/main"], cwd=repo).strip() or
                _cmd(["git", "status", "--porcelain"], cwd=repo).strip()):
            raise RuntimeError("pushed D21 seal is not clean and synchronized")
        _status(state_path, "d21_sealed_and_pushed", root=root, seed0=seed0)
        for seed, run in ((1, args.seed1_run_dir), (2, args.seed2_run_dir)):
            if seed == 2:
                _verify_complete(args.seed1_run_dir, seed=1, root_sha=root)
            if run.exists() and any(run.iterdir()):
                raise RuntimeError(f"optional run directory occupied: {run}")
            run.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(run.parent).free < 35 * 1024**3:
                raise RuntimeError("insufficient free storage for fresh C3 run")
            _gpu()
            active = []
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    cmdline = proc.cmdline()
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
                if (proc.pid != os.getpid() and "sinklab" in " ".join(cmdline).lower()
                        and "train" in cmdline):
                    active.append(proc.pid)
            if active:
                raise RuntimeError("another sinklab trainer is active")
            corpus = _one_json_path(replicas / f"corpus/seed{seed}", "owt-corpus-*.json")
            panels = _one_json_path(replicas / f"panels/seed{seed}", "owt-panels-*.json")
            init = _one_json_path(replicas / f"initialization/seed{seed}",
                                  f"init-seed{seed}-*.json")
            command = [python, "-u", "-m", "sinklab", "train",
                       "--config", f"configs/production/s1/c3_seed{seed}_rtx4080super.json",
                       "--protocol-lock", "protocols/protocol.lock.json",
                       "--hardware-plan", "protocols/hardware.lock.json",
                       "--corpus", str(corpus), "--panels", str(panels),
                       "--student-config", str(production / "student-config/config.json"),
                       "--initialization", str(init),
                       "--teacher-dir", str(production / "teacher"),
                       "--run-dir", str(run), "--seed", str(seed),
                       "--stop-after", "10000", "--mse-scale", MSE_SCALE]
            _status(state_path, f"seed{seed}_running", root=root, command=command,
                    run_dir=str(run))
            _cmd(command, cwd=repo, env=env, log=ops / f"seed{seed}.launcher.log")
            result = _verify_complete(run, seed=seed, root_sha=root)
            _status(state_path, f"seed{seed}_complete", root=root, result=result)
        _status(state_path, "queue_complete", root=root,
                seed1_run_dir=str(args.seed1_run_dir),
                seed2_run_dir=str(args.seed2_run_dir))
    except Exception as exc:
        _status(state_path, "stopped_failed_gate", error=str(exc),
                launch_armed=False)
        raise


if __name__ == "__main__":
    main()
