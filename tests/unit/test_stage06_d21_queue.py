"""The queued launcher refuses incomplete 10k runs before starting another seed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from sinklab.checkpoint import save_checkpoint
from sinklab.evaluate import RETAINED_FULL
from sinklab.provenance import canonical_json_bytes, seal_payload


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "stage06_d21_queue", ROOT / "scripts/queue_stage06_c3_seed12.py")
assert SPEC and SPEC.loader
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


def _fixture(run: Path) -> int:
    run.mkdir()
    identity = {"study": "S1", "condition": "C3", "seed": 1,
                "run_id": run.name, "protocol_hash": "a" * 64,
                "data_hash": "b" * 64, "gpu_uuid": QUEUE.NODI_UUID,
                "microbatch": 4}
    save_checkpoint(run / "checkpoints", step=10000, kind="final",
                    model=torch.nn.Linear(2, 2), state={"optimizer": {}},
                    identity=identity)
    with (run / "train.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(1, 10001):
            stream.write(json.dumps({
                "event": "update", "step": step, "loss": 2.0, "ce": 3.0,
                "kd": 1.0, "attention": 0.1, "grad_norm": 0.5, "lr": 1e-4,
                "input_tokens": step * 8192, "target_tokens": step * 8128}) + "\n")
    evaluation = run / "evaluation"
    evaluation.mkdir()
    required = {(step, "owt_dense64", role)
                for step in range(0, 10001, 100) for role in ("student", "teacher")}
    required |= {(step, "owt_full300", role)
                 for step in RETAINED_FULL for role in ("student", "teacher")}
    required |= {(step, "owt_lm2000", "student") for step in (0, 10000)}
    for index, (step, panel, role) in enumerate(sorted(required)):
        ids = [str(number) for number in range(
            2000 if panel == "owt_lm2000" else 300 if panel == "owt_full300" else 64)]
        operation = {"status": "complete", "complete_item_ids": ids,
                     "failed_item_ids": [], "missing_item_ids": [], "metrics": {}}
        operations = {"clean": operation} if panel == "owt_lm2000" else {
            name: operation for name in ("clean", "delete", "relocate")}
        record = seal_payload({"key": {"run_id": run.name, "step": step, "panel": panel,
                                       "model_role": role,
                                       "run_identity": {"seed": 1,
                                                        "protocol_sha256": "a" * 64,
                                                        "corpus_sha256": "b" * 64}},
                               "operations": operations})
        (evaluation / f"aggregate-{index:04}.json").write_bytes(
            canonical_json_bytes(record) + b"\n")
    return len(required)


def test_queue_requires_final_state_all_updates_and_complete_evaluations(tmp_path):
    run = tmp_path / "s1-c3-seed1-rtx4080super"
    count = _fixture(run)
    assert QUEUE._verify_complete(run, seed=1, root_sha="a" * 64) == {
        "seed": 1, "updates": 10000, "aggregate_count": count,
        "final_manifest_sha256": QUEUE._hash(
            run / "checkpoints/final-010000/manifest.json")}
    (run / "evaluation/aggregate-0000.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        QUEUE._verify_complete(run, seed=1, root_sha="a" * 64)
    with pytest.raises(ValueError, match="checkpoint"):
        QUEUE._verify_complete(run, seed=1, root_sha="c" * 64)


def test_sequential_launcher_builds_one_explicit_c3_seed_without_rel_scale(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "d21_sequential_launcher", ROOT / "scripts/launch_stage06_c3_seed12_sequential.py")
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    inputs = {name: tmp_path / name for name in
              ("corpus", "panels", "student_config", "initialization", "teacher_dir")}
    command = launcher._train_command("python", 1, tmp_path / "run", inputs)
    assert command[command.index("--seed") + 1] == "1"
    assert command[command.index("--stop-after") + 1] == "10000"
    assert command[command.index("--mse-scale") + 1] == QUEUE.MSE_SCALE
    assert "--rel-scale" not in command
    assert "c3_seed1_rtx4080super.json" in command[command.index("--config") + 1]
    with pytest.raises(ValueError, match="only C3 seeds"):
        launcher._train_command("python", 0, tmp_path / "run", inputs)
