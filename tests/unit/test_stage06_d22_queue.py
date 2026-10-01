"""D22 queue completion checks respect each objective and never repeat a run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sinklab.evaluate import RETAINED_FULL
from sinklab.provenance import canonical_json_bytes, seal_payload


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "stage06_d22_queue", ROOT / "scripts/launch_stage06_c0_c2_seed12_queue.py")
assert SPEC and SPEC.loader
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


def _run_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str) -> Path:
    run = tmp_path / f"s1-{condition.lower()}-seed1-rtx4080super"
    final = run / "checkpoints/final-010000"
    final.mkdir(parents=True)
    (final / "manifest.json").write_text("{}\n", encoding="utf-8")
    identity = {"study": "S1", "condition": condition, "seed": 1,
                "run_id": run.name, "protocol_hash": "a" * 64,
                "data_hash": "b" * 64, "gpu_uuid": QUEUE.NODI_UUID,
                "microbatch": 4}
    monkeypatch.setattr("sinklab.checkpoint.verify_checkpoint", lambda _: {
        "step": 10000, "kind": "final", "identity": identity})
    with (run / "train.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(1, 10001):
            stream.write(json.dumps({
                "event": "update", "step": step, "loss": 2.0, "ce": 3.0,
                "kd": None if condition == "C0" else 1.0,
                "attention": None if condition == "C0" else 0.1,
                "relation": None, "grad_norm": 0.5, "lr": 1e-4,
                "input_tokens": step * 8192,
                "target_tokens": step * 8128}) + "\n")
    evaluation = run / "evaluation"
    evaluation.mkdir()
    expected = {(step, "owt_dense64", role)
                for step in range(0, 10001, 100) for role in ("student", "teacher")}
    expected |= {(step, "owt_full300", role)
                 for step in RETAINED_FULL for role in ("student", "teacher")}
    expected |= {(step, "owt_lm2000", "student") for step in (0, 10000)}
    for index, (step, panel, role) in enumerate(sorted(expected)):
        record = seal_payload({"key": {"run_id": run.name, "step": step,
                                       "panel": panel, "model_role": role,
                                       "run_identity": {"seed": 1,
                                                        "protocol_sha256": "a" * 64,
                                                        "corpus_sha256": "b" * 64}},
                               "operations": {"clean": {"status": "complete",
                                                         "missing_item_ids": [],
                                                         "failed_item_ids": []}}})
        (evaluation / f"aggregate-{index:04}.json").write_bytes(
            canonical_json_bytes(record) + b"\n")
    return run


@pytest.mark.parametrize("condition", ["C0", "C2"])
def test_queue_verifies_objective_specific_update_fields(tmp_path, monkeypatch, condition):
    run = _run_fixture(tmp_path, monkeypatch, condition)
    result = QUEUE._verify_complete(run, condition=condition, seed=1, root_sha="a" * 64)
    assert result["updates"] == 10000
    assert result["aggregate_count"] == 222


@pytest.mark.parametrize("condition,field,value", [
    ("C0", "kd", 1.0), ("C0", "ce", float("nan")),
    ("C2", "attention", None), ("C2", "input_tokens", 0),
])
def test_queue_rejects_bad_update_fields(tmp_path, monkeypatch, condition, field, value):
    run = _run_fixture(tmp_path, monkeypatch, condition)
    path = run / "train.jsonl"
    with path.open("r+", encoding="utf-8") as stream:
        first = json.loads(stream.readline())
        rest = stream.read()
        first[field] = value
        stream.seek(0)
        stream.write(json.dumps(first) + "\n" + rest)
        stream.truncate()
    with pytest.raises(ValueError, match="update record differs at 1"):
        QUEUE._verify_complete(run, condition=condition, seed=1, root_sha="a" * 64)


def test_queue_resumes_only_verified_completed_prefix(tmp_path, monkeypatch):
    jobs = [{"run_id": f"run-{index}", "condition": "C0", "seed": index}
            for index in (1, 2, 3)]
    (tmp_path / "run-1").mkdir()
    (tmp_path / "run-1" / "train.jsonl").touch()
    verified = []

    def check(run, *, condition, seed, root_sha):
        verified.append((run.name, condition, seed, root_sha))
        return {"seed": seed}

    monkeypatch.setattr(QUEUE, "_verify_complete", check)
    assert QUEUE._verified_completed_prefix(jobs, tmp_path, "a" * 64) == [{"seed": 1}]
    assert verified == [("run-1", "C0", 1, "a" * 64)]
    (tmp_path / "run-3").mkdir()
    (tmp_path / "run-3" / "train.jsonl").touch()
    with pytest.raises(RuntimeError, match="occupied after an unstarted job"):
        QUEUE._verified_completed_prefix(jobs, tmp_path, "a" * 64)
