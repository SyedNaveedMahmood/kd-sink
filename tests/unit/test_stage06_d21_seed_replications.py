"""D21 must preserve D20 while binding each optional C3 data lineage."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sinklab.config import ConfigError, production_binding_for, resolve_config
from sinklab.provenance import canonical_json_bytes, seal_payload
from sinklab.stage06_readiness import (
    D20_PROTOCOL_ROOT, ReadinessError, build_d21_lock_set,
)
from sinklab import stage06_readiness
from sinklab.training_entry import _check_seed_replication_artifacts, _require_s1_data_lock


REPO = Path(__file__).resolve().parents[2]
SOURCE_SHA = "d36784aaf0521f6e96d40d603e0361744397b23df90dc505e29b0b9cf18360eb"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(document) + b"\n")


def _fixture(tmp_path: Path, monkeypatch) -> tuple[dict, dict, dict]:
    old = {name: _read(REPO / f"protocols/{name}.lock.json") for name in
           ("artifact", "environment", "hardware", "calibration", "protocol")}
    if old["protocol"]["sha256"] != D20_PROTOCOL_ROOT:
        old["protocol"] = _read(REPO / "protocols/superseded" /
                                f"s1-protocol-{D20_PROTOCOL_ROOT}.json")
    assert old["protocol"]["sha256"] == D20_PROTOCOL_ROOT
    data = old["protocol"]["payload"]["protocol"]["data"]
    rows = {
        str(seed): {"corpus_sha256": str(seed) * 64,
                    "corpus_file_sha256": "a" * 64,
                    "panels_sha256": str(seed + 2) * 64,
                    "panels_file_sha256": "b" * 64,
                    "order_sha256": str(seed + 4) * 64,
                    "order_file_sha256": "c" * 64,
                    "initialization_sha256": str(seed + 6) * 64,
                    "initialization_metadata_file_sha256": "d" * 64,
                    "initialization_weights_file_sha256": "e" * 64}
        for seed in (1, 2)}
    evidence = seal_payload({
        "kind": "s1-c3-seed12-repacked-artifacts-v1",
        "predecessor_root_sha256": D20_PROTOCOL_ROOT,
        "source_sha256": SOURCE_SHA,
        "dataset_revision": data["dataset_revision"],
        "tokenizer_revision": data["tokenizer_revision"],
        "tokenizer_sha256": data["tokenizer_files_sha256"],
        "seeds": rows})
    decision = seal_payload({
        "kind": "s1-d21-c3-only-seed12-replication-v1",
        "decision_id": "D21",
        "status": "approved_prospective_optional_replication",
        "approved_on_utc_date": "2026-09-29",
        "predecessor_production_root": D20_PROTOCOL_ROOT,
        "replication_evidence_sha256": evidence["sha256"],
        "condition": "C3", "seeds": [1, 2], "device_role": "rtx4080super",
        "document_packing": "repack_training_evaluation_calibration_per_seed",
        "evaluation_panels": "separate_per_seed_not_directly_paired",
        "common_schedule": {"microbatch": 4, "accumulation": 16,
                            "effective_sequences": 64, "optimizer_updates": 10000},
        "s_MSE": 68.00580071126464, "prospective_only": True,
        "preserve_d20_seed0_run": True, "calibration_changed": False})
    _write(tmp_path / stage06_readiness.D21_EVIDENCE_PATH, evidence)
    _write(tmp_path / stage06_readiness.D21_PATH, decision)
    monkeypatch.setattr(stage06_readiness, "_d20", lambda _: None)
    return old, evidence, decision


def test_d21_binds_two_distinct_c3_seeds_and_preserves_d20(tmp_path, monkeypatch):
    old, evidence, decision = _fixture(tmp_path, monkeypatch)
    new = build_d21_lock_set(tmp_path, old, "f" * 40)
    body = new["protocol"]["payload"]["protocol"]
    assert [new[name]["sha256"] for name in ("artifact", "environment", "hardware", "calibration")] == [
        old[name]["sha256"] for name in ("artifact", "environment", "hardware", "calibration")]
    assert body["predecessor_protocol_root_sha256"] == D20_PROTOCOL_ROOT
    assert body["d21_researcher_amendment_sha256"] == decision["sha256"]
    assert body["seed_replications_sha256"] == evidence["sha256"]
    assert body["run_plan"] == old["protocol"]["payload"]["protocol"]["run_plan"]
    assert body["objectives"] == old["protocol"]["payload"]["protocol"]["objectives"]
    binding = production_binding_for(new["protocol"]["payload"], "rtx4080super", 1)
    assert binding["initialization_sha256"] == evidence["payload"]["seeds"]["1"]["initialization_sha256"]
    assert "seed0_initialization_sha256" not in binding
    raw = _read(REPO / "configs/production/s1/c3_rtx4080super.json")
    raw["protocol_digest"] = new["protocol"]["sha256"]
    raw["production_binding"] = binding
    assert resolve_config(raw, seed=1, protocol_lock=new["protocol"], production=True).seed == 1
    wrong = copy.deepcopy(raw)
    wrong["production_binding"]["corpus_sha256"] = "0" * 64
    with pytest.raises(ConfigError, match="binding differs"):
        resolve_config(wrong, seed=1, protocol_lock=new["protocol"], production=True)
    wrong = copy.deepcopy(raw)
    wrong["condition"], wrong["variant"] = "C2", "cosine_soft_jsd_v2"
    with pytest.raises(ConfigError, match="only C3"):
        resolve_config(wrong, seed=1, protocol_lock=new["protocol"], production=True)


def test_d21_rejects_missing_or_reused_replication_evidence(tmp_path, monkeypatch):
    old, evidence, _ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(OSError):
        build_d21_lock_set(tmp_path / "absent", old, "f" * 40)
    bad = copy.deepcopy(evidence["payload"])
    bad["seeds"]["2"]["corpus_sha256"] = bad["seeds"]["1"]["corpus_sha256"]
    _write(tmp_path / stage06_readiness.D21_EVIDENCE_PATH, seal_payload(bad))
    with pytest.raises(ReadinessError, match="seed data evidence"):
        build_d21_lock_set(tmp_path, old, "f" * 40)


def test_d21_data_guard_requires_matching_seed_and_hashes():
    protocol = {"data": {"recipe": "owt-upstream-gpt2-pack-v1",
                         "dataset_revision": "a" * 40,
                         "tokenizer_revision": "b" * 40,
                         "tokenizer_files_sha256": "c" * 64},
                "seed_replications": {"1": {"corpus_sha256": "d" * 64,
                                            "panels_sha256": "e" * 64}}}
    corpus = {"seed": 1, "dataset": {"revision": "a" * 40},
              "tokenizer": {"revision": "b" * 40, "files_sha256": "c" * 64}}
    _require_s1_data_lock(protocol, corpus, "d" * 64, "e" * 64, seed=1)
    with pytest.raises(ValueError, match="seed, corpus"):
        _require_s1_data_lock(protocol, {**corpus, "seed": 0}, "d" * 64, "e" * 64, seed=1)
    with pytest.raises(ValueError, match="seed, corpus"):
        _require_s1_data_lock(protocol, corpus, "f" * 64, "e" * 64, seed=1)


def test_external_replication_files_must_match_committed_bytes(tmp_path):
    repo, root = tmp_path / "repo", tmp_path / "data"
    seed = 1
    order_document = seal_payload({"seed": seed})
    row = {"corpus_sha256": "1" * 64, "panels_sha256": "2" * 64,
           "order_sha256": order_document["sha256"], "initialization_sha256": "4" * 64}
    documents = {
        "corpus": (f"corpus/seed{seed}/owt-corpus-{row['corpus_sha256']}.json", b"corpus"),
        "panels": (f"panels/seed{seed}/owt-panels-{row['panels_sha256']}.json", b"panels"),
        "order": (f"order/seed{seed}/update-order-{row['order_sha256']}.json",
                  canonical_json_bytes(order_document) + b"\n"),
        "initialization": (f"initialization/seed{seed}/init-seed{seed}-{row['initialization_sha256']}.json",
                           canonical_json_bytes(seal_payload({
                               "seed": seed, "tensor_content_sha256": row["initialization_sha256"]})) + b"\n"),
        "weights": (f"initialization/seed{seed}/init-seed{seed}-{row['initialization_sha256']}.safetensors",
                    b"weights"),
    }
    for name, (relative, contents) in documents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        field = {
            "corpus": "corpus_file_sha256", "panels": "panels_file_sha256",
            "order": "order_file_sha256",
            "initialization": "initialization_metadata_file_sha256",
            "weights": "initialization_weights_file_sha256"}[name]
        row[field] = hashlib.sha256(contents).hexdigest()
    evidence = seal_payload({"seeds": {"1": row}})
    _write(repo / stage06_readiness.D21_EVIDENCE_PATH, evidence)
    lock = {"protocol": {"seed_replications_sha256": evidence["sha256"]}}
    args = SimpleNamespace(**{name: root / documents[name][0]
                              for name in ("corpus", "panels", "initialization")})
    _check_seed_replication_artifacts(repo, root, lock,
                                      SimpleNamespace(seed=seed), args)
    (root / documents["corpus"][0]).write_bytes(b"altered")
    with pytest.raises(ValueError, match="differs from D21"):
        _check_seed_replication_artifacts(repo, root, lock,
                                          SimpleNamespace(seed=seed), args)
