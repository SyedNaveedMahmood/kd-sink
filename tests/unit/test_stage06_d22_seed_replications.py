"""D22 extends the sealed seed artifacts only to paired C0/C2 replications."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sinklab.config import ConfigError, production_binding_for, resolve_config
from sinklab.provenance import validate_protocol_lock
from sinklab.stage06_readiness import (
    D21_PROTOCOL_ROOT, LOCK_NAMES, build_d22_lock_set,
)


REPO = Path(__file__).resolve().parents[2]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _d21_predecessor() -> dict[str, dict]:
    historical = REPO / "protocols/superseded" / f"s1-protocol-{D21_PROTOCOL_ROOT}.json"
    protocol = _read(historical if historical.exists() else REPO / "protocols/protocol.lock.json")
    assert protocol["sha256"] == D21_PROTOCOL_ROOT
    return {**{name: _read(REPO / f"protocols/{name}.lock.json")
               for name in LOCK_NAMES[:-1]}, "protocol": protocol}


def _config(condition: str, lock: dict, seed: int) -> dict:
    row = _read(REPO / f"configs/production/s1/{condition.lower()}_rtx4080super.json")
    row["protocol_digest"] = lock["sha256"]
    row["production_binding"] = production_binding_for(lock["payload"], "rtx4080super", seed)
    return row


def test_d22_preserves_science_and_authorizes_only_c0_c2_seed12() -> None:
    predecessor = _d21_predecessor()
    successor = build_d22_lock_set(REPO, predecessor, "1" * 40)
    payload, digest = validate_protocol_lock(successor["protocol"])
    body, old = payload["protocol"], predecessor["protocol"]["payload"]
    assert digest == successor["protocol"]["sha256"]
    assert body["predecessor_protocol_root_sha256"] == D21_PROTOCOL_ROOT
    assert body["optional_replication_conditions"] == ["C0", "C2"]
    assert body["production_config_binding_schema"] == 5
    assert body["seed_replications"] == old["protocol"]["seed_replications"]
    assert [successor[name]["sha256"] for name in LOCK_NAMES[:-1]] == [
        predecessor[name]["sha256"] for name in LOCK_NAMES[:-1]]
    for seed in (1, 2):
        for condition in ("C0", "C2"):
            spec = resolve_config(_config(condition, successor["protocol"], seed), seed=seed,
                                  protocol_lock=successor["protocol"], production=True)
            assert (spec.condition, spec.seed, spec.device_role) == (
                condition, seed, "rtx4080super")
        with pytest.raises(ConfigError, match="only C0/C2"):
            resolve_config(_config("C3", successor["protocol"], seed), seed=seed,
                           protocol_lock=successor["protocol"], production=True)


def test_d22_rejects_stale_or_tampered_decision_binding() -> None:
    predecessor = _d21_predecessor()
    bad = copy.deepcopy(predecessor)
    bad["protocol"]["sha256"] = "0" * 64
    with pytest.raises(Exception):
        build_d22_lock_set(REPO, bad, "1" * 40)
