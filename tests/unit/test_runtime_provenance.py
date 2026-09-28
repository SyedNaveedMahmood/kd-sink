"""The production runner must match its versioned, immutable source tree."""

import json
import subprocess
from pathlib import Path

import pytest

from sinklab.config import ConfigError, production_binding_for, resolve_config
from sinklab.provenance import seal_payload
from sinklab.runtime_provenance import RuntimeSourceError, validate_runtime_source
from sinklab.stage06_readiness import (CALIBRATION_SOURCE_COMMIT, ReadinessError,
                                       SUPERSEDED_PROTOCOL_ROOT, validate_final_lock_set)


ROOT = Path(__file__).resolve().parents[2]


def _git(repo, *args):
    result = subprocess.run(("git", "-C", str(repo), *args), capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / "source"
    package = repo / "src/sinklab"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "trainer.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (repo / ".gitignore").write_text("__pycache__/\nignored_*.py\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.test")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "source milestone")
    return repo, _git(repo, "rev-parse", "HEAD")


def _check(source_repo, source=None):
    repo, milestone = source_repo
    return validate_runtime_source(repo, source or milestone, 1,
                                   loaded_package_dir=repo / "src/sinklab")


def test_exact_source_milestone_passes(source_repo):
    assert _check(source_repo)["current_head"] == source_repo[1]


def test_documentation_only_descendant_passes(source_repo):
    repo, milestone = source_repo
    for name in ("protocols/protocol.lock.json", "configs/production/job.json",
                 "reports/stage06.json", "docs/README.md"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "seal generated records")
    assert _check(source_repo)["production_runtime_source_commit"] == milestone


def test_modified_runtime_file_fails(source_repo):
    (source_repo[0] / "src/sinklab/trainer.py").write_text("VALUE = 2\n")
    with pytest.raises(RuntimeSourceError, match="dirty"):
        _check(source_repo)


def test_deleted_runtime_file_fails(source_repo):
    (source_repo[0] / "src/sinklab/trainer.py").unlink()
    with pytest.raises(RuntimeSourceError, match="dirty"):
        _check(source_repo)


def test_added_tracked_runtime_file_fails(source_repo):
    repo = source_repo[0]
    (repo / "src/sinklab/added.py").write_text("VALUE = 2\n")
    _git(repo, "add", "src/sinklab/added.py")
    _git(repo, "commit", "-qm", "new runtime file")
    with pytest.raises(RuntimeSourceError, match="committed tree differs"):
        _check(source_repo)


def test_untracked_importable_python_file_fails_even_when_ignored(source_repo):
    (source_repo[0] / "src/sinklab/ignored_extra.py").write_text("VALUE = 2\n")
    with pytest.raises(RuntimeSourceError, match="untracked importable"):
        _check(source_repo)


@pytest.mark.parametrize("name", ["pyproject.toml", "uv.lock"])
def test_dirty_dependency_source_fails(source_repo, name):
    (source_repo[0] / name).write_text("changed = true\n")
    with pytest.raises(RuntimeSourceError, match="dirty"):
        _check(source_repo)


def test_runtime_source_must_be_ancestor(source_repo):
    repo, milestone = source_repo
    original_branch = _git(repo, "branch", "--show-current")
    _git(repo, "switch", "-qc", "sibling")
    (repo / "sibling.md").write_text("sibling\n")
    _git(repo, "add", "sibling.md")
    _git(repo, "commit", "-qm", "sibling commit")
    sibling = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-q", original_branch)
    (repo / "main.md").write_text("main\n")
    _git(repo, "add", "main.md")
    _git(repo, "commit", "-qm", "main descendant")
    assert milestone != sibling
    with pytest.raises(RuntimeSourceError, match="not an ancestor"):
        _check(source_repo, sibling)


def test_wrong_loaded_package_and_path_set_fail(source_repo):
    repo, milestone = source_repo
    with pytest.raises(RuntimeSourceError, match="loaded sinklab"):
        validate_runtime_source(repo, milestone, 1, loaded_package_dir=repo / "other")
    with pytest.raises(RuntimeSourceError, match="path-set version"):
        validate_runtime_source(repo, milestone, 2,
                                loaded_package_dir=repo / "src/sinklab")


def test_protocol_and_config_runtime_bindings_must_agree():
    payload = {
        "status": "approved", "production_ready": True, "study": "S1",
        "condition_variants": {"C0": "ce_only"},
        "allowed_device_roles": {"C0": ["rtx4080super"]},
        "source_commit": "f" * 40, "production_runtime_source_commit": "f" * 40,
        "calibration_source_commit": CALIBRATION_SOURCE_COMMIT,
        "execution_critical_path_set_version": 1,
        "approval": {"researcher": "fixture", "approved_at_utc": "2026-09-28T00:00:00Z",
                     "approval_sha256": "a" * 64,
                     "decision_ids": [f"D{i:02d}" for i in range(1, 19)]},
        "artifact_lock_digest": "b" * 64, "environment_lock_digest": "c" * 64,
        "hardware_lock_digest": "d" * 64, "calibration_lock_digest": "e" * 64,
        "protocol": {"production_config_binding_schema": 2,
                     "teacher": {"layers": 36, "heads": 20, "width": 1280},
                     "student": {"layers": 24, "heads": 16, "width": 1024,
                                 "initialization": "random_from_config"},
                     "hardware": {"gpu_uuids": {"rtx4080super": "GPU-fixture"}},
                     "data": {"seed0_initialization_sha256": "1" * 64,
                              "seed0_order_sha256": "2" * 64,
                              "production_corpus_sha256": "3" * 64,
                              "frozen_panels_sha256": "4" * 64},
                     "training": {"microbatch": 4, "accumulation": 16,
                                  "effective_sequences": 64, "sequence_length": 128,
                                  "primary_optimizer_updates": 10000}},
    }
    lock = seal_payload(payload)
    config = {"schema_version": 1, "study": "S1", "condition": "C0",
              "variant": "ce_only", "device_role": "rtx4080super",
              "teacher": payload["protocol"]["teacher"],
              "student": payload["protocol"]["student"],
              "protocol_digest": lock["sha256"],
              "production_binding": production_binding_for(payload, "rtx4080super", 0)}
    resolve_config(config, seed=0, protocol_lock=lock, production=True)
    config["production_binding"]["production_runtime_source_commit"] = "0" * 40
    with pytest.raises(ConfigError, match="production config binding"):
        resolve_config(config, seed=0, protocol_lock=lock, production=True)


def test_superseded_root_is_nonlaunchable(tmp_path):
    historical = ROOT / "protocols/superseded" / f"s1-protocol-{SUPERSEDED_PROTOCOL_ROOT}.json"
    if not historical.exists():
        historical = ROOT / "protocols/protocol.lock.json"
    old = json.loads(historical.read_text(encoding="utf-8"))
    assert old["sha256"] == SUPERSEDED_PROTOCOL_ROOT
    from sinklab.provenance import validate_protocol_lock
    assert validate_protocol_lock(old)[0]["source_commit"] == CALIBRATION_SOURCE_COMMIT
    assert "production_runtime_source_commit" not in old["payload"]
    for name in ("artifact", "environment", "hardware", "calibration"):
        (tmp_path / f"{name}.lock.json").write_bytes(
            (ROOT / "protocols" / f"{name}.lock.json").read_bytes())
    (tmp_path / "protocol.lock.json").write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(ReadinessError, match="superseded"):
        # The old root is rejected before any external artifact or model read.
        validate_final_lock_set(tmp_path)
