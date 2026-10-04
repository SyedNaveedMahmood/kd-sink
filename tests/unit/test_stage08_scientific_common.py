import pytest

from scripts.stage08_scientific_common import (
    REPO, Stage08Blocked, failure_status, output_path_is_external,
    write_or_validate_run_manifest,
)


def test_run_manifest_is_stable_across_resume_time_and_fails_closed_on_binding_change(tmp_path):
    path = tmp_path / "RUN_MANIFEST.json"
    first = {"study": "S4", "source_sha256": "a" * 64,
             "device_uuid": "GPU-test", "created_utc": "2026-10-04T00:00:00Z",
             "repository_commit": "a" * 40}
    first_sha = write_or_validate_run_manifest(path, first)
    resume = {**first, "created_utc": "2026-10-05T00:00:00Z",
              "repository_commit": "b" * 40}
    assert write_or_validate_run_manifest(path, resume) == first_sha
    changed = {**resume, "device_uuid": "GPU-other"}
    with pytest.raises(ValueError, match="resume run manifest conflicts"):
        write_or_validate_run_manifest(path, changed)


def test_scientific_output_must_stay_outside_git_and_source_runs(tmp_path):
    source = tmp_path / "source-run"
    source.mkdir()
    output_path_is_external(tmp_path / "analysis", [source])
    with pytest.raises(ValueError, match="outside Git"):
        output_path_is_external(REPO / "analysis", [source])
    with pytest.raises(ValueError, match="original source run"):
        output_path_is_external(source / "analysis", [source])


def test_failure_status_separates_blockers_from_integrity_failures():
    assert failure_status(Stage08Blocked("GPU unavailable")) == "SKIPPED_BLOCKED"
    assert failure_status(FileNotFoundError("approved artifact unavailable")) == "SKIPPED_BLOCKED"
    assert failure_status(ValueError("identity mismatch")) == "FAILED_INTEGRITY_CHECK"
