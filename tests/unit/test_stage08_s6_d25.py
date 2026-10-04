import json
from pathlib import Path

from sinklab.provenance import verify_envelope
from sinklab.s6 import validate_s6_source_metadata


ROOT = Path(__file__).resolve().parents[2]


def _read_sealed(path):
    return verify_envelope(json.loads((ROOT / path).read_text(encoding="utf-8")))


def test_d25_is_a_separate_prospective_addendum_to_unchanged_d24():
    decision, decision_sha = _read_sealed(
        "protocols/s1_researcher_amendment_d25_s6_external_datasets_20261004.json")
    lock, lock_sha = _read_sealed("protocols/s6_external_dataset_sources_d25.json")
    d24, d24_sha = _read_sealed("protocols/s1_researcher_amendment_d24_seed0_followups_20261004.json")
    assert decision["decision_id"] == "D25"
    assert decision["prospective_only"] is True
    assert decision["source_lock_sha256"] == lock_sha
    assert decision["d24_policy_sha256"] == d24_sha
    assert d24["decision_id"] == "D24"
    assert decision["historical_preservation"]["S1_roots_and_results"] == "unchanged"
    assert decision["historical_preservation"]["S2"] == "unchanged"
    assert decision["license_policy"]["legal_determination"] is False
    assert decision["license_policy"]["redistribution_permission_asserted"] is False
    assert len(decision_sha) == 64
    assert lock["decision_id"] == "D25"


def test_d25_source_lock_has_exact_fields_revisions_hashes_and_license_statuses():
    lock, _ = _read_sealed("protocols/s6_external_dataset_sources_d25.json")
    domains = validate_s6_source_metadata(lock["domains"])
    expected = {
        "sst2": ("nyu-mll/glue", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "validation", "sentence", "upstream_ambiguous"),
        "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "test", "question", "upstream_stated"),
        "humaneval": ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544", "test", "prompt", "upstream_stated"),
    }
    for domain, (repository, revision, split, field, license_status) in expected.items():
        metadata = domains[domain]
        assert (metadata["repository"], metadata["revision"], metadata["split"], metadata["field"]) == (
            repository, revision, split, field)
        assert metadata["license"]["status"] == license_status
        assert len(metadata["file_sha256"]) == 64
        assert len(metadata["license"]["evidence_sha256"]) == 64
        assert metadata["model_input_fields"] == [field]
    assert domains["sst2"]["license"]["upstream_value"] == "other"
    assert domains["gsm8k"]["license"]["upstream_value"] == "mit"
    assert domains["humaneval"]["license"]["upstream_value"] == "mit"
    assert lock["selection"]["contexts"] == [40, 128]
    assert lock["selection"]["optional_contexts"] == {"512": "disabled", "1024": "disabled"}
