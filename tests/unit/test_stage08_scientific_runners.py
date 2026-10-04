import hashlib

import pytest

from scripts.run_stage08_s4 import _verify_pinned_file
from scripts.run_stage08_s6 import _expected_record_counts


def test_teacher_config_pin_verifies_exact_source_file_bytes(tmp_path):
    path = tmp_path / "config.json"
    raw = b'{"n_layer":36,"n_head":20,"n_embd":1280}\n'
    path.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    assert _verify_pinned_file(path, expected, "teacher config") == expected
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        _verify_pinned_file(path, hashlib.sha256(raw.rstrip()).hexdigest(), "teacher config")


def test_s6_final_audit_counts_one_all_operations_aggregate_per_panel():
    assert _expected_record_counts() == (50_400, 168)
