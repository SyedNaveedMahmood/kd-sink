import json
import subprocess
import sys
from pathlib import Path

import pytest

from sinklab.provenance import LockError, _no_duplicate_keys


def test_no_default_seed_or_sweep(tmp_path):
    config = Path(__file__).resolve().parents[2] / "configs" / "s1_c2_draft.json"
    missing = subprocess.run(
        [sys.executable, "-m", "sinklab", "validate", "--config", str(config)],
        capture_output=True, text=True,
    )
    assert missing.returncode == 2 and "--seed" in missing.stderr
    valid = subprocess.run(
        [sys.executable, "-m", "sinklab", "validate", "--config", str(config), "--seed", "0"],
        capture_output=True, text=True,
    )
    assert valid.returncode == 0, valid.stderr
    result = json.loads(valid.stdout)
    assert result["action"] == "validated_only"
    assert result["seed"] == 0 and result["condition"] == "C2"
    blocked = subprocess.run(
        [sys.executable, "-m", "sinklab", "validate", "--config", str(config),
         "--seed", "0", "--production"], capture_output=True, text=True,
    )
    assert blocked.returncode == 2 and "approved protocol" in blocked.stderr


def test_duplicate_json_key_is_rejected():
    with pytest.raises(LockError, match="duplicate"):
        json.loads('{"status":"draft","status":"approved"}', object_pairs_hook=_no_duplicate_keys)
