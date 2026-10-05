import json
import sys
from pathlib import Path

import pytest

from scripts import run_stage08_s7
from scripts.run_stage08_s7_supplement import validate_analysis_lock
from sinklab.provenance import seal_payload
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from test_s7_utility import _bundle, _sha


def test_analysis_command_writes_traceable_external_result(tmp_path, monkeypatch):
    source, _ = _bundle(tmp_path)
    monkeypatch.setattr(run_stage08_s7, "S5_AUDIT_SHA256", _sha(source / "STAGE08_S5_AUDIT.json"))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "s7"
    audit = run_stage08_s7.run(s5_bundle=source, output=output)
    assert audit["status"] == "descriptive_s5_reuse_only"
    assert audit["science_complete"] is False
    assert audit["s7_analysis_sha256"] == _sha(output / "S7_ANALYSIS.json")
    assert json.loads((output / "S7_ANALYSIS.json").read_text())["training_seed_count"] == 1
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    with pytest.raises(FileExistsError):
        run_stage08_s7.run(s5_bundle=source, output=output)


def test_unapproved_draft_and_forged_lock_cannot_launch_new_s7_inference(tmp_path, monkeypatch):
    draft = json.loads(Path("protocols/s7_utility_analysis_draft.json").read_text())
    with pytest.raises(ValueError):
        validate_analysis_lock(draft)
    approved = json.loads(Path("protocols/s7_utility_analysis_approved.json").read_text())
    forged_payload = {**approved["payload"], "training_seed": 2}
    forged = seal_payload(forged_payload)
    fake_checked_in = tmp_path / "forged.json"
    fake_checked_in.write_text(json.dumps(forged), encoding="utf-8")
    monkeypatch.setattr("scripts.run_stage08_s7_supplement.APPROVED_LOCK_PATH", fake_checked_in)
    with pytest.raises(ValueError, match="exact researcher-approved"):
        validate_analysis_lock(forged)
