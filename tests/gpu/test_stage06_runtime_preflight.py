"""Real-device production entry checks that never start a training job."""

from argparse import Namespace
from pathlib import Path

import pytest

from sinklab.runtime_provenance import RuntimeSourceError, validate_runtime_source
from sinklab.stage06_readiness import validate_final_lock_set
from sinklab import training_entry


ROOT = Path(__file__).resolve().parents[2]


def test_4080_runtime_source_and_preload_refusal(selected_cuda, gpu_evidence, monkeypatch):
    assert selected_cuda.type == "cuda"
    lock = validate_final_lock_set(ROOT / "protocols")["protocol"]["payload"]
    evidence = validate_runtime_source(
        ROOT, lock["production_runtime_source_commit"],
        lock["execution_critical_path_set_version"],
        loaded_package_dir=ROOT / "src/sinklab")
    assert evidence["tracked_tree_equal"] and evidence["working_tree_clean"]
    with pytest.raises(RuntimeSourceError, match="cat-file"):
        validate_runtime_source(ROOT, "0" * 40, 1,
                                loaded_package_dir=ROOT / "src/sinklab")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("model, GPU identity, or training was reached before source guard")

    monkeypatch.setattr(training_entry, "_gpu_identity", forbidden)
    monkeypatch.setattr(training_entry.GPT2LMHeadModel, "from_pretrained", forbidden)
    monkeypatch.setattr("sinklab.runtime_provenance.validate_runtime_source",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            RuntimeSourceError("preload runtime source refusal")))
    args = Namespace(protocol_lock=ROOT / "protocols/protocol.lock.json",
                     config=ROOT / "configs/production/s1/c0_rtx4080super.json",
                     hardware_plan=ROOT / "protocols/hardware.lock.json", seed=0)
    with pytest.raises(RuntimeSourceError, match="preload runtime source refusal"):
        training_entry.run_approved_training(args)
    gpu_evidence[0]["measurements"]["stage06_runtime_preflight"] = {
        "runtime_source_commit": lock["production_runtime_source_commit"],
        "clean_runtime_source_passed": True,
        "invalid_source_refused": True,
        "production_entry_refused_before_model_load": True,
        "training_started": False,
    }
