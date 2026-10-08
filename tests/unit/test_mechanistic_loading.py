import json

import pytest
import torch
from safetensors.torch import save_file

from sinklab.mechanistic_admission import load_model
from sinklab.mechanistic_run import sha256_file,evaluate_item
from sinklab.mechanism_trace import MechanisticGPT2Adapter
from test_calibrated_probes import model_fixture
from test_mechanistic_run import fixture


def source(tmp_path,mutation=None):
    model=model_fixture()
    config=tmp_path/"config.json";config.write_text(model.config.to_json_string())
    tensors={key:value.detach().clone() for key,value in model.state_dict().items()}
    if mutation=="missing_tied": del tensors["lm_head.weight"]
    if mutation=="missing_required": del tensors["transformer.h.0.attn.c_attn.bias"]
    if mutation=="extra": tensors["fake_parameter"]=torch.zeros(1)
    if mutation=="conflicting_tied": tensors["lm_head.weight"]=tensors["lm_head.weight"]+1.
    weights=tmp_path/"model.safetensors";save_file(tensors,str(weights))
    return model,{"weights_path":str(weights),"weights_sha256":sha256_file(weights),
        "config_path":str(config),"config_sha256":sha256_file(config)}


@pytest.mark.parametrize("mutation",[None,"missing_tied"])
def test_local_weights_match_source_freeze_eval_and_preserve_rng(tmp_path,mutation):
    original,reference=source(tmp_path,mutation)
    rng=torch.get_rng_state().clone()
    loaded=load_model(reference,"cpu")
    assert torch.equal(torch.get_rng_state(),rng)
    assert not loaded.training and all(not p.requires_grad and p.dtype==torch.float32 for p in loaded.parameters())
    assert loaded.lm_head.weight.data_ptr()==loaded.transformer.wte.weight.data_ptr()
    for key,value in loaded.state_dict().items(): assert torch.equal(value,original.state_dict()[key])


@pytest.mark.parametrize("mutation",["missing_required","extra","conflicting_tied"])
def test_incomplete_or_ambiguous_model_state_rejected(tmp_path,mutation):
    _,reference=source(tmp_path,mutation)
    with pytest.raises(ValueError): load_model(reference,"cpu")


def test_stale_source_rejected_before_tensor_deserialization(tmp_path,monkeypatch):
    _,reference=source(tmp_path)
    (tmp_path/"model.safetensors").write_bytes(b"tamper")
    def forbidden(*args,**kwargs):raise AssertionError("deserialized before pin check")
    monkeypatch.setattr("safetensors.torch.load_file",forbidden)
    with pytest.raises(ValueError,match="changed after admission"): load_model(reference,"cpu")


def test_e1_api_rejects_half_and_autocast():
    items,settings,_=fixture("E1")
    with pytest.raises(ValueError,match="FP32"):
        evaluate_item(MechanisticGPT2Adapter(model_fixture().half()),items[0],"E1",settings)
    with torch.autocast("cpu",dtype=torch.bfloat16),pytest.raises(ValueError,match="autocast"):
        evaluate_item(MechanisticGPT2Adapter(model_fixture()),items[0],"E1",settings)
