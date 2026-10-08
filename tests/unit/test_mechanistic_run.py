import copy
import json

import pytest
import torch

from sinklab.mechanistic_run import (VERSION, run_state, verify_bundle, evaluate_item,
    operation_ids, read_json, sha256_file, write_json, validate_items)
from sinklab.mechanism_trace import MechanisticGPT2Adapter
from test_calibrated_probes import model_fixture


def fixture(phase):
    items = [{"id": "long", "input_ids": [1,2,3,4,5], "attention_mask": [True]*5},
             {"id": "short", "input_ids": [6,7,8,0,0], "attention_mask": [True]*3+[False]*2}]
    settings = {"atol":2e-6,"rtol":2e-5,"token_chunk":2,"denominator_floor":1e-8}
    if phase == "E1":
        settings.update(alphas=[0.,.25,.5,.75,1.],control_seed=17,scopes=[{"name":"native","layers":[0,1]}])
    else:
        settings.update(layers=[0,1])
    if phase == "E3":
        settings.update(etas=[0.,.01,.05],norm_floor=1e-10,control_seed=313,
            reference="clean_residual_input_before_ln_1",query_min=2,nonsink_keys=[1,2],orders=[[0,1],[1,0]])
    identity = {"schema_version":VERSION,"phase":phase,"seed":0,"state":"unit_random_model",
                "run_id":"explicit-unit-fixture","engineering_only":True,"context_length":5}
    return items, settings, identity


@pytest.mark.parametrize("phase",["E1","E2","E3"])
def test_full_item_artifacts_and_independent_token_weighting(tmp_path,phase):
    items, settings, identity = fixture(phase)
    model = model_fixture()
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    result = run_state(MechanisticGPT2Adapter(model),items,phase=phase,settings=settings,identity=identity,output=tmp_path/phase)
    assert verify_bundle(tmp_path/phase) == result
    assert len(result["operations"]) == len(operation_ids(phase,settings))
    records = [read_json(p) for p in sorted((tmp_path/phase).glob("item-*.json"))]
    for index, summary in enumerate(result["operations"].values()):
        rows = [r["operations"][index] for r in records]
        assert summary["behavior"]["valid_targets"] == 6
        clean = sum(r["behavior"]["clean_nll_sum_nats"] for r in rows)/6
        edited = sum(r["behavior"]["edited_nll_sum_nats"] for r in rows)/6
        assert summary["behavior"]["delta_ce_nats"] == pytest.approx(edited-clean)
        assert summary["geometry"]["delta_nll_sum_nats"] == pytest.approx((edited-clean)*6,abs=3e-6)
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in model.state_dict().items(): assert torch.equal(before[key],value)
    with pytest.raises(FileExistsError):
        run_state(MechanisticGPT2Adapter(model),items,phase=phase,settings=settings,identity=identity,output=tmp_path/phase)


@pytest.mark.parametrize("corruption",["item","summary","extra","missing","marker","failure","traversal"])
def test_corrupt_bundle_rejected(tmp_path,corruption):
    items, settings, identity = fixture("E2")
    root = tmp_path/"bundle"
    run_state(MechanisticGPT2Adapter(model_fixture()),items,phase="E2",settings=settings,identity=identity,output=root)
    if corruption == "item":
        (root/"item-00000.json").write_text("{}")
    elif corruption == "summary":
        summary = read_json(root/"summary.json"); summary["items"].reverse()
        (root/"summary.json").write_text(json.dumps(summary))
        # Even self-consistently rehashed corruption must fail reaggregation.
        manifest = read_json(root/"manifest.json"); manifest["files"]["summary.json"] = sha256_file(root/"summary.json")
        (root/"manifest.json").write_text(json.dumps(manifest)); (root/"COMPLETE").write_text(sha256_file(root/"manifest.json"))
    elif corruption == "extra": (root/"untracked.json").write_text("{}")
    elif corruption == "missing": (root/"item-00000.json").unlink()
    elif corruption == "marker": (root/"COMPLETE").write_text("0"*64)
    elif corruption == "failure": (root/"FAILED.json").write_text("{}")
    else:
        manifest = read_json(root/"manifest.json"); manifest["files"]["../outside"] = "0"*64
        (root/"manifest.json").write_text(json.dumps(manifest)); (root/"COMPLETE").write_text(sha256_file(root/"manifest.json"))
    with pytest.raises((ValueError,FileNotFoundError)): verify_bundle(root)


def test_failed_item_preserves_partial_evidence_and_does_not_complete(tmp_path,monkeypatch):
    import sinklab.mechanistic_run as module
    items, settings, identity = fixture("E2")
    actual = module.evaluate_item
    def failing(adapter,item,*args,**kwargs):
        if item["id"] == "short": raise RuntimeError("deliberate failure")
        return actual(adapter,item,*args,**kwargs)
    monkeypatch.setattr(module,"evaluate_item",failing)
    with pytest.raises(RuntimeError,match="deliberate"):
        run_state(MechanisticGPT2Adapter(model_fixture()),items,phase="E2",settings=settings,identity=identity,output=tmp_path/"failed")
    assert (tmp_path/"failed/item-00000.json").exists()
    assert not (tmp_path/"failed/COMPLETE").exists()
    assert read_json(tmp_path/"failed/FAILED.json")["completed_items"] == 1


def test_degenerate_direction_not_recorded_as_zero_response():
    model = model_fixture()
    with torch.no_grad():
        for block in model.transformer.h: block.attn.c_proj.weight.zero_()
    items, settings, _ = fixture("E3")
    record = evaluate_item(MechanisticGPT2Adapter(model),items[0],"E3",settings)
    unavailable = [r for r in record["operations"] if "/eta0.01" in r["operation"]]
    assert len(unavailable)==8 and all(r["status"]=="unavailable" and r["behavior"] is None for r in unavailable)
    noops = [r for r in record["operations"] if r["operation"].endswith("/eta0")]
    assert all(r["status"]=="measured" for r in noops)


@pytest.mark.parametrize("mutation",["duplicate","padding","nonbool","token"])
def test_invalid_panel(mutation):
    items,_,_ = fixture("E2")
    if mutation=="duplicate": items[1]["id"] = items[0]["id"]
    if mutation=="padding": items[1]["attention_mask"] = [True,True,False,True,False]
    if mutation=="nonbool": items[0]["attention_mask"] = [1]*5
    if mutation=="token": items[0]["input_ids"][0] = 31
    with pytest.raises(ValueError): validate_items(items,length=5,vocab_size=31)


def test_json_duplicate_keys_nonfinite_and_exclusive_writes(tmp_path):
    path = tmp_path/"data.json"
    path.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError): read_json(path)
    path.write_text('{"a":NaN}')
    with pytest.raises(ValueError): read_json(path)
    with pytest.raises(FileExistsError): write_json(path,{"a":3})
