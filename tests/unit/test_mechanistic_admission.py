import copy
import json

import pytest

from sinklab.followup_policy import D24_SHA256
from sinklab.mechanistic_admission import admit_job, E1_GRID, E2_GRID
from sinklab.mechanistic_run import VERSION, sha256_file
from sinklab.provenance import payload_digest, seal_payload


def write(path,data):
    path.write_text(json.dumps(data),encoding="utf-8")
    return {"path":str(path),"sha256":sha256_file(path)}


def protocol(tmp_path,monkeypatch,phase="E2"):
    runtime = {"engineering_test_runtime_only":True}
    monkeypatch.setattr("sinklab.mechanistic_admission.runtime_identity",lambda *args:runtime)
    monkeypatch.setattr("sinklab.mechanistic_admission.read_json",lambda path: {"test":True} if str(path).endswith("artifact.lock.json") else json.loads(path.read_text()))
    monkeypatch.setattr("sinklab.mechanistic_panel.prepare_frozen_panel",lambda **kwargs:json.loads((tmp_path/"panel.json").read_text()))
    grid = ["teacher",*(E1_GRID if phase=="E1" else E2_GRID)]
    if phase=="E3": grid=["teacher","C2/step500"]
    sources, settings = {}, {}
    for state in grid:
        teacher = state=="teacher"
        label = state.replace("/","_")
        folder = tmp_path/label; folder.mkdir()
        config = write(folder/"config.json",dict(zip(("n_layer","n_head","n_embd"),(36,20,1280) if teacher else (24,16,1024))) | {"vocab_size":50257,"n_positions":1024})
        (folder/"model.safetensors").write_bytes(b"fake pinned weights never deserialized in admission tests")
        source = {"config":config,"weights":{"path":str(folder/"model.safetensors"),"sha256":sha256_file(folder/"model.safetensors")}}
        if teacher: source.update(model_id="openai-community/gpt2-large",revision="1"*40)
        else:
            condition,step = state.split("/step")
            identity={"study":"S1","condition":condition,"seed":0,"protocol_hash":"a"*64,"run_id":f"original-{condition}"}
            source.update(identity=identity,initialization="configuration_random")
            source["manifest"] = write(folder/"manifest.json",{"schema_version":1,"kind":"weights","step":int(step),
                "identity":identity,"identity_sha256":payload_digest(identity),"files":{"model.safetensors":source["weights"]["sha256"]}})
            (folder/"COMPLETE").write_text(source["manifest"]["sha256"])
        sources[state] = source
        current = {"atol":1e-6,"rtol":1e-5,"token_chunk":16,"denominator_floor":1e-8}
        if phase=="E1":
            current.update(alphas=[0.,.25,.5,.75,1.],control_seed=17,scopes=[{"name":"native","layers":list(range(36 if teacher else 24))}])
            if teacher: current["scopes"].append({"name":"mapped_teacher","layers":list(range(24))})
        else: current["layers"]=list(range(36 if teacher else 24)) if phase=="E2" else ([0,18,35] if teacher else [0,12,23])
        if phase=="E3":
            current.update(etas=[0.,.01],norm_floor=1e-10,control_seed=23,reference="clean_residual_input_before_ln_1",
                nonsink_keys=[1,2],query_min=2,orders=[list(range(36 if teacher else 24)),list(reversed(range(36 if teacher else 24)))])
        settings[state]=current
    panel = {"context_length":128,"tokenizer":{"model_id":"gpt2","artifact_sha256":"b"*64},
             "preparation":{"corpus":{},"registered_panels":{}}}
    for i,name in enumerate(("discovery","confirmation")):
        panel[name]=[{"id":name,"document_ids":[name],"document_hashes":[str(i)*64],
                      "input_ids":[1]*128,"attention_mask":[True]*128}]
    payload={"schema_version":VERSION,"status":"approved","production_ready":True,
        "approval":{"authority":"researcher","record":"unit-test simulated approval; NOT a scientific approval"},
        "phase":phase,"seed":0,"d24_sha256":D24_SHA256,"run_id":"test-explicit-job","grid":grid,"sources":sources,
        "settings":settings,"runtime":runtime,"panel":write(tmp_path/"panel.json",panel),
        "panel_counts":{"discovery":1,"confirmation":1},"teacher_layer_map":list(range(24))}
    payload["qualification"] = write(tmp_path/"qualification.json",{"status":"qualified","scientific_production_shape":True,
        "phase":phase,"runtime":runtime,"architectures":["gpt2-large","gpt2-medium"],"parity_passed":True,
        "headroom_passed":True,"settings_sha256":payload_digest(settings)})
    if phase=="E3":
        payload["selection"]={"method":"prespecified_coarse","selected_states":grid,
                              "selected_layers":{s:settings[s]["layers"] for s in grid}}
    return payload


def admit(payload,phase=None,seed=0,panel="confirmation"):
    doc=seal_payload(payload)
    return admit_job(doc,approved_sha256=doc["sha256"],phase=phase or payload["phase"],state="C2/step500",
                     panel_name=panel,seed=seed,repo="unused",device="cpu")


@pytest.mark.parametrize("phase",["E1","E2","E3"])
def test_declared_complete_grid_admitted_metadata_only(tmp_path,monkeypatch,phase):
    payload = protocol(tmp_path,monkeypatch,phase)
    identity,settings,items,source,teacher = admit(payload)
    assert identity["source"]["source_identity"]["protocol_hash"] == "a"*64
    assert identity["protocol_sha256"] != "a"*64 and identity["engineering_only"] is False
    assert teacher["role"]=="teacher" and source["role"]=="student" and len(items)==1
    assert len(E1_GRID)==13 and len(E2_GRID)==15


@pytest.mark.parametrize("mutation",["draft","scope","seed","d24","runtime","qualification","panel_overlap",
    "panel_count","missing_source","wrong_step","source_seed","pretrained","weights","complete","config","settings"])
def test_gate_rejects_changed_or_unapproved_scope_before_model_load(tmp_path,monkeypatch,mutation):
    payload=protocol(tmp_path,monkeypatch)
    source=payload["sources"]["C2/step500"]
    if mutation=="draft": payload["status"]="draft"
    elif mutation=="scope": payload["grid"].remove("C3/step2000")
    elif mutation=="seed": payload["seed"]=1
    elif mutation=="d24": payload["d24_sha256"]="f"*64
    elif mutation=="runtime": payload["runtime"]={"different":True}
    elif mutation=="qualification": payload["qualification"]=write(tmp_path/"qualification.json",{"status":"synthetic_only"})
    elif mutation=="panel_overlap":
        path=tmp_path/"panel.json"; panel=json.loads(path.read_text()); panel["confirmation"][0]["document_ids"]=["discovery"]
        payload["panel"]=write(path,panel)
    elif mutation=="panel_count": payload["panel_counts"]["discovery"]=300
    elif mutation=="missing_source": del payload["sources"]["C2/step10000"]
    elif mutation in {"wrong_step","source_seed","config"}:
        if mutation=="config":
            path=tmp_path/"C2_step500/config.json"; config=json.loads(path.read_text()); config["n_head"]=12
            source["config"]=write(path,config)
        else:
            path=tmp_path/"C2_step500/manifest.json"; data=json.loads(path.read_text())
            if mutation=="wrong_step": data["step"]=2000
            else:
                source["identity"]["seed"]=1; data["identity"]=source["identity"]; data["identity_sha256"]=payload_digest(source["identity"])
            source["manifest"]=write(path,data); path.with_name("COMPLETE").write_text(source["manifest"]["sha256"])
    elif mutation=="pretrained": source["initialization"]="pretrained"
    elif mutation=="weights": (tmp_path/"C2_step500/model.safetensors").write_bytes(b"tamper")
    elif mutation=="complete": (tmp_path/"C2_step500/COMPLETE").write_text("f"*64)
    else: payload["settings"]["C2/step500"]["atol"]=1
    with pytest.raises(ValueError): admit(payload)


def test_missing_external_approval_digest_cannot_self_authorize(tmp_path,monkeypatch):
    payload=protocol(tmp_path,monkeypatch);doc=seal_payload(payload)
    with pytest.raises(ValueError,match="externally approved"):
        admit_job(doc,approved_sha256="0"*64,phase="E2",state="C2/step500",panel_name="discovery",seed=0,repo="unused",device="cpu")


def test_confirmation_cannot_choose_layers_and_discovery_requires_alternate_order(tmp_path,monkeypatch):
    payload=protocol(tmp_path,monkeypatch,"E3")
    payload["selection"]["method"]="discovery_only"
    payload["selection"]["discovery_evidence"]=write(tmp_path/"selection.json",{"panel":"confirmation","phase":"E2","panel_sha256":payload["panel"]["sha256"]})
    with pytest.raises(ValueError,match="leakage"): admit(payload)
    payload["selection"]["method"]="prespecified_coarse"
    payload["settings"]["C2/step500"]["orders"] = [list(range(24))]
    payload["qualification"]=write(tmp_path/"qualification.json",{"status":"qualified","scientific_production_shape":True,
        "phase":"E3","runtime":payload["runtime"],"architectures":["gpt2-large","gpt2-medium"],"parity_passed":True,
        "headroom_passed":True,"settings_sha256":payload_digest(payload["settings"])})
    with pytest.raises(ValueError,match="alternate"): admit(payload,panel="discovery")
