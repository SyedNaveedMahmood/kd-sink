import copy
import importlib.util
from pathlib import Path

import pytest

from sinklab.mechanistic_admission import E1_GRID,E2_GRID
from sinklab.provenance import payload_digest,seal_payload

spec=importlib.util.spec_from_file_location("mechanistic_report_script",Path(__file__).resolve().parents[2]/"scripts/report_mechanistic.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture(monkeypatch,phase="E2"):
    grid=["teacher",*(E1_GRID if phase=="E1" else E2_GRID)]
    runtime={"unit_fixture":True}
    payload={"status":"approved","production_ready":True,"phase":phase,"grid":grid,"runtime":runtime,
        "panel":{"sha256":"a"*64},"panel_counts":{"confirmation":2},"sources":{},"settings":{}}
    for state in grid:
        payload["sources"][state]={"weights":{"sha256":"b"*64},"config":{"sha256":"c"*64},"identity":None if state=="teacher" else {"run_id":state}}
        payload["settings"][state]={"alphas":[0.,.25,.5,.75,1.]}
    lock=seal_payload(payload);summaries={}
    for state in grid:
        summaries[state]={"identity":{"state":state,"phase":phase,"seed":0,"panel":"confirmation","engineering_only":False,
            "protocol_sha256":lock["sha256"],"panel_sha256":"a"*64,"runtime_sha256":payload_digest(runtime),
            "source":{"weights_sha256":"b"*64,"config_sha256":"c"*64,"source_identity":payload["sources"][state]["identity"]}},
            "settings":payload["settings"][state],"items":["item-a","item-b"],"operations":{}}
    monkeypatch.setattr(module,"verify_bundle",lambda root:summaries[root])
    monkeypatch.setattr(module,"sha256_file",lambda path:"d"*64)
    return grid,lock,summaries


@pytest.mark.parametrize("phase",["E1","E2"])
def test_exact_full_grid_join_preserves_static_teacher_and_original_sources(monkeypatch,phase):
    grid,lock,summaries=fixture(monkeypatch,phase)
    result=module.complete_join(grid,lock=lock,approved_sha256=lock["sha256"],panel_name="confirmation")
    assert set(result["states"])==set(grid) and len(result["states"])==(14 if phase=="E1" else 16)


@pytest.mark.parametrize("change",["missing","duplicate","panel","items","runtime","source","settings","engineering"])
def test_complete_join_rejects_partial_or_mixed_identity(monkeypatch,change):
    grid,lock,summaries=fixture(monkeypatch)
    selected=summaries["C2/step500"]
    if change=="missing": grid=grid[:-1]
    elif change=="duplicate": grid=grid+[grid[0]]
    elif change=="panel": selected["identity"]["panel"]="discovery"
    elif change=="items": selected["items"].reverse()
    elif change=="runtime": selected["identity"]["runtime_sha256"]="f"*64
    elif change=="source": selected["identity"]["source"]["weights_sha256"]="f"*64
    elif change=="settings": selected["settings"]={"other":True}
    else: selected["identity"]["engineering_only"]=True
    with pytest.raises(ValueError): module.complete_join(grid,lock=lock,approved_sha256=lock["sha256"],panel_name="confirmation")


def test_dose_matching_uses_mapped_scope_and_actual_overlap(monkeypatch):
    grid,lock,summaries=fixture(monkeypatch,"E1")
    for state,summary in summaries.items():
        scope="mapped_teacher" if state=="teacher" else "native"
        for route in ("q_bias","k_top3",*(f"k_random{i}" for i in range(5))):
            for alpha in summary["settings"]["alphas"]:
                summary["operations"][f"{route}/{scope}/{alpha:g}"]={"sink_removed_fraction_item_mean":alpha,
                    "scope_attention_output_delta_rms_item_mean":alpha*2}
    join=module.complete_join(grid,lock=lock,approved_sha256=lock["sha256"],panel_name="confirmation")
    matches=module.dose_comparisons(join,metric="sink_removed_fraction",targets=[.5,2.])
    assert len(matches)==13*7*2
    assert all(m["status"]=="observed_overlap" for m in matches if m["target"]==.5)
    assert all(m["status"]=="dose_unmatched" for m in matches if m["target"]==2.)
    summaries["teacher"]["operations"]["q_bias/mapped_teacher/0.5"]["sink_removed_fraction_item_mean"]=None
    assert module.dose_comparisons(join,metric="sink_removed_fraction",targets=[.5])[0]["status"]=="dose_unavailable"
