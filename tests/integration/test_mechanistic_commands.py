import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]


def run(*args):
    env=os.environ.copy();env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="1")
    return subprocess.run([sys.executable,str(ROOT/"scripts/run_mechanistic.py"),*map(str,args)],
        cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)


@pytest.mark.parametrize("phase",["E1","E2","E3"])
def test_128_token_engineering_roundtrip_and_readonly_report(tmp_path,phase):
    output=tmp_path/phase
    result=run("engineering","--phase",phase,"--seed",0,"--device","cpu","--output",output)
    assert result.returncode==0,result.stdout+result.stderr
    assert "engineering_only=True" in result.stdout and "COMPLETE" in result.stdout
    summary=json.loads((output/"summary.json").read_text())
    assert summary["identity"]["engineering_only"] is True and summary["identity"]["context_length"]==128
    assert summary["identity"]["protocol_sha256"] is None
    assert next(iter(summary["operations"].values()))["behavior"]["valid_targets"]==243
    verify=run("verify","--output",output)
    assert verify.returncode==0,verify.stdout+verify.stderr
    report=tmp_path/f"{phase}-report.json"
    command=subprocess.run([sys.executable,str(ROOT/"scripts/report_mechanistic.py"),"--bundle",str(output),"--output",str(report),"--figure-prefix",str(tmp_path/f"{phase}-figure")],
        cwd=ROOT,capture_output=True,text=True,timeout=120)
    assert command.returncode==0,command.stdout+command.stderr
    assert json.loads(report.read_text())["summary"]==summary
    assert (tmp_path/f"{phase}-figure.png").stat().st_size>1000
    assert "<svg" in (tmp_path/f"{phase}-figure.svg").read_text()
    before=(output/"COMPLETE").read_bytes()
    repeat=run("engineering","--phase",phase,"--seed",0,"--device","cpu","--output",output)
    assert repeat.returncode!=0 and (output/"COMPLETE").read_bytes()==before


def test_no_default_seed_hidden_campaign_or_internal_output(tmp_path):
    result=run("engineering","--phase","E2","--device","cpu","--output",tmp_path/"no-seed")
    assert result.returncode!=0 and "--seed" in result.stderr
    result=run("engineering","--phase","E2","--seed",1,"--device","cpu","--output",tmp_path/"seed1")
    assert result.returncode!=0 and not (tmp_path/"seed1").exists()
    result=run("engineering","--phase","E2","--seed",0,"--device","cpu","--output",ROOT/"should-not-exist")
    assert result.returncode!=0 and not (ROOT/"should-not-exist").exists()


def test_draft_scientific_lock_fails_before_model_load_or_output(tmp_path):
    from sinklab.provenance import seal_payload
    lock=seal_payload({"schema_version":"mechanistic-followup-records-v1","status":"draft"})
    path=tmp_path/"draft.json";path.write_text(json.dumps(lock))
    result=run("scientific","--phase","E2","--seed",0,"--device","cpu","--state","C2/step500",
        "--panel","discovery","--lock",path,"--approved-sha256",lock["sha256"],"--output",tmp_path/"rejected")
    assert result.returncode!=0 and "draft or unapproved" in result.stderr and not (tmp_path/"rejected").exists()
