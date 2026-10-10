"""Migration boundaries: scientific changes, provenance and restart ownership."""
import copy
import os
from pathlib import Path
import sys

import psutil
import pytest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
import e3_nodi_recovery as recovery
from audit_mechanistic_scientific import read,write,digest
from mechanistic_campaign_supervisor import append,ledger
from monitor_e3_nodi_recovery import decision
from compare_e3_migration import numeric_diff


def successor():
    old=read(REPO/'protocols/mechanistic_e1_e3_approved_20261009/E3.approved.json')
    new=copy.deepcopy(old); value=new['payload']
    value['runtime']['hardware']['uuid']='GPU-real-successor-test-device'
    value['panel']['path']='C:/relocated/panel.json'
    value['panel']['sha256']='a'*64
    value['approval']={'authority':'researcher','record':'explicit migration instruction'}
    value['migration']={'authority':'researcher-authorized E3 migration'}
    for source in value['sources'].values():
        for key in ('config','manifest','weights'):
            if key in source: source[key]['path']='C:/relocated/'+key
    new['sha256']=digest(value)
    return old,new


def test_relocation_and_real_successor_device_preserve_science():
    old,new=successor()
    assert recovery.compatibility(old,new)['scientific_field_diff']==[]
    assert old['payload']['runtime']['hardware']['uuid']!=new['payload']['runtime']['hardware']['uuid']


@pytest.mark.parametrize('change',['dose','layers','source_weights','seed','grid','initialization','norm_reference','runtime_code','tf32'])
def test_successor_rejects_material_or_unqualified_changes(change):
    old,new=successor(); p=new['payload']; state='C1/step500'
    if change=='dose':p['settings'][state]['etas'][-1]=.2
    elif change=='layers':p['settings'][state]['layers'].reverse()
    elif change=='source_weights':p['sources'][state]['weights']['sha256']='b'*64
    elif change=='seed':p['seed']=1
    elif change=='grid':p['grid'].reverse()
    elif change=='initialization':p['sources'][state]['initialization']='pretrained'
    elif change=='norm_reference':p['settings'][state]['reference']='post_ln_1'
    elif change=='runtime_code':p['runtime']['code_files']['src/sinklab/mechanism_injection.py']='c'*64
    elif change=='tf32':p['runtime']['environment']['tf32_matmul']=True
    new['sha256']=digest(p)
    with pytest.raises(ValueError):recovery.compatibility(old,new)


@pytest.mark.parametrize('change',['token','mask','order','document','tokenizer'])
def test_portable_panel_cannot_change_scientific_membership(change):
    old={'preparation':{'corpus':{'path':'old','sha256':'a'},'registered_panels':{'path':'old','sha256':'b'}},
         'tokenizer':{'model_id':'gpt2','artifact_sha256':'c'},
         'discovery':[{'id':'one','input_ids':[1,2],'attention_mask':[True,True],'document_hashes':['original']},
                      {'id':'two','input_ids':[2,3],'attention_mask':[True,True],'document_hashes':['other']}], 'confirmation':[]}
    new=copy.deepcopy(old)
    new['preparation']['corpus']['path']='new';new['preparation']['registered_panels']['path']='new'
    assert recovery.panel_view(old)==recovery.panel_view(new)
    if change=='token':new['discovery'][0]['input_ids'][0]=4
    elif change=='mask':new['discovery'][0]['attention_mask'][1]=False
    elif change=='order':new['discovery'].reverse()
    elif change=='document':new['discovery'][0]['document_hashes']=['forged']
    elif change=='tokenizer':new['tokenizer']['artifact_sha256']='changed'
    assert recovery.panel_view(old)!=recovery.panel_view(new)


def test_monitor_never_calls_uptime_or_partial_coverage_complete():
    manifest={'jobs':[{'id':'one'},{'id':'two'}]}
    statuses={'one':{'status':'complete'}}
    assert decision(manifest,statuses,True,None)=='observe'
    assert decision(manifest,statuses,False,None)=='recover_supervisor'
    statuses['two']={'status':'invalidated'}
    assert decision(manifest,statuses,False,None)=='stop_for_diagnosis'
    statuses['two']={'status':'complete'}
    assert decision(manifest,statuses,False,{'status':'INCOMPLETE'})=='recover_supervisor'
    assert decision(manifest,statuses,False,{'status':'COMPLETE_INDEPENDENTLY_AUDITED'})=='complete'


def test_numeric_comparator_detects_effect_and_structural_changes():
    old={'effect':.1,'count':300,'order':[0,1]}
    exact=numeric_diff(old,copy.deepcopy(old));assert exact['different_numeric_fields']==0
    changed=numeric_diff(old,{'effect':.11,'count':299,'order':[1,0]})
    assert changed['different_numeric_fields']==1 and len(changed['structural_differences'])==3


def test_imported_E2_is_not_emitted_as_Nodi_coverage(tmp_path,monkeypatch):
    supervisor=recovery.RecoverySupervisor.__new__(recovery.RecoverySupervisor)
    supervisor.manifest={'jobs':[{'id':'E3_one'}],'health_interval_seconds':60,'stale_warning_seconds':100}
    supervisor.root=tmp_path; supervisor.path=tmp_path/'manifest';supervisor.path.write_text('fixed')
    supervisor.current=None;supervisor.last_health=0;supervisor.health_failure=None
    supervisor.statuses={'E3_one':{'status':'complete','origin':'AdritaPC'},'E2_one':{'status':'complete','origin':'AdritaPC_dependency'}}
    monkeypatch.setattr(recovery.Supervisor,'beat',lambda self,**kwargs:write(tmp_path/'observed.json',self.statuses))
    supervisor.beat()
    assert list(read(tmp_path/'observed.json'))==['E3_one']
    assert 'E2_one' in supervisor.statuses


def test_supervisor_io_failure_does_not_launch_second_worker(tmp_path,monkeypatch):
    supervisor=recovery.RecoverySupervisor.__new__(recovery.RecoverySupervisor)
    jobs=[{'id':str(i),'phase':'E3','panel':'confirmation','state':'test'} for i in range(2)]
    supervisor.manifest={'jobs':jobs,'frozen_files':{}};supervisor.root=tmp_path
    supervisor.ledger_path=tmp_path/'state_ledger.jsonl';supervisor.statuses={};supervisor.current=None
    monkeypatch.setattr(supervisor,'beat',lambda **kwargs:None)
    monkeypatch.setattr(supervisor,'recover',lambda:None)
    monkeypatch.setattr(supervisor,'phase_report',lambda *args:None)
    calls=[]
    def fail(job):
        calls.append(job['id']);supervisor.current={'job_id':job['id'],'worker_pid':os.getpid(),'worker_created':psutil.Process().create_time()}
        supervisor.event(job,'running');raise PermissionError('durable heartbeat denied')
    monkeypatch.setattr(supervisor,'run_job',fail)
    with pytest.raises(RuntimeError,match='live worker preserved'):supervisor.run()
    assert calls==['0'] and ledger(supervisor.ledger_path)['0']['status']=='running'
    assert read(tmp_path/'supervisor_interruptions.jsonl')['status']=='LIVE_WORKER_PRESERVED'


def test_interrupted_attempt_is_retained_and_scheduled_fresh(tmp_path,monkeypatch):
    supervisor=recovery.RecoverySupervisor.__new__(recovery.RecoverySupervisor)
    job={'id':'E3_confirmation_test','phase':'E3','panel':'confirmation','state':'test'}
    attempt=tmp_path/'old_attempt';attempt.mkdir();partial=attempt/'item-00000.json';partial.write_text('preserved')
    supervisor.manifest={'jobs':[job],'reuse':{}};supervisor.root=tmp_path;supervisor.ledger_path=tmp_path/'state_ledger.jsonl'
    supervisor.statuses={job['id']:{'status':'running','attempt':str(attempt),'worker_pid':999999999,'worker_created':0}}
    monkeypatch.setattr(supervisor,'beat',lambda **kwargs:None)
    supervisor.recover()
    assert partial.read_text()=='preserved'
    assert supervisor.statuses[job['id']]['status']=='failed' and supervisor.statuses[job['id']]['retryable']
    assert supervisor.current is None


@pytest.mark.parametrize('mutation',['none','item','marker','extra','missing'])
def test_cryptographically_cached_dual_audit_requires_unchanged_bytes(tmp_path,mutation):
    bundle=tmp_path/'bundle';bundle.mkdir();item=bundle/'item-00000.json';item.write_text('verified recorded item')
    from audit_mechanistic_scientific import sha
    write(bundle/'manifest.json',{'files':{item.name:sha(item)}})
    (bundle/'COMPLETE').write_text(sha(bundle/'manifest.json'))
    if mutation=='item':item.write_text('changed measurement')
    elif mutation=='marker':(bundle/'COMPLETE').write_text('forged')
    elif mutation=='extra':(bundle/'extra.json').write_text('{}')
    elif mutation=='missing':item.unlink()
    if mutation=='none':assert recovery.check_manifest_bytes(bundle)['files'][item.name]==sha(item)
    else:
        with pytest.raises(ValueError):recovery.check_manifest_bytes(bundle)


def test_windows_actual_worker_matches_exact_attempt_and_creation(tmp_path):
    event={'attempt':str(tmp_path/'fresh'),'worker_created':100.}
    workers=[{'pid':i,'created':created,'argv':['python.exe','scripts/run_mechanistic.py','scientific','--output',str(tmp_path/name/'bundle')]} for i,name,created in
        [(1,'fresh',101.),(2,'fresh',99.),(3,'other',101.)]]
    assert [w['pid'] for w in recovery.same_attempt_workers(event,workers)]==[1]


def test_existing_scientific_child_prevents_new_work(tmp_path,monkeypatch):
    supervisor=recovery.RecoverySupervisor.__new__(recovery.RecoverySupervisor);supervisor.statuses={}
    calls=[]
    monkeypatch.setattr(recovery,'scientific_processes',lambda:[{'pid':1}])
    monkeypatch.setattr(recovery.Supervisor,'run_job',lambda self,job:calls.append(job))
    with pytest.raises(ValueError,match='another CUDA scientific worker'):supervisor.run_job({'id':'new'})
    assert calls==[]
