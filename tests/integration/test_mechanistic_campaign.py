"""Operational recovery and independent arithmetic checks on real tiny bundles."""
import copy
import os
from pathlib import Path
import shutil
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts'))
import audit_mechanistic_scientific as independent
from mechanistic_campaign_supervisor import Supervisor, append, transient, check_pins, exclusive, validate_manifest
import mechanistic_campaign_supervisor as campaign
from report_mechanistic_scientific import distribution,matched_effects
from run_mechanistic import engineering_fixture
from sinklab.mechanistic_run import run_state, VERSION
from sinklab.mechanism_trace import MechanisticGPT2Adapter
from sinklab.provenance import payload_digest


@pytest.fixture(scope='module')
def bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp('campaign-engineering')
    for phase in ('E1','E2','E3'):
        model, items, settings = engineering_fixture(phase, 'cpu')
        identity = {'schema_version':VERSION,'phase':phase,'seed':0,'state':'synthetic_random_gpt2',
                    'panel':'synthetic','context_length':128,'run_id':'campaign-tests','engineering_only':True}
        run_state(MechanisticGPT2Adapter(model),items,phase=phase,settings=settings,identity=identity,output=root/phase)
    return root


@pytest.mark.parametrize('phase', ['E1','E2','E3'])
def test_independent_real_bundle(bundles, phase):
    receipt = independent.audit(bundles/phase)
    assert receipt['status'] == 'PASS' and receipt['items'] == 2
    assert receipt['max_geometry_error'] < 1e-10


def reseal_bundle(root):
    m = independent.read(root/'manifest.json')
    m['files'] = {name: independent.sha(root/name) for name in m['files']}
    (root/'manifest.json').write_bytes(independent.canonical(m))
    (root/'COMPLETE').write_text(independent.sha(root/'manifest.json'))


@pytest.mark.parametrize('mutation', ['geometry','coverage','no_op','missing_item','nan','failed','norm','factor'])
def test_adversarial_bundle_rejected(bundles, tmp_path, mutation):
    root = tmp_path/'bundle'; shutil.copytree(bundles/'E3',root)
    r = independent.read(root/'item-00000.json')
    if mutation == 'geometry': r['operations'][0]['geometry']['target_records'][0]['log_normalizer_change'] += .01
    elif mutation == 'coverage': r['operations'].pop()
    elif mutation == 'no_op':
        row = next(o for o in r['operations'] if o['operation'].endswith('/eta0'))
        row['behavior']['flip_count'] = 1
    elif mutation == 'missing_item': r['item_id'] = 'wrong'
    elif mutation == 'nan': r['operations'][0]['behavior']['self_kl_sum_nats'] = float('nan')
    elif mutation == 'failed': (root/'FAILED.json').write_text('{}')
    elif mutation == 'norm':
        row = next(o for o in r['operations'] if 'injection/sink' in o['operation'])
        row['injection']['directions']['sink']['requested_norm_sum'] += 1
    elif mutation == 'factor':r['operations'][0]['factors']['all_q_ge1']['projected_delta_norm_mean'] += 1
    (root/'item-00000.json').write_text(__import__('json').dumps(r))
    reseal_bundle(root)
    with pytest.raises(ValueError): independent.audit(root)


@pytest.mark.parametrize('message', ['parity mismatch','hash mismatch','CUDA out of memory','No module named foo','runtime lock mismatch','geometry failure'])
def test_integrity_failures_never_retry(message):
    assert not transient(message)


@pytest.mark.parametrize('message', ['CUDA initialization error','CUDA driver initialization failed','WinError 1450'])
def test_bounded_transient_classification(message):
    assert transient(message)


def test_source_change_rejected(tmp_path):
    path=tmp_path/'code.py'; path.write_text('first')
    manifest={'frozen_files':{str(path):independent.sha(path)}}
    check_pins(manifest)
    path.write_text('second')
    with pytest.raises(ValueError): check_pins(manifest)


def test_exclusive_lock(tmp_path):
    with exclusive(tmp_path/'gpu.lock'):
        with pytest.raises(RuntimeError):
            with exclusive(tmp_path/'gpu.lock'): pass


def test_atomic_windows_reader_sharing_preserves_complete_snapshot(tmp_path, monkeypatch):
    import threading
    import time
    path = tmp_path/'heartbeat.json'
    campaign.atomic(path, {'version': 1})
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.CreateFileW(str(path), 0x80000000, 0, None, 3, 0, None)
        assert handle != ctypes.c_void_p(-1).value
        def release():
            time.sleep(.15)
            assert kernel.CloseHandle(handle)
        reader = threading.Thread(target=release)
        reader.start()
        try:
            campaign.atomic(path, {'version': 2})
        finally:
            reader.join()
    else:
        original = campaign.os.replace
        calls = []
        def sharing(source, target):
            calls.append(True)
            if len(calls) < 3:
                assert independent.read(path) == {'version': 1}
                raise PermissionError('reader temporarily denies delete sharing')
            return original(source, target)
        monkeypatch.setattr(campaign.os, 'replace', sharing)
        campaign.atomic(path, {'version': 2})
    assert independent.read(path) == {'version': 2}


def test_atomic_permanent_denial_is_bounded_and_preserves_old_file(tmp_path, monkeypatch):
    path = tmp_path/'heartbeat.json'
    campaign.atomic(path, {'version': 1})
    def denied(*args):
        raise PermissionError('persistent denial')
    ticks = iter([0., 0., 3.])
    monkeypatch.setattr(campaign.os, 'replace', denied)
    monkeypatch.setattr(campaign.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(campaign.time, 'sleep', lambda _: None)
    with pytest.raises(PermissionError, match='persistent denial'):
        campaign.atomic(path, {'version': 2})
    assert independent.read(path) == {'version': 1}


def test_supervisor_io_failure_preserves_live_worker_and_stops_future_jobs(tmp_path, monkeypatch):
    import psutil
    jobs = [{'id':f'E1_test_{n}', 'phase':'E1', 'panel':'synthetic', 'state':'synthetic'} for n in range(2)]
    manifest = {'root':str(tmp_path), 'repo':str(REPO), 'python':sys.executable,
                'jobs':jobs, 'frozen_files':{}, 'health_interval_seconds':60,
                'stale_warning_seconds':1800, 'poll_seconds':.05}
    path=tmp_path/'manifest.json'
    independent.write(path, {'schema_version':1, 'payload':manifest, 'sha256':payload_digest(manifest)})
    supervisor = Supervisor(path)
    monkeypatch.setattr(supervisor, 'beat', lambda **kwargs: None)
    calls = []
    def interrupted(job):
        calls.append(job['id'])
        supervisor.current = {'job_id':job['id'], 'attempt':str(tmp_path/'attempt_001'),
                              'worker_pid':os.getpid(), 'worker_created':psutil.Process().create_time()}
        supervisor.event(job, 'running', **{k:v for k,v in supervisor.current.items() if k != 'job_id'})
        raise PermissionError('heartbeat replacement denied')
    monkeypatch.setattr(supervisor, 'run_job', interrupted)
    with pytest.raises(RuntimeError, match='live worker; ownership preserved'):
        supervisor.run()
    assert calls == [jobs[0]['id']]
    assert campaign.ledger(tmp_path/'state_ledger.jsonl')[jobs[0]['id']]['status'] == 'running'
    assert jobs[1]['id'] not in campaign.ledger(tmp_path/'state_ledger.jsonl')
    evidence = independent.read(tmp_path/'supervisor_interruptions.jsonl')
    assert evidence['status'] == 'LIVE_WORKER_PRESERVED'
    assert evidence['current']['worker_pid'] == os.getpid()


def test_recovery_reverifies_complete_worker(bundles,tmp_path):
    attempt=tmp_path/'attempts/E1_synthetic/attempt_001'
    attempt.mkdir(parents=True);shutil.copytree(bundles/'E1',attempt/'bundle')
    job={'id':'E1_synthetic','phase':'E1','state':'synthetic_random_gpt2','panel':'synthetic'}
    manifest={'root':str(tmp_path),'repo':str(REPO),'python':sys.executable,
              'environment':{'PYTHONPATH':str(REPO/'src'),'HF_HUB_OFFLINE':'1'},'jobs':[job],
              'health_interval_seconds':60,'stale_warning_seconds':1800,'poll_seconds':.05,'frozen_files':{}}
    path=tmp_path/'manifest.json';independent.write(path,{'schema_version':1,'payload':manifest,'sha256':payload_digest(manifest)})
    append(tmp_path/'state_ledger.jsonl',{**job,'job_id':job['id'],'status':'running','attempt':str(attempt),'worker_pid':999999999,'worker_created':0})
    supervisor=Supervisor(path);supervisor.recover()
    assert supervisor.statuses[job['id']]['status']=='complete'
    assert supervisor.statuses[job['id']]['worker_exit_code'] is None
    assert (attempt/'INDEPENDENT_AUDIT.json').exists()
    supervisor.recover()  # re-audit, no destructive replacement
    (attempt/'bundle/summary.json').write_text('{}')
    supervisor.recover()
    assert supervisor.statuses[job['id']]['status']=='invalidated'


def test_e3_requires_e2_verified_state(tmp_path):
    manifest={'root':str(tmp_path),'repo':str(REPO),'python':sys.executable,'jobs':[], 'frozen_files':{},
              'health_interval_seconds':60,'stale_warning_seconds':1800,'poll_seconds':.05}
    path=tmp_path/'manifest.json';independent.write(path,{'schema_version':1,'payload':manifest,'sha256':payload_digest(manifest)})
    supervisor=Supervisor(path)
    job={'id':'E3_test','phase':'E3','state':'C2/step500','panel':'discovery','dependency':'E2_same'}
    supervisor.run_job(job)
    assert supervisor.statuses['E3_test']['status']=='blocked'
    assert not (tmp_path/'attempts').exists()


def test_distribution_keeps_null_negative_and_linear_quantiles():
    result=distribution([None,-2,0,2,4])
    assert result == {'available_items':4,'median':1,'q25':-.5,'q75':2.5,'iqr':3.,'p90':3.4000000000000004}


def test_matched_effects_only_use_already_observed_dose_brackets():
    fields=('delta_ce_nats','self_kl_nats','prediction_flip_fraction','absolute_target_logprob_change_nats')
    teacher={'operations':{f'q_bias/mapped_teacher/{a:g}':{'behavior':dict.fromkeys(fields,v)} for a,v in ((0,0),(.25,4))}}
    student={'operations':{f'q_bias/native/{a:g}':{'behavior':dict.fromkeys(fields,v)} for a,v in ((.5,1),(.75,3))}}
    join={'states':{'teacher':{'summary':teacher},'C2/step500':{'summary':student}}}
    match={'status':'observed_overlap','state':'C2/step500','route':'q_bias',
           'left_candidates':[{'bracket':[0,.25],'weight':.5}], 'right_candidates':[{'bracket':[.5,.75],'weight':.25}]}
    result=matched_effects(join,[match,{'status':'dose_unmatched'}])
    assert result[0]['student_minus_teacher']['delta_ce_nats']==-.5
    assert result[1]=={'status':'dose_unmatched'}


def test_archive_crc_hashes_extraction_and_original_preservation(tmp_path):
    from finalize_mechanistic_campaign import archive
    root=tmp_path/'derived';root.mkdir()
    independent.write(root/'CAMPAIGN_MANIFEST.json',{'source_model_reference':'weights stay outside this derived tree'})
    nested=root/'reports';nested.mkdir();(nested/'figure.svg').write_text('<svg/>')
    before=independent.sha(root/'CAMPAIGN_MANIFEST.json')
    result=archive(root,tmp_path/'archive.zip')
    assert result['CRC']==result['all_file_hashes']==result['test_extraction']=='PASS'
    assert result['sha256']==independent.sha(tmp_path/'archive.zip')
    assert independent.sha(root/'CAMPAIGN_MANIFEST.json')==before
    assert (nested/'figure.svg').exists()


def test_archive_never_overwrites_existing_attempt(tmp_path):
    from finalize_mechanistic_campaign import archive
    root=tmp_path/'derived';root.mkdir();destination=tmp_path/'existing.zip';destination.write_bytes(b'previous')
    with pytest.raises(ValueError):archive(root,destination)
    assert destination.read_bytes()==b'previous'


@pytest.mark.parametrize('mutation',['none','missing','seed','tf32','dependency','order'])
def test_scientific_manifest_exact_grid_and_command_gate(tmp_path,mutation):
    manifest={'scientific':True,'python':sys.executable,'approved_lock_hashes':{},'jobs':[]}
    for phase in ('E1','E2','E3'):
        grid=['teacher',*[f'{c}/step{s}' for c in ('C1','C2','C3','C5','C6') for s in
                         ((500,10000) if phase=='E1' and c in ('C1','C5') else (500,2000,10000))]]
        payload={'status':'approved','production_ready':True,'seed':0,'grid':grid}
        lock=tmp_path/f'{phase}.json';independent.write(lock,{'schema_version':1,'payload':payload,'sha256':payload_digest(payload)})
        manifest['approved_lock_hashes'][phase]=payload_digest(payload)
        for panel in ('discovery','confirmation'):
            for state in grid:
                job={'id':f'{phase}_{panel}_{state.replace("/","-")}','phase':phase,'panel':panel,'state':state,'seed':0,'lock':str(lock)}
                job['argv']=[sys.executable,'scripts/run_mechanistic.py','scientific','--phase',phase,'--seed','0','--state',state,'--panel',panel,'--device','cuda:0','--disable-tf32','--lock',str(lock),'--approved-sha256',payload_digest(payload),'--output','{OUTPUT}']
                if phase=='E3':job['dependency']=f'E2_{panel}_{state.replace("/","-")}'
                manifest['jobs'].append(job)
    if mutation=='missing':manifest['jobs'].pop()
    elif mutation=='seed':manifest['jobs'][0]['argv'][6]='1'
    elif mutation=='tf32':manifest['jobs'][0]['argv'].remove('--disable-tf32')
    elif mutation=='dependency':manifest['jobs'][-1]['dependency']='wrong'
    elif mutation=='order':manifest['jobs'].reverse()
    if mutation=='none':validate_manifest(manifest)
    else:
        with pytest.raises(ValueError):validate_manifest(manifest)
