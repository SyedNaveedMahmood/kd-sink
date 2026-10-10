"""Independent admission of the E3 migration plan, before scientific launch.

No recovery-supervisor or numerical-executor imports. Source/model SHA checks,
authority, panel equality, audit coverage and actual runtime are checked here.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):result.update(block)
    return result.hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def envelope(document):
    require(document['sha256']==digest(document['payload']),'envelope digest mismatch')
    return document['payload']
def write(path,value):
    with Path(path).open('xb') as stream:stream.write(canonical(value)+b'\n')


def verify(path,root,output):
    document=read(path);plan=envelope(document);receipts={}
    for name,expected in plan['frozen_files'].items():require(sha(name)==expected,'frozen file changed: '+name)
    receipts['frozen_files']=len(plan['frozen_files'])
    old_doc,new_doc=read(plan['original_lock']),read(plan['successor_lock']);old,new=envelope(old_doc),envelope(new_doc)
    require(old_doc['sha256']=='5673eaefc30ff238ee338b0cff973032f6f2579e985592f0c837a04c6f24cf33','original E3 approval')
    require(new['status']=='approved' and new['production_ready'] and new['approval']['authority']=='researcher' and
            sha(new['approval']['record'])==new['approval']['record_sha256'],'successor authority')
    science=copy.deepcopy(old);relocated=copy.deepcopy(new)
    for value in (science,relocated):
        for field in ('status','production_ready','approval','candidate_version','runtime','qualification','run_id','panel','prospective_amendment','migration'):value.pop(field,None)
        value['selection_receipt'].pop('path')
        for source in value['sources'].values():
            for key in ('config','weights','manifest'):
                if key in source:source[key].pop('path')
    require(science==relocated,'scientific field change')
    receipts['scientific_field_diff']=[]
    for state,source in new['sources'].items():
        for key in ('config','weights','manifest'):
            if key in source:require(sha(source[key]['path'])==source[key]['sha256']==old['sources'][state][key]['sha256'],'original source bytes: '+state+'/'+key)
        if state!='teacher':
            m=read(source['manifest']['path']);complete=Path(source['manifest']['path']).parent/'COMPLETE'
            require(complete.read_text().strip()==source['manifest']['sha256'] and m['identity']==source['identity'] and
                    m['files']['model.safetensors']==source['weights']['sha256'] and m['step']==int(state.split('step')[1]) and
                    source['initialization']=='configuration_random' and m['identity']['seed']==0,'source checkpoint identity')
    receipts['verified_original_models']=len(new['sources'])
    # Independently bind every installed scientific package and real device.
    import torch
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    props=torch.cuda.get_device_properties('cuda:0');runtime=new['runtime']
    require(runtime['code_files']==old['runtime']['code_files'] and runtime['code_sha256']==digest(runtime['code_files']),'executor changed')
    require(runtime['environment']['python']==platform.python_version() and runtime['environment']['platform']==platform.platform() and
            all(importlib.metadata.version(k)==v for k,v in runtime['environment']['packages'].items()) and
            runtime['environment']['torch_cuda']==torch.version.cuda,'installed runtime mismatch')
    require(runtime['hardware']=={'device':'cuda:0','name':props.name,'uuid':str(props.uuid),'total_memory_bytes':props.total_memory} and
            runtime['precision']=='float32' and runtime['backend']=='eager' and not runtime['environment']['tf32_matmul'] and
            not runtime['environment']['tf32_cudnn'],'actual hardware/numerics')
    query=subprocess.run(['nvidia-smi','--query-gpu=uuid,driver_version,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    require(plan['gpu_uuid'] in query.stdout,'physical GPU UUID')
    receipts['actual_runtime_sha256']=digest(runtime);receipts['nvidia_smi']=query.stdout.strip()
    qual=read(new['qualification']['path']);require(sha(new['qualification']['path'])==new['qualification']['sha256'],'qualification bytes')
    require(qual['status']=='qualified' and qual['runtime']==runtime and qual['settings_sha256']==digest(new['settings']) and
            qual['parity_passed'] and qual['headroom_passed'] and qual['checkpoint_grid_hashes_verified']==15,'fresh production qualification')
    qual_complete=read(root/'qualification01/QUALIFICATION_COMPLETE.json')
    for name,expected in qual_complete['files'].items():require(sha(root/'qualification01'/name)==expected,'qualification manifest member')
    for phase in ('E1','E2','E3'):
        for role in ('student','teacher'):
            receipt=read(root/f'QUALIFICATION_AUDIT_{phase}_{role}.json');require(receipt['status']=='PASS' and receipt['items']==2,'fresh independent qualification audit')
    receipts['production_shape_qualification']='PASS; six independent phase/model audits'
    original=read(plan['original_panel']);local=read(new['panel']['path'])
    require(sha(plan['original_panel'])==old['panel']['sha256'] and sha(new['panel']['path'])==new['panel']['sha256'],'panel file pins')
    before,after=copy.deepcopy(original),copy.deepcopy(local)
    for value in (before,after):
        for key in ('corpus','registered_panels'):value['preparation'][key].pop('path')
    require(before==after and len(local['discovery'])==len(local['confirmation'])==300,'fixed600 changed')
    strict=read(root/'STRICT_PANEL_RECONSTRUCTION.json')
    require(strict['exit_code']==0 and strict['exact_prepared_panel_match'] and read(root/'strict_panel01/panel.json')==local,'strict corpus admission')
    receipts['fixed600_strict_corpus_reconstruction']='PASS; original source ownership/text disjointness independently reconstructed'
    inventory=read(root/'reconstruction01/ALL_92_JOBS.json')['jobs']
    require(len(inventory)==92 and len({r['job_id'] for r in inventory})==92,'original92 coverage')
    for phase,expected in (('E1',28),('E2',32)):
        require(sum(r['phase']==phase and r['status']=='independently_complete_valid' for r in inventory)==expected,'original phase incomplete')
    expected_jobs=[f"E3_{p}_{s.replace('/','-')}" for p in ('discovery','confirmation') for s in old['grid']]
    require([j['id'] for j in plan['jobs']]==expected_jobs and len(expected_jobs)==32 and len(plan['E2_dependencies'])==32,'E3 exact32 manifest')
    for job in plan['jobs']:
        require(job['phase']=='E3' and job['seed']==0 and job['argv']==[plan['python'],'scripts/run_mechanistic.py','scientific','--phase','E3','--seed','0','--state',job['state'],'--panel',job['panel'],'--device','cuda:0','--disable-tf32','--lock',plan['successor_lock'],'--approved-sha256',new_doc['sha256'],'--output','{OUTPUT}'],'explicit single-state command')
        require(job['dependency']==job['id'].replace('E3_','E2_',1),'E2 dependency identity')
        dependency=plan['E2_dependencies'][job['dependency']];receipt=read(dependency['audit_path']);identity=receipt['identity']
        require(sha(dependency['audit_path'])==dependency['audit_sha256'] and receipt['status']=='PASS' and receipt['items']==300 and
            identity['phase']=='E2' and identity['state']==job['state'] and identity['panel']==job['panel'] and identity['seed']==0 and
            identity['protocol_sha256']=='1ba9dff94bdd6735cb1cd1c871a655094a91bd1b96c746e9f26f14f942c3d586' and
            identity['panel_sha256']==old['panel']['sha256'] and identity['runtime_sha256']==digest(old['runtime']) and
            identity['source']['weights_sha256']==new['sources'][job['state']]['weights']['sha256'],'matching original E2 proof')
        if job['id'] in plan['reuse']:
            proof=plan['reuse'][job['id']];receipt=read(proof['audit_path']);identity=receipt['identity']
            require(sha(proof['audit_path'])==proof['audit_sha256'] and receipt['status']=='PASS' and receipt['items']==300 and
                identity['phase']=='E3' and identity['state']==job['state'] and identity['panel']==job['panel'] and
                identity['protocol_sha256']==old_doc['sha256'] and identity['runtime_sha256']==digest(old['runtime']) and
                identity['source']['weights_sha256']==new['sources'][job['state']]['weights']['sha256'] and
                receipt['manifest_sha256']==sha(Path(proof['attempt'])/'bundle/manifest.json'),'original reuse provenance')
    require(len(plan['reuse'])==17,'expected reusable17 inventory')
    deps=read(root/'ARCHIVED_E2_E3_DEPENDENCY_PARITY.json');require(deps['status']=='PASS' and len(deps['jobs'])==17,'complete archived dependency parity')
    bridge=read(plan['comparability_receipt']);require(sha(plan['comparability_receipt'])==plan['comparability_sha256'] and bridge['status']=='PASS' and
        bridge['fixed_item_count']==12 and all(x['all_measurements']['different_numeric_fields']==0 and not x['all_measurements']['structural_differences'] for x in bridge['items']),'cross-machine comparator')
    receipts['comparison_numeric_fields']=sum(x['all_measurements']['numeric_fields'] for x in bridge['items'])
    tests=ET.parse(root/'migration-focused-tests-v3.xml');suites=tests.findall('.//testsuite')
    test_count=sum(int(s.get('tests','0')) for s in suites)
    require(test_count>=66 and all(s.get('failures')=='0' and s.get('errors')=='0' for s in suites),'focused tests')
    receipts['focused_tests']=str(test_count)+' passed'
    report=read(root/'portable_join_qualification02/REPORT_COMPLETE.json')
    require(report['status']=='COMPLETE_AUDITED_MIXED_RUNTIME_EXPORT','portable scientific report qualification')
    for name,expected in report['files'].items():require(sha(root/'portable_join_qualification02'/name)==expected,'portable report pin')
    receipts['portable_report']='16-state discovery join validated against own original seals and all E2 dependencies'
    archive=read(root/'ARCHIVE_VERIFICATION.json')
    # Archive verification is descriptive evidence, not a claim of a new original seal.
    require(archive['status']=='PASS' and archive['zip64_valid'] and archive['crc_all_members_passed'] and
            archive['original_inventory_present'] and archive['source_archive_unchanged'] and
            sha(archive['archive'])==archive['sha256'],'full source archive verification')
    receipts['archive_verification_sha256']=sha(root/'ARCHIVE_VERIFICATION.json')
    write(output,{'status':'PASS','verdict':'READY_FOR_PORTABLE_E3_CONTINUATION','execution_plan_sha256':document['sha256'],
        'successor_lock_sha256':new_doc['sha256'],'reusable_E3_bundles':17,'new_E3_bundles':15,
        'first_remaining':'E3_confirmation_C1-step500','verification_receipts':receipts,
        'independence':'separate process and implementation; no recovery/numerical-executor imports; standard SHA/runtime/coverage checks plus separate scientific auditor receipts'})
    print('INDEPENDENT READY_FOR_PORTABLE_E3_CONTINUATION',new_doc['sha256'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for key in ('plan','root','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();verify(args.plan.resolve(),args.root.resolve(),args.output.resolve())
