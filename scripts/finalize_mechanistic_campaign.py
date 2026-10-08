"""Independent final coverage audit and checksum-verified ZIP64 preservation.

Run only after supervisor and monitoring turns finish. Final archive-hash receipt
is a sidecar, avoiding a self-referential archive hash. Originals are preserved.
"""
import argparse
import math
import os
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import psutil
from audit_mechanistic_scientific import read, write, sha, envelope, audit, require
from mechanistic_campaign_supervisor import ledger, utc


def archive(root, destination):
    require(not destination.exists() and not destination.resolve().is_relative_to(root.resolve()), 'fresh external archive path')
    files = sorted(p for p in root.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
    require(all(not p.is_symlink() for p in files), 'symlink in derived archive')
    inventory = {p.relative_to(root).as_posix(): {'sha256':sha(p), 'bytes':p.stat().st_size} for p in files}
    total = sum(v['bytes'] for v in inventory.values())
    inventory_path = root/'FINAL_ARTIFACT_INVENTORY.json'
    if inventory_path.exists(): inventory_path=root/f'FINAL_ARTIFACT_INVENTORY_retry_{time.time_ns()}.json'
    write(inventory_path, {'schema_version':1,'files':inventory,'file_count':len(inventory),'bytes':total,
                           'scope':'generated campaign artifacts only; source models/corpus/historical trees are identities, not archive members',
                           'late_sidecars':'final archive checksum/completion receipts are outside their own hash inventory'})
    require(psutil.disk_usage(str(destination.parent)).free > total + 2**30, 'insufficient space for archive and extraction check')
    files.append(inventory_path)
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for path in files:
            name = path.relative_to(root).as_posix()
            z.write(path,name)
            if name in inventory: require(sha(path) == inventory[name]['sha256'], 'artifact changed during archive')
    with zipfile.ZipFile(destination) as z:
        require(z.testzip() is None, 'archive CRC failure')
        require(len(z.namelist()) == len(set(z.namelist())) == len(files), 'archive membership')
        import hashlib
        for name,item in inventory.items():
            with z.open(name) as stream:
                require(hashlib.file_digest(stream,'sha256').hexdigest() == item['sha256'], 'archive per-file SHA')
        # Test extraction of representative nonempty files; every entry is already
        # streamed and hash verified above, avoiding a duplicate bulk result tree.
        chosen = sorted(inventory, key=lambda n: inventory[n]['bytes'], reverse=True)[:1]
        chosen += [n for n in ('CAMPAIGN_MANIFEST.json','CAMPAIGN_HANDOFF.json','SCIENTIFIC_FINAL_AUDIT.json') if n in inventory]
        with tempfile.TemporaryDirectory(prefix='mechanistic-extraction-',dir=destination.parent) as temp:
            for name in chosen:
                path = Path(z.extract(name,temp)); require(sha(path) == inventory[name]['sha256'], 'test extraction SHA')
    return {'path':str(destination),'sha256':sha(destination),'bytes':destination.stat().st_size,
            'CRC':'PASS','all_file_hashes':'PASS','test_extraction':'PASS','file_count':len(files),
            'inventory_sha256':sha(inventory_path)}


def finalize(manifest_path, destination):
    manifest = envelope(read(manifest_path)); root=Path(manifest['root'])
    ready=read(root/'FINALIZATION_READY.json')
    require((root/'SCIENTIFIC_FINDINGS.md').exists(), 'monitor scientific interpretation missing')
    heartbeat=read(root/'heartbeat.json')
    alive=psutil.pid_exists(heartbeat['supervisor_pid']) and psutil.Process(heartbeat['supervisor_pid']).create_time()==heartbeat.get('supervisor_created')
    require(not alive, 'supervisor still active; preserve stable final snapshot')
    states=ledger(root/'state_ledger.jsonl')
    results, failures = {}, []
    for job in manifest['jobs']:
        event=states.get(job['id'],{'status':'pending'})
        if event['status'] != 'complete':
            failures.append({'job_id':job['id'],**event});continue
        attempt=Path(event['attempt'])
        runner=subprocess.run([manifest['python'],'scripts/run_mechanistic.py','verify','--output',str(attempt/'bundle')],
                               cwd=manifest['repo'],env={**os.environ,**manifest['environment']},capture_output=True,text=True)
        require(runner.returncode == 0, 'final runner verification failed '+job['id'])
        receipt=audit(attempt/'bundle',read(job['lock']),job)
        require(receipt['manifest_sha256'] == read(attempt/'INDEPENDENT_AUDIT.json')['manifest_sha256'], 'previous audit differs')
        results[job['id']]=receipt
    reports={}
    for phase in ('E1','E2','E3'):
        for panel in ('discovery','confirmation'):
            candidates=sorted((root/'reports').glob(f'{phase}_{panel}*/REPORT_COMPLETE.json'))
            valid=[]
            for p in candidates:
                r=read(p)
                require(r['protocol_sha256'] == manifest['approved_lock_hashes'][phase], 'report lock identity')
                for name,value in r['files'].items(): require(sha(p.parent/name) == value,'report/export SHA')
                valid.append(str(p))
            reports[f'{phase}_{panel}']=valid
    status={}
    coverage={}
    for phase in ('E1','E2','E3'):
        jobs=[j for j in manifest['jobs'] if j['phase']==phase]
        n=sum(j['id'] in results for j in jobs)
        coverage[phase]={'verified':n,'expected':len(jobs),
            'panels':{p:sum(j['id'] in results and j['panel']==p for j in jobs) for p in ('discovery','confirmation')}}
        status[phase]='COMPLETE' if n==len(jobs) and all(reports[f'{phase}_{p}'] for p in ('discovery','confirmation')) else 'PARTIAL_BLOCKED'
        if any(states.get(j['id'],{}).get('status')=='invalidated' for j in jobs): status[phase]='FAILED_INTEGRITY'
    for job in manifest['jobs']:
        if job['id'] in results and job.get('dependency'): require(job['dependency'] in results,'missing E2 evidence for E3')
    audit_receipt={'utc':utc(),'phase_status':status,'coverage':coverage,'verified_bundles':results,'failures':failures,'reports':reports,
                   'approved_lock_sha256':manifest['approved_lock_hashes'],'source_commit':manifest['scientific_source_commit'],
                   'runtime_sha256':manifest['runtime_sha256'],'code_sha256':manifest['code_sha256'],
                   'max_parity_error':max((r['max_parity_absolute_error'] for r in results.values()),default=None),
                   'max_geometry_error':max((r['max_geometry_error'] for r in results.values()),default=None),
                   'max_norm_error':max((r['max_norm_absolute_error'] for r in results.values()),default=None),
                   'engineering_gate_scope':'independently verified recorded scalar algebra and declared policies plus runtime live-tensor gates; no dumped tensor reconstruction',
                   'constraints':manifest['constraints'],'final_git':ready.get('final_git'),'scientific_findings_sha256':sha(root/'SCIENTIFIC_FINDINGS.md')}
    audit_path=root/'SCIENTIFIC_FINAL_AUDIT.json'
    if audit_path.exists():audit_path=root/f'SCIENTIFIC_FINAL_AUDIT_retry_{time.time_ns()}.json'
    write(audit_path,audit_receipt)
    try:archive_receipt=archive(root,destination)
    except ValueError as error:
        if str(error)!='insufficient space for archive and extraction check':raise
        archive_receipt={'status':'NOT_CREATED_DISK_HEADROOM','reason':str(error),'original_artifacts_preserved':True,
                         'recovery':'when enough external disk is available, create fresh ZIP64 from the sealed inventory; do not delete sources'}
    completion={**{k:v for k,v in audit_receipt.items() if k!='verified_bundles'},'archive':archive_receipt,
                'scientific_final_audit_sha256':sha(audit_path),'scientific_final_audit_path':str(audit_path),
                'conclusion':'E1–E3 SCIENTIFIC EXECUTION COMPLETE AND INDEPENDENTLY AUDITED' if all(s=='COMPLETE' for s in status.values()) else 'Partial campaign; see coverage and blockers',
                'preservation':'original directories retained; archive contains generated artifacts only; final archive-hash/completion sidecars follow the immutable archived inventory'}
    write(root/'MECHANISTIC_E1_E3_SCIENTIFIC_COMPLETION.json',completion)
    write(destination.with_suffix('.verification.json'),completion)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args()
    finalize(args.manifest,args.archive)
