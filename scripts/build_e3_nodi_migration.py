"""Seal the explicitly authorized E3-only migration plan from verified evidence."""
import argparse
import copy
from pathlib import Path
import statistics
import subprocess

from audit_mechanistic_scientific import read,write as write_exclusive,sha,digest,envelope,require
from e3_nodi_recovery import compatibility,validate_recovery,ORIGINAL_E3


def seal(value): return {'schema_version':1,'payload':value,'sha256':digest(value)}


def write(path,value):
    if path.exists():require(read(path)==value,'existing migration evidence differs; choose a fresh version: '+str(path))
    else:write_exclusive(path,value)


def build(repo,root,setup,plan_output):
    authority=root/'RESEARCHER_MIGRATION_AUTHORITY.md'
    authority.write_text('# Researcher authorization received in this Codex session\n\n'
        '2026-10-10: CODEX — KD-SINK E3 EMERGENCY MIGRATION AND RESUMPTION.\n\n'
        'Exact excerpts from the current instruction: “The researcher explicitly authorizes migration-related amendments and superseding execution procedures.” '
        '“You are explicitly authorized to create a NodiPC-specific successor execution envelope.” '
        '“Once a READY verdict has been independently established, proceed with the selected validated E3 continuation or rerun plan without waiting for an additional administrative approval.”\n\n'
        'Authority applies to documented E3 migration and continuation, preserving original scientific inputs and definitions. '
        'This is a record of an instruction received in-session, not a fabricated historical signature or preregistration. '
        'Historical approved protocols and Adrita evidence remain unchanged.\n',encoding='utf-8')
    old_path=root/'imported_adrita/campaign/input_provenance/E3.approved.json'
    old=read(old_path);protocol=copy.deepcopy(envelope(read(setup/'candidates_for_researcher_approval/E3_new_pc_v1.candidate.json')))
    qualification=root/'qualification01/E3_QUALIFICATION.json'
    protocol.update(status='approved',production_ready=True,candidate_version='E3-NodiPC-migration-v1',
        run_id='mechanistic-E3-NodiPC-recovery-20261010',approval={'authority':'researcher','record':str(authority),
        'record_sha256':sha(authority),'scope':'E3 scientific continuation; researcher-authorized runtime/path migration'},
        qualification={'path':str(qualification),'sha256':sha(qualification)})
    protocol['migration']={'version':1,'original_protocol_sha256':ORIGINAL_E3,
        'authority':'researcher-authorized migration for E3 continuation',
        'scientific_field_diff':[], 'execution_code_changes':[],
        'archive_sha256':read(root/'ARCHIVE_VERIFICATION.json')['sha256'],
        'comparison_receipt':{'path':str(root/'COMPARABILITY_INDEPENDENT.json'),'sha256':sha(root/'COMPARABILITY_INDEPENDENT.json')},
        'strict_panel_reconstruction':{'path':str(root/'STRICT_PANEL_RECONSTRUCTION.json'),'sha256':sha(root/'STRICT_PANEL_RECONSTRUCTION.json')},
        'auditor_amendment':'Correct eligible versus common injection support scalar audit; all numerical executor gates and thresholds unchanged; historical failed receipts preserved.'}
    lock=seal(protocol);locks=root/'protocols';locks.mkdir(exist_ok=True)
    lock_path=locks/'E3.NodiPC.v1.approved.json';write(lock_path,lock)
    compat=compatibility(old,lock);old_protocol=envelope(old)
    relocation=[]
    for state,source in protocol['sources'].items():
        for key in ('config','weights','manifest'):
            if key in source:
                relocation.append({'state':state,'artifact':key,'original_path':old_protocol['sources'][state][key]['path'],
                    'new_path':source[key]['path'],'sha256':source[key]['sha256']})
    for key in ('selection_receipt','qualification','panel'):
        relocation.append({'artifact':key,'original':old_protocol[key],'successor':protocol[key]})
    for key in ('corpus','registered_panels'):
        relocation.append({'artifact':key,'original':read(root/'imported_adrita/campaign/input_provenance/panel.json')['preparation'][key],
            'successor':read(protocol['panel']['path'])['preparation'][key]})
    portability={'version':1,**compat,'authority':protocol['migration']['authority'],'authority_record_sha256':sha(authority),
        'original_runtime':old_protocol['runtime'],'successor_runtime':protocol['runtime'],
        'original_runtime_sha256':digest(old_protocol['runtime']),'new_runtime_sha256':digest(protocol['runtime']),
        'source_path_relocation_table':relocation,'scientific_parameters':'unchanged all source identities, seed0,600 blocks, eta0/.01/.03/.10, pre-ln_1 reference, coarse layers, control seed/support, numerical gates and both telescopes',
        'qualification':protocol['qualification'],'comparability':protocol['migration']['comparison_receipt'],
        'execution_code_changes':[], 'operational_changes':['E3-only serial recovery','separate mixed-identity join','independent norm audit population correction'],
        'reason':'Adrita unavailable; Nodi hardware UUID differs; exact original weights/corpus recovered; full strict admission retained.',
        'scientific_implications':'No changed estimand. Seed0 descriptive inference; cross-machine records explicitly labelled, never independent seed replication.',
        'limitations':['Fixed comparator sample is not proof of all-item bitwise reproducibility.','Historical metadata lacks per-query residual norms; live sample reconstructs those norms, full run retains original live per-query gates.']}
    write(root/'E3_NODIPC_PORTABILITY_v1.json',seal(portability))
    results=read(root/'reconstruction01/ALL_92_JOBS.json')['jobs'];jobs=[];deps={};reuse={}
    for row in results:
        if row['phase']=='E2':
            selected=row['selected'];deps[row['job_id']]={'attempt':selected['attempt'],'audit_path':selected['independent_audit_path'],
                'audit_sha256':sha(selected['independent_audit_path'])}
        elif row['phase']=='E3' and row['status']=='independently_complete_valid':
            selected=row['selected'];reuse[row['job_id']]={'attempt':selected['attempt'],'audit_path':selected['independent_audit_path'],
                'audit_sha256':sha(selected['independent_audit_path']), 'runner_receipt_path':selected['runner_receipt_path'],
                'runner_receipt_sha256':sha(selected['runner_receipt_path'])}
    python=str(repo/'.venv/Scripts/python.exe')
    for panel in ('discovery','confirmation'):
        for state in protocol['grid']:
            identifier=f"E3_{panel}_{state.replace('/','-')}"
            argv=[python,'scripts/run_mechanistic.py','scientific','--phase','E3','--seed','0','--state',state,'--panel',panel,
                '--device','cuda:0','--disable-tf32','--lock',str(lock_path),'--approved-sha256',lock['sha256'],'--output','{OUTPUT}']
            jobs.append({'id':identifier,'phase':'E3','state':state,'panel':panel,'seed':0,'lock':str(lock_path),
                'approved_sha256':lock['sha256'],'dependency':identifier.replace('E3_','E2_',1),'argv':argv})
    resources=read(root/'qualification01/RESOURCE_SUMMARY.json')
    evidence_files=[root/name for name in ('ARCHIVE_VERIFICATION.json','COMPARISON_PLAN.json','COMPARABILITY_INDEPENDENT.json',
        'STRICT_PANEL_RECONSTRUCTION.json','STRICT_PANEL_RECONSTRUCTION_MEASUREMENT_ADDENDUM.json','E3_NODIPC_PORTABILITY_v1.json',
        'RESEARCHER_MIGRATION_AUTHORITY.md','ARCHIVED_E2_E3_DEPENDENCY_PARITY.json','migration-focused-tests-v3.xml','EXISTING_OUTPUT_GATE.json')]
    evidence_files += [old_path,lock_path,qualification,Path(protocol['panel']['path']),Path(protocol['selection_receipt']['path']),
        root/'imported_adrita/campaign/input_provenance/panel.json',root/'reconstruction01/ALL_92_JOBS.json',root/'reconstruction01/E3_RECOVERY_MATRIX.json']
    evidence_files += [repo/name for name in ('scripts/audit_mechanistic_scientific.py','scripts/mechanistic_campaign_supervisor.py',
        'scripts/e3_nodi_recovery.py','scripts/monitor_e3_nodi_recovery.py','scripts/compare_e3_migration.py','scripts/build_e3_nodi_migration.py','scripts/verify_e3_nodi_readiness.py')]
    evidence_files += [root/f'QUALIFICATION_AUDIT_{phase}_{role}.json' for phase in ('E1','E2','E3') for role in ('teacher','student')]
    evidence_files += [Path(x['audit_path']) for x in list(deps.values())+list(reuse.values())]
    evidence_files += [Path(x['runner_receipt_path']) for x in reuse.values()]
    frozen={str(repo/name):value for name,value in protocol['runtime']['code_files'].items()}
    frozen.update({str(repo/name):value for name,value in protocol['specification_files_sha256'].items()})
    frozen.update({str(p):sha(p) for p in evidence_files})
    manifest={'schema_version':'e3-nodipc-recovery-v1','scientific':True,'repo':str(repo),'root':str(root/'campaign_nodipc'),
        'python':python,'original_lock':str(old_path),'successor_lock':str(lock_path),'approved_lock_hashes':{'E3':lock['sha256']},
        'original_panel':str(root/'imported_adrita/campaign/input_provenance/panel.json'), 'jobs':jobs,'reuse':reuse,'E2_dependencies':deps,
        'comparability_receipt':str(root/'COMPARABILITY_INDEPENDENT.json'),'comparability_sha256':sha(root/'COMPARABILITY_INDEPENDENT.json'),
        'archived_dependency_comparison':str(root/'ARCHIVED_E2_E3_DEPENDENCY_PARITY.json'),
        'archived_dependency_comparison_sha256':sha(root/'ARCHIVED_E2_E3_DEPENDENCY_PARITY.json'),
        'frozen_files':frozen,'environment':{'PYTHONPATH':str(repo/'src'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'},
        'gpu_uuid':'GPU-'+protocol['runtime']['hardware']['uuid'],'gpu_lock_path':str(root/('GPU-'+protocol['runtime']['hardware']['uuid']+'.lock')),
        'qualified_reserved_bytes':resources['peak_reserved_bytes'],'gpu_headroom_bytes':resources['required_gpu_headroom_bytes'],
        'minimum_ram_bytes':40*2**30,'minimum_disk_bytes':30*2**30,'max_attempts':2,'poll_seconds':2,
        'health_interval_seconds':15,'stale_warning_seconds':900,'monitor_poll_seconds':10,'monitor_max_supervisor_starts':3}
    validate_recovery(manifest)
    write(plan_output,seal(manifest))
    item_times=[x['wall_seconds'] for x in read(root/'comparability01/COMPARISON_RUN.json')['items'] if x['state']!='teacher']
    inference=statistics.mean(item_times)*300
    print({'lock_sha256':lock['sha256'],'reusable':len(reuse),'remaining':32-len(reuse),
        'first_remaining':next(j['id'] for j in jobs if j['id'] not in reuse),'estimated_seconds_per_bundle':inference+read(root/'STRICT_PANEL_RECONSTRUCTION.json')['wall_seconds'],
        'plan_sha256':digest(manifest)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for key in ('repo','root','setup'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--plan-output',type=Path,required=True)
    args=parser.parse_args();build(args.repo.resolve(),args.root.resolve(),args.setup.resolve(),args.plan_output.resolve())
