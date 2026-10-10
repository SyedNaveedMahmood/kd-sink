"""E3-only recovery with immutable Adrita evidence and explicit mixed provenance.

The frozen numerical executor and its strict single-state admission are unchanged.
This separately pinned operational layer authorizes only the sealed migration.
"""
import argparse
import copy
import csv
import os
from pathlib import Path
import subprocess
import time
import traceback

import psutil

from audit_mechanistic_scientific import audit, read, write, sha, digest, envelope, require
from mechanistic_campaign_supervisor import (Supervisor, append, atomic, exclusive,
    health, ledger, check_pins, verify_attempt, utc)

ORIGINAL_E3 = '5673eaefc30ff238ee338b0cff973032f6f2579e985592f0c837a04c6f24cf33'


def scientific_processes():
    """Windows venv launchers have base-interpreter children: inspect both."""
    found=[]
    for process in psutil.process_iter(['pid','create_time','cmdline']):
        try:
            command=process.info['cmdline'] or []
            if 'scripts/run_mechanistic.py' in command and 'scientific' in command and 'cuda:0' in command:
                found.append({'pid':process.pid,'created':process.info['create_time'],'argv':command})
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    return found


def same_attempt_workers(event,workers):
    target=str(Path(event['attempt'])/'bundle')
    return [w for w in workers if '--output' in w['argv'] and
            w['argv'][w['argv'].index('--output')+1]==target and
            w['created']>=event['worker_created']-.01]


def scientific_view(protocol):
    value = copy.deepcopy(protocol)
    for key in ('status', 'production_ready', 'approval', 'candidate_version', 'runtime',
                'qualification', 'run_id', 'panel', 'prospective_amendment', 'migration'):
        value.pop(key, None)
    value['selection_receipt'].pop('path')
    for source in value['sources'].values():
        for key in ('config', 'weights', 'manifest'):
            if key in source: source[key].pop('path')
    return value


def panel_view(panel):
    value = copy.deepcopy(panel)
    for key in ('corpus', 'registered_panels'):
        value['preparation'][key].pop('path')
    return value


def compatibility(original, successor):
    old, new = envelope(original), envelope(successor)
    require(original['sha256'] == ORIGINAL_E3, 'wrong original E3 approval')
    require(new['status'] == 'approved' and new['production_ready'] and
            new['approval']['authority'] == 'researcher' and new['approval']['record'], 'unapproved successor')
    require(scientific_view(old) == scientific_view(new), 'E3 scientific fields changed')
    require(old['runtime']['code_files'] == new['runtime']['code_files'] and
            old['runtime']['precision'] == new['runtime']['precision'] == 'float32' and
            old['runtime']['backend'] == new['runtime']['backend'] == 'eager' and
            not new['runtime']['environment']['tf32_matmul'] and
            not new['runtime']['environment']['tf32_cudnn'], 'unqualified numerical executor change')
    return {'scientific_field_diff': [], 'scientific_view_sha256': digest(scientific_view(old)),
            'changed_top_level_fields': sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k)),
            'original_protocol_sha256': original['sha256'], 'successor_protocol_sha256': successor['sha256']}


def validate_recovery(manifest):
    require(manifest['schema_version'] == 'e3-nodipc-recovery-v1' and manifest['scientific'], 'recovery schema')
    old, new = read(manifest['original_lock']), read(manifest['successor_lock'])
    compatibility(old, new)
    require(new['sha256'] == manifest['approved_lock_hashes']['E3'], 'successor digest')
    protocol = envelope(new)
    jobs = manifest['jobs']
    expected = [f"E3_{panel}_{state.replace('/', '-')}" for panel in ('discovery', 'confirmation') for state in protocol['grid']]
    require(len(expected) == len(set(expected)) == 32 and [j['id'] for j in jobs] == expected, 'exact32 E3 coverage/order')
    require(set(manifest['reuse']).issubset(expected) and len(manifest['E2_dependencies']) == 32, 'reuse/dependency inventory')
    for job in jobs:
        dependency = f"E2_{job['panel']}_{job['state'].replace('/', '-')}"
        argv = [manifest['python'], 'scripts/run_mechanistic.py', 'scientific', '--phase', 'E3', '--seed', '0',
                '--state', job['state'], '--panel', job['panel'], '--device', 'cuda:0', '--disable-tf32',
                '--lock', manifest['successor_lock'], '--approved-sha256', new['sha256'], '--output', '{OUTPUT}']
        require(job['phase'] == 'E3' and job['seed'] == 0 and job['state'] in protocol['grid'] and
                job['dependency'] == dependency and job['argv'] == argv and job['lock'] == manifest['successor_lock'], 'command/dependency changed')
        evidence = manifest['E2_dependencies'][dependency]
        receipt = read(evidence['audit_path'])
        identity = receipt['identity']
        require(sha(evidence['audit_path']) == evidence['audit_sha256'] and receipt['status'] == 'PASS' and
                receipt['items'] == 300 and identity['phase'] == 'E2' and identity['state'] == job['state'] and
                identity['panel'] == job['panel'] and identity['seed'] == 0 and not identity['engineering_only'] and
                identity['source']['weights_sha256'] == protocol['sources'][job['state']]['weights']['sha256'], 'unverified matching E2 dependency')
    original_panel, local_panel = read(manifest['original_panel']), read(protocol['panel']['path'])
    require(sha(manifest['original_panel']) == envelope(old)['panel']['sha256'] and
            sha(protocol['panel']['path']) == protocol['panel']['sha256'] and panel_view(original_panel) == panel_view(local_panel), 'fixed600 panel changed')
    bridge = read(manifest['comparability_receipt'])
    require(sha(manifest['comparability_receipt']) == manifest['comparability_sha256'] and
            bridge['status'] == 'PASS' and bridge['scientific_effect_agreement'] != 'not_assessed', 'missing independent comparability gate')


def selected_audit(manifest, job, event, heartbeat=None):
    imported = event.get('origin') == 'AdritaPC'
    lock = read(manifest['original_lock'] if imported else manifest['successor_lock'])
    return audit(Path(event['attempt']) / 'bundle', lock, job, heartbeat=heartbeat,
                 panel_path=manifest['original_panel'] if imported else None)


def check_manifest_bytes(bundle):
    """Revalidate every byte covered by already completed dual audits."""
    bundle=Path(bundle);record=read(bundle/'manifest.json')
    require(not (bundle/'FAILED.json').exists() and
            (bundle/'COMPLETE').read_text().strip()==sha(bundle/'manifest.json'),'unsealed or failed imported bundle')
    names=set(record['files'])
    require({p.name for p in bundle.iterdir()}==names|{'manifest.json','COMPLETE'},'unmanifested/missing imported evidence')
    for name,expected in record['files'].items():
        require(Path(name).name==name and not (bundle/name).is_symlink() and sha(bundle/name)==expected,'imported evidence bytes changed: '+name)
    return record


def verified_import(manifest,job,proof):
    bundle=Path(proof['attempt'])/'bundle';record=check_manifest_bytes(bundle)
    receipt=read(proof['audit_path']);runner=read(proof['runner_receipt_path'])
    require(sha(proof['audit_path'])==proof['audit_sha256'] and sha(proof['runner_receipt_path'])==proof['runner_receipt_sha256'] and
            receipt['status']==runner['status']=='PASS' and receipt['items']==runner['items']==300 and
            receipt['manifest_sha256']==runner['manifest_sha256']==sha(bundle/'manifest.json'),'imported dual proofs changed')
    identity=receipt['identity'];protocol=envelope(read(manifest['original_lock']))
    require(identity==record['identity'] and identity['phase']=='E3' and identity['state']==job['state'] and
            identity['panel']==job['panel'] and identity['protocol_sha256']==ORIGINAL_E3 and
            identity['runtime_sha256']==digest(protocol['runtime']) and identity['source']['weights_sha256']==protocol['sources'][job['state']]['weights']['sha256'],
            'imported provenance changed')
    return receipt


def audit_E2_dependency(manifest, job, event):
    """Check matched state/item deletion effects against preserved E2 bytes.

    The migration's conservative fixed-sample gate establishes exact execution
    agreement; this additionally checks the actual coarsely selected E3 layers.
    Any discrepancy is retained for diagnosis rather than assigned a new margin.
    """
    from compare_e3_migration import numeric_diff
    evidence = manifest['E2_dependencies'][job['dependency']]
    require(sha(evidence['audit_path']) == evidence['audit_sha256'], 'E2 receipt changed')
    e2_bundle = Path(evidence['attempt']) / 'bundle'; e3_bundle = Path(event['attempt']) / 'bundle'
    m2, m3 = read(e2_bundle/'manifest.json'), read(e3_bundle/'manifest.json')
    settings = envelope(read(manifest['successor_lock']))['settings'][job['state']]
    count = 0; numeric_fields = 0
    for name in sorted(n for n in m3['files'] if n.startswith('item-')):
        require(name in m2['files'] and sha(e2_bundle/name) == m2['files'][name] and
                sha(e3_bundle/name) == m3['files'][name], 'E2/E3 dependency item changed')
        left, right = read(e2_bundle/name), read(e3_bundle/name)
        require(left['item_id'] == right['item_id'], 'E2/E3 item mismatch')
        for layer in settings['layers']:
            original = next(r for r in left['operations'] if r['operation'] == f'delete/layer{layer}')
            successor = next(r for r in right['operations'] if r['operation'] == f'single_delete/layer{layer}')
            keys = ('behavior','geometry','factors')
            diff = numeric_diff({k:original[k] for k in keys}, {k:successor[k] for k in keys})
            require(not diff['structural_differences'] and diff['different_numeric_fields'] == 0,
                    f'archived E2/E3 deletion comparator differs: {job["id"]}/{name}/layer{layer}; {diff}')
            numeric_fields += diff['numeric_fields']
        count += 1
    require(count == 300, 'E2/E3 full300 dependency coverage')
    return {'status':'PASS','job_id':job['id'],'E2_dependency':job['dependency'],'items':count,
            'numeric_fields_compared':numeric_fields,'comparison':'exact measured deletion effects, geometry and factors',
            'E2_manifest_sha256':sha(e2_bundle/'manifest.json'),'E3_manifest_sha256':sha(e3_bundle/'manifest.json')}


def audited_summary(manifest,job,event):
    """Reuse completed arithmetic audits only after every covered byte rehashes.

    This is cryptographic reuse of a verified computation, not a skipped audit.
    Historical and successor receipts retain distinct sealed execution identity.
    """
    bundle=Path(event['attempt'])/'bundle';record=check_manifest_bytes(bundle)
    imported=event.get('origin')=='AdritaPC'
    if imported:receipt=verified_import(manifest,job,manifest['reuse'][job['id']])
    else:
        receipt=read(Path(event['attempt'])/'INDEPENDENT_AUDIT.json')
        exits=[read(p) for p in Path(event['attempt']).glob('VERIFIER_EXIT_*.json')]
        require(any(r['exit_code']==0 for r in exits),'missing original verifier success receipt')
        require(receipt['status']=='PASS' and receipt['items']==300 and receipt['manifest_sha256']==sha(bundle/'manifest.json') and
                receipt['manifest_sha256']==event['audit'],'successor completed dual audit differs')
    protocol_doc=read(manifest['original_lock'] if imported else manifest['successor_lock']);protocol=envelope(protocol_doc)
    identity=receipt['identity'];summary=read(bundle/'summary.json')
    require(identity==record['identity']==summary['identity'] and len(summary['items'])==300 and identity['state']==job['state'] and
            identity['panel']==job['panel'] and identity['protocol_sha256']==protocol_doc['sha256'] and
            identity['runtime_sha256']==digest(protocol['runtime']) and
            identity['source']['weights_sha256']==protocol['sources'][job['state']]['weights']['sha256'],'own-seal reporting identity')
    dependency=manifest['E2_dependencies'][job['dependency']];e2_bundle=Path(dependency['attempt'])/'bundle'
    check_manifest_bytes(e2_bundle)
    require(read(dependency['audit_path'])['manifest_sha256']==sha(e2_bundle/'manifest.json'),'matching E2 manifest changed')
    if imported:
        prior=read(manifest['archived_dependency_comparison'])
        require(sha(manifest['archived_dependency_comparison'])==manifest['archived_dependency_comparison_sha256'],'archived dependency comparison changed')
        pair=next(r for r in prior['jobs'] if r['job_id']==job['id'])
    else:
        file=Path(event['attempt'])/'E2_DEPENDENCY_COMPARISON.json'
        if file.exists():
            require(sha(file)==event.get('E2_comparison_sha256'),'new dependency comparison changed')
            pair=read(file)
        else:pair=audit_E2_dependency(manifest,job,event)
    require(pair['status']=='PASS' and pair['items']==300 and pair['E2_manifest_sha256']==sha(e2_bundle/'manifest.json') and
            pair['E3_manifest_sha256']==receipt['manifest_sha256'],'completed E2/E3 comparison differs')
    return summary,receipt,pair


def portable_phase_report(path, panel, output):
    """A separate join: every source is audited against its own sealed lock."""
    manifest = envelope(read(path)); validate_recovery(manifest); check_pins(manifest)
    statuses = ledger(Path(manifest['root']) / 'state_ledger.jsonl')
    jobs = [j for j in manifest['jobs'] if j['panel'] == panel]
    require(len(jobs) == 16 and all(statuses.get(j['id'], {}).get('status') == 'complete' for j in jobs), 'incomplete16-state panel')
    output.mkdir(parents=True, exist_ok=False)
    rows, joins, identities, proofs = [], {}, [], {}
    for job in jobs:
        event = statuses[job['id']]; bundle = Path(event['attempt']) / 'bundle'
        summary,receipt,pair = audited_summary(manifest,job,event)
        require(receipt['items'] == 300, 'full300 coverage')
        dependency = manifest['E2_dependencies'][job['dependency']]
        require(sha(dependency['audit_path']) == dependency['audit_sha256'], 'E2 dependency changed')
        identities.append(tuple(summary['items']))
        joins[job['state']] = {'bundle': str(bundle), 'origin': event.get('origin', 'NodiPC'),
                              'manifest_sha256': receipt['manifest_sha256'], 'summary': summary}
        proofs[job['state']] = {'audit':receipt, 'E2_dependency_comparison':pair,
            'verification':'original full runner and independent arithmetic audits, followed by current all-member SHA revalidation'}
        for operation, pooled in summary['operations'].items():
            rows.append({'state': job['state'], 'panel': panel, 'operation': operation,
                         'origin': event.get('origin', 'NodiPC'),
                         'protocol_sha256': summary['identity']['protocol_sha256'],
                         'runtime_sha256': summary['identity']['runtime_sha256'],
                         'measured_items': pooled['measured_items'], 'unavailable_items': pooled['unavailable_items'],
                         **(pooled['behavior'] or {})})
    require(len(set(identities)) == 1, 'item order differs between states')
    write(output / 'PORTABLE_JOIN.json', {'phase': 'E3', 'panel': panel, 'states': joins,
          'compatibility': compatibility(read(manifest['original_lock']), read(manifest['successor_lock'])),
          'interpretation': 'Explicit Adrita/Nodi mixed execution; identical scientific definitions; descriptive seed0; no independent seed replication'})
    write(output / 'INDEPENDENT_PHASE_AUDIT.json', {'status': 'PASS', 'states': proofs})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (output / 'results.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (output / 'PROVENANCE.md').write_text('Each state is independently audited against its own original or successor lock. '
        'PORTABLE_JOIN.json retains every intervention, geometry, norm/factor and telescope summary. '
        'The original complete_join remains strict and is not used to disguise mixed identities. '
        'Results are descriptive seed0; engineering tolerances do not establish scientific equivalence.\n', encoding='utf-8')
    write(output / 'REPORT_COMPLETE.json', {'status': 'COMPLETE_AUDITED_MIXED_RUNTIME_EXPORT', 'phase': 'E3', 'panel': panel,
          'files': {p.name: sha(p) for p in output.iterdir() if p.is_file()}})


class RecoverySupervisor(Supervisor):
    def __init__(self, path):
        self.path = Path(path).resolve(); self.manifest = envelope(read(path)); validate_recovery(self.manifest)
        self.root = Path(self.manifest['root']); self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.root / 'state_ledger.jsonl'; self.statuses = ledger(self.ledger_path)
        self.last_health = 0; self.current = None; self.health_failure = None
        # Private prerequisite statuses are never emitted as Nodi execution.
        self.statuses.update({key: {'status': 'complete', 'origin': 'AdritaPC_dependency'} for key in self.manifest['E2_dependencies']})

    def beat(self, force=False):
        if self.current:
            tree=same_attempt_workers(self.current,scientific_processes())
            if tree:
                self.current['executor_processes']=tree
                path=Path(self.current['attempt'])/'WORKER_PROCESS_TREE.json'
                if not path.exists():write(path,{'wrapper_pid':self.current['worker_pid'],'executor_processes':tree})
        all_statuses = self.statuses
        self.statuses = {j['id']: all_statuses[j['id']] for j in self.manifest['jobs'] if j['id'] in all_statuses}
        try: super().beat(force=force)
        finally: self.statuses = all_statuses

    def recover(self):
        for job in self.manifest['jobs']:
            previous = self.statuses.get(job['id'])
            reuse = self.manifest['reuse'].get(job['id'])
            if not previous and reuse:
                receipt = verified_import(self.manifest,job,reuse)
                self.event(job, 'complete', attempt=reuse['attempt'], origin='AdritaPC',
                           audit=receipt['manifest_sha256'], protocol_sha256=receipt['identity']['protocol_sha256'],
                           runtime_sha256=receipt['identity']['runtime_sha256'], execution='reused_original; never executed on Nodi')
                continue
            if not previous: continue
            if previous['status'] == 'complete':
                if previous.get('origin') == 'AdritaPC':
                    require(reuse and previous['attempt']==reuse['attempt'],'imported attempt changed')
                    verified_import(self.manifest,job,reuse)
                else: selected_audit(self.manifest, job, previous, heartbeat=self.beat)
            elif previous['status'] == 'running':
                self.current = previous
                try: process = psutil.Process(previous['worker_pid'])
                except psutil.NoSuchProcess: process = None
                if process is not None:
                    require(abs(process.create_time() - previous['worker_created']) < .01, 'orphan PID reused; diagnose before scheduling')
                    require('scripts/run_mechanistic.py' in process.cmdline(), 'orphan worker command differs')
                    while process.is_running(): self.beat(); time.sleep(self.manifest['poll_seconds'])
                # A killed venv launcher does not prove its scientific child died.
                # Match exact fresh output path and creation time, preserve and wait.
                while same_attempt_workers(previous,scientific_processes()):
                    self.beat();time.sleep(self.manifest['poll_seconds'])
                attempt = Path(previous['attempt'])
                if (attempt / 'bundle/COMPLETE').exists() and not (attempt / 'bundle/FAILED.json').exists():
                    try:
                        receipt = verify_attempt(self.manifest, job, attempt, heartbeat=self.beat)
                        self.event(job, 'complete', attempt=str(attempt), origin='NodiPC', worker_exit_code=None,
                                   wrapper_status='recovered; exit status unavailable', audit=receipt['manifest_sha256'])
                    except Exception as error: self.event(job, 'invalidated', attempt=str(attempt), reason=str(error))
                else: self.event(job, 'failed', attempt=str(attempt), retryable=True, reason='interrupted Nodi attempt retained; fresh full-state retry')
        self.current = None

    def run_job(self, job):
        prior = self.statuses.get(job['id'], {})
        if prior.get('status')!='complete':
            require(not scientific_processes(),'another CUDA scientific worker is active; preserve it and do not schedule conflicting work')
        super().run_job(job)
        event = self.statuses.get(job['id'], {})
        if prior.get('status') != 'complete' and event.get('status') == 'complete' and event.get('origin') != 'AdritaPC':
            try:
                receipt = audit_E2_dependency(self.manifest,job,event)
                file = Path(event['attempt']) / 'E2_DEPENDENCY_COMPARISON.json'; write(file,receipt)
                self.event(job,'complete',attempt=event['attempt'],origin='NodiPC',worker_exit_code=event.get('worker_exit_code'),
                           audit=event['audit'],E2_comparison_sha256=sha(file))
            except Exception as error:
                self.event(job,'invalidated',attempt=event['attempt'],origin='NodiPC',reason=str(error))

    def phase_report(self, phase, panel):
        jobs = [j for j in self.manifest['jobs'] if j['panel'] == panel]
        if any(self.statuses.get(j['id'], {}).get('status') != 'complete' for j in jobs):
            append(self.root / 'phase_ledger.jsonl', {'phase': 'E3', 'panel': panel, 'status': 'PARTIAL'}); return
        parent = self.root / 'reports'; parent.mkdir(exist_ok=True)
        folder = parent / f'E3_{panel}'
        if (folder / 'REPORT_COMPLETE.json').exists(): return
        if folder.exists(): folder = parent / f'E3_{panel}_retry_{time.time_ns()}'
        command = [self.manifest['python'], 'scripts/e3_nodi_recovery.py', 'report', '--manifest', str(self.path), '--panel', panel, '--output', str(folder)]
        with (parent / f'{folder.name}.log').open('xb') as stream:
            worker = subprocess.Popen(command, cwd=self.manifest['repo'], env={**os.environ, **self.manifest['environment']}, stdout=stream, stderr=subprocess.STDOUT)
            code = self.wait_worker(worker)
        append(self.root / 'phase_ledger.jsonl', {'phase': 'E3', 'panel': panel, 'status': 'REPORT_COMPLETE' if code == 0 else 'REPORT_FAILED', 'exit_code': code, 'directory': str(folder)})
        require(code == 0, 'portable phase report failed')

    def run(self):
        check_pins(self.manifest); self.recover()
        for panel in ('discovery', 'confirmation'):
            for job in self.manifest['jobs']:
                if job['panel'] != panel: continue
                try: self.run_job(job)
                except Exception as error:
                    append(self.root / 'supervisor_errors.jsonl', {'job_id': job['id'], 'error': str(error), 'traceback': traceback.format_exc()})
                    if self.current:
                        try: p = psutil.Process(self.current['worker_pid'])
                        except psutil.NoSuchProcess: p = None
                        if p and p.is_running() and abs(p.create_time() - self.current['worker_created']) < .01:
                            append(self.root / 'supervisor_interruptions.jsonl', {'status': 'LIVE_WORKER_PRESERVED', 'current': self.current, 'error': str(error)})
                            raise RuntimeError('live worker preserved; no further scheduling') from error
                    self.event(job, 'blocked', reason=str(error))
            self.phase_report('E3', panel)
        self.current = None; self.beat(force=True)
        states = {j['id']: self.statuses.get(j['id'], {'status': 'absent'}) for j in self.manifest['jobs']}
        complete = len(states) == 32 and all(r['status'] == 'complete' for r in states.values())
        receipt = {'status': 'COMPLETE_INDEPENDENTLY_AUDITED' if complete else 'INCOMPLETE', 'utc': utc(),
                   'expected': 32, 'states': states, 'reused_Adrita': sum(r.get('origin') == 'AdritaPC' for r in states.values()),
                   'new_Nodi': sum(r['status'] == 'complete' and r.get('origin') != 'AdritaPC' for r in states.values()),
                   'all_E2_dependencies_verified': True, 'manifest_sha256': sha(self.path),
                   'original_protocol_sha256': ORIGINAL_E3, 'successor_protocol_sha256': self.manifest['approved_lock_hashes']['E3'],
                   'reports': {p.relative_to(self.root).as_posix(): sha(p) for p in (self.root / 'reports').rglob('*') if p.is_file()}}
        write(self.root / f'FINAL_E3_AUDIT_{time.time_ns()}.json', receipt)
        if complete:
            files = {p.relative_to(self.root).as_posix(): sha(p) for p in self.root.rglob('*')
                     if p.is_file() and p.name not in {'heartbeat.json', 'health.jsonl', 'monitor_heartbeat.json', 'monitor_ledger.jsonl'}
                     and not p.name.startswith(('monitor-', 'supervisor-'))}
            write(self.root / f'DERIVED_ARTIFACT_INVENTORY_{time.time_ns()}.json', {'status': 'COMPLETE', 'files': files})


def main():
    parser = argparse.ArgumentParser(__doc__); commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('supervise'); run.add_argument('--manifest', type=Path, required=True)
    report = commands.add_parser('report'); report.add_argument('--manifest', type=Path, required=True)
    report.add_argument('--panel', choices=('discovery', 'confirmation'), required=True); report.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'report': portable_phase_report(args.manifest, args.panel, args.output); return
    supervisor = RecoverySupervisor(args.manifest)
    with exclusive(Path(supervisor.manifest['gpu_lock_path'])): supervisor.run()


if __name__ == '__main__': main()
