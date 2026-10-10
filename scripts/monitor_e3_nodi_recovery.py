"""Detached, bounded monitor for one sealed E3 recovery supervisor.

No administrative approvals, scientific retries or outcome selection occur here.
An orphan scientific worker is adopted by the recovery supervisor, never doubled.
"""
import argparse
import os
from pathlib import Path
import subprocess
import time

import psutil

from audit_mechanistic_scientific import read, sha, envelope, require, digest
from mechanistic_campaign_supervisor import append, atomic, exclusive, check_pins, ledger, utc
from e3_nodi_recovery import validate_recovery


def decision(manifest, statuses, supervisor_alive, terminal):
    """Explicit terminal semantics: uptime alone is never scientific completion."""
    jobs = manifest['jobs']
    complete = all(statuses.get(j['id'], {}).get('status') == 'complete' for j in jobs)
    if complete and terminal and terminal.get('status') == 'COMPLETE_INDEPENDENTLY_AUDITED': return 'complete'
    if supervisor_alive: return 'observe'
    if any(statuses.get(j['id'], {}).get('status') in {'invalidated','blocked'} for j in jobs): return 'stop_for_diagnosis'
    if any(statuses.get(j['id'], {}).get('status') == 'failed' and not statuses[j['id']].get('retryable') for j in jobs): return 'stop_for_diagnosis'
    return 'recover_supervisor'


def monitor(path):
    manifest = envelope(read(path)); validate_recovery(manifest); check_pins(manifest)
    readiness = read(manifest['independent_readiness'])
    checked_plan = {k:v for k,v in manifest.items() if k not in ('independent_readiness','independent_readiness_sha256')}
    require(readiness['status'] == 'PASS' and readiness['verdict'].startswith('READY_') and
            sha(manifest['independent_readiness']) == manifest['independent_readiness_sha256'] and
            readiness['execution_plan_sha256'] == digest(checked_plan), 'independent READY required before launch')
    root = Path(manifest['root']); root.mkdir(parents=True, exist_ok=True)
    with exclusive(root / 'monitor.lock'):
        worker = None; adopted = None; starts = 0
        beat = root / 'heartbeat.json'
        if beat.exists():
            prior = read(beat)
            require(prior['manifest_sha256'] == sha(path), 'prior supervisor identity differs')
            try: process = psutil.Process(prior['supervisor_pid'])
            except psutil.NoSuchProcess: process = None
            if process and abs(process.create_time()-prior['supervisor_created']) < .01:
                require('scripts/e3_nodi_recovery.py' in process.cmdline(), 'prior supervisor command differs')
                adopted = process
        while True:
            statuses = ledger(root/'state_ledger.jsonl')
            receipts = sorted(root.glob('FINAL_E3_AUDIT_*.json'))
            terminal = read(receipts[-1]) if receipts else None
            alive = (worker is not None and worker.poll() is None) or (adopted is not None and adopted.is_running())
            action = decision(manifest, statuses, alive, terminal)
            snapshot = {'utc':utc(), 'monitor_pid':os.getpid(), 'manifest_sha256':sha(path), 'action':action,
                'supervisor_starts_this_monitor':starts, 'coverage':{s:sum(e.get('status')==s for e in statuses.values())
                for s in ('complete','running','failed','blocked','invalidated')}, 'expected':32,
                'supervisor_pid':worker.pid if worker else adopted.pid if adopted else None}
            if beat.exists(): snapshot['supervisor_heartbeat'] = read(beat)
            atomic(root/'monitor_heartbeat.json',snapshot)
            if action == 'complete':
                # Final terminal receipt is linked, independently generated
                # reports and inventory remain separate verifiable artifacts.
                atomic(root/'MONITOR_COMPLETE.json', {**snapshot,'final_audit':str(receipts[-1]),'final_audit_sha256':sha(receipts[-1])})
                append(root/'monitor_ledger.jsonl', snapshot); return
            if action == 'stop_for_diagnosis':
                atomic(root/'MONITOR_DIAGNOSIS_REQUIRED.json',snapshot)
                append(root/'monitor_ledger.jsonl',snapshot); return
            if action == 'recover_supervisor':
                # The Windows venv launcher may exit while its real interpreter
                # continues. Adopt the creation-verified heartbeat owner first.
                if beat.exists():
                    prior=read(beat)
                    try:process=psutil.Process(prior['supervisor_pid'])
                    except psutil.NoSuchProcess:process=None
                    if process and process.is_running() and abs(process.create_time()-prior['supervisor_created'])<.01:
                        command=process.cmdline()
                        if 'scripts/e3_nodi_recovery.py' in command and str(path) in command:
                            adopted=process;worker=None
                            append(root/'monitor_ledger.jsonl',{**snapshot,'action':'actual_supervisor_adopted','pid':process.pid})
                            time.sleep(manifest['monitor_poll_seconds']);continue
                require(starts < manifest['monitor_max_supervisor_starts'], 'bounded supervisor restart limit reached')
                check_pins(manifest)
                starts += 1
                command = [manifest['python'],'scripts/e3_nodi_recovery.py','supervise','--manifest',str(path)]
                flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                with (root/f'supervisor-{time.time_ns()}.log').open('xb') as stream:
                    worker = subprocess.Popen(command,cwd=manifest['repo'],env={**os.environ,**manifest['environment']},
                        stdout=stream,stderr=subprocess.STDOUT,creationflags=flags)
                adopted = None
                append(root/'monitor_ledger.jsonl', {**snapshot,'action':'supervisor_started','pid':worker.pid,
                    'created':psutil.Process(worker.pid).create_time(),'argv':command})
            time.sleep(manifest['monitor_poll_seconds'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__); parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args(); monitor(args.manifest.resolve())
