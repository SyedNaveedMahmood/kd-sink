"""Durable serial supervisor for a sealed, explicit E1-E3 inference manifest.

Operational recovery only: no numerical edits, protocol amendments or training.
Run detached on Windows; a live conversation is not needed for progress.
"""
import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import psutil
from audit_mechanistic_scientific import read, write, sha, envelope, require, audit, canonical


def utc():
    return datetime.now(timezone.utc).isoformat()


def append(path, event):
    with path.open('ab') as stream:
        stream.write(canonical({'utc': utc(), **event}) + b'\n')
        stream.flush(); os.fsync(stream.fileno())


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        stream.write(canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    # Windows readers can briefly deny replacement. Keep the prior complete
    # heartbeat until the new complete file can replace it; never truncate it.
    deadline = time.monotonic() + 2.0
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


@contextlib.contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+b')
    stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        stream.close(); raise RuntimeError('another supervisor owns this GPU/campaign')
    try:
        yield
    finally:
        stream.close()


def ledger(path):
    events = []
    if path.exists():
        # A torn append is an integrity failure requiring preserved repair,
        # never silently discard a potentially complete state transition.
        for line in path.read_bytes().splitlines():
            events.append(json.loads(line))
    latest = {}
    for event in events:
        if 'job_id' in event: latest[event['job_id']] = event
    return latest


def progress(bundle):
    path = bundle / 'events.jsonl'
    if not path.exists(): return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 8192))
        lines = stream.read().splitlines()
    try:
        return {'updated_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                **json.loads(lines[-1])}
    except (ValueError, IndexError): return None


def health(root, gpu_uuid=None):
    ram = psutil.virtual_memory()
    result = {'utc': utc(), 'supervisor_pid': os.getpid(), 'ram_available_bytes': ram.available,
              'disk_free_bytes': psutil.disk_usage(str(root)).free}
    if gpu_uuid:
        query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,utilization.gpu,temperature.gpu,memory.used,memory.total,driver_version',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
        require(query.returncode == 0, 'CUDA health query failed')
        rows = [r.split(',') for r in query.stdout.splitlines()]
        selected = [r for r in rows if r[0].strip() == gpu_uuid]
        require(len(selected) == 1, 'fixed GPU UUID unavailable')
        r = [v.strip() for v in selected[0]]
        result['gpu'] = {'uuid': r[0], 'utilization_percent': float(r[1]), 'temperature_c': float(r[2]),
                         'used_bytes': int(r[3]) * 2**20, 'total_bytes': int(r[4]) * 2**20, 'driver': r[5]}
    return result


def check_pins(manifest):
    for name, expected in manifest['frozen_files'].items():
        require(sha(name) == expected, f'frozen source/lock changed: {name}')


def validate_manifest(manifest):
    if not manifest.get('scientific'): return
    jobs=manifest['jobs']
    require(len(jobs)==len({j['id'] for j in jobs})==92, 'scientific manifest must cover exactly 92 logical states')
    ordered=[]
    for phase in ('E1','E2','E3'):
        phase_jobs=[j for j in jobs if j['phase']==phase]
        require(phase_jobs, 'missing phase')
        lock=read(phase_jobs[0]['lock']);payload=envelope(lock)
        require(payload['status']=='approved' and payload['production_ready'] and payload['seed']==0 and
                lock['sha256']==manifest['approved_lock_hashes'][phase], 'unapproved manifest phase')
        for panel in ('discovery','confirmation'):
            for state in payload['grid']:
                job_id=f'{phase}_{panel}_{state.replace("/","-")}'
                matched=[j for j in jobs if j['id']==job_id]
                require(len(matched)==1, 'missing approved state/panel')
                j=matched[0]
                expected=[manifest['python'],'scripts/run_mechanistic.py','scientific','--phase',phase,'--seed','0',
                          '--state',state,'--panel',panel,'--device','cuda:0','--disable-tf32','--lock',j['lock'],
                          '--approved-sha256',lock['sha256'],'--output','{OUTPUT}']
                require(j['argv']==expected and j['seed']==0 and j['lock']==phase_jobs[0]['lock'] and
                        all(j[k]==v for k,v in (('phase',phase),('state',state),('panel',panel))), 'command changed from explicit approved single-state contract')
                if phase=='E3':require(j['dependency']==f'E2_{panel}_{state.replace("/","-")}', 'E3 dependency changed')
                ordered.append(job_id)
    require([j['id'] for j in jobs]==ordered, 'campaign order changed')


def transient(log):
    # Deliberately narrow. Integrity/parity/OOM/import/config failures are
    # deterministic until diagnosed; they cannot retry into acceptance.
    text = log.lower()
    if any(t in text for t in ('parity', 'hash mismatch', 'geometry', 'out of memory', 'runtime lock mismatch', 'no module named')):
        return False
    return any(t in text for t in ('cuda initialization error', 'cuda driver initialization failed', 'winerror 1450'))


def verify_attempt(manifest, job, attempt, heartbeat=None):
    suffix = str(time.time_ns())
    stdout, stderr = attempt / f'verify-{suffix}.stdout.txt', attempt / f'verify-{suffix}.stderr.txt'
    with stdout.open('xb') as out, stderr.open('xb') as err:
        result = subprocess.Popen([manifest['python'], 'scripts/run_mechanistic.py', 'verify', '--output', str(attempt / 'bundle')],
                                cwd=manifest['repo'], env={**os.environ, **manifest['environment']}, stdout=out, stderr=err)
        while result.poll() is None:
            if heartbeat: heartbeat()
            time.sleep(.5)
    write(attempt / f'VERIFIER_EXIT_{suffix}.json', {'exit_code':result.returncode,'pid':result.pid,'utc':utc()})
    require(result.returncode == 0, f'runner verifier exit {result.returncode}')
    lock = read(job['lock']) if 'lock' in job else None
    receipt = audit(attempt / 'bundle', lock, job if lock else None, heartbeat=heartbeat)
    if (attempt / 'INDEPENDENT_AUDIT.json').exists():
        require(read(attempt / 'INDEPENDENT_AUDIT.json')['manifest_sha256'] == receipt['manifest_sha256'], 'audit identity changed')
        write(attempt / f'RECOVERY_AUDIT_{suffix}.json', receipt)
    else: write(attempt / 'INDEPENDENT_AUDIT.json', receipt)
    return receipt


class Supervisor:
    def __init__(self, path):
        self.path = path.resolve()
        self.manifest = envelope(read(path))
        validate_manifest(self.manifest)
        self.root = Path(self.manifest['root'])
        self.ledger_path = self.root / 'state_ledger.jsonl'
        self.statuses = ledger(self.ledger_path)
        self.last_health = 0
        self.current = None
        self.health_failure = None

    def event(self, job, status, **fields):
        event = {'job_id': job['id'], 'phase': job['phase'], 'state': job['state'], 'panel': job['panel'], 'status': status, **fields}
        append(self.ledger_path, event)
        self.statuses[job['id']] = event
        self.beat(force=True)

    def beat(self, force=False):
        snapshot = {'utc': utc(), 'supervisor_pid': os.getpid(), 'manifest_sha256': sha(self.path),
                    'supervisor_created': psutil.Process().create_time(),
                    'current': self.current, 'coverage': {s: sum(e['status'] == s for e in self.statuses.values())
                     for s in ('complete', 'running', 'failed', 'blocked', 'invalidated')},
                    'expected': len(self.manifest['jobs'])}
        if self.current:
            snapshot['progress'] = progress(Path(self.current['attempt']) / 'bundle')
            p = snapshot['progress']
            last = datetime.fromisoformat(p['updated_utc']).timestamp() if p else self.current.get('worker_created', time.time())
            snapshot['stale_progress_warning'] = time.time() - last > self.manifest['stale_warning_seconds']
        if force or time.monotonic() - self.last_health >= self.manifest['health_interval_seconds']:
            try:
                sample = health(self.root, self.manifest.get('gpu_uuid'))
                sample.update(current=self.current, coverage=snapshot['coverage'], progress=snapshot.get('progress'))
                append(self.root / 'health.jsonl', sample)
                snapshot['health'] = sample
                self.health_failure = None
            except Exception as error:
                self.health_failure = str(error)
                append(self.root / 'health.jsonl', {'error': str(error), 'current': self.current})
            self.last_health = time.monotonic()
        snapshot['health_error'] = self.health_failure
        atomic(self.root / 'heartbeat.json', snapshot)

    def wait_worker(self, worker):
        while worker.poll() is None:
            self.beat(); time.sleep(self.manifest['poll_seconds'])
        self.beat(force=True)
        return worker.returncode

    def recover(self):
        for job in self.manifest['jobs']:
            previous = self.statuses.get(job['id'])
            if not previous: continue
            if previous['status'] == 'complete':
                attempt = Path(previous['attempt'])
                try:
                    receipt = audit(attempt / 'bundle', read(job['lock']) if 'lock' in job else None, job if 'lock' in job else None, heartbeat=self.beat)
                    require(receipt['manifest_sha256'] == read(attempt / 'INDEPENDENT_AUDIT.json')['manifest_sha256'], 'previously audited bundle changed')
                except Exception as error:
                    self.event(job, 'invalidated', attempt=str(attempt), reason=str(error))
            elif previous['status'] == 'running':
                attempt = Path(previous['attempt'])
                self.current = previous
                pid = previous['worker_pid']
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    require(process.create_time() == previous['worker_created'], 'worker PID reused; manual diagnosis required')
                    while process.is_running():
                        self.beat(); time.sleep(self.manifest['poll_seconds'])
                if (attempt / 'bundle/COMPLETE').exists():
                    try:
                        receipt = verify_attempt(self.manifest, job, attempt, heartbeat=self.beat)
                        self.event(job, 'complete', attempt=str(attempt), worker_exit_code=None, wrapper_status='recovered; exit status unavailable', audit=receipt['manifest_sha256'])
                        continue
                    except Exception as error:
                        self.event(job, 'invalidated', attempt=str(attempt), reason=str(error))
                else:
                    self.event(job, 'failed', attempt=str(attempt), reason='interrupted worker; preserve partial attempt', retryable=True)
        self.current = None

    def run_job(self, job):
        previous = self.statuses.get(job['id'])
        if previous and previous['status'] in ('complete', 'blocked', 'invalidated'): return
        if job.get('dependency') and self.statuses.get(job['dependency'], {}).get('status') != 'complete':
            self.event(job, 'blocked', reason='required same-state/panel E2 parity evidence is unavailable'); return
        count = len(list((self.root / 'attempts' / job['id']).glob('attempt_*')))
        if previous and previous['status'] == 'failed' and not previous.get('retryable'):
            self.event(job, 'blocked', reason='deterministic failed attempt requires diagnosis'); return
        while count < self.manifest['max_attempts']:
            check_pins(self.manifest)
            sample = health(self.root, self.manifest.get('gpu_uuid'))
            if 'minimum_disk_bytes' in self.manifest:
                require(sample['disk_free_bytes'] >= self.manifest['minimum_disk_bytes'] and sample['ram_available_bytes'] >= self.manifest['minimum_ram_bytes'], 'RAM/disk readiness gate')
                gpu = sample['gpu']
                require(gpu['total_bytes'] - gpu['used_bytes'] >= self.manifest['qualified_reserved_bytes'] + self.manifest['gpu_headroom_bytes'], 'free VRAM readiness gate')
            count += 1
            attempt = self.root / 'attempts' / job['id'] / f'attempt_{count:03d}'
            attempt.mkdir(parents=True, exist_ok=False)
            write(attempt / 'job.json', job)
            argv = [s.replace('{OUTPUT}', str(attempt / 'bundle')) for s in job['argv']]
            write(attempt / 'command.json', {'argv': argv, 'environment': self.manifest['environment'], 'cwd': self.manifest['repo']})
            with (attempt / 'worker.stdout.txt').open('xb') as out, (attempt / 'worker.stderr.txt').open('xb') as err:
                worker = subprocess.Popen(argv, cwd=self.manifest['repo'], env={**os.environ, **self.manifest['environment']}, stdout=out, stderr=err)
                self.current = {'job_id': job['id'], 'phase': job['phase'], 'state': job['state'], 'panel': job['panel'], 'attempt': str(attempt), 'worker_pid': worker.pid, 'worker_created':psutil.Process(worker.pid).create_time()}
                self.event(job, 'running', **{k: v for k, v in self.current.items() if k not in ('job_id', 'phase', 'state', 'panel')})
                code = self.wait_worker(worker)
            write(attempt / 'WORKER_EXIT.json', {'exit_code': code, 'utc': utc(), 'pid': worker.pid})
            if code == 0:
                try:
                    receipt = verify_attempt(self.manifest, job, attempt, heartbeat=self.beat)
                    self.event(job, 'complete', attempt=str(attempt), worker_exit_code=code, audit=receipt['manifest_sha256'])
                    self.current = None; return
                except Exception as error:
                    self.event(job, 'invalidated', attempt=str(attempt), worker_exit_code=code, reason=str(error)); self.current = None; return
            retryable = transient((attempt / 'worker.stderr.txt').read_text(errors='replace'))
            self.event(job, 'failed', attempt=str(attempt), worker_exit_code=code, retryable=retryable, reason='see preserved worker stdout/stderr')
            self.current = None
            if not retryable: break
        self.event(job, 'blocked', reason='bounded attempts exhausted or deterministic failure')

    def phase_report(self, phase, panel):
        jobs = [j for j in self.manifest['jobs'] if j['phase'] == phase and j['panel'] == panel]
        states = {j['id']: self.statuses.get(j['id'], {'status': 'pending'}) for j in jobs}
        if any(e['status'] != 'complete' for e in states.values()):
            append(self.root / 'phase_ledger.jsonl', {'phase': phase, 'panel': panel, 'status': 'PARTIAL_BLOCKED', 'states': states}); return
        folder = self.root / 'reports' / f'{phase}_{panel}'
        if (folder / 'REPORT_COMPLETE.json').exists(): return
        # Fresh attempts preserve failed reporting attempts as well.
        parent = self.root / 'reports'
        parent.mkdir(exist_ok=True)
        if folder.exists(): folder = parent / f'{phase}_{panel}_retry_{time.time_ns()}'
        argv = [self.manifest['python'], 'scripts/report_mechanistic_scientific.py', '--manifest', str(self.path), '--phase', phase, '--panel', panel, '--output', str(folder)]
        with (parent / f'{folder.name}.stdout.txt').open('xb') as out, (parent / f'{folder.name}.stderr.txt').open('xb') as err:
            worker = subprocess.Popen(argv, cwd=self.manifest['repo'], env={**os.environ, **self.manifest['environment']}, stdout=out, stderr=err)
            code = self.wait_worker(worker)
        append(self.root / 'phase_ledger.jsonl', {'phase': phase, 'panel': panel, 'status': 'REPORT_COMPLETE' if code == 0 else 'REPORT_FAILED', 'exit_code': code, 'directory': str(folder)})

    def run(self):
        check_pins(self.manifest)
        self.recover()
        for phase, panel in dict.fromkeys((j['phase'], j['panel']) for j in self.manifest['jobs']):
            for job in self.manifest['jobs']:
                if job['phase'] == phase and job['panel'] == panel:
                    try: self.run_job(job)
                    except Exception as error:
                        append(self.root / 'supervisor_errors.jsonl', {'job_id': job['id'], 'error': str(error), 'traceback': traceback.format_exc()})
                        if self.current and 'worker_pid' in self.current:
                            pid = self.current['worker_pid']
                            try:
                                worker = psutil.Process(pid)
                                alive = worker.is_running() and worker.create_time() == self.current['worker_created']
                            except psutil.NoSuchProcess:
                                alive = False
                            if alive:
                                # A supervisor I/O failure does not terminate a
                                # worker. Preserve its running ownership and
                                # stop scheduling before another GPU job starts.
                                append(self.root / 'supervisor_interruptions.jsonl', {
                                    'job_id': job['id'], 'status': 'LIVE_WORKER_PRESERVED',
                                    'current': self.current, 'error': str(error),
                                    'recovery': 'diagnose supervisor failure; restart the sealed manifest to wait/audit the exact orphan'})
                                raise RuntimeError('supervisor interrupted with a live worker; ownership preserved, no further jobs launched') from error
                        self.event(job, 'blocked', reason=str(error))
            if self.manifest.get('scientific'): self.phase_report(phase, panel)
        self.current = None
        self.beat(force=True)
        write(self.root / f'SUPERVISOR_TERMINAL_{time.time_ns()}.json', {'utc': utc(), 'states': self.statuses,
            'all_bundles_complete': len(self.statuses) == len(self.manifest['jobs']) and all(e['status'] == 'complete' for e in self.statuses.values()),
            'next': 'independent monitor must finish cross-phase reporting, audit and archive; this receipt alone is not scientific completion'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    supervisor = Supervisor(args.manifest)
    with exclusive(Path(supervisor.manifest['gpu_lock_path'])):
        supervisor.run()
