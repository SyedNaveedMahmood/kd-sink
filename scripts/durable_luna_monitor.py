"""Detached Luna monitoring turns around the independent scientific supervisor.

The wrapper survives the launching shell. Each actual Codex CLI thread records
its model/effort and a human-readable diagnosis; it is not a fictitious persistent
collaboration-tool conversation. Supervisor execution does not depend on API uptime.
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

import psutil
from audit_mechanistic_scientific import read, write, sha, envelope
from mechanistic_campaign_supervisor import atomic, append, exclusive, utc


def monitor(manifest_path, codex, prompt_path, interval):
    manifest = envelope(read(manifest_path))
    root = Path(manifest['root'])
    directory = root / 'luna_monitor'
    directory.mkdir(exist_ok=True)
    consecutive_failures = 0
    counter = len(list(directory.glob('turn_*')))
    with exclusive(root/'luna_monitor.lock'):
        while not (root/'MECHANISTIC_E1_E3_SCIENTIFIC_COMPLETION.json').exists():
            counter += 1
            folder = directory/f'turn_{counter:05d}'
            folder.mkdir(exist_ok=False)
            prompt = prompt_path.read_text(encoding='utf-8') + f'\nMonitoring turn {counter}; UTC {utc()}. Manifest {manifest_path}.\n'
            argv = [str(codex), 'exec', '--ignore-user-config', '-m', 'gpt-6-luna', '-c', 'model_reasoning_effort=max',
                    '-c', 'approval_policy=never', '-s', 'danger-full-access', '-C', manifest['repo'], '--json',
                    '--output-last-message', str(folder/'last_message.txt'), '-']
            write(folder/'INVOCATION.json', {'argv':argv,'requested_model':'gpt-6-luna','requested_effort':'max',
                'prompt_sha256':sha(prompt_path),'utc':utc(),'wrapper_pid':os.getpid()})
            with (folder/'events.jsonl').open('xb') as out, (folder/'stderr.txt').open('xb') as err:
                process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=out, stderr=err)
                process.stdin.write(prompt.encode('utf-8')); process.stdin.close()
                while process.poll() is None:
                    atomic(root/'LUNA_MONITOR_HEARTBEAT.json', {'utc':utc(),'wrapper_pid':os.getpid(),
                        'codex_pid':process.pid,'turn':counter,'model':'gpt-6-luna','effort':'max','status':'monitoring',
                        'campaign_heartbeat':read(root/'heartbeat.json') if (root/'heartbeat.json').exists() else None})
                    time.sleep(5)
            write(folder/'EXIT.json', {'utc':utc(),'pid':process.pid,'exit_code':process.returncode})
            append(directory/'monitor_ledger.jsonl', {'turn':counter,'exit_code':process.returncode,'folder':str(folder)})
            consecutive_failures = consecutive_failures + 1 if process.returncode else 0
            if (root/'FINALIZATION_READY.json').exists() and process.returncode == 0:
                atomic(root/'LUNA_MONITOR_HEARTBEAT.json', {'utc':utc(),'wrapper_pid':os.getpid(),'status':'finalizing_after_last_monitor_turn'})
                argv=[manifest['python'],'scripts/finalize_mechanistic_campaign.py','--manifest',str(manifest_path),
                      '--archive',str(root.parent/f'{root.name}_artifacts_{time.time_ns()}.zip')]
                # These live wrapper diagnostics are outside the immutable
                # archive root. All completed turn logs are already closed.
                stamp=time.time_ns()
                with (root.parent/f'{root.name}_finalization_{stamp}.stdout.txt').open('xb') as out, (root.parent/f'{root.name}_finalization_{stamp}.stderr.txt').open('xb') as err:
                    result=subprocess.run(argv,cwd=manifest['repo'],env={**os.environ,**manifest['environment']},stdout=out,stderr=err)
                if result.returncode != 0:
                    write(root/f'FINALIZATION_FAILED_{time.time_ns()}.json',{'utc':utc(),'exit_code':result.returncode,'diagnostics':'external finalization stdout/stderr; preserve old failed archive/receipt and diagnose conservatively'})
                    (root/'FINALIZATION_READY.json').rename(root/f'FINALIZATION_READY_failed_{stamp}.json')
                    # Preserve failure, resume diagnosis on the next monitoring
                    # turn; no repeated deterministic finalization without a
                    # newly written readiness receipt from the owner.
                    continue
                return 0
            if consecutive_failures >= 3:
                write(root/f'LUNA_MONITOR_BLOCKED_{time.time_ns()}.json', {'reason':'three consecutive CLI failures; supervisor retained', 'utc':utc()})
                return 1
            deadline = time.monotonic() + interval
            while time.monotonic() < deadline and not (root/'MECHANISTIC_E1_E3_SCIENTIFIC_COMPLETION.json').exists():
                atomic(root/'LUNA_MONITOR_HEARTBEAT.json', {'utc':utc(),'wrapper_pid':os.getpid(),'turn':counter,
                    'model':'gpt-6-luna','effort':'max','status':'between_monitoring_turns'})
                time.sleep(5)
        atomic(root/'LUNA_MONITOR_HEARTBEAT.json', {'utc':utc(),'wrapper_pid':os.getpid(),'status':'terminal_outcome_recorded'})
    return 0


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--codex',type=Path,required=True)
    parser.add_argument('--prompt',type=Path,required=True)
    parser.add_argument('--interval',type=int,default=90)
    args=parser.parse_args()
    raise SystemExit(monitor(args.manifest,args.codex,args.prompt,args.interval))
