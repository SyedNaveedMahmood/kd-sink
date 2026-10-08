"""Freeze an explicit serial 92-job manifest after approved-lock readiness."""
import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from audit_mechanistic_scientific import read, write, sha, digest, envelope, require
from mechanistic_campaign_supervisor import health


def build(repo, root, python, tests):
    require(root.resolve().is_relative_to(Path('D:/KD-SINK-central/analysis')), 'external campaign root required')
    suites = ET.parse(tests).getroot()
    require(sum(int(s.get('failures', 0)) + int(s.get('errors', 0)) + int(s.get('skipped', 0)) for s in suites.iter('testsuite')) == 0, 'readiness tests failed/skipped')
    locks = {p: repo / f'protocols/mechanistic_e1_e3_approved_20261009/{p}.approved.json' for p in ('E1','E2','E3')}
    documents = {p: read(path) for p, path in locks.items()}
    protocols = {p: envelope(d) for p, d in documents.items()}
    fresh = root / 'readiness_qualification01'
    require(read(root/'READINESS_INDEPENDENT_QUALIFICATION.json')['status'] == 'PASS', 'missing independent qualification')
    require(read(root/'PANEL_INDEPENDENT_AUDIT.json')['status'] == 'PASS', 'missing independent panel audit')
    for p, protocol in protocols.items():
        q = read(fresh / f'{p}_QUALIFICATION.json')
        require(protocol['status'] == 'approved' and protocol['production_ready'] and protocol['runtime'] == q['runtime'] and
                digest(protocol['settings']) == q['settings_sha256'] and q['parity_passed'] and q['headroom_passed'], 'fresh qualification differs')
    env = {'PYTHONPATH': str(repo/'src'), 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
           'PIP_NO_INDEX':'1', 'UV_OFFLINE':'1', 'PYTHONUNBUFFERED':'1'}
    jobs = []
    for phase in ('E1','E2','E3'):
        for panel in ('discovery','confirmation'):
            for state in protocols[phase]['grid']:
                job_id = f'{phase}_{panel}_{state.replace("/", "-")}'
                argv = [str(python), 'scripts/run_mechanistic.py', 'scientific', '--phase', phase, '--seed', '0',
                        '--state', state, '--panel', panel, '--device', 'cuda:0', '--disable-tf32', '--lock', str(locks[phase]),
                        '--approved-sha256', documents[phase]['sha256'], '--output', '{OUTPUT}']
                job = {'id': job_id, 'phase':phase, 'panel':panel, 'state':state, 'seed':0,
                       'lock':str(locks[phase]), 'approved_sha256':documents[phase]['sha256'], 'argv':argv}
                if phase == 'E3': job['dependency'] = f'E2_{panel}_{state.replace("/", "-")}'
                jobs.append(job)
    require(len(jobs) == len({j['id'] for j in jobs}) == 92, 'coverage count')
    files = {str(repo/name): value for name, value in protocols['E1']['runtime']['code_files'].items()}
    for path in [*locks.values(), *repo.glob('scripts/*mechanistic*campaign*.py'),
                 repo/'scripts/audit_mechanistic_scientific.py', repo/'scripts/report_mechanistic_scientific.py',
                 repo/'scripts/durable_luna_monitor.py', root/'RESEARCHER_AUTHORIZATION.md', tests,
                 repo/'scripts/finalize_mechanistic_campaign.py',
                 root/'READINESS_INDEPENDENT_QUALIFICATION.json', root/'PANEL_INDEPENDENT_AUDIT.json']:
        files[str(path)] = sha(path)
    resources = read(fresh/'RESOURCE_SUMMARY.json')
    snapshot = health(root, 'GPU-2a5c25d0-1f73-919b-fd8b-f6f0df709aaf')
    require(snapshot['disk_free_bytes'] >= 10*2**30 and snapshot['ram_available_bytes'] >= 4*2**30, 'disk/RAM headroom')
    require(snapshot['gpu']['total_bytes'] - snapshot['gpu']['used_bytes'] >= resources['peak_reserved_bytes'] + resources['required_gpu_headroom_bytes'], 'GPU headroom')
    payload = {'schema_version':1, 'scientific':True, 'repo':str(repo), 'root':str(root), 'python':str(python),
               'scientific_source_commit': protocols['E1']['source_commit'],
               'operational_source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
               'runtime_sha256':digest(protocols['E1']['runtime']), 'code_sha256':protocols['E1']['runtime']['code_sha256'],
               'approved_lock_hashes':{p:d['sha256'] for p,d in documents.items()}, 'frozen_files':files, 'jobs':jobs,
               'environment':env, 'gpu_uuid':snapshot['gpu']['uuid'],
               'gpu_lock_path':'D:/KD-SINK-central/analysis/mechanistic_adrita_cuda0.supervisor.lock',
               'max_attempts':2, 'poll_seconds':2, 'health_interval_seconds':60, 'stale_warning_seconds':1800,
               'minimum_disk_bytes':10*2**30, 'minimum_ram_bytes':4*2**30,
               'qualified_reserved_bytes':resources['peak_reserved_bytes'], 'gpu_headroom_bytes':resources['required_gpu_headroom_bytes'],
               'readiness_health':snapshot, 'readiness_qualification_complete_sha256':sha(fresh/'QUALIFICATION_COMPLETE.json'),
               'constraints':'Exact approved v2 only; immutable panels/read-only sources; no training/seed1/2/E4/E5/Stage09/JVP/fine extension/tolerance changes; descriptive seed0; E3 requires independently verified E2 same state/panel; never include invalid outputs'}
    write(root/'CAMPAIGN_MANIFEST.json', {'schema_version':1,'payload':payload,'sha256':digest(payload)})
    write(root/'LAUNCH_READINESS.json', {'status':'PASSED', 'manifest_sha256':sha(root/'CAMPAIGN_MANIFEST.json'), 'health':snapshot,
          'logical_bundles':92, 'locks':payload['approved_lock_hashes'], 'source':payload['scientific_source_commit'],
          'runtime_sha256':payload['runtime_sha256'], 'tests_sha256':sha(tests)})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--tests',type=Path,required=True)
    args=parser.parse_args()
    build(Path(__file__).resolve().parents[1],args.root,args.python,args.tests)
