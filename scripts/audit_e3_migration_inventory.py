"""Read-only reconstruction of all original jobs, with fresh dual audits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import traceback

from audit_mechanistic_scientific import audit, read, write, sha, envelope, require


def audit_job(task):
    imported, output, job, original_root = task
    imported, output = Path(imported), Path(output)
    from sinklab.mechanistic_run import verify_bundle
    lock = read(imported / f"campaign/input_provenance/{job['phase']}.approved.json")
    protocol = envelope(lock)
    require(lock['sha256'] == job['approved_sha256'], 'original manifest/lock differs')
    panel = imported / 'campaign/input_provenance/panel.json'
    attempts = []
    for attempt in sorted((imported / 'campaign/attempts' / job['id']).glob('attempt_*')):
        bundle = attempt / 'bundle'
        item_paths = sorted(bundle.glob('item-*.json'))
        row = {'attempt': str(attempt), 'original_attempt': str(Path(original_root) / 'attempts' / job['id'] / attempt.name),
               'items': len(item_paths), 'complete_marker': (bundle / 'COMPLETE').exists(),
               'failed_marker': (bundle / 'FAILED.json').exists(), 'status': 'partially_completed' if item_paths else 'interrupted'}
        folder = output / job['id'] / attempt.name
        folder.mkdir(parents=True, exist_ok=True)
        if (attempt / 'INDEPENDENT_AUDIT.json').exists():
            row['historical_audit_sha256'] = sha(attempt / 'INDEPENDENT_AUDIT.json')
        if (bundle / 'invocation.json').exists():
            identity = read(bundle / 'invocation.json')['identity']
            row.update(protocol_sha256=identity.get('protocol_sha256'), runtime_sha256=identity.get('runtime_sha256'),
                       source_weights_sha256=identity.get('source', {}).get('weights_sha256'))
        if row['complete_marker'] and not row['failed_marker']:
            try:
                if (folder / 'RUNNER_VERIFICATION.json').exists() and (folder / 'INDEPENDENT_AUDIT.json').exists():
                    runner = read(folder / 'RUNNER_VERIFICATION.json')
                    receipt = read(folder / 'INDEPENDENT_AUDIT.json')
                    require(runner['status'] == receipt['status'] == 'PASS' and
                            runner['manifest_sha256'] == receipt['manifest_sha256'] == sha(bundle / 'manifest.json') and
                            runner['items'] == receipt['items'] == 300, 'saved verified audit changed')
                else:
                    summary = verify_bundle(bundle)
                    require(len(summary['items']) == protocol['panel_counts'][job['panel']] == 300, 'full panel coverage')
                    runner = {'status': 'PASS', 'manifest_sha256': sha(bundle / 'manifest.json'), 'items': len(summary['items']),
                              'scope': 'original frozen numerical executor verify_bundle; complete hash/coverage/reaggregation'}
                    if not (folder / 'RUNNER_VERIFICATION.json').exists(): write(folder / 'RUNNER_VERIFICATION.json', runner)
                    receipt = audit(bundle, lock, job, panel_path=panel)
                    write(folder / 'INDEPENDENT_AUDIT.json', receipt)
                row.update(status='independently_complete_valid', manifest_sha256=runner['manifest_sha256'],
                           independent_audit_path=str(folder / 'INDEPENDENT_AUDIT.json'),
                           runner_receipt_path=str(folder / 'RUNNER_VERIFICATION.json'))
            except Exception as error:
                row.update(status='invalidated', error=str(error))
                write(folder / 'AUDIT_FAILURE.json', {'status': 'FAIL', 'detail': str(error), 'traceback': traceback.format_exc()})
        elif row['failed_marker']:
            row['failure'] = read(bundle / 'FAILED.json')
        attempts.append(row)
    valid = [r for r in attempts if r['status'] == 'independently_complete_valid']
    # Multiple admissible attempts with different scientific item bytes conflict.
    if len(valid) > 1:
        signatures = [tuple(sha(p) for p in sorted((Path(r['attempt']) / 'bundle').glob('item-*.json'))) for r in valid]
        require(len(set(signatures)) == 1, f"conflicting valid scientific attempts: {job['id']}")
    chosen = valid[-1] if valid else attempts[-1] if attempts else None
    result = {'job_id': job['id'], 'phase': job['phase'], 'state': job['state'], 'panel': job['panel'],
              'status': chosen['status'] if chosen else 'absent', 'attempts': attempts, 'selected': chosen,
              'original_protocol_sha256': lock['sha256'], 'original_runtime_sha256': __import__('audit_mechanistic_scientific').digest(protocol['runtime']),
              'source_checkpoint_sha256': protocol['sources'][job['state']]['weights']['sha256']}
    print(f"AUDITED {job['id']} {result['status']}", flush=True)
    return result


def reconstruct(imported, output, workers):
    require((imported.parent / 'ARCHIVE_VERIFICATION.json').is_file(), 'archive verification required')
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / 'RECOVERY_INVENTORY_COMPLETE.json').exists(), 'already completed reconstruction')
    manifest = envelope(read(imported / 'campaign/CAMPAIGN_MANIFEST.json'))
    require(len(manifest['jobs']) == 92 and len({j['id'] for j in manifest['jobs']}) == 92, 'original exact92 inventory')
    latest = {}
    for line in (imported / 'campaign/state_ledger.jsonl').read_text(encoding='utf-8').splitlines():
        event = __import__('json').loads(line); latest[event['job_id']] = event
    tasks = [(str(imported), str(output / 'audits'), j, manifest['root']) for j in manifest['jobs']]
    # Windows worker recycling stalled with an empty pool in the first attempt.
    # Keep a bounded persistent pool; tasks release records before returning.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(audit_job, tasks))
    by_id = {r['job_id']: r for r in results}
    for row in results:
        row['historical_ledger_status'] = latest.get(row['job_id'], {}).get('status', 'absent')
        row['historical_ledger_reason'] = latest.get(row['job_id'], {}).get('reason')
    matrix = []
    for row in results:
        if row['phase'] != 'E3': continue
        dependency = by_id[row['job_id'].replace('E3_', 'E2_', 1)]
        chosen = row['selected'] or {}
        matrix.append({'job_id': row['job_id'], 'state': row['state'], 'panel': row['panel'],
                       'old_attempt_path': chosen.get('original_attempt'), 'imported_attempt_path': chosen.get('attempt'),
                       'item_count': chosen.get('items', 0), 'verified_status': row['status'],
                       'old_protocol_sha256': row['original_protocol_sha256'], 'old_runtime_sha256': row['original_runtime_sha256'],
                       'source_checkpoint_sha256': row['source_checkpoint_sha256'], 'E2_dependency': dependency['job_id'],
                       'E2_dependency_status': dependency['status'], 'E2_audit_path': (dependency['selected'] or {}).get('independent_audit_path'),
                       'historical_ledger_status': row['historical_ledger_status'],
                       'action': 'reuse_after_comparability_gate' if row['status'] == 'independently_complete_valid' else 'fresh_entire_state_panel_attempt'})
    require(len(matrix) == 32, 'exact E3 matrix')
    write(output / 'ALL_92_JOBS.json', {'status': 'RECONSTRUCTED', 'original_manifest_sha256': sha(imported / 'campaign/CAMPAIGN_MANIFEST.json'), 'jobs': results})
    write(output / 'E3_RECOVERY_MATRIX.json', {'rows': matrix})
    with (output / 'E3_RECOVERY_MATRIX.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix[0])); writer.writeheader(); writer.writerows(matrix)
    counts = {phase: {status: sum(r['phase'] == phase and r['status'] == status for r in results)
                     for status in sorted({r['status'] for r in results})} for phase in ('E1', 'E2', 'E3')}
    write(output / 'RECOVERY_INVENTORY_COMPLETE.json', {'status': 'PASS', 'counts': counts,
          'all_E2_dependencies_valid': all(r['E2_dependency_status'] == 'independently_complete_valid' for r in matrix),
          'files': {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}})
    print(counts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--imported', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    reconstruct(args.imported.resolve(), args.output.resolve(), args.workers)
