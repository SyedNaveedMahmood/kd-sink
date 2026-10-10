"""CPU-only descriptive export of sealed Adrita E1/E2 results; no inference."""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

PROTOCOLS = {
    'E1': 'aa280a9a1244c7e00bff5e88fe342548ac587553788c24dd072eeab8542528f6',
    'E2': '1ba9dff94bdd6735cb1cd1c871a655094a91bd1b96c746e9f26f14f942c3d586',
}
RUNTIME = 'df8b0b3258a9e9bd236e8a9d7281e95a68d3e5555733e76404fda338db5ada0b'
PANEL = '509a90c6039cd90d7d4f6986ac7fe3b75468609073b4808baa8b5bae5294c7f7'
CONDITIONS = ('C1', 'C2', 'C3', 'C5', 'C6')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def expected_states(phase):
    return {'teacher'} | {f'{c}/step{s}' for c in CONDITIONS for s in (500, 2000, 10000)
                          if phase == 'E2' or c not in ('C1', 'C5') or s != 2000}


def verify_behavior(behavior, geometry):
    require(behavior['item_count'] == 300 and behavior['valid_targets'] == 38100,
            'unexpected item/target population')
    ce = behavior['edited_ce_nats'] - behavior['clean_ce_nats']
    require(math.isclose(ce, behavior['delta_ce_nats'], abs_tol=1e-12), 'CE arithmetic')
    require(math.isclose(math.exp(ce), behavior['ppl_ratio'], rel_tol=1e-12), 'PPL units')
    require(math.isclose(geometry['delta_nll_sum_nats'] / 38100, ce, abs_tol=1e-6),
            'independent double-precision geometry does not agree with FP32 CE')
    signed = geometry['positive_delta_nll_sum_nats'] + geometry['negative_delta_nll_sum_nats']
    require(math.isclose(signed, geometry['delta_nll_sum_nats'], abs_tol=1e-8), 'signed loss closure')


def load_verified(root):
    """Rehash consumed bytes; use SHA-bound prior full runner/independent audits."""
    reconstruction = root / 'reconstruction01'
    seal = read(reconstruction / 'RECOVERY_INVENTORY_COMPLETE.json')
    require(seal['status'] == 'PASS', 'campaign reconstruction did not pass')

    def sealed(path):
        relative = path.relative_to(reconstruction).as_posix()
        require(sha(path) == seal['files'][relative], f'reconstruction receipt changed: {relative}')

    inventory_path = reconstruction / 'ALL_92_JOBS.json'
    sealed(inventory_path)
    summaries, sources, audits = {}, [], []
    for job in read(inventory_path)['jobs']:
        if job['phase'] not in PROTOCOLS:
            continue
        require(job['status'] == 'independently_complete_valid', f'unverified job {job["job_id"]}')
        selected = job['selected']
        bundle = Path(selected['attempt']) / 'bundle'
        manifest_sha = sha(bundle / 'manifest.json')
        require(manifest_sha == selected['manifest_sha256'], 'manifest identity changed')
        require((bundle / 'COMPLETE').read_text().strip() == manifest_sha and not (bundle / 'FAILED.json').exists(),
                'unsealed or failed bundle')
        manifest = read(bundle / 'manifest.json')
        require(sha(bundle / 'summary.json') == manifest['files']['summary.json'], 'summary bytes changed')
        summary = read(bundle / 'summary.json')
        identity = summary['identity']
        key = (job['phase'], job['panel'], job['state'])
        require(key not in summaries, 'duplicate logical job')
        require(identity == manifest['identity'], 'summary/manifest identity mismatch')
        require(identity['phase'] == key[0] and identity['panel'] == key[1] and identity['state'] == key[2],
                'wrong logical identity')
        require(identity['protocol_sha256'] == PROTOCOLS[key[0]] and identity['runtime_sha256'] == RUNTIME
                and identity['panel_sha256'] == PANEL and identity['seed'] == 0
                and identity['context_length'] == 128 and identity['engineering_only'] is False,
                'non-original scientific provenance')
        require(identity['source']['weights_sha256'] == job['source_checkpoint_sha256'], 'source mismatch')
        require(len(summary['items']) == 300 and len(set(summary['items'])) == 300, 'item coverage')
        receipts = {}
        for label in ('independent_audit_path', 'runner_receipt_path'):
            path = Path(selected[label])
            sealed(path)
            receipt = read(path)
            require(receipt['status'] == 'PASS' and receipt['items'] == 300
                    and receipt['manifest_sha256'] == manifest_sha, 'unbound/failed audit')
            if label == 'independent_audit_path':
                require(receipt['identity'] == identity, 'audit identity mismatch')
                audits.append(receipt)
            receipts[label] = {'path': str(path), 'sha256': sha(path)}
        for operation in summary['operations'].values():
            require(operation['measured_items'] == 300 and operation['unavailable_items'] == 0,
                    'unexpected unavailable operation')
            verify_behavior(operation['behavior'], operation['geometry'])
        summaries[key] = summary
        sources.append({'job_id': job['job_id'], 'bundle': str(bundle), 'items': 300,
                        'manifest_sha256': manifest_sha, 'summary_sha256': sha(bundle / 'summary.json'),
                        'weights_sha256': job['source_checkpoint_sha256'], **receipts})
    for phase in PROTOCOLS:
        for panel in ('discovery', 'confirmation'):
            require({k[2] for k in summaries if k[:2] == (phase, panel)} == expected_states(phase), 'grid coverage')
    for panel in ('discovery', 'confirmation'):
        order = summaries['E1', panel, 'teacher']['items']
        for (phase, population, state), summary in summaries.items():
            if population != panel:
                continue
            require(summary['items'] == order, 'within-panel order mismatch')
            if phase == 'E1':
                other = summaries['E2', panel, state]
                require(summary['identity']['source'] == other['identity']['source'], 'cross-phase weights')
                baseline = summary['operations']['none']['behavior']
                require(all(abs(baseline['clean_ce_nats'] - op['behavior']['clean_ce_nats']) < 1e-12
                            for op in other['operations'].values()), 'cross-phase clean CE')
    require(set(summaries['E1', 'discovery', 'teacher']['items']).isdisjoint(
            summaries['E1', 'confirmation', 'teacher']['items']), 'panel item overlap')
    return summaries, sources, audits


def load_exports(root, summaries):
    """Check archived descriptive exports against seals and current source summaries."""
    distributions, temporal, doses, provenance = [], [], [], []
    for phase in PROTOCOLS:
        for panel in ('discovery', 'confirmation'):
            directory = root / 'imported_adrita/campaign/reports' / f'{phase}_{panel}'
            seal = read(directory / 'REPORT_COMPLETE.json')
            require(seal['status'] == 'COMPLETE_AUDITED_EXPORT' and seal['protocol_sha256'] == PROTOCOLS[phase],
                    'wrong report identity')
            names = ['complete_join.json', 'item_distributions.csv', 'paired_temporal_contrasts.csv']
            if phase == 'E1':
                names.append('observed_dose_comparisons.json')
            for name in names:
                require(sha(directory / name) == seal['files'][name], f'report export changed: {name}')
            join = read(directory / 'complete_join.json')
            require(set(join['states']) == expected_states(phase), 'report state coverage')
            for state, row in join['states'].items():
                require(row['summary'] == summaries[phase, panel, state], 'archived report/source mismatch')
            for name, target in [('item_distributions.csv', distributions), ('paired_temporal_contrasts.csv', temporal)]:
                with (directory / name).open(encoding='utf-8', newline='') as stream:
                    target.extend({'phase': phase, 'panel': panel, **row} for row in csv.DictReader(stream))
            if phase == 'E1':
                doses.extend({'panel': panel, **row} for row in read(directory / 'observed_dose_comparisons.json'))
            provenance.append({'phase': phase, 'panel': panel, 'seal_sha256': sha(directory / 'REPORT_COMPLETE.json'),
                               'consumed_files': {name: sha(directory / name) for name in names}})
    return distributions, temporal, doses, provenance


def average(values):
    values = list(values)
    return math.fsum(values) / len(values)


def tables(summaries):
    operations, factors, endpoints = [], [], []
    for (phase, panel, state), summary in sorted(summaries.items()):
        for label, row in summary['operations'].items():
            operations.append({'phase': phase, 'panel': panel, 'state': state, 'operation': label,
                               **row['behavior'], **row['geometry'],
                               **{k: v for k, v in row.items() if k.endswith('_item_mean')},
                               **row.get('sink_fingerprint', {})})
        if phase != 'E2':
            continue
        ops = summary['operations']
        baseline = next(iter(ops.values()))['behavior']
        peak = max(ops, key=lambda k: ops[k]['behavior']['delta_ce_nats'])
        endpoint = {'panel': panel, 'state': state, 'clean_ce_nats': baseline['clean_ce_nats'],
                    'clean_ppl': baseline['clean_ppl'], 'clean_accuracy_fraction': baseline['clean_accuracy_fraction'],
                    'native_layers': len(ops), 'isolated_layer_mean_delta_ce_nats': average(
                        x['behavior']['delta_ce_nats'] for x in ops.values()),
                    'peak_delta_ce_layer': int(peak.split('layer')[1]),
                    'peak_delta_ce_nats': ops[peak]['behavior']['delta_ce_nats'],
                    'peak_self_kl_nats': ops[peak]['behavior']['self_kl_nats'],
                    'positive_delta_ce_layers': sum(x['behavior']['delta_ce_nats'] > 0 for x in ops.values())}
        for support in ('all_q_ge1', 'second_half_queries'):
            for field in ('head_sink_mass_mean', 'relative_projected_delta_mean', 'cancellation_ratio_mean'):
                endpoint[support + '_' + field] = average(x[support][field] for x in
                                                          summary['diagnostics']['factor_summary'].values())
        for layer, supports in summary['diagnostics']['factor_summary'].items():
            for support, row in supports.items():
                factors.append({'panel': panel, 'state': state, 'layer': int(layer), 'support': support,
                                **{k: v for k, v in row.items() if not isinstance(v, list)},
                                **{k + '_head_mean': average(v) for k, v in row.items() if isinstance(v, list)}})
        e1 = summaries.get(('E1', panel, state))
        if e1:
            endpoint['clean_sink_later_half'] = e1['operations']['none']['sink_fingerprint']['baseline_sink']
            endpoint['clean_teacher_kl_nats'] = e1['operations']['none']['behavior']['teacher_kl_nats']
        endpoints.append(endpoint)
    return operations, factors, endpoints


def export_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(output, summaries, endpoints):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    colors = dict(zip(CONDITIONS, ('#555555', '#0072B2', '#D55E00', '#009E73', '#CC79A7')))
    end = {(x['panel'], x['state']): x for x in endpoints}
    for panel, style in [('discovery', '--'), ('confirmation', '-')]:
        for condition in CONDITIONS:
            steps = [s for s in (500, 2000, 10000) if ('E1', panel, f'{condition}/step{s}') in summaries]
            y = [summaries['E1', panel, f'{condition}/step{s}']['operations']['none']['sink_fingerprint']['baseline_sink'] for s in steps]
            axes[0, 0].plot(steps, y, style, marker='o', color=colors[condition], label=condition if panel == 'confirmation' else None)
            y = [100 * summaries['E1', panel, f'{condition}/step{s}']['operations']['q_bias/native/1']['sink_removed_fraction_item_mean'] for s in steps]
            axes[0, 1].plot(steps, y, style, marker='o', color=colors[condition])
            for axis, field in [(axes[1, 0], 'isolated_layer_mean_delta_ce_nats'), (axes[1, 1], 'peak_delta_ce_nats')]:
                axis.plot((500, 2000, 10000), [end[panel, f'{condition}/step{s}'][field] for s in (500, 2000, 10000)],
                          style, marker='o', color=colors[condition])
        axes[0, 0].axhline(summaries['E1', panel, 'teacher']['operations']['none']['sink_fingerprint']['baseline_sink'], color='black', linestyle=style, linewidth=1)
        axes[0, 1].axhline(100 * summaries['E1', panel, 'teacher']['operations']['q_bias/native/1']['sink_removed_fraction_item_mean'], color='black', linestyle=style, linewidth=1)
        for axis, field in [(axes[1, 0], 'isolated_layer_mean_delta_ce_nats'), (axes[1, 1], 'peak_delta_ce_nats')]:
            axis.axhline(end[panel, 'teacher'][field], color='black', linestyle=style, linewidth=1)
    for axis, title, unit in zip(axes.flat,
        ('E1: clean sink pattern', 'E1: Q-bias removal, alpha=1', 'E2: mean isolated-layer loss change', 'E2: largest isolated-layer loss change'),
        ('Later-half key-0 attention mass', 'Sink removed (%)', 'Delta CE (nats/target)', 'Delta CE (nats/target)')):
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_xlabel('Optimizer updates')
        axis.set_xticks((500, 2000, 10000))
        axis.grid(alpha=.2)
    axes[0, 0].legend(ncol=3, fontsize=8)
    figure.suptitle('Seed 0: solid = confirmation; dashed = discovery; black = static teacher\nLayer averages and maxima are descriptive, not simultaneous-deletion effects.', fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(output / 'trajectories.svg', metadata={'Date': None})
    svg = output / 'trajectories.svg'
    svg.write_bytes(b'\n'.join(line.rstrip(b' \t\r') for line in svg.read_bytes().splitlines()) + b'\n')
    figure.savefig(output / 'trajectories.png', dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summaries, sources, audits = load_verified(args.root)
    distributions, temporal, doses, archived_exports = load_exports(args.root, summaries)
    operations, factors, endpoints = tables(summaries)
    args.output.mkdir(parents=True, exist_ok=True)
    export_csv(args.output / 'operations.csv', operations)
    export_csv(args.output / 'e2_factors.csv', factors)
    export_csv(args.output / 'state_endpoints.csv', endpoints)
    export_csv(args.output / 'e1_dose_matches.csv', doses)
    plot_summary(args.output, summaries, endpoints)
    # Small machine-readable companion; bulk item distributions remain external.
    dump(args.output / 'analysis.json', {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'analysis': 'descriptive-only; existing Adrita results; no model inference', 'protocols': PROTOCOLS,
         'runtime_sha256': RUNTIME, 'panel_sha256': PANEL, 'sources': sources, 'archived_exports': archived_exports,
         'bundle_counts': {'E1': 28, 'E2': 32}, 'item_records': {'E1': 8400, 'E2': 9600},
         'operation_rows': len(operations), 'unavailable_item_operations': sum(x['unavailable_item_operations'] for x in audits),
         'max_parity_absolute_error': max(x['max_parity_absolute_error'] for x in audits),
         'max_geometry_error': max(x['max_geometry_error'] for x in audits),
         'no_op_max_absolute_delta_ce': max(abs(row['behavior']['delta_ce_nats']) for key, summary in summaries.items()
             if key[0] == 'E1' for label, row in summary['operations'].items()
             if label == 'none' or label.startswith('reapply_q/') or label.endswith('/0')),
         'endpoints': endpoints,
         'published_distributions': [x for x in distributions if x['phase'] == 'E2' and x['metric'] == 'delta_ce_nats'
             and (x['state'] == 'teacher' or x['state'].endswith('10000'))
             and x['operation'] == 'delete/layer' + str(next(y['peak_delta_ce_layer'] for y in endpoints
                    if y['panel'] == x['panel'] and y['state'] == x['state']))],
         'c2_paired_temporal': [x for x in temporal if x['condition'] == 'C2' and x['metric'] == 'delta_ce_nats'
             and ((x['phase'] == 'E1' and x['operation'] in ('q_bias/native/1', 'k_top3/native/1'))
                  or (x['phase'] == 'E2' and x['operation'] == 'delete/layer13'))],
         'matched_dose_counts': {panel: {status: sum(x['status'] == status and x['panel'] == panel for x in doses)
            for status in sorted({x['status'] for x in doses})} for panel in ('discovery', 'confirmation')}})
    dump(args.output / 'FILES_SHA256.json', {p.name: sha(p) for p in sorted(args.output.iterdir())
         if p.is_file() and p.name != 'FILES_SHA256.json'})
    print(json.dumps({'status': 'PASS', 'bundles': len(sources), 'operations': len(operations),
                      'factor_rows': len(factors), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
