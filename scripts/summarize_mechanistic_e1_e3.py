"""Read-only, CPU-only scientific E3 and combined E1-E3 reporting."""
import argparse
import csv
import datetime
import io
import json
import math
from pathlib import Path

from audit_mechanistic_scientific import audit_injection_norms, envelope
from summarize_mechanistic_e1_e2 import (
    CONDITIONS, dump, export_csv, load_exports, load_verified, read, require, sha,
    tables, verify_behavior,
)

DIRECTIONS = ('sink', 'random', 'orthogonal', 'non_sink')
PANELS = ('discovery', 'confirmation')
DATE = '20261011'


def quantile(values, p):
    values = sorted(values)
    require(bool(values) and 0 <= p <= 1, 'invalid quantile population')
    position = (len(values) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def describe(values):
    return {'mean': math.fsum(values) / len(values), 'median': quantile(values, .5),
            'q25': quantile(values, .25), 'q75': quantile(values, .75), 'p90': quantile(values, .9)}


def repool_behavior(rows):
    n = sum(row['valid_targets'] for row in rows)
    require(n > 0, 'no scored targets')
    clean = math.fsum(row['clean_nll_sum_nats'] for row in rows) / n
    edited = math.fsum(row['edited_nll_sum_nats'] for row in rows) / n
    return {'valid_targets': n, 'clean_ce_nats': clean, 'edited_ce_nats': edited,
            'delta_ce_nats': edited - clean,
            'self_kl_nats': math.fsum(row['self_kl_sum_nats'] for row in rows) / n,
            'prediction_flip_fraction': sum(row['flip_count'] for row in rows) / n,
            'absolute_target_logprob_change_nats': math.fsum(
                row['absolute_target_logprob_change_sum_nats'] for row in rows) / n}


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row)), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def check_e1_e2(repo, root):
    receipt = read(repo / 'reports/mechanistic_e1_e2_verification_20261010.json')
    require(sha(repo / receipt['report']) == receipt['report_sha256'], 'historical E1/E2 report changed')
    require(sha(repo / 'scripts/summarize_mechanistic_e1_e2.py') == receipt['exporter_sha256'], 'E1/E2 exporter changed')
    directory = repo / 'reports/mechanistic_e1_e2_results_20261010'
    require(sha(directory / 'FILES_SHA256.json') == receipt['artifact_seal_sha256'], 'E1/E2 export seal changed')
    for name, expected in read(directory / 'FILES_SHA256.json').items():
        require(sha(directory / name) == expected, 'E1/E2 companion changed: ' + name)
    summaries, sources, audits = load_verified(root)
    _, _, doses, _ = load_exports(root, summaries)
    operations, factors, endpoints = tables(summaries)
    for name, rows in [('operations.csv', operations), ('e2_factors.csv', factors),
                       ('state_endpoints.csv', endpoints), ('e1_dose_matches.csv', doses)]:
        require((directory / name).read_bytes() == csv_bytes(rows), 'fresh E1/E2 numeric export differs: ' + name)
    require(read(directory / 'analysis.json')['sources'] == sources, 'E1/E2 selected sources changed')
    return summaries, endpoints, {'status': 'PASS', 'bundles': len(sources), 'operation_rows': len(operations),
        'factor_rows': len(factors), 'report_sha256': receipt['report_sha256'],
        'scope': 'Fresh consumed-summary/receipt/export hashes, all four numeric CSVs regenerated exactly, arithmetic and archived join checks. Prior raw-item and full semantic audits reused.'}


def state_rows(summary, panel, state, origin):
    ops, layers = summary['operations'], summary['settings']['layers']
    all_delete = ops[f'all_delete/layer{layers[0]}']
    require(all(ops[f'all_delete/layer{layer}'] == all_delete for layer in layers),
            'duplicated all-layer deletions disagree')
    base = {'panel': panel, 'state': state, 'origin': origin}
    endpoint = {**base, 'clean_ce_nats': all_delete['behavior']['clean_ce_nats'],
                'all_delete_delta_ce_nats': all_delete['behavior']['delta_ce_nats'],
                'all_delete_self_kl_nats': all_delete['behavior']['self_kl_nats'],
                'all_delete_flip_fraction': all_delete['behavior']['prediction_flip_fraction'],
                'all_delete_ppl_ratio': all_delete['behavior']['ppl_ratio']}
    for direction in DIRECTIONS:
        for metric in ('delta_ce_nats', 'self_kl_nats'):
            endpoint[f'{direction}_eta01_coarse_mean_{metric}'] = math.fsum(
                ops[f'injection/{direction}/layer{layer}/eta0.1']['behavior'][metric] for layer in layers) / len(layers)
    rescues = []
    for third, layer in enumerate(layers):
        isolated = ops[f'single_delete/layer{layer}']['behavior']['delta_ce_nats']
        conditional = ops[f'conditional_rescue/layer{layer}']['behavior']['delta_ce_nats']
        rescues.append({**base, 'third': third, 'layer': layer, 'single_delete_delta_ce_nats': isolated,
            'single_rescue_delta_ce_nats': ops[f'single_rescue/layer{layer}']['behavior']['delta_ce_nats'],
            'all_delete_delta_ce_nats': endpoint['all_delete_delta_ce_nats'],
            'conditional_rescue_delta_ce_nats': conditional,
            'conditional_loss_reduction_nats': endpoint['all_delete_delta_ce_nats'] - conditional})
    telescope_rows = []
    for order_index, telescope in enumerate(summary['diagnostics']['telescopes']):
        require(math.isclose(math.fsum(telescope['increments_ce_nats']), telescope['all_layer_delta_ce_nats'],
                             abs_tol=1e-12), 'telescope sum does not close')
        for ordinal, (layer, delta) in enumerate(zip(telescope['order'], telescope['increments_ce_nats'], strict=True)):
            telescope_rows.append({**base, 'order': 'forward' if order_index == 0 else 'reverse',
                'ordinal': ordinal, 'layer_added': layer, 'increment_ce_nats': delta,
                'all_layer_delta_ce_nats': telescope['all_layer_delta_ce_nats']})
    return endpoint, rescues, telescope_rows


def load_e3(root, e12):
    campaign = root / 'campaign_nodipc'
    manifest_path = campaign / 'E3_RECOVERY_MANIFEST.json'
    manifest = envelope(read(manifest_path))
    marker = read(campaign / 'MONITOR_COMPLETE.json')
    require(marker['manifest_sha256'] == sha(manifest_path), 'completion manifest changed')
    require(sha(marker['final_audit']) == marker['final_audit_sha256'], 'final audit changed')
    final = read(marker['final_audit'])
    require(final['status'] == 'COMPLETE_INDEPENDENTLY_AUDITED' and final['expected'] == 32
            and final['all_E2_dependencies_verified'] is True and len(final['states']) == 32, 'E3 not independently complete')
    completion = campaign / 'final_completion'
    for name, expected in read(completion / 'DERIVED_INVENTORY.json')['files'].items():
        require(sha(completion / name) == expected, 'final report artifact changed: ' + name)
    join = read(completion / 'ALL_PHASE_PORTABLE_JOIN.json')
    require(join['status'] == 'PASS' and join['compatibility']['scientific_field_diff'] == []
            and len(join['sources']) == 92, 'incompatible/incomplete cross-phase join')
    for key, summary in e12.items():
        phase, panel, state = key
        require(join['summaries'][f'{phase}_{panel}_{state.replace("/", "-")}'] == summary, 'E1/E2 cross-phase source differs')
    summaries, operations, supports, distributions, endpoints, rescues, telescopes = {}, [], [], [], [], [], []
    sources, audits, item_effects = [], [], {}
    max_repool = 0.; raw_files = 0; numeric_checks = 0
    for panel in PANELS:
        directory = campaign / 'reports' / f'E3_{panel}'
        seal = read(directory / 'REPORT_COMPLETE.json')
        require(seal['status'] == 'COMPLETE_AUDITED_MIXED_RUNTIME_EXPORT', 'unverified E3 phase export')
        for name, expected in seal['files'].items():
            require(sha(directory / name) == expected, 'E3 phase export changed: ' + name)
        phase_join = read(directory / 'PORTABLE_JOIN.json')
        proofs = read(directory / 'INDEPENDENT_PHASE_AUDIT.json')
        require(proofs['status'] == 'PASS', 'phase audit failed')
        jobs = [job for job in manifest['jobs'] if job['panel'] == panel]
        require(len(jobs) == 16, 'E3 panel grid incomplete')
        for job in jobs:
            jid, state = job['id'], job['state']; event = final['states'][jid]
            require(event['status'] == 'complete', 'partial bundle cannot be reported')
            bundle = Path(event['attempt']) / 'bundle'; record = read(bundle / 'manifest.json')
            require((bundle / 'COMPLETE').read_text().strip() == sha(bundle / 'manifest.json')
                    and not (bundle / 'FAILED.json').exists(), 'failed/unsealed E3')
            require({p.name for p in bundle.iterdir()} == set(record['files']) | {'manifest.json', 'COMPLETE'}, 'E3 extra/missing files')
            for name, expected in record['files'].items():
                if not name.startswith('item-'):
                    require(sha(bundle / name) == expected, 'E3 metadata changed: ' + name)
            summary = read(bundle / 'summary.json'); origin = event.get('origin', 'NodiPC')
            require(summary == join['summaries'][jid] == phase_join['states'][state]['summary'], 'E3 reporting summary differs')
            require(summary['identity'] == record['identity'] and summary['items'] == e12['E2', panel, state]['items'], 'E3 identity/order')
            require(summary['identity']['source']['weights_sha256'] == e12['E2', panel, state]['identity']['source']['weights_sha256'], 'E2/E3 weight mismatch')
            proof = proofs['states'][state]; audit = proof['audit']; pair = proof['E2_dependency_comparison']
            require(audit['status'] == pair['status'] == 'PASS' and audit['items'] == pair['items'] == 300
                    and audit['identity'] == summary['identity'] and audit['manifest_sha256'] == sha(bundle / 'manifest.json')
                    and pair['E3_manifest_sha256'] == audit['manifest_sha256'], 'E3 audit/dependency binding')
            if origin == 'AdritaPC':
                prior = manifest['reuse'][jid]
                require(sha(prior['audit_path']) == prior['audit_sha256'] and read(prior['audit_path']) == audit
                        and sha(prior['runner_receipt_path']) == prior['runner_receipt_sha256'], 'Adrita dual receipts changed')
            else:
                require(read(bundle.parent / 'INDEPENDENT_AUDIT.json') == audit
                        and any(read(p)['exit_code'] == 0 for p in bundle.parent.glob('VERIFIER_EXIT_*.json')), 'Nodi dual audit differs')
                require(sha(bundle.parent / 'E2_DEPENDENCY_COMPARISON.json') == event['E2_comparison_sha256'], 'E2 comparison changed')
            for layer in summary['settings']['layers']:
                original = e12['E2', panel, state]['operations'][f'delete/layer{layer}']
                successor = summary['operations'][f'single_delete/layer{layer}']
                require(all(original[key] == successor[key] for key in ('behavior', 'geometry')), 'E2/E3 measured deletion differs')
            accum = {label: [] for label in summary['operations']}; geom = {label: [] for label in accum}
            norms = {}; effects = {label: [] for label in accum}
            names = sorted(name for name in record['files'] if name.startswith('item-'))
            require(len(names) == 300, 'E3 raw coverage')
            for ordinal, name in enumerate(names):
                data = (bundle / name).read_bytes()
                import hashlib
                require(hashlib.sha256(data).hexdigest() == record['files'][name], 'raw item changed')
                item = json.loads(data); raw_files += 1
                require(item['item_id'] == summary['items'][ordinal], 'raw item order mismatch')
                raw_ops = {row['operation']: row for row in item['operations']}
                require(len(raw_ops) == len(item['operations']) and set(raw_ops) == set(accum), 'operation coverage')
                for label, row in raw_ops.items():
                    require(row['status'] == 'measured' and row['behavior']['valid_targets'] == 127, 'unavailable/wrong targets')
                    b = row['behavior']; accum[label].append(b); geom[label].append(row['geometry'])
                    effects[label].append((b['edited_nll_sum_nats'] - b['clean_nll_sum_nats']) / b['valid_targets'])
                    if label.startswith('injection/sink/'):
                        injection = row['injection']; errors = audit_injection_norms(injection, summary['settings'])
                        for direction in DIRECTIONS:
                            other = raw_ops[label.replace('injection/sink/', f'injection/{direction}/')]
                            require(other['injection'] == injection, 'equal-norm controls do not share support')
                        group = norms.setdefault(label, {'eligible_positions': 0, 'common_available_positions': 0,
                            'reference_below_floor_positions': 0, 'natural_sink_degenerate_positions': 0,
                            'max_norm_error': 0., **{d + '_unavailable_positions': 0 for d in DIRECTIONS}})
                        for field in ('eligible_positions', 'common_available_positions', 'reference_below_floor_positions', 'natural_sink_degenerate_positions'):
                            group[field] += injection[field]
                        group['max_norm_error'] = max(group['max_norm_error'], max(errors, default=0.))
                        for d in DIRECTIONS: group[d + '_unavailable_positions'] += injection['directions'][d]['unavailable_direction_positions']
                for t in item['diagnostics']['telescopes']:
                    require([x['layer_added'] for x in t['increments']] == t['order'] and
                            math.isclose(math.fsum(x['increment_ce_nats'] for x in t['increments']), t['all_layer_delta_ce_nats'], abs_tol=1e-12), 'raw telescope closure')
            for label, pooled in summary['operations'].items():
                verify_behavior(pooled['behavior'], pooled['geometry'])
                computed = repool_behavior(accum[label])
                values = [(computed[k], pooled['behavior'][k]) for k in computed]
                values += [(math.fsum(row[k] for row in geom[label]), pooled['geometry'][k])
                           for k in ('delta_nll_sum_nats', 'positive_delta_nll_sum_nats', 'negative_delta_nll_sum_nats', 'centered_logit_displacement_sum')]
                for actual, expected in values:
                    max_repool = max(max_repool, abs(actual - expected)); numeric_checks += 1
                    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), 'raw-to-summary repooling differs')
                require(pooled['measured_items'] == 300 and pooled['unavailable_items'] == 0, 'incomplete operation summary')
                operations.append({'panel': panel, 'state': state, 'origin': origin, 'operation': label,
                    **pooled['behavior'], **pooled['geometry']})
                distributions.append({'panel': panel, 'state': state, 'operation': label, 'items': 300,
                    **{f'delta_ce_{k}': v for k, v in describe(effects[label]).items()}})
            for label, counts in norms.items():
                supports.append({'panel': panel, 'state': state, 'operation': label, **counts})
            endpoint, rescue, tele = state_rows(summary, panel, state, origin)
            endpoints.append(endpoint); rescues.extend(rescue); telescopes.extend(tele)
            summaries[panel, state] = summary; item_effects[panel, state] = effects; audits.append(audit)
            sources.append({'job_id': jid, 'origin': origin, 'bundle': str(bundle), 'manifest_sha256': sha(bundle / 'manifest.json'),
                'summary_sha256': sha(bundle / 'summary.json'), 'protocol_sha256': summary['identity']['protocol_sha256'],
                'runtime_sha256': summary['identity']['runtime_sha256'], 'weights_sha256': summary['identity']['source']['weights_sha256'],
                'E2_numeric_fields_exact': pair['numeric_fields_compared']})
            print(f'{jid}: 300 raw files, 60 operation pools and support metadata verified', flush=True)
    paired = []
    for panel in PANELS:
        for label in item_effects[panel, 'C2/step500']:
            delta = [b - a for a, b in zip(item_effects[panel, 'C2/step500'][label], item_effects[panel, 'C2/step10000'][label], strict=True)]
            paired.append({'panel': panel, 'condition': 'C2', 'from_step': 500, 'to_step': 10000,
                'operation': label, 'items': len(delta), **describe(delta)})
    verification = {'status': 'PASS', 'raw_E3_files_rehashed': raw_files, 'operation_pools_recomputed': len(operations),
        'pooled_numeric_checks': numeric_checks, 'max_absolute_repool_discrepancy': max_repool,
        'max_parity_absolute_error': max(a['max_parity_absolute_error'] for a in audits),
        'max_injection_norm_error': max(a['max_norm_absolute_error'] for a in audits),
        'max_double_geometry_error': max(a['max_geometry_error'] for a in audits),
        'unavailable_item_operations': sum(a['unavailable_item_operations'] for a in audits),
        'final_audit_sha256': marker['final_audit_sha256'], 'successor_scientific_field_diff': [],
        'independent_scope': 'New complete raw-byte and scalar repooling/support/telescope reporting checks; original tensor gates and full semantic audits reused. No inference.'}
    return summaries, operations, supports, distributions, endpoints, rescues, telescopes, paired, sources, verification


def plots(output, e12, e3, e2_end, endpoints):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(CONDITIONS, ('#555555', '#0072B2', '#D55E00', '#009E73', '#CC79A7')))
    ep = {(r['panel'], r['state']): r for r in endpoints}; e2 = {(r['panel'], r['state']): r for r in e2_end}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for panel, style in [('discovery', '--'), ('confirmation', '-')]:
        for condition in CONDITIONS:
            steps = (500, 2000, 10000)
            for index, field in [(2, 'peak_delta_ce_nats'), (3, 'all_delete_delta_ce_nats'),
                                 (4, 'sink_eta01_coarse_mean_delta_ce_nats'), (5, 'sink_eta01_coarse_mean_self_kl_nats')]:
                source = e2 if index == 2 else ep
                axes.flat[index].plot(steps, [source[panel, f'{condition}/step{s}'][field] for s in steps],
                    style, marker='o', color=colors[condition], label=condition if panel == 'confirmation' and index == 3 else None)
            steps = [s for s in steps if ('E1', panel, f'{condition}/step{s}') in e12]
            for index, label, field, scale in [(0, 'none', 'baseline_sink', 1), (1, 'q_bias/native/1', 'sink_removed_fraction_item_mean', 100)]:
                vals = [e12['E1', panel, f'{condition}/step{s}']['operations'][label] for s in steps]
                y = [scale * (v['sink_fingerprint'][field] if index == 0 else v[field]) for v in vals]
                axes.flat[index].plot(steps, y, style, marker='o', color=colors[condition])
        refs = [e12['E1', panel, 'teacher']['operations']['none']['sink_fingerprint']['baseline_sink'],
            100 * e12['E1', panel, 'teacher']['operations']['q_bias/native/1']['sink_removed_fraction_item_mean'],
            e2[panel, 'teacher']['peak_delta_ce_nats'], ep[panel, 'teacher']['all_delete_delta_ce_nats'],
            ep[panel, 'teacher']['sink_eta01_coarse_mean_delta_ce_nats'], ep[panel, 'teacher']['sink_eta01_coarse_mean_self_kl_nats']]
        for ax, val in zip(axes.flat, refs): ax.axhline(val, linestyle=style, color='black', linewidth=1)
    for ax, title, ylabel in zip(axes.flat,
        ('E1: clean sink pattern', 'E1: Q-bias route removal', 'E2: observed peak isolated deletion',
         'E3: simultaneous all-layer deletion', 'E3: sink direction, eta=0.10', 'E3: sink direction, eta=0.10'),
        ('Later-half key-0 mass', 'Removed (%)', 'Delta CE (nats/target)', 'Delta CE (nats/target)', 'Three-probe mean delta CE', 'Three-probe mean self-KL')):
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel('Optimizer updates'); ax.set_xticks((500, 2000, 10000)); ax.grid(alpha=.2)
    axes[1, 0].legend(ncol=3, fontsize=8)
    fig.suptitle('Seed0: confirmation solid; discovery dashed; static teacher black\nE2 maxima are outcome-selected; E3 injection averages describe three separate interventions.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93)); save(fig, output, 'combined_trajectories'); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    dc = dict(zip(DIRECTIONS, ('#0072B2', '#555555', '#CC79A7', '#D55E00')))
    for row, state in enumerate(('teacher', 'C2/step10000')):
        s = e3['confirmation', state]
        for third, layer in enumerate(s['settings']['layers']):
            ax = axes[row, third]
            for direction in DIRECTIONS:
                doses = s['settings']['etas']
                y = [s['operations'][f'injection/{direction}/layer{layer}/eta{eta:g}']['behavior']['delta_ce_nats'] for eta in doses]
                ax.plot(doses, y, '-o', color=dc[direction], label=direction)
            ax.set_title(f'{state}, layer {layer}'); ax.set_xlabel('eta (fraction of clean entering residual norm)'); ax.set_ylabel('Delta CE (nats/target)'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8); fig.suptitle('Confirmation: all approved doses and equal-support direction controls\nTeacher/student rows use different architecture-local directions; coarse thirds do not establish circuit homology.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .92)); save(fig, output, 'e3_dose_response'); plt.close(fig)


def save(fig, output, name):
    fig.savefig(output / (name + '.png'), dpi=160)
    fig.savefig(output / (name + '.svg'), metadata={'Date': None})
    p = output / (name + '.svg')
    p.write_bytes(b'\n'.join(line.rstrip(b' \t\r') for line in p.read_bytes().splitlines()) + b'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); repo = Path(__file__).resolve().parents[1]
    e12, e2_end, check12 = check_e1_e2(repo, args.root)
    e3, ops, support, dist, endpoints, rescue, tele, paired, sources, verification = load_e3(args.root, e12)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('e3_operations.csv', ops), ('e3_injection_support.csv', support),
        ('e3_item_distributions.csv', dist), ('e3_state_endpoints.csv', endpoints), ('e3_rescues.csv', rescue),
        ('e3_telescopes.csv', tele), ('e3_C2_paired_temporal.csv', paired)]: export_csv(args.output / name, rows)
    matched = []
    e2_index = {(r['panel'], r['state']): r for r in e2_end}
    for row in endpoints:
        panel, state = row['panel'], row['state']; s = e12.get(('E1', panel, state)); a = e2_index[panel, state]
        matched.append({**row, 'weights_sha256': e3[panel, state]['identity']['source']['weights_sha256'],
            'E1_available': s is not None, 'E1_clean_sink': s['operations']['none']['sink_fingerprint']['baseline_sink'] if s else None,
            'E1_q_bias_removed_fraction': s['operations']['q_bias/native/1']['sink_removed_fraction_item_mean'] if s else None,
            'E1_k_top3_removed_fraction': s['operations']['k_top3/native/1']['sink_removed_fraction_item_mean'] if s else None,
            'E2_native_layer_mean_delta_ce_nats': a['isolated_layer_mean_delta_ce_nats'],
            'E2_observed_peak_layer': a['peak_delta_ce_layer'], 'E2_observed_peak_delta_ce_nats': a['peak_delta_ce_nats']})
    export_csv(args.output / 'matched_E1_E2_E3.csv', matched)
    plots(args.output, e12, e3, e2_end, endpoints)
    analysis = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'E1_E2_recheck': check12,
        'E3_recheck': verification, 'bundle_counts': {'E1': 28, 'E2': 32, 'E3': 32}, 'sources': sources,
        'E3_origin_counts': {origin: sum(s['origin'] == origin for s in sources) for origin in ('AdritaPC', 'NodiPC')},
        'E2_numeric_fields_exact': sum(s['E2_numeric_fields_exact'] for s in sources),
        'max_eta0_absolute_delta_ce': max(abs(r['delta_ce_nats']) for r in ops if r['operation'].endswith('/eta0')),
        'max_single_rescue_absolute_delta_ce': max(abs(r['delta_ce_nats']) for r in ops if r['operation'].startswith('single_rescue/')),
        'support_excluded_positions': sum(r['eligible_positions'] - r['common_available_positions'] for r in support if r['operation'].endswith('/eta0.1')),
        'endpoints': endpoints, 'interpretation': 'Descriptive seed0. Within-state equal relative norm; cross-state direction/basis differs. No model inference, hypothesis tests or mediation/equivalence claims.'}
    dump(args.output / 'analysis.json', analysis)
    dump(args.output / 'FILES_SHA256.json', {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name != 'FILES_SHA256.json'})
    print(json.dumps({'status': 'PASS', 'E3_operation_rows': len(ops), 'support_rows': len(support), 'verification': verification}), flush=True)


if __name__ == '__main__':
    main()
