"""Independent stdlib-only seal, identity, coverage and scalar arithmetic audit.

No executor imports. This checks recorded measurements; it cannot reconstruct
unserialized tensors or independently reperform a componentwise relative gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def envelope(document):
    require(set(document) == {'schema_version', 'payload', 'sha256'} and document['schema_version'] == 1, 'envelope schema')
    require(digest(document['payload']) == document['sha256'], 'envelope SHA mismatch')
    return document['payload']


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value) + b'\n')


def near(a, b):
    # Arithmetic reduction check, not a scientific/engineering acceptance gate.
    require(math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-12), f'scalar reduction differs: {a} != {b}')


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite measurement')
    elif isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, list):
        for v in value: finite(v)


def audit(root, lock=None, job=None, heartbeat=None):
    root = Path(root)
    require(not (root / 'FAILED.json').exists(), 'failed attempt')
    m = read(root / 'manifest.json')
    require((root / 'COMPLETE').read_text().strip() == sha(root / 'manifest.json'), 'COMPLETE hash')
    require({p.name for p in root.iterdir()} == set(m['files']) | {'manifest.json', 'COMPLETE'}, 'missing/extra artifact')
    for name, expected in m['files'].items():
        require(Path(name).name == name and not (root / name).is_symlink(), 'unsafe file path')
        require(sha(root / name) == expected, f'file SHA: {name}')
    invocation, summary = read(root / 'invocation.json'), read(root / 'summary.json')
    identity, settings = invocation['identity'], invocation['settings']
    require(identity == summary['identity'] == m['identity'], 'bundle identity mismatch')
    require(summary['settings'] == settings and identity['seed'] == 0, 'settings/seed mismatch')
    panel_items = None
    if lock is not None:
        payload = envelope(lock)
        require(payload['status'] == 'approved' and payload['production_ready'] is True, 'unapproved protocol')
        require(identity['engineering_only'] is False and identity['protocol_sha256'] == lock['sha256'], 'protocol identity')
        require(identity['phase'] == payload['phase'] and identity['state'] in payload['grid'], 'phase/state')
        require(job is not None and all(identity[k] == job[k] for k in ('phase', 'state', 'panel')), 'requested state identity')
        require(identity['runtime_sha256'] == digest(payload['runtime']), 'runtime identity')
        require(identity['panel_sha256'] == sha(payload['panel']['path']) == payload['panel']['sha256'], 'panel identity')
        require(settings == payload['settings'][identity['state']], 'settings differ from lock')
        panel_items = read(payload['panel']['path'])[identity['panel']]
        require(invocation['items'] == [p['id'] for p in panel_items], 'panel membership/order')
        for key, state in (('source', identity['state']), ('teacher_source', 'teacher')):
            reference, observed = payload['sources'][state], identity[key]
            require(observed['weights_sha256'] == reference['weights']['sha256'] and
                    observed['config_sha256'] == reference['config']['sha256'] and
                    Path(observed['weights_path']).resolve() == Path(reference['weights']['path']).resolve() and
                    observed['source_identity'] == reference.get('identity'), 'checkpoint/teacher identity')
    records = []
    for name in sorted(m['files']):
        if name.startswith('item-'):
            records.append(read(root / name))
            if heartbeat: heartbeat()
    require([r['item_id'] for r in records] == summary['items'] == invocation['items'], 'item coverage/order')
    require(len(set(summary['items'])) == len(records) > 0, 'duplicate/empty items')
    expected_ops = list(summary['operations'])
    phase = identity['phase']
    if phase == 'E1':
        declared = {'none'} | {f'reapply_q/{s["name"]}' for s in settings['scopes']}
        declared |= {f'{route}/{s["name"]}/{a:g}' for s in settings['scopes'] for a in settings['alphas']
                     for route in ['q_bias','k_top3',*[f'k_random{i}' for i in range(5)]]}
        declared |= {f'{route}/{a:g}' for a in settings['alphas'] for route in ('epe_transport','position0_to1')}
    elif phase == 'E2': declared = {f'delete/layer{i}' for i in settings['layers']}
    else:
        declared = {f'{route}/layer{i}' for i in settings['layers'] for route in ('single_delete','single_rescue','all_delete','conditional_rescue')}
        declared |= {f'injection/{d}/layer{i}/eta{a:g}' for i in settings['layers'] for a in settings['etas'] for d in ('sink','random','orthogonal','non_sink')}
    require(set(expected_ops) == declared, 'operations differ from declared specification')
    parity, geometry, norm, unavailable = [], [], [], 0
    for index, record in enumerate(records):
        if heartbeat: heartbeat()
        finite(record)
        require(record['identity'] == identity, 'per-item identity')
        operations = {r['operation']: r for r in record['operations']}
        require(len(operations) == len(record['operations']) and set(operations) == set(expected_ops), 'operation coverage')
        targets = None if panel_items is None else [(0, q) for q in range(127)
            if panel_items[index]['attention_mask'][q] and panel_items[index]['attention_mask'][q + 1]]
        for op, row in operations.items():
            require(row['status'] in ('measured', 'unavailable'), 'unknown measurement status')
            for check in row.get('parity_checks', []):
                require(check['atol'] == settings['atol'] and check['rtol'] == settings['rtol'] and check['max_absolute_error'] >= 0, 'parity policy')
                parity.append(check['max_absolute_error'])
            if row['status'] == 'unavailable':
                require(identity['phase'] == 'E3' and row['reason'] == 'no_common_normalizable_direction_support' and
                        row['injection']['common_available_positions'] == 0 and row['injection']['eta'] != 0 and
                        row['behavior'] is None and row['geometry'] is None, 'unjustified unavailable result')
                unavailable += 1
                continue
            g, b = row['geometry'], row['behavior']
            ts = g['target_records']
            require(len(ts) == g['valid_targets'] == b['valid_targets'], 'target count')
            if targets is not None:
                require([(t['batch'], t['query']) for t in ts] == targets, 'target mask/order')
            for t in ts:
                require(t['target_position'] == t['query'] + 1, 'next-token shift')
                actual = t['edited_target_nll_nats'] - t['clean_target_nll_nats']
                predicted = -t['target_logit_change'] + t['log_normalizer_change']
                error = abs(actual - predicted)
                require(error <= settings.get('geometry_atol', settings['atol']), 'double loss geometry gate')
                near(actual, t['delta_nll_nats']); geometry.append(error)
            for name, values in (
                ('delta_nll_sum_nats', [t['delta_nll_nats'] for t in ts]),
                ('positive_delta_nll_sum_nats', [t['delta_nll_nats'] for t in ts if t['delta_nll_nats'] > 0]),
                ('negative_delta_nll_sum_nats', [t['delta_nll_nats'] for t in ts if t['delta_nll_nats'] < 0]),
                ('centered_logit_displacement_sum', [t['centered_logit_displacement_l2'] for t in ts])):
                near(math.fsum(values), g[name])
            if 'injection' in row:
                injection = row['injection']
                require(all(injection[k] == settings[k] for k in ('control_seed','norm_floor','query_min','nonsink_keys','reference')), 'injection definition changed')
                require(float(op.split('/eta')[1]) == injection['eta'], 'delivered eta label')
                support = injection['common_support_mask'][0]
                require(all(s in (0, 1) for s in support), 'nonbinary support')
                require(sum(support) == injection['common_available_positions'], 'common support')
                require(all(not s or q >= injection['query_min'] for q, s in enumerate(support)), 'injection causal support')
                if panel_items is not None:
                    require(all(not s or panel_items[index]['attention_mask'][q] for q, s in enumerate(support)), 'injection padding support')
                requested = injection['eta'] * injection['reference_residual_norm_sum']
                for direction in injection['directions'].values():
                    near(requested, direction['requested_norm_sum'])
                    require(direction['injected_positions'] == sum(support), 'norm direction support')
                    require(direction['max_norm_error'] >= 0 and direction['actual_norm_sum'] >= 0, 'invalid norm measurement')
                    require(abs(direction['actual_norm_sum'] - requested) <= len(support)*settings.get('norm_atol', settings['atol']) + settings.get('norm_rtol', settings['rtol'])*requested, 'aggregate norm agreement')
                    norm.append(direction['max_norm_error'])
            if op == 'none' or op.startswith('single_rescue/') or op.endswith('/eta0'):
                require(b['clean_nll_sum_nats'] == b['edited_nll_sum_nats'] and b['flip_count'] == 0, 'no-op/restoration closure')
        for telescope in record['diagnostics'].get('telescopes', []):
            require([i['layer_added'] for i in telescope['increments']] == telescope['order'], 'telescope order')
            near(math.fsum(i['increment_ce_nats'] for i in telescope['increments']), telescope['all_layer_delta_ce_nats'])
    for op, pooled in summary['operations'].items():
        rows = [next(r for r in record['operations'] if r['operation'] == op) for record in records]
        measured = [r for r in rows if r['status'] == 'measured']
        require(pooled['measured_items'] == len(measured) and pooled['unavailable_items'] == len(rows) - len(measured), 'undefined count')
        if not measured:
            require(pooled['behavior'] is None, 'unavailable aggregate')
            continue
        n = sum(r['behavior']['valid_targets'] for r in measured)
        require(n == pooled['behavior']['valid_targets'], 'pooled target count')
        for name, field in (('clean_ce_nats', 'clean_nll_sum_nats'), ('edited_ce_nats', 'edited_nll_sum_nats'),
                            ('self_kl_nats', 'self_kl_sum_nats'), ('prediction_flip_fraction', 'flip_count'),
                            ('absolute_target_logprob_change_nats', 'absolute_target_logprob_change_sum_nats')):
            near(math.fsum(r['behavior'][field] for r in measured) / n, pooled['behavior'][name])
        near(pooled['behavior']['edited_ce_nats'] - pooled['behavior']['clean_ce_nats'], pooled['behavior']['delta_ce_nats'])
        for field in ('delta_nll_sum_nats', 'positive_delta_nll_sum_nats', 'negative_delta_nll_sum_nats', 'centered_logit_displacement_sum'):
            near(math.fsum(r['geometry'][field] for r in measured), pooled['geometry'][field])
    if phase in ('E2','E3'):
        for layer in settings['layers']:
            fs = [next(o['factors'] for o in record['operations'] if o.get('layer') == layer and 'factors' in o) for record in records]
            for support in ('all_q_ge1','second_half_queries'):
                entries=[f[support] for f in fs]
                pooled=summary['diagnostics']['factor_summary'][str(layer)][support]
                n=sum(e['real_query_count'] for e in entries)
                require(n == pooled['real_query_count'] and n > 0, 'factor query support')
                for field in ('head_sink_mass_mean','projected_delta_norm_mean','relative_projected_delta_mean'):
                    near(math.fsum(e[field]*e['real_query_count'] for e in entries)/n,pooled[field])
                for field in ('per_head_conditional_value_norm_mean','per_head_value_contrast_norm_mean','per_head_local_delta_norm_mean'):
                    for h,value in enumerate(pooled[field]): near(math.fsum(e[field][h]*e['real_query_count'] for e in entries)/n,value)
                for h,value in enumerate(pooled['per_head_sink_value_norm_item_mean']):near(math.fsum(e['per_head_sink_value_norm'][h] for e in entries)/len(entries),value)
                count=sum(e['cancellation_ratio_defined_queries'] for e in entries)
                require(count == pooled['cancellation_ratio_defined_queries'], 'cancellation support count')
                if count:near(math.fsum(e['cancellation_ratio_mean']*e['cancellation_ratio_defined_queries'] for e in entries if e['cancellation_ratio_defined_queries'])/count,pooled['cancellation_ratio_mean'])
                else:require(pooled['cancellation_ratio_mean'] is None, 'undefined cancellation')
                for e in entries:
                    for value,count in zip(e['per_head_cosine_with_projected_sum_mean'],e['per_head_cosine_defined_queries'],strict=True):
                        require((value is None and count == 0) or (value is not None and count > 0 and -1.0000001 <= value <= 1.0000001), 'head orientation/support')
    finite(summary)
    return {'status': 'PASS', 'identity': identity, 'manifest_sha256': sha(root / 'manifest.json'),
            'items': len(records), 'operations': len(expected_ops), 'unavailable_item_operations': unavailable,
            'max_parity_absolute_error': max(parity, default=0), 'max_geometry_error': max(geometry, default=0),
            'max_norm_absolute_error': max(norm, default=0),
            'parity_absolute_bound_sufficient': max(parity, default=0) <= settings['atol'],
            'norm_absolute_bound_sufficient': max(norm, default=0) <= settings.get('norm_atol', settings['atol']),
            'tensor_gate_scope': 'runner componentwise gates; independent policy and recorded scalars, no unserialized tensor reconstruction',
            'auditor_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write(args.output, audit(args.bundle, read(args.lock) if args.lock else None, read(args.job) if args.job else None))
