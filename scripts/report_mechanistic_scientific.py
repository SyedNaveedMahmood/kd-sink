"""Complete-grid scientific exports, descriptive item/paired statistics only."""
import argparse
import csv
import math
from pathlib import Path

from audit_mechanistic_scientific import read, write, sha, envelope, audit, require
from mechanistic_campaign_supervisor import ledger
from report_mechanistic import complete_join, dose_comparisons, plot_bundle


def quantile(values, p):
    values = sorted(values)
    if not values: return None
    index = (len(values) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def distribution(values):
    values = [v for v in values if v is not None]
    q1, q3 = quantile(values, .25), quantile(values, .75)
    return {'available_items': len(values), 'median': quantile(values, .5),
            'q25': q1, 'q75': q3, 'iqr': None if not values else q3 - q1, 'p90': quantile(values, .9)}


def metrics(row):
    if row['status'] != 'measured': return {}
    b, g = row['behavior'], row['geometry']
    n = b['valid_targets']
    result = {'delta_ce_nats': (b['edited_nll_sum_nats'] - b['clean_nll_sum_nats']) / n,
              'self_kl_nats': b['self_kl_sum_nats'] / n,
              'prediction_flip_fraction': b['flip_count'] / n,
              'absolute_target_logprob_change_nats': b['absolute_target_logprob_change_sum_nats'] / n,
              'centered_logit_displacement_l2': g['centered_logit_displacement_sum'] / n,
              'positive_delta_nll_nats': g['positive_delta_nll_sum_nats'] / n,
              'negative_delta_nll_nats': g['negative_delta_nll_sum_nats'] / n}
    if 'sink_fingerprint' in row:
        result.update(row['sink_fingerprint'])
        result['sink_removed_fraction'] = row['sink_removed_fraction']
    return {k: v for k, v in result.items() if isinstance(v, (int, float)) or v is None}


def export_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def matched_effects(join, matches):
    """Evaluate endpoints at the already located sink-dose brackets only."""
    result=[]
    for match in matches:
        row=dict(match)
        if match['status']=='observed_overlap':
            for label,state,scope,candidates in (('teacher','teacher','mapped_teacher',match['left_candidates']),
                                                  ('student',match['state'],'native',match['right_candidates'])):
                require(len(candidates)==1,'ambiguous dose cannot be interpreted')
                bracket=candidates[0];a,b=bracket['bracket'];w=bracket['weight']
                summary=join['states'][state]['summary']
                left=summary['operations'][f'{match["route"]}/{scope}/{a:g}']['behavior']
                right=summary['operations'][f'{match["route"]}/{scope}/{b:g}']['behavior']
                row[label+'_interpolated_behavior']={k:left[k]+w*(right[k]-left[k]) for k in
                    ('delta_ce_nats','self_kl_nats','prediction_flip_fraction','absolute_target_logprob_change_nats')}
            row['student_minus_teacher']={k:v-row['teacher_interpolated_behavior'][k] for k,v in row['student_interpolated_behavior'].items()}
        result.append(row)
    return result


def report(manifest_path, phase, panel, output):
    manifest = envelope(read(manifest_path)); root = Path(manifest['root'])
    jobs = [j for j in manifest['jobs'] if j['phase'] == phase and j['panel'] == panel]
    statuses = ledger(root / 'state_ledger.jsonl')
    require(jobs and all(statuses.get(j['id'], {}).get('status') == 'complete' for j in jobs), 'incomplete grid cannot report')
    lock = read(jobs[0]['lock'])
    bundles = [Path(statuses[j['id']]['attempt']) / 'bundle' for j in jobs]
    audits = {j['state']: audit(b, lock, j) for j, b in zip(jobs, bundles, strict=True)}
    join = complete_join(bundles, lock=lock, approved_sha256=lock['sha256'], panel_name=panel)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'INDEPENDENT_PHASE_AUDIT.json', {'status': 'PASS', 'phase': phase, 'panel': panel, 'states': audits})
    write(output / 'complete_join.json', join)
    if phase == 'E1':
        matches=dose_comparisons(join, metric='sink_removed_fraction', targets=[.1, .25, .5])
        write(output / 'observed_dose_comparisons.json', matched_effects(join,matches))
    raw, distributions, factors, summaries = {}, [], [], []
    for job, bundle in zip(jobs, bundles, strict=True):
        state = job['state']
        records = [read(p) for p in sorted(bundle.glob('item-*.json'))]
        raw[state] = {r['item_id']: {o['operation']: metrics(o) for o in r['operations']} for r in records}
        summary = join['states'][state]['summary']
        for op, pooled in summary['operations'].items():
            summaries.append({'phase': phase, 'panel': panel, 'state': state, 'operation': op,
                              'measured_items': pooled['measured_items'], 'unavailable_items': pooled['unavailable_items'],
                              **(pooled['behavior'] or {})})
            fields = set(k for r in raw[state].values() for k in r[op])
            for field in sorted(fields):
                ds = distribution([r[op].get(field) for r in raw[state].values()])
                distributions.append({'phase': phase, 'panel': panel, 'state': state, 'operation': op,
                                      'metric': field, 'undefined_items': len(records) - ds['available_items'], **ds})
        # Extended head/value/cancellation scalars are pooled independently of
        # the executor's deliberately smaller legacy factor summary.
        for layer in summary['settings'].get('layers', []):
            if phase == 'E1': continue
            values = [next(o['factors'] for o in r['operations'] if o.get('layer') == layer and 'factors' in o) for r in records]
            for support in ('all_q_ge1', 'second_half_queries'):
                entries = [v[support] for v in values]
                for field in entries[0]:
                    if not field.endswith(('_mean', '_norm')): continue
                    xs = [v[field] for v in entries]
                    columns = zip(*xs, strict=True) if isinstance(xs[0], list) else [xs]
                    for head, column in enumerate(columns):
                        weights = [v['real_query_count'] for v in entries]
                        if 'cosine' in field: weights = [v['per_head_cosine_defined_queries'][head] for v in entries]
                        elif field == 'cancellation_ratio_mean': weights = [v['cancellation_ratio_defined_queries'] for v in entries]
                        pairs = [(x, w) for x, w in zip(column, weights, strict=True) if x is not None and w > 0]
                        n = sum(w for _, w in pairs)
                        factors.append({'state': state, 'layer': layer, 'support': support, 'metric': field,
                                        'head': head if isinstance(xs[0], list) else '', 'defined_queries': n,
                                        'query_weighted_mean': math.fsum(x*w for x,w in pairs)/n if n else None})
        plot_bundle(summary, output / state.replace('/', '-'))
    pairs = []
    for condition in ('C1', 'C2', 'C3', 'C5', 'C6'):
        a, b = f'{condition}/step500', f'{condition}/step10000'
        for op in join['states'][a]['summary']['operations']:
            for field in sorted(set(k for r in raw[a].values() for k in r[op])):
                diffs = [raw[b][item][op].get(field) - left[op][field] for item, left in raw[a].items()
                         if left[op].get(field) is not None and raw[b][item][op].get(field) is not None]
                ds = distribution(diffs)
                pairs.append({'condition': condition, 'contrast': 'step10000 minus step500; paired items',
                              'operation': op, 'metric': field, 'undefined_pairs': len(raw[a]) - ds['available_items'], **ds})
    export_csv(output / 'results.csv', summaries)
    export_csv(output / 'item_distributions.csv', distributions)
    export_csv(output / 'paired_temporal_contrasts.csv', pairs)
    export_csv(output / 'factor_head_data.csv', factors)
    write(output / 'figure_data.json', {'aggregate_results': summaries, 'factor_head_data': factors, 'paired_item_distributions': pairs})
    # Cross-state trajectories retain intermediate checkpoints and static
    # teacher scopes separately. All per-operation controls are exported.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if phase == 'E1':
        operations = ['q_bias/native/1', 'epe_transport/1', 'position0_to1/1', 'k_top3/native/1', *(f'k_random{i}/native/1' for i in range(5))]
    elif phase == 'E2':
        operations = [f'delete/layer{i}' for i in range(24)]
    else:
        operations = [f'{name}/layer{layer}' for name in ('single_delete', 'single_rescue', 'all_delete', 'conditional_rescue') for layer in (3,11,19)]
        operations += [f'injection/{direction}/layer{layer}/eta{eta:g}' for direction in ('sink','random','orthogonal','non_sink') for layer in (3,11,19) for eta in (.01,.03,.1)]
    for ordinal, op in enumerate(operations):
        fig, axes = plt.subplots(1, 2, figsize=(10,4), layout='constrained')
        for condition in ('C1','C2','C3','C5','C6'):
            states = [s for s in join['states'] if s.startswith(condition+'/')]
            states.sort(key=lambda s: int(s.split('step')[1]))
            for axis, field in zip(axes, ('delta_ce_nats','self_kl_nats'), strict=True):
                axis.plot([int(s.split('step')[1]) for s in states],
                          [(join['states'][s]['summary']['operations'][op]['behavior'] or {}).get(field, math.nan) for s in states], marker='o', label=condition)
                axis.set(xlabel='Optimizer update (seed0)', ylabel=field+' (nats/target)'); axis.grid(alpha=.2)
        teacher = join['states']['teacher']['summary']
        teacher_ops = [op]
        if phase == 'E1' and '/native/' in op: teacher_ops += [op.replace('/native/','/mapped_teacher/')]
        elif phase == 'E3':
            teacher_ops = [op.replace('/layer3','/layer5').replace('/layer11','/layer17').replace('/layer19','/layer29')]
        for teacher_op in teacher_ops:
            for axis, field in zip(axes, ('delta_ce_nats','self_kl_nats'), strict=True):
                behavior = teacher['operations'][teacher_op]['behavior']
                if behavior: axis.axhline(behavior[field], linestyle='--', label='Static teacher '+teacher_op)
        axes[-1].legend(fontsize=7); fig.suptitle(f'{phase} {panel}: {op}; descriptive seed0')
        for extension in ('png','svg'):
            with (output / f'trajectory-{ordinal:03d}.{extension}').open('xb') as stream: fig.savefig(stream, format=extension, dpi=150)
        plt.close(fig)
    (output / 'REPORT.md').write_text(f'''# {phase} {panel}: independently audited scientific measurements

Coverage: {len(jobs)}/{len(jobs)} locked states, 300 paired blocks per state. Protocol {lock['sha256']}.
Discovery spans46 source documents; confirmation spans32. Blocks, checkpoints and route components are dependent descriptive observations from one training seed.

results.csv contains token-weighted behavior. item_distributions.csv reports item median, q25/q75, IQR and p90; unavailable support is counted. paired_temporal_contrasts.csv retains signed paired step10000-minus-step500 differences. factor_head_data.csv retains native-head value/projection/cancellation/orientation data and query support. Full per-item target-loss geometry and interventions remain in hashed bundles.

All states/controls and nonmonotonic or contradictory measurements are retained. E1 effective doses use only observed brackets; unavailable/ambiguous matches cannot support route absence. Isolated E2 contributions are not additive nonlinear all-layer explanations. E3 equal-relative-norm directions differ across checkpoint residual bases; conditional rescue and telescope increments are order-dependent, not mediation fractions. Numerical tolerances establish engineering closure, not scientific equivalence. E0's fixed prior mixed trajectory is unchanged.
''', encoding='utf-8')
    write(output / 'REPORT_COMPLETE.json', {'status': 'COMPLETE_AUDITED_EXPORT', 'phase': phase, 'panel': panel,
         'protocol_sha256': lock['sha256'], 'files': {p.name: sha(p) for p in output.iterdir() if p.is_file()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--phase', choices=['E1','E2','E3'], required=True)
    parser.add_argument('--panel', choices=['discovery','confirmation'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report(args.manifest, args.phase, args.panel, args.output)
