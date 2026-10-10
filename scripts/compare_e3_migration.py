"""Execute a prospectively sealed E3 comparator; independently compare its records.

This produces a small migration sample, never a complete scientific bundle.
The conservative reuse gate requires exact numeric agreement on the fixed sample.
"""
import argparse
import gc
import math
from pathlib import Path
import time

from audit_mechanistic_scientific import read, write, sha, digest, envelope, require, audit_injection_norms


def numeric_diff(old, new, path='', result=None):
    result = result if result is not None else {'numeric_fields': 0, 'different_numeric_fields': 0,
        'max_absolute_difference': 0., 'max_relative_difference': 0., 'structural_differences': [], 'largest_difference_path': None}
    if type(old) is float and type(new) is float:
        require(math.isfinite(old) and math.isfinite(new), 'nonfinite comparison')
        result['numeric_fields'] += 1
        if old != new:
            error = abs(old-new); result['different_numeric_fields'] += 1
            if error > result['max_absolute_difference']:
                result['max_absolute_difference'] = error; result['largest_difference_path'] = path
            result['max_relative_difference'] = max(result['max_relative_difference'], error / max(abs(old), abs(new), 1e-300))
    elif type(old) is dict and type(new) is dict and old.keys() == new.keys():
        for key in old: numeric_diff(old[key], new[key], path+'/'+key, result)
    elif type(old) is list and type(new) is list and len(old) == len(new):
        for index, (left, right) in enumerate(zip(old, new)): numeric_diff(left, right, path+'/'+str(index), result)
    elif type(old) != type(new) or old != new: result['structural_differences'].append(path)
    return result


def run(plan_path, output):
    import torch
    from sinklab.mechanistic_admission import _source, load_model, runtime_identity
    from sinklab.mechanistic_run import evaluate_item
    from sinklab.mechanism_injection import evaluation_mode, equal_norm_injections
    from sinklab.mechanism_trace import MechanisticGPT2Adapter, ParityTolerance
    from mechanistic_campaign_supervisor import exclusive
    plan_doc = read(plan_path); plan = envelope(plan_doc)
    require(plan['scope'] == 'prospectively_fixed_migration_comparison_only', 'comparison scope')
    require(sha(plan['local_panel']) == plan['local_panel_sha256'], 'comparison panel changed')
    protocol = envelope(read(plan['candidate']))
    panel = read(plan['local_panel'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    require(runtime_identity(Path(plan['repo']), 'cuda:0') == protocol['runtime'], 'comparison runtime changed')
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); receipts = []
    with exclusive(Path(plan['gpu_lock_path'])):
        teacher_source = _source(protocol['sources']['teacher'], 'teacher')
        teacher = load_model(teacher_source, 'cuda:0')
        for state in dict.fromkeys(row['state'] for row in plan['items']):
            source = _source(protocol['sources'][state], state)
            model = teacher if state == 'teacher' else load_model(source, 'cuda:0')
            adapter = MechanisticGPT2Adapter(model); settings = protocol['settings'][state]
            for row in [r for r in plan['items'] if r['state'] == state]:
                require(sha(row['original_item']) == row['original_item_sha256'], 'original comparison item changed')
                item = panel[row['panel']][row['ordinal']]
                require(item['id'] == row['item_id'] and digest(item) == row['input_sha256'], 'fixed item changed')
                ids = torch.tensor([item['input_ids']], dtype=torch.long, device='cuda:0')
                mask = torch.tensor([item['attention_mask']], dtype=torch.bool, device='cuda:0')
                item_start = time.monotonic()
                with evaluation_mode(teacher): logits = teacher(input_ids=ids, attention_mask=mask, use_cache=False).logits
                measurement = evaluate_item(adapter, item, 'E3', settings, teacher_logits=logits)
                norms = []
                # Independently recover the exact common-position reference from
                # live entering residuals, absent from historical scalar metadata.
                with evaluation_mode(model):
                    for layer in settings['layers']:
                        traced = adapter.traced_forward(input_ids=ids, attention_mask=mask, trace_layers=(layer,))
                        residual_norms = traced.traces[layer].residual_input.double().norm(dim=-1)
                        for eta in settings['etas']:
                            edits, metadata = equal_norm_injections(traced.traces[layer], mask, eta=eta,
                                norm_floor=settings['norm_floor'], control_seed=settings['control_seed'],
                                query_min=settings['query_min'], nonsink_keys=tuple(settings['nonsink_keys']),
                                tolerance=ParityTolerance(settings['norm_atol'], settings['norm_rtol']))
                            common = torch.tensor(metadata['common_support_mask'], dtype=torch.bool, device='cuda:0')
                            expected = eta * residual_norms[common]
                            comparisons = []
                            for direction, edit in edits.items():
                                actual = edit.double().norm(dim=-1)[common]
                                check = ParityTolerance(settings['norm_atol'], settings['norm_rtol']).check(actual, expected, 'independent_live_common_reference') if common.any() else None
                                recorded = metadata['directions'][direction]['requested_norm_sum']
                                require(abs(recorded-float(expected.sum())) <= 2e-12*max(1.,abs(recorded)), 'common reference sum mismatch')
                                comparisons.append({'direction': direction, 'check': check, 'requested_norm_sum': recorded})
                            norms.append({'layer': layer, 'eta': eta, 'common_positions': int(common.sum()),
                                'eligible_positions': metadata['eligible_positions'], 'common_residual_reference_sum': float(residual_norms[common].sum()),
                                'directions': comparisons})
                file = output / (row['id']+'.json')
                write(file, {'scope': plan['scope'], 'plan_sha256': plan_doc['sha256'], 'runtime': protocol['runtime'],
                    'source': source, **measurement})
                receipt = {**row, 'new_item': str(file), 'new_item_sha256': sha(file),
                           'wall_seconds': time.monotonic()-item_start, 'live_common_norm_checks': norms}
                receipts.append(receipt)
                print('COMPARISON',row['id'],receipt['wall_seconds'],flush=True)
                del logits, traced, edits, measurement; gc.collect(); torch.cuda.empty_cache()
            if model is not teacher: del adapter, model; gc.collect(); torch.cuda.empty_cache()
        del teacher
    write(output / 'COMPARISON_RUN.json', {'scope': plan['scope'], 'plan_sha256': plan_doc['sha256'],
        'items': receipts, 'wall_seconds': time.monotonic()-start})


def compare(plan_path, run_path, output):
    """Read-only comparison; no model/executor imports and no shared forward code."""
    plan_document = read(plan_path); plan = envelope(plan_document); result = read(run_path)
    require(result['plan_sha256'] == plan_document['sha256'] and len(result['items']) == len(plan['items']), 'prospective selection changed')
    protocol = envelope(read(plan['candidate'])); rows = []
    for expected, receipt in zip(plan['items'], result['items']):
        require(all(receipt[k] == v for k,v in expected.items()), 'comparator selection changed')
        require(sha(receipt['original_item']) == receipt['original_item_sha256'] and sha(receipt['new_item']) == receipt['new_item_sha256'], 'comparison bytes changed')
        old, new = read(receipt['original_item']), read(receipt['new_item'])
        require(old['identity']['state'] == expected['state'] and old['identity']['panel'] == expected['panel'] and
                old['identity']['source']['weights_sha256'] == new['source']['weights_sha256'], 'different scientific source')
        diff = numeric_diff({k:old[k] for k in ('item_id','operations','diagnostics')}, {k:new[k] for k in ('item_id','operations','diagnostics')})
        behavior = numeric_diff([r['behavior'] for r in old['operations']], [r['behavior'] for r in new['operations']])
        for record in new['operations']:
            if 'injection' in record: audit_injection_norms(record['injection'], protocol['settings'][expected['state']])
        for norm in receipt['live_common_norm_checks']:
            for direction in norm['directions']:
                require(abs(direction['requested_norm_sum'] - norm['eta']*norm['common_residual_reference_sum']) <=
                        2e-12*max(1.,abs(direction['requested_norm_sum'])), 'live residual reference receipt mismatch')
                check = direction['check']
                require(check is not None or norm['common_positions'] == 0, 'missing live norm gate')
        rows.append({'id':expected['id'], 'all_measurements':diff, 'scientific_behavior_effects':behavior,
            'live_norm_records':len(receipt['live_common_norm_checks'])})
    exact = all(not r['all_measurements']['different_numeric_fields'] and not r['all_measurements']['structural_differences'] for r in rows)
    write(output, {'status':'PASS' if exact else 'REQUIRES_COMPLETE_NODI_RERUN_ASSESSMENT', 'plan_sha256':plan_document['sha256'],
        'run_sha256':sha(run_path), 'items':rows, 'fixed_item_count':len(rows),
        'numerical_correctness':'original unchanged live gates plus independent scalar audits and live common-reference check',
        'cross_machine_reproducibility':'exact on prospectively fixed sample' if exact else 'quantified differences; not exact',
        'scientific_effect_agreement':'exact on fixed sample' if exact else 'differences require assessment',
        'limitation':'A finite engineering sample does not prove global bitwise reproducibility or constitute an independent seed replication.'})
    require(exact, 'fixed-sample exact comparability gate failed; assess complete E3 rerun')


def main():
    parser = argparse.ArgumentParser(__doc__); sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run'); run_parser.add_argument('--plan',type=Path,required=True); run_parser.add_argument('--output',type=Path,required=True)
    compare_parser = sub.add_parser('compare'); compare_parser.add_argument('--plan',type=Path,required=True); compare_parser.add_argument('--run',type=Path,required=True); compare_parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command == 'run': run(args.plan,args.output)
    else: compare(args.plan,args.run,args.output)


if __name__ == '__main__': main()
