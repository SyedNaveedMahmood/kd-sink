"""Report integrity and metric-unit regressions; stdlib-only, no GPU."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / 'scripts/summarize_mechanistic_e1_e2.py'
spec = importlib.util.spec_from_file_location('e1_e2_summary', path)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def behavior(delta):
    return {'item_count': 300, 'valid_targets': 38100, 'clean_ce_nats': 3.,
            'edited_ce_nats': 3. + delta, 'delta_ce_nats': delta,
            'ppl_ratio': math.exp(delta), 'clean_ppl': math.exp(3.),
            'clean_accuracy_fraction': .4, 'self_kl_nats': .02, 'teacher_kl_nats': .7}


def geometry(delta):
    return {'delta_nll_sum_nats': 38100 * delta,
            'positive_delta_nll_sum_nats': 38100 * max(delta, 0),
            'negative_delta_nll_sum_nats': 38100 * min(delta, 0)}


@pytest.mark.parametrize('delta', [0., .1, -.1])
def test_signed_loss_and_perplexity_units(delta):
    report.verify_behavior(behavior(delta), geometry(delta))


def test_incorrect_target_denominator_is_rejected():
    row = behavior(.1)
    row['valid_targets'] = 38400  # Input tokens are not scored shifted targets.
    with pytest.raises(ValueError, match='population'):
        report.verify_behavior(row, geometry(.1))


def test_relative_ce_is_not_perplexity_ratio():
    row = behavior(.1)
    row['ppl_ratio'] = 1 + .1
    with pytest.raises(ValueError, match='PPL units'):
        report.verify_behavior(row, geometry(.1))


def test_sign_cancellation_must_close():
    geo = geometry(.1)
    geo['negative_delta_nll_sum_nats'] = -100.
    with pytest.raises(ValueError, match='signed loss closure'):
        report.verify_behavior(behavior(.1), geo)


def test_protocol_exclusions_are_not_missing_results():
    assert len(report.expected_states('E1')) == 14
    assert len(report.expected_states('E2')) == 16
    assert report.expected_states('E2') - report.expected_states('E1') == {'C1/step2000', 'C5/step2000'}


def test_clean_teacher_kl_is_not_edited_teacher_kl():
    b = behavior(.1)
    op = {'behavior': b, 'geometry': geometry(.1)}
    support = {'head_sink_mass_mean': .3, 'relative_projected_delta_mean': .2,
               'cancellation_ratio_mean': .5}
    summaries = {
        ('E1', 'confirmation', 'teacher'): {'operations': {'none': {
            **op, 'behavior': {**behavior(0.), 'teacher_kl_nats': 0.},
            'sink_fingerprint': {'baseline_sink': .4}}}},
        ('E2', 'confirmation', 'teacher'): {'operations': {'delete/layer0': op},
            'diagnostics': {'factor_summary': {'0': {'all_q_ge1': support, 'second_half_queries': support}}}},
    }
    _, _, endpoints = report.tables(summaries)
    assert endpoints[0]['clean_teacher_kl_nats'] == 0.
    assert endpoints[0]['peak_delta_ce_nats'] == .1


def test_summary_tampering_is_detected_before_cached_audits(tmp_path):
    reconstruction = tmp_path / 'reconstruction01'
    reconstruction.mkdir()
    attempt = tmp_path / 'attempt_001'
    bundle = attempt / 'bundle'
    bundle.mkdir(parents=True)
    summary = bundle / 'summary.json'
    summary.write_text('{}', encoding='utf-8')
    report.dump(bundle / 'manifest.json', {'files': {'summary.json': report.sha(summary)}})
    manifest_sha = report.sha(bundle / 'manifest.json')
    (bundle / 'COMPLETE').write_text(manifest_sha, encoding='utf-8')
    inventory = reconstruction / 'ALL_92_JOBS.json'
    report.dump(inventory, {'jobs': [{'job_id': 'E1_confirmation_teacher',
        'phase': 'E1', 'status': 'independently_complete_valid',
        'selected': {'attempt': str(attempt), 'manifest_sha256': manifest_sha}}]})
    report.dump(reconstruction / 'RECOVERY_INVENTORY_COMPLETE.json',
                {'status': 'PASS', 'files': {'ALL_92_JOBS.json': report.sha(inventory)}})
    summary.write_text('{"tampered": true}', encoding='utf-8')
    with pytest.raises(ValueError, match='summary bytes changed'):
        report.load_verified(tmp_path)


def test_csv_preserves_signed_numbers_and_lf(tmp_path):
    target = tmp_path / 'numbers.csv'
    report.export_csv(target, [{'delta': -.01, 'unavailable': None}])
    assert b'\r\n' not in target.read_bytes()
    assert '-0.01,' in target.read_text()
