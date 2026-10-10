"""Scientific-report regressions: populations, signed rescues and layer orders."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import summarize_mechanistic_e1_e3 as reporting


def test_repool_uses_targets_rather_than_item_means():
    rows = [dict(valid_targets=n, clean_nll_sum_nats=2*n, edited_nll_sum_nats=edited*n,
                 self_kl_sum_nats=.2*n, flip_count=flips, absolute_target_logprob_change_sum_nats=.3*n)
            for n, edited, flips in [(1, 1, 0), (3, 4, 1)]]
    pooled = reporting.repool_behavior(rows)
    assert pooled['valid_targets'] == 4
    assert pooled['delta_ce_nats'] == 1.25
    assert pooled['prediction_flip_fraction'] == .25
    assert pooled['self_kl_nats'] == pytest.approx(.2)


def test_zero_target_population_rejected():
    with pytest.raises(ValueError, match='no scored targets'):
        reporting.repool_behavior([{'valid_targets': 0}])


def test_signed_quantiles_keep_improvements_and_heterogeneity():
    assert reporting.describe([-2., 0., 1., 5.]) == {
        'mean': 1., 'median': .5, 'q25': -.5, 'q75': 2., 'p90': pytest.approx(3.8)}


def fixture_summary():
    layers = [3, 11, 19]
    base = {'behavior': {'clean_ce_nats': 3., 'delta_ce_nats': 1., 'self_kl_nats': .7,
                        'prediction_flip_fraction': .2, 'ppl_ratio': 2.718}}
    operations = {}
    for layer in layers:
        operations[f'all_delete/layer{layer}'] = base
        for name, effect in [('single_delete', .1), ('single_rescue', 0.), ('conditional_rescue', 1.2)]:
            operations[f'{name}/layer{layer}'] = {'behavior': {'delta_ce_nats': effect}}
        for direction in reporting.DIRECTIONS:
            operations[f'injection/{direction}/layer{layer}/eta0.1'] = {'behavior': {
                'delta_ce_nats': layer/100, 'self_kl_nats': layer/1000}}
    return {'settings': {'layers': layers}, 'operations': operations, 'diagnostics': {'telescopes': [
        {'order': [0, 1, 2], 'increments_ce_nats': [.2, -.1, .9], 'all_layer_delta_ce_nats': 1.},
        {'order': [2, 1, 0], 'increments_ce_nats': [.7, .4, -.1], 'all_layer_delta_ce_nats': 1.}]}}


def test_three_probe_average_and_all_deletion_deduplicated():
    endpoint, rescues, _ = reporting.state_rows(fixture_summary(), 'confirmation', 'C2/step10000', 'NodiPC')
    assert endpoint['all_delete_delta_ce_nats'] == 1.
    assert endpoint['sink_eta01_coarse_mean_delta_ce_nats'] == pytest.approx(.11)
    assert len(rescues) == 3


def test_conditional_restoration_can_worsen_loss():
    _, rescues, _ = reporting.state_rows(fixture_summary(), 'confirmation', 'C2/step10000', 'NodiPC')
    assert all(row['conditional_loss_reduction_nats'] == pytest.approx(-.2) for row in rescues)


def test_conflicting_duplicate_all_layer_results_rejected():
    summary = fixture_summary()
    summary['operations']['all_delete/layer11'] = {'behavior': {'delta_ce_nats': 9.}}
    with pytest.raises(ValueError, match='all-layer deletions disagree'):
        reporting.state_rows(summary, 'discovery', 'teacher', 'AdritaPC')


def test_reverse_order_preserves_layer_assignment_and_negative_increment():
    _, _, rows = reporting.state_rows(fixture_summary(), 'discovery', 'teacher', 'AdritaPC')
    reverse = [r for r in rows if r['order'] == 'reverse']
    assert [r['layer_added'] for r in reverse] == [2, 1, 0]
    assert reverse[-1]['increment_ce_nats'] == -.1


def test_nonclosing_telescope_is_rejected():
    summary = fixture_summary()
    summary['diagnostics']['telescopes'][1]['increments_ce_nats'][-1] = -.2
    with pytest.raises(ValueError, match='does not close'):
        reporting.state_rows(summary, 'discovery', 'teacher', 'AdritaPC')


def test_csv_is_utf8_with_lf_and_retains_negative_values():
    result = reporting.csv_bytes([{'metric': '\u0394CE', 'value': -.2}])
    assert b'\r' not in result
    assert result.decode('utf-8') == 'metric,value\n\u0394CE,-0.2\n'
