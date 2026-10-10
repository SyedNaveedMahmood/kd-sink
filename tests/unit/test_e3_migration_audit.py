"""Regression: degenerate queries must not be audited as injected queries."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('migration_auditor', REPO / 'scripts/audit_mechanistic_scientific.py')
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)
ROWS = json.loads((REPO / 'tests/fixtures/e3_common_support_auditor_regression.json').read_text())['rows']


@pytest.mark.parametrize('row', ROWS)
def test_archived_common_support_norms_pass_without_changing_thresholds(row):
    injection = row['injection']
    assert injection['common_available_positions'] < injection['eligible_positions']
    with pytest.raises(ValueError, match='scalar reduction differs'):
        auditor.near(injection['eta'] * injection['reference_residual_norm_sum'],
                     injection['directions']['sink']['requested_norm_sum'])
    assert len(auditor.audit_injection_norms(injection, row['settings'])) == 4


@pytest.mark.parametrize('corruption', ['request', 'actual', 'support', 'eligible_reference'])
def test_wrong_support_or_delivered_dose_still_fails(corruption):
    row = copy.deepcopy(ROWS[0]); injection = row['injection']
    if corruption == 'request': injection['directions']['random']['requested_norm_sum'] += 1
    if corruption == 'actual': injection['directions']['sink']['actual_norm_sum'] += 1
    if corruption == 'support': injection['common_support_mask'][0][0] = 1
    if corruption == 'eligible_reference': injection['reference_residual_norm_sum'] = 1
    with pytest.raises(ValueError):
        auditor.audit_injection_norms(injection, row['settings'])


def test_identical_populations_require_exact_reference_dose():
    row = copy.deepcopy(ROWS[0]); injection = row['injection']
    injection['eligible_positions'] = injection['common_available_positions']
    with pytest.raises(ValueError, match='scalar reduction differs'):
        auditor.audit_injection_norms(injection, row['settings'])


def test_empty_common_support_is_a_zero_dose_with_no_norm_maximum():
    row = copy.deepcopy(ROWS[0]); injection = row['injection']
    injection.update(common_available_positions=0, common_support_mask=[[0] * 128], eta=0.)
    for direction in injection['directions'].values():
        direction.update(injected_positions=0, actual_norm_sum=0., requested_norm_sum=0., max_norm_error=None)
    assert auditor.audit_injection_norms(injection, row['settings']) == []
