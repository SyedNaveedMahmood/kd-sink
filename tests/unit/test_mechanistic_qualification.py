"""Qualification measurements and real execution interfaces on CPU fixtures."""
import runpy
from pathlib import Path
import pytest
import torch
from test_calibrated_probes import model_fixture, inputs
from sinklab.mechanism_trace import MechanisticGPT2Adapter

q=runpy.run_path(str(Path(__file__).parents[2]/'scripts/qualify_mechanistic_production_shape.py'))


def test_error_distribution_matches_independent_values():
    tolerance=q['DistributionTolerance'](.2,0.)
    actual=torch.tensor([0.,.1,.2],dtype=torch.float64)
    row=tolerance.check(actual,torch.zeros_like(actual),'independent')
    assert row['max_absolute_error']==.2 and row['element_count']==3
    assert row['mean_absolute_error']==pytest.approx(.1)
    assert row['sampled_p50']==.1 and row['max_gate_fraction']==1
    with pytest.raises(ValueError):tolerance.check(actual+.1,torch.zeros_like(actual),'failure')


def test_synthetic_inputs_are_deterministic_and_variable_length():
    a=q['synthetic_items'](1729)
    assert a==q['synthetic_items'](1729) and a!=q['synthetic_items'](1730)
    assert [sum(i['attention_mask']) for i in a]==[128,117]
    assert all(len(i['input_ids'])==128 for i in a)
    assert q['normalized_uuid']('GPU-AB-CD')==q['normalized_uuid']('ab-cd')


def test_production_route_comparator_interface_on_fixture():
    model=model_fixture();adapter=MechanisticGPT2Adapter(model);ids,mask=inputs()
    item={'input_ids':ids[0].tolist(),'attention_mask':mask[0].tolist()}
    tolerance=q['DistributionTolerance'](1e-3,1e-4)
    rows=q['route_diagnostics'](adapter,item,{'control_seed':20260927,'alphas':[0.,.25,.5,.75,1.]},tolerance,'cpu')
    assert len(rows)==9 and all(r['legacy_logit_parity']=='bitwise_equal' for r in rows)
    assert len(tolerance.observations)==5
