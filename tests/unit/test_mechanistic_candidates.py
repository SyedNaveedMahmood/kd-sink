import runpy
from pathlib import Path
import pytest
from sinklab.mechanistic_admission import E1_GRID,E2_GRID

settings=runpy.run_path(str(Path(__file__).parents[2]/"scripts/build_mechanistic_candidates.py"))["phase_settings"]


@pytest.mark.parametrize("phase",["E1","E2","E3"])
def test_full_declared_grids_and_settings(phase):
    grid=E1_GRID if phase=="E1" else E2_GRID
    assert len(grid)==(13 if phase=="E1" else 15)
    for state in ("teacher",*grid):
        value=settings(phase,state)
        assert value['token_chunk']==16
        if phase=='E2':assert value['layers']==list(range(36 if state=='teacher' else 24))
        if phase=='E3':
            assert value['reference']=='clean_residual_input_before_ln_1'
            assert value['etas']==[0.,.01,.03,.10]
            assert value['layers']==([5,17,29] if state=='teacher' else [3,11,19])
            assert value['orders'][1]==list(reversed(value['orders'][0]))


def test_unknown_phase_rejected():
    with pytest.raises(ValueError):settings('E4','teacher')
