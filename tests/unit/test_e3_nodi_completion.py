"""A live/partial/forged terminal receipt cannot trigger final publication."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from finalize_e3_nodi_recovery import completion_gate


def receipts():
    manifest={'file_sha256':'sealed','jobs':[{'id':str(i)} for i in range(32)]}
    monitor={'action':'complete','manifest_sha256':'sealed'}
    final={'status':'COMPLETE_INDEPENDENTLY_AUDITED','expected':32,'manifest_sha256':'sealed',
        'states':{str(i):{'status':'complete'} for i in range(32)},'reused_Adrita':17,'new_Nodi':15}
    return manifest,monitor,final


def test_complete_exact_coverage_can_finalize():
    completion_gate(*receipts())


@pytest.mark.parametrize('mutation',['partial','missing','duplicate_identity','wrong_origin_counts','old_manifest','uptime','unqualified_status'])
def test_scientific_completion_requires_every_gate(mutation):
    manifest,monitor,final=receipts()
    if mutation=='partial':final['states']['0']['status']='running'
    elif mutation=='missing':final['states'].pop('0')
    elif mutation=='duplicate_identity':final['states']['other']=final['states'].pop('0')
    elif mutation=='wrong_origin_counts':final['new_Nodi']=16
    elif mutation=='old_manifest':final['manifest_sha256']='previous plan'
    elif mutation=='uptime':monitor['action']='observe'
    elif mutation=='unqualified_status':final['status']='RUNNER_COMPLETE_ONLY'
    with pytest.raises(ValueError):completion_gate(manifest,monitor,final)
