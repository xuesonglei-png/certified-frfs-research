"""Fresh certificates after actual removal, full training votes and budgets."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from guarded_postmetric_pruning import fit_postmetric_pruning
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate
from audit_simplex_source_budget_diagnostic import independent_loo


def test_constant_training_coordinates_can_be_removed_with_fresh_certificates():
    A=np.zeros((16,6)); y=np.repeat([0,1],8); A[:,0]=.1+.8*y
    w=np.ones(6)/np.sqrt(6)
    fs,result,fit=fit_postmetric_pruning(A,y,list(range(6)),w)
    assert len(fs)==3 and fit['status']=='target_reached'
    assert 0 in fs and fit['guard_reference']=='original_full'
    assert fit['initial_hard_LOO_BA']==fit['final_hard_LOO_BA']==1.
    assert independent_loo(A,y,fs,result)==1. and abs(result.sum()-np.sqrt(6))<1e-12
    cert=weight_certificate(A,y,fs,w,result)
    assert cert['feasible'] and classwise_dependency_certificate(y,cert)['feasible_all_classes']
    for i in fit['chosen_ids']:
        p=fit['checks'][i]; assert p['eligible'] and p['accepted'] and p['nonworsening_training_BA']


def test_tight_budget_records_honest_partial_search():
    A=np.zeros((16,6)); y=np.repeat([0,1],8); A[:,0]=.1+.8*y
    fs,w,fit=fit_postmetric_pruning(A,y,list(range(6)),np.ones(6)/np.sqrt(6),max_checks=1)
    assert len(fit['checks'])==1 and len(fs)==5
    assert fit['status']=='guard_budget_exhausted' and fit['endpoint_guard']['accepted']
    assert fit['final_hard_LOO_BA']==independent_loo(A,y,fs,w)


def test_control_anchor_reference_is_preserved_without_claiming_full_admission():
    A=np.zeros((16,6)); y=np.repeat([0,1],8); A[:,0]=.1+.8*y
    _,_,fit=fit_postmetric_pruning(A,y,[1,2],np.array([np.sqrt(6)/2]*2),fraction=1.,require_original=False)
    assert fit['guard_reference']=='own_anchor' and not fit['original_initial_guard']['accepted']
    assert fit['initial_guard']['accepted'] and fit['endpoint_guard']['accepted']
