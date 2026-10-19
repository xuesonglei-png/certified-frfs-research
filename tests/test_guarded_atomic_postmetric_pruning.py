import numpy as np
from guarded_atomic_postmetric_pruning import fit_atomic_pruning
from independent_atomic_pruning import check_atomic_pruning


def data():
    x=np.r_[np.linspace(.05,.1,6),np.linspace(.9,.95,6)]
    return np.c_[x,x,np.full(12,.5)],np.repeat([0,1],6)


def test_guarded_path_reaches_exact_omission_and_parent_accuracy_is_nondecreasing():
    A,y=data(); fs,w,fit=fit_atomic_pruning(A,y,[0,1,2],np.ones(3)/np.sqrt(3),fraction=1/3,max_checks=48)
    assert len(fs)==1 and fit['status']=='target_reached'
    assert abs(w.sum()-np.sqrt(3))<1e-8 and fit['endpoint_guard']['accepted']
    assert fit['guard_reference']=='original_full' and fit['final_hard_LOO_BA']==1.
    assert all(fit['checks'][i]['eligible'] for i in fit['chosen_ids'])
    assert any(p['kind']=='atomic' for p in fit['checks'])
    assert all(fit['checks'][i]['certificate']['required']==fit['initial_guard']['certificate']['required'] for i in fit['chosen_ids'])
    assert check_atomic_pruning(A,y,[0,1,2],np.ones(3)/np.sqrt(3),fit)==len(fit['checks'])


def test_shared_budget_stops_proposals_at_the_declared_limit():
    A,y=data(); fs,w,fit=fit_atomic_pruning(A,y,[0,1,2],np.ones(3)/np.sqrt(3),fraction=1/3,max_checks=1)
    assert len(fit['checks'])==1
    assert fit['status']=='guard_budget_exhausted'
    assert fit['final_hard_LOO_BA']>=fit['initial_hard_LOO_BA']
    assert check_atomic_pruning(A,y,[0,1,2],np.ones(3)/np.sqrt(3),fit)==1


def test_control_own_reference_stays_fixed_across_deletions():
    A,y=data(); weights=np.array([.01,.01,np.sqrt(3)-.02])
    fs,w,fit=fit_atomic_pruning(A,y,[0,1,2],weights,fraction=1/3,require_original=False)
    assert fit['guard_reference']=='own_anchor'
    assert not fit['original_initial_guard']['accepted'] and fit['initial_guard']['accepted']
    assert fit['endpoint_guard']['accepted']
    assert fit['final_hard_LOO_BA']>=fit['initial_hard_LOO_BA']
    assert check_atomic_pruning(A,y,[0,1,2],weights,fit)==len(fit['checks'])
