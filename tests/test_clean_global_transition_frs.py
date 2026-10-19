from collections import Counter
import numpy as np
import pytest
from capped_third_enemy_hinge import CappedThirdEnemyHinge
from fit_sparse_global_transition_frs import fit_sparse_global_transition_target
from fit_clean_global_transition_frs import fit_clean_global_transition_target
from audit_clean_global_transition_frs import check_clean_global_transition_fit


def fixture():
    A=np.array([[x,c] for c in (0.,1.) for x in (0.,0.,.1,.1,.2,.2)])
    return A,np.repeat([0,1],6),np.ones(2)/np.sqrt(2)


def test_nominal_loss_and_unchanged_robust_FRS_constraints_are_separate_and_replayed():
    A,y,anchor=fixture()
    point,fit=fit_clean_global_transition_target(A,y,[0,1],anchor,full_reference=anchor,reference_scope='original_full',optimization_eps=0.)
    assert fit['training_witnesses']['eps']==fit['optimization_eps']==0.
    assert fit['class_cell']['eps']==fit['certificate_eps']==fit['polytope']['eps']==.01
    assert fit['class_cell']['eta']==.1 and fit['polytope']['retention']==.9
    assert fit['class_cell_final']['feasible']
    check_clean_global_transition_fit(A,y,{'features':[0,1],'anchor':anchor.tolist(),'target':point.tolist(),'fit':fit},Counter())
    with pytest.raises(ValueError):
        fit_clean_global_transition_target(A,y,[0,1],anchor,full_reference=anchor,reference_scope='original_full',optimization_eps=.1)


def test_eps01_exactly_recovers_previous_optimizer_and_nominal_loss_is_lower():
    A,y,anchor=fixture();kwargs={'full_reference':anchor,'reference_scope':'original_full'}
    old,oldfit=fit_sparse_global_transition_target(A,y,[0,1],anchor,**kwargs)
    new,newfit=fit_clean_global_transition_target(A,y,[0,1],anchor,optimization_eps=.01,**kwargs)
    np.testing.assert_array_equal(old,new)
    assert oldfit['steps']==newfit['steps'] and oldfit['training_objective_final']==newfit['training_objective_final']
    normalized=anchor/np.sqrt(2)
    clean=CappedThirdEnemyHinge(A,y,normalized,eps=0.)
    robust=CappedThirdEnemyHinge(A,y,normalized,eps=.01)
    for w in (normalized,np.array([0.,1.]),np.array([1.,0.]),np.array([.9,.1])):
        assert np.all(clean.losses(w)<=robust.losses(w)+1e-10)
