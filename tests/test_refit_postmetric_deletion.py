import numpy as np
import pytest
from refit_postmetric_deletion import fit_refitted_deletion


def test_relearning_cannot_admit_the_dependency_preserving_bad_classifier():
    X=np.array([[0,0,0,0]]*2+[[1,0,0,0]]*2+[[.5,1,1,1]]*4,float)
    y=np.array([0]*4+[1]*4)
    result=fit_refitted_deletion(X,y,[0],[2.],1.)
    assert result['selected_id'] is None
    assert all(not r['accepted'] and r['hard_training_LOO_BA']==.5 for r in result['checks'])
    assert all(r['certificate']['required']==[4,4] for r in result['checks'])


def test_unchanged_good_metric_remains_available_and_failed_fit_is_visible():
    rng=np.random.default_rng(812)
    y=np.repeat([0,1],6)
    X=np.c_[.1+.8*y+rng.uniform(-.02,.02,12),np.full(12,.5)]
    result=fit_refitted_deletion(X,y,[0],[np.sqrt(2)],1.)
    assert result['selected_id']==0
    assert result['checks'][0]['eligible']
    assert result['refit']['status']=='unique_simplex'
    assert result['guard_reference']=='original_full'


def test_invalid_mass_and_parent_are_rejected():
    X=np.zeros((8,2)); y=np.repeat([0,1],4)
    with pytest.raises(ValueError): fit_refitted_deletion(X,y,[0],[1.],1.)
    with pytest.raises(ValueError): fit_refitted_deletion(X,y,[0],[np.sqrt(2)],1.1)


def test_relearning_recovers_a_badly_rescaled_support_with_original_guards():
    # Mechanical QA example, not a benchmark performance claim.
    from audit_simplex_source_budget_diagnostic import independent_loo
    from independent_soft_topk_objective import independent_objective
    X=np.array([[.05,.1,.5],[.06,.9,.5],[.07,.3,.5],[.08,.7,.5],
                [.09,.4,.5],[.10,.6,.5],[.9,.1,.5],[.91,.9,.5],
                [.92,.3,.5],[.93,.7,.5],[.94,.4,.5],[.95,.6,.5]])
    y=np.repeat([0,1],6); mass=np.sqrt(3); anchor=np.array([.01,mass-.01])
    result=fit_refitted_deletion(X,y,[0,1],anchor,1.)
    assert result['refit']['optimizer_success']
    assert not result['checks'][0]['eligible']
    selected=result['checks'][result['selected_id']]
    assert selected['eligible'] and selected['accepted']
    assert independent_loo(X,y,[0,1],selected['weights'])==1.
    assert selected['certificate']['feasible'] and selected['class_certificate']['feasible_all_classes']
    target=np.array(result['target_weights'])
    assert independent_objective(X[:,[0,1]],y,target/mass)<independent_objective(X[:,[0,1]],y,anchor/mass)
