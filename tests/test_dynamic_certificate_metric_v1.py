import copy
from collections import Counter
import numpy as np
import pytest
from dynamic_certificate_metric_v1 import fit_dynamic_certificate,original_protection,refreshed_cell
from audit_dynamic_certificate_metric_v1 import check_dynamic_fit, independent_soft_objective
from crossgroup_soft5_terms import crossgroup_terms
from soft_topk_metric_objective import soft_topk_objective


def table():
    rng=np.random.default_rng(721);y=np.repeat(np.arange(3),8)
    A=np.clip(np.column_stack((np.repeat([.15,.5,.85],8)+rng.normal(0,.008,24),rng.uniform(0,1,24))),0,1)
    return A,y


def test_refreshed_certificate_optimizer_audit_and_tamper():
    A,y=table();anchor=np.ones(2)/np.sqrt(2)
    for refresh in (False,True):
        w,fit=fit_dynamic_certificate(A,y,[0,1],anchor,anchor,anchor,refresh=refresh,max_steps=3,lp_time_limit=3.)
        assert fit['available'];counts=Counter();check_dynamic_fit(A,y,fit,counts)
        assert fit['final_objective']<=fit['initial_objective'] and counts['independent_direction_primal_dual_replays']>0
        bad=copy.deepcopy(fit);bad['steps'][0]['gradient'][0]+=1.
        with pytest.raises(AssertionError):check_dynamic_fit(A,y,bad,Counter())


def test_original_reference_never_replaced_and_objective_gradient():
    A,y=table();protected=original_protection(A,y);assert protected.any()
    # Removing the informative coordinate cannot silently redefine protection.
    assert refreshed_cell(A[:,[1]],y,np.array([np.sqrt(2)]),protected) is None
    w=np.array([.43,.57])
    for eps in (0.,.01):
        terms,_=crossgroup_terms(A,y,eps=eps)
        value,gradient=soft_topk_objective(w,terms)
        independent,igrad=independent_soft_objective(A,y,w,eps)
        np.testing.assert_allclose(value,independent,atol=1e-9)
        np.testing.assert_allclose(gradient,igrad,atol=1e-8)
        h=1e-6;direction=np.array([1.,-1.])
        numeric=(soft_topk_objective(w+h*direction,terms)[0]-soft_topk_objective(w-h*direction,terms)[0])/(2*h)
        np.testing.assert_allclose(numeric,gradient@direction,atol=1e-7)


def test_fixed_zero_face_and_infeasible_original_reference_branch():
    A,y=table();anchor=np.ones(2)/np.sqrt(2);seed=np.array([np.sqrt(2),0.])
    w,fit=fit_dynamic_certificate(A,y,[0,1],seed,anchor,seed,fixed_zeros=[1],max_steps=2,lp_time_limit=3.)
    assert fit['available'] and w[1]==0.
    check_dynamic_fit(A,y,fit,Counter())
    badseed=np.array([0.,np.sqrt(2)])
    w,fit=fit_dynamic_certificate(A,y,[0,1],anchor,anchor,badseed,fixed_zeros=[0],max_steps=2)
    assert not fit['available'] and fit['status']=='original_full_majority_reference_unavailable'
    np.testing.assert_array_equal(w,badseed)
