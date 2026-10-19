import numpy as np
from crossgroup_soft5_terms import crossgroup_terms
from density_calibrated_soft5 import calibrate,objective
from independent_density_calibration import calibration


def test_training_gap_calibration_independent_and_ties():
    for A in [np.random.default_rng(3).uniform(size=(24,4)),np.zeros((24,4))]:
        y=np.arange(24)%3;w=np.array([.1,.2,.3,.4]);terms,_=crossgroup_terms(A,y)
        t,info=calibrate(terms,w);fresh=calibration(A,y,w)
        assert set(info)==set(fresh)
        for key in info:
            if isinstance(info[key],float):np.testing.assert_allclose(info[key],fresh[key],atol=1e-12,rtol=1e-12)
            else:assert info[key]==fresh[key]
        assert .0005<=t<=.05
        import json
        json.dumps(info)
        json.dumps(fresh)


def test_calibrated_soft5_gradient_and_scalar_loss():
    from independent_density_objective import independent_objective
    A=np.random.default_rng(32).uniform(size=(24,4));y=np.arange(24)%3
    w=np.array([.1,.2,.3,.4]);terms,_=crossgroup_terms(A,y);t,_=calibrate(terms,w)
    np.testing.assert_allclose(objective(w,terms,t)[0],independent_objective(A,y,w,t),atol=1e-8,rtol=0)
    h=1e-6
    numeric=np.array([(objective(w+np.eye(4)[i]*h,terms,t)[0]-objective(w-np.eye(4)[i]*h,terms,t)[0])/(2*h) for i in range(4)])
    np.testing.assert_allclose(numeric,objective(w,terms,t)[1],atol=2e-4,rtol=1e-4)


def test_saturated_top5_preserves_tails_and_implicit_gradient():
    from density_calibrated_soft5 import balanced_membership
    from scipy.special import log_expit,logsumexp
    distance=np.array([0.,.01,.02,.03,.04,.105,.12,.13,.14,.15])
    coefficients=np.column_stack([distance,np.linspace(.005,.015,10)])[None,:,:]
    same=np.zeros((1,10),bool);same[0,5]=True
    mask=np.ones((1,10),bool);terms=(coefficients,same,mask,np.array([0]))
    w=np.array([1.,.01]);D=(coefficients@w)[0];tau=.001
    q,slopes=balanced_membership(D[None,:],mask,tau,return_slope=True)
    # At least one saturated coordinate must retain a positive derivative.
    assert np.any((q==1)&(slopes>0)) or np.any(abs(slopes-q*(1-q))>1e-18)
    # Tiny outside mass and inside deficit must balance in log space.
    outside=q[0,5:].sum();assert outside>0
    # Central differences assess the actual regularized objective near tails.
    h=1e-7
    numeric=np.array([(objective(w+np.eye(2)[i]*h,terms,tau)[0]-objective(w-np.eye(2)[i]*h,terms,tau)[0])/(2*h) for i in range(2)])
    np.testing.assert_allclose(numeric,objective(w,terms,tau)[1],atol=2e-6,rtol=1e-5)
