import numpy as np
from certified_frfs.robust_linear_refit import fit_robust_linear
from certified_frfs.linear_retention import box_class_margins

def test_refitted_deployed_coefficients_retain_teacher_margin():
    X=np.array([[.05,.05],[.1,.1],[.9,.9],[.95,.95]])
    y=np.array([0,0,1,1]);W=np.array([[0.,0.],[2.,2.]]);b=np.array([0.,-2.])
    r=fit_robust_linear(X,y,W,b,np.array([0,1]),[0])
    assert r['feasible'] and r['protected_count']==4
    assert np.all(np.array(r['coef'])[:,1]==0)
    margin=box_class_margins(X,y,np.array(r['coef']),np.array(r['intercept']),np.array([0,1]))
    assert np.all(margin>=.5*np.array(r['original_margins'])-1e-8)
