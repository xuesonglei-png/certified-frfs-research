import numpy as np
from certified_frfs.exact_box_linear_refit import fit_exact_box_linear
from certified_frfs.linear_retention import box_class_margins

def test_pair_coefficients_and_boundary_clipping_protect_deployed_model():
    X=np.array([[0.,0.],[.1,.1],[.9,.9],[1.,1.]])
    y=np.array([0,0,1,1]);W=np.array([[10.,10.],[12.,12.]]);b=np.array([0.,-2.])
    result=fit_exact_box_linear(X,y,W,b,[0,1],[0])
    assert result['feasible']
    margin=box_class_margins(X,y,np.array(result['coef']),np.array(result['intercept']),[0,1])
    assert np.all(margin>=.5*np.array(result['original_margins'])-1e-8)
    assert np.all(np.array(result['coef'])[:,1]==0)
