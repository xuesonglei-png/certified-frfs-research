import numpy as np
from certified_frfs.triangular_robust_refit import fit_triangular_robust
from certified_frfs.triangular_classifier import triangular_box_margins

def test_redundant_raw_attribute_can_be_removed_inside_coupled_margin_LP():
    X=np.array([[0.,0.],[.1,.1],[.9,.9],[1.,1.]])
    y=np.array([0,0,1,1]);W=np.zeros((2,2,3));W[1]=[[-1.,0.,1.],[-1.,0.,1.]];b=np.zeros(2)
    r=fit_triangular_robust(X,y,W,b,[0,1],[0])
    assert r['feasible'] and r['protected_count']==4
    assert np.all(np.array(r['coef'])[:,1]==0)
    margin=triangular_box_margins(X,y,np.array(r['coef']),np.array(r['intercept']),[0,1])
    assert np.all(margin>=.5*np.array(r['original_margins'])-1e-8)
    other=fit_triangular_robust(X,y,W+7,b+3,[0,1],[0])
    assert other['feasible'];np.testing.assert_allclose(r['objective'],other['objective'],atol=1e-10)
