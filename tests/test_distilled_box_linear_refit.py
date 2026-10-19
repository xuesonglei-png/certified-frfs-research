import numpy as np
import pytest
from certified_frfs.distilled_box_linear_refit import fit_distilled_box_linear
from certified_frfs.linear_retention import box_class_margins

def test_distillation_ignores_common_teacher_score_gauge():
    X = np.array([[0.,0.],[.1,.1],[.9,.9],[1.,1.]])
    y = np.array([0,0,1,1]); W = np.array([[0.,0.],[2.,2.]])
    b = np.array([0.,-2.])
    a = fit_distilled_box_linear(X,y,W,b,[0,1],[0])
    shifted = fit_distilled_box_linear(X,y,W+7,b+3,[0,1],[0])
    assert a['feasible'] and shifted['feasible']
    np.testing.assert_allclose(a['coef'], shifted['coef'], atol=1e-10)
    np.testing.assert_allclose(a['intercept'], shifted['intercept'], atol=1e-10)
    assert np.all(np.array(a['coef'])[:,1] == 0)
    m = box_class_margins(X,y,np.array(a['coef']),np.array(a['intercept']),[0,1])
    assert np.all(m >= .5*np.array(a['original_margins'])-1e-8)
    with pytest.raises(ValueError):
        fit_distilled_box_linear(X,y,W,b,[0,1],[0],distillation=-1)
