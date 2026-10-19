import numpy as np
from certified_frfs.classwise_box_linear_refit import fit_classwise_box_linear
from certified_frfs.distilled_box_linear_refit import fit_distilled_box_linear
from certified_frfs.linear_retention import box_class_margins

def test_classwise_retention_can_relax_contradictory_positive_teacher_anchors():
    X=np.vstack([np.tile([.1,0.],(20,1)),np.tile([.9,1.],(20,1))])
    X[19,0]=.9;X[39,0]=.1;y=np.r_[np.zeros(20,int),np.ones(20,int)]
    W=np.array([[0.,0.],[0.,2.]]);b=np.array([0.,-1.]);classes=[0,1]
    assert not fit_distilled_box_linear(X,y,W,b,classes,[0])['feasible']
    result=fit_classwise_box_linear(X,y,W,b,classes,[0],retention=.9)
    assert result['feasible'] and result['required']==[18,18]
    margins=box_class_margins(X,y,np.array(result['coef']),np.array(result['intercept']),classes)
    original=np.array(result['original_margins']);retained=(margins>=.5*original-1e-8)&(margins>1e-10)
    assert [int(sum(retained[y==c])) for c in classes]==result['retained_counts']
    assert all(c>=18 for c in result['retained_counts'])
    assert not fit_classwise_box_linear(X,y,W,b,classes,[0],retention=1)['feasible']

def test_classwise_model_is_teacher_gauge_invariant():
    X=np.array([[0.,0.],[.1,.1],[.9,.9],[1.,1.]])
    y=np.array([0,0,1,1]);W=np.array([[0.,0.],[2.,2.]]);b=np.array([0.,-2.])
    a=fit_classwise_box_linear(X,y,W,b,[0,1],[0])
    other=fit_classwise_box_linear(X,y,W+7,b+3,[0,1],[0])
    assert a['feasible'] and other['feasible']
    np.testing.assert_allclose(a['coef'],other['coef'],atol=1e-10)
    np.testing.assert_allclose(a['intercept'],other['intercept'],atol=1e-10)
