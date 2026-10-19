from itertools import product
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from certified_frfs.linear_retention import box_class_margins,projected_parameters,linear_parameters,linear_retention_certificate


def test_box_margins_equal_exhaustive_vertex_minima():
    rng=np.random.default_rng(7701);X=rng.uniform(0,1,(8,4));y=np.arange(8)%3
    W=rng.normal(size=(3,4));b=rng.normal(size=3);eps=.15
    actual=box_class_margins(X,y,W,b,np.arange(3),eps=eps)
    expected=[]
    for x,q in zip(X,y):
        lo=np.maximum(0,x-eps);hi=np.minimum(1,x+eps)
        values=[]
        for mask in product((0,1),repeat=4):
            z=np.where(mask,hi,lo);scores=W@z+b
            values.extend(scores[q]-scores[r] for r in range(3) if r!=q)
        expected.append(min(values))
    np.testing.assert_allclose(actual,expected,atol=1e-12)


def test_projection_is_center_imputation_and_does_not_read_removed_values():
    W=np.array([[1.,2.,-3.],[-2.,1.,4.]]);b=np.array([.1,-.3]);center=np.array([.2,.4,.5])
    R,r=projected_parameters(W,b,[0,2],center)
    X=np.array([[.1,.9,.2],[.8,.0,.6]])
    filled=X.copy();filled[:,1]=center[1]
    np.testing.assert_allclose(X@R.T+r,filled@W.T+b)
    X[:,1]=1e12
    np.testing.assert_allclose(X@R.T+r,filled@W.T+b)


def test_binary_class_order_and_certificate_nonvacuous():
    X=np.array([[0.,0.],[.1,.1],[.9,.9],[1.,1.]]);y=np.array([4,4,9,9])
    model=LogisticRegression(C=10).fit(X,y);W,b,classes=linear_parameters(model)
    np.testing.assert_array_equal(classes[(X@W.T+b).argmax(axis=1)],model.predict(X))
    cert=linear_retention_certificate(X,y,W,b,classes,[0,1],X.mean(axis=0))
    assert cert['feasible'] and cert['protected_count']==4 and cert['required']==[2,2]
    empty=linear_retention_certificate(X,y,W,b,classes,[],X.mean(axis=0))
    assert not empty['feasible']
    with pytest.raises(ValueError):box_class_margins(X,y,W,b,classes,eps=-1)
