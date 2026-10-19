from itertools import product
import numpy as np
from certified_frfs.triangular_classifier import triangular_memberships,triangular_box_margins

def test_exact_granule_margin_matches_all_raw_breakpoint_vertices():
    rng=np.random.default_rng(33);X=rng.random((5,3));y=np.arange(5)%3;W=rng.normal(size=(3,3,3));b=rng.normal(size=3);eps=.2
    actual=triangular_box_margins(X,y,W,b,[0,1,2],eps=eps);expected=[]
    for x,label in zip(X,y):
        points=[[max(0,t-eps),min(1,t+eps),np.clip(.5,max(0,t-eps),min(1,t+eps))] for t in x]
        values=[]
        for raw in product(*points):
            scores=np.einsum('qdk,dk->q',W,triangular_memberships(raw))+b
            values.extend(scores[label]-scores[r] for r in range(3) if r!=label)
        expected.append(min(values))
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    relaxed=triangular_box_margins(X,y,W,b,[0,1,2],eps=eps,independent_memberships=True)
    assert np.all(relaxed<=actual+1e-12)

def test_shared_membership_coupling_certifies_when_independent_boxes_cannot():
    X=np.array([[.5]]);W=np.array([[[0.,0.,0.]],[[-1.,0.,-1.]]]);b=np.array([0.,.03])
    exact=triangular_box_margins(X,[1],W,b,[0,1],eps=.01)
    relaxed=triangular_box_margins(X,[1],W,b,[0,1],eps=.01,independent_memberships=True)
    np.testing.assert_allclose(exact,[.01],atol=1e-12);np.testing.assert_allclose(relaxed,[-.01],atol=1e-12)
    np.testing.assert_allclose(triangular_memberships(X).sum(axis=-1),1)
