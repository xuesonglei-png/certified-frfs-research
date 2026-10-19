import numpy as np
from certified_frfs.triplet_metric import learn_triplet_weights


def test_triplet_metric_prefers_separating_coordinate_and_is_normalized():
    X=np.array([[0,.0],[0,.1],[0,.2],[1,.0],[1,.1],[1,.2]],float)
    y=np.array([0,0,0,1,1,1])
    w=learn_triplet_weights(X,y,margin=.02,reg=.01,seed=3)
    assert np.isclose(np.linalg.norm(w),1)
    assert np.all(w>0)
    assert w[0]>w[1]


def test_triplet_metric_is_deterministic():
    X=np.random.default_rng(4).random((30,5)); y=np.repeat([0,1,2],10)
    a=learn_triplet_weights(X,y,seed=8); b=learn_triplet_weights(X,y,seed=8)
    assert np.allclose(a,b)
