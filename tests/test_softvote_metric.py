import numpy as np
from scipy.optimize import check_grad
from scipy.special import softmax
from certified_frfs.softvote_metric import softvote_terms,softvote_objective,learn_softvote_weights


def test_gradient_matches_finite_differences():
    rng=np.random.default_rng(17);X=rng.uniform(.1,.9,(12,3));y=np.repeat([0,1,2],4)
    args=softvote_terms(X,y);w=np.array([.2,.3,.5])
    error=check_grad(lambda w:softvote_objective(w,*args)[0],lambda w:softvote_objective(w,*args)[1],w)
    assert error<1e-5


def test_mass_lower_bound_under_shared_box_perturbations():
    rng=np.random.default_rng(7);X=rng.uniform(.1,.9,(12,3));y=np.repeat([0,1,2],4)
    C,same,mask,labels=softvote_terms(X,y);w=np.array([.2,.3,.5])
    bound=np.sum(softmax(np.where(mask,-C@w/.05,-np.inf),axis=1)*same,axis=1)
    for _ in range(100):
        Z=X+rng.uniform(-.01,.01,X.shape)
        distance=np.sum(abs(Z[:,None,:]-Z[None,:,:])*w,axis=2)
        mass=np.sum(softmax(np.where(mask,-distance/.05,-np.inf),axis=1)*same,axis=1)
        assert np.all(bound<=mass+1e-12)


def test_informative_coordinate_receives_more_weight():
    rng=np.random.default_rng(11);y=np.repeat([0,1],20)
    X=np.column_stack([.1+.8*y,rng.uniform(.1,.9,40)])
    w=learn_softvote_weights(X,y)
    assert w[0]>w[1] and abs(np.linalg.norm(w)-1)<1e-10
