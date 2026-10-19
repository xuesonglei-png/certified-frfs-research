import numpy as np
from certified_frfs.shared_metric_attack import dependency_gradient,attack_shared_metric
from certified_frfs.core import gamma


def test_dependency_gradient_matches_finite_difference():
    rng=np.random.default_rng(123);X=rng.uniform(.1,.9,(10,3));y=np.repeat([0,1],5);w=np.array([.2,.3,.25])
    value,g=dependency_gradient(X,y,w);assert abs(value-gamma(X,y,weights=w))<1e-12
    for i,a in [(0,0),(2,1),(7,2)]:
        plus,minus=X.copy(),X.copy();plus[i,a]+=1e-7;minus[i,a]-=1e-7
        diff=(dependency_gradient(plus,y,w)[0]-dependency_gradient(minus,y,w)[0])/2e-7
        assert abs(diff-g[i,a])<1e-8


def test_attack_is_shared_box_lower_witness():
    rng=np.random.default_rng(63);X=rng.random((12,4));y=np.repeat([0,1],6);wf=np.ones(4)/2;wr=np.array([1.,1.,0.,0.])
    fit=attack_shared_metric(X,y,wf,wr,starts=3,steps=5);Z=np.array(fit['perturbed_table'])
    assert np.all(Z>=np.maximum(0,X-.01)) and np.all(Z<=np.minimum(1,X+.01))
    actual=gamma(Z,y,weights=wf)-gamma(Z,y,weights=wr)
    assert abs(actual-fit['observed_loss_lower'])<1e-12
    assert actual>=gamma(X,y,weights=wf)-gamma(X,y,weights=wr)-1e-12
    assert all(np.all(np.diff(r['trace'])>=0) for r in fit['runs'])
