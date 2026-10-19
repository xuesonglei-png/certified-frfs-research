import numpy as np
from certified_frfs.witness_mixture import mixture_envelope,coordinate_envelopes,witness_mixture_loss_bound
from certified_frfs.core import gamma


def test_opposed_witnesses_strictly_tighten_shared_anchor():
    fit=mixture_envelope([np.array([[0.,1.],[1.,0.]])])
    assert np.isclose(fit['upper'],.5)
    assert fit['single_witness_upper']==1
    assert np.allclose(fit['lambda'],[.5,.5])


def test_piecewise_knots_dominate_continuous_anchor_samples():
    lo=np.array([[.4,.1],[0.,.2],[.9,.7]]);hi=np.array([[.6,.4],[.1,.3],[1.,.9]])
    wf=np.array([.8,.6]);wr=np.array([.3,.9]);witnesses=np.array([1,2])
    grids,_=coordinate_envelopes(lo,hi,0,1,witnesses,wf,wr);fit=mixture_envelope(grids)
    rng=np.random.default_rng(7)
    for _ in range(100):
        z=rng.uniform(lo,hi)
        actual=min(abs(z[0]-z[k])@wf-abs(z[0]-z[1])@wr for k in witnesses)
        assert actual<=fit['upper']+1e-12


def test_dependency_samples_and_old_bound_domination():
    rng=np.random.default_rng(92);X=rng.random((12,4));y=np.repeat([0,1],6)
    wf=np.ones(4)/2;fs=[0,1,3];wr=np.array([.8,.5,.7])
    result=witness_mixture_loss_bound(X,y,fs,wf,wr,eps=.05)
    assert result['upper_bound']<=result['redistributed_bound']
    wide=np.zeros(4);wide[fs]=wr
    for _ in range(40):
        Z=np.clip(X+rng.uniform(-.05,.05,X.shape),0,1)
        assert gamma(Z,y,weights=wf)-gamma(Z,y,fs,weights=wide)<=result['upper_bound']+1e-9
