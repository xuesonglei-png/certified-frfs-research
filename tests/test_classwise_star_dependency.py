from itertools import product
import numpy as np
from classwise_star_dependency import signed_star_maximum,classwise_star_dependency
from certified_frfs.core import positive_region


def test_signed_star_formula_matches_independent_full_knot_enumeration():
    rng=np.random.default_rng(812)
    for _ in range(12):
        lo=rng.uniform(0,.7,3);hi=lo+rng.uniform(.01,.3,3);grid=np.unique(np.r_[lo,hi]);edges=[(0,2),(1,2)]
        states=[grid[(grid>=lo[i])&(grid<=hi[i])] for i in range(3)]
        for coefficient in (.7,-.4):
            brute=max(coefficient*(abs(x[0]-x[2])+abs(x[1]-x[2])) for x in product(*states))
            fit=signed_star_maximum(lo,hi,edges,coefficient)
            np.testing.assert_allclose(fit['maximum'],brute,atol=1e-12,rtol=0)


def test_same_threshold_class_certificate_strictly_improves_and_is_exact_on_case():
    A=np.array([[0,.03],[1,0],[.45,.4]]);y=np.array([0,0,1]);wf=np.ones(2)/np.sqrt(2);wr=np.array([np.sqrt(2)])
    result=classwise_star_dependency(A,y,[1],wf,wr)
    assert not result['original_class_certificate']['feasible_all_classes']
    assert result['class_certificate']['feasible_all_classes'] and result['exact_all_class_cells']
    expected=[.27/(2*np.sqrt(2)),.11/np.sqrt(2)]
    np.testing.assert_allclose([r['raw_signed_class_loss_bound'] for r in result['class_reports']],expected,atol=1e-12,rtol=0)
    maximum=np.zeros(2)
    lo,hi=np.maximum(0,A-.01),np.minimum(1,A+.01)
    for corner in product([0,1],repeat=A.size):
        Z=np.where(np.array(corner).reshape(A.shape),hi,lo)
        loss=positive_region(Z,y,weights=wf)-positive_region(Z,y,[1],weights=np.array([0,np.sqrt(2)]))
        maximum=np.maximum(maximum,[loss[y==c].mean() for c in (0,1)])
    np.testing.assert_allclose(maximum,expected,atol=1e-12,rtol=0)


def test_shared_random_tables_and_arbitrary_priors_obey_class_bounds():
    rng=np.random.default_rng(113);A=rng.uniform(0,1,(15,3));y=np.repeat([0,1,2],5)
    wf=np.ones(3)/np.sqrt(3);wr=np.array([.9,np.sqrt(3)-.9]);result=classwise_star_dependency(A,y,[0,1],wf,wr)
    for _ in range(96):
        Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
        loss=positive_region(Z,y,weights=wf)-positive_region(Z,y,[0,1],weights=np.r_[wr,0.])
        cls=np.array([loss[y==c].mean() for c in (0,1,2)])
        assert np.all(cls<=np.array(result['class_certificate']['upper_bounds'])+1e-10)
        prior=rng.dirichlet(np.ones(3));assert prior@cls<=result['worst_prior_upper_bound']+1e-10
