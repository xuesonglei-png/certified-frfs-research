import numpy as np
from certified_frfs.adaptive_witness import adaptive_bounds,adaptive_path
from certified_frfs.majority_metric import MajorityTerms


def test_full_space_equivalence_and_fixed_weight_scale():
    rng=np.random.default_rng(19);X=rng.uniform(.1,.9,(16,3));y=np.repeat([0,1],8);w=np.array([.4,.5,.7])
    expected=MajorityTerms(X,y,weights=w).bounds(range(3),allowed_enemies=2)
    np.testing.assert_allclose(adaptive_bounds(X,y,[0,1,2],w),expected)
    direct=MajorityTerms(X[:,[0,2]],y,weights=w[[0,2]]).bounds(range(2),allowed_enemies=2)
    np.testing.assert_allclose(adaptive_bounds(X,y,[0,2],w),direct)


def test_refreshed_path_retains_required_training_objects():
    rng=np.random.default_rng(20);y=np.repeat([0,1],12)
    X=np.column_stack([.1+.8*y,rng.uniform(.1,.9,24),rng.uniform(.1,.9,24)])
    w=np.array([1.,.05,.05]);path=adaptive_path(X,y,[1,2,0],w)
    assert path['candidates'][-1]==[0]
    assert all(c>=r for step in path['trace'] if step['feasible'] for c,r in zip(step['retained'],path['required']))
