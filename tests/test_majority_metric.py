import numpy as np
from itertools import product
from certified_frfs.majority_metric import MajorityTerms,learn_majority_weights,protected_majority_path
from certified_frfs.knn_decision import certified_knn_coverage


def test_majority_terms_match_independent_certificate():
    rng=np.random.default_rng(14);X=rng.random((16,4));y=np.repeat([0,1],8)
    w=np.array([.2,.4,.7,.3]);fs=[0,2,3]
    workspace=MajorityTerms(X,y,weights=w)
    # Different nominal friend pools are possible after deleting features.
    full=workspace.bounds(range(4))
    other=certified_knn_coverage(X,y,range(4),weights=w)
    assert np.allclose(full,other['lower_bounds'])


def test_learned_weights_and_protected_path_remove_redundant_noise():
    rng=np.random.default_rng(3);y=np.repeat([0,1],8)
    X=np.column_stack([y*.8+.1,rng.uniform(.3,.7,16)])
    w=learn_majority_weights(X,y)
    assert w[0]>w[1] and np.isclose(np.linalg.norm(w),1)
    result=protected_majority_path(X,y,[1,0],w)
    assert [0] in result['candidates']
    assert all(c>=r for c,r in zip(result['trace'][0]['retained'],result['required']))


def test_shared_scalar_box_corners_obey_majority_bound():
    X=np.array([[.05],[.10],[.15],[.2],[.8],[.85],[.9],[.95]])
    y=np.repeat([0,1],4);workspace=MajorityTerms(X,y,eps=.02)
    bounds=workspace.bounds([0]);assert np.all(bounds>0)
    for signs in product([-1,1],repeat=8):
        Z=X[:,0]+.02*np.array(signs)
        for i in range(8):
            friends=np.flatnonzero((y==y[i])&(np.arange(8)!=i))
            enemies=np.flatnonzero(y!=y[i])
            actual=min(abs(Z[i]-Z[e])-abs(Z[i]-Z[f]) for e in enemies for f in friends)
            assert actual>=bounds[i]-1e-12


def test_two_preceding_enemies_still_certify_five_neighbor_majority():
    X=np.array([[.1],[.2],[.3],[.4],[.11],[.12],[.9],[.95]])
    y=np.repeat([0,1],4);workspace=MajorityTerms(X,y,eps=.005)
    assert workspace.bounds([0])[0]<0
    bound=workspace.bounds([0],allowed_enemies=2)[0];assert bound>0
    for signs in product([-1,1],repeat=8):
        Z=X[:,0]+.005*np.array(signs)
        distances=np.abs(Z-Z[0]);distances[0]=np.inf
        nearest=np.argsort(distances,kind='stable')[:5]
        assert np.sum(y[nearest]==0)>=3
        enemies=np.abs(Z[4:]-Z[0]);friends=np.abs(Z[1:4]-Z[0])
        actual=np.sort(enemies-friends.max())[2]
        assert actual>=bound-1e-12


def test_three_preceding_enemies_are_not_certified():
    X=np.array([[.1],[.3],[.4],[.5],[.11],[.12],[.13],[.95]])
    y=np.repeat([0,1],4)
    assert MajorityTerms(X,y,eps=0).bounds([0],allowed_enemies=2)[0]<0


def test_nominal_budget_matches_strict_majority_on_tie_free_multiclass_table():
    rng=np.random.default_rng(56);X=rng.random((24,4));y=np.repeat([0,1,2],8)
    w=np.array([.1,.3,.4,.7]);workspace=MajorityTerms(X,y,weights=w,eps=0)
    D=np.abs(X[:,None,:]-X[None,:,:])@w;np.fill_diagonal(D,np.inf)
    nearest=np.argsort(D,axis=1,kind='stable')[:,:5]
    strict=(y[nearest]==y[:,None]).sum(axis=1)>=3
    assert np.array_equal(workspace.bounds(range(4),allowed_enemies=2)>1e-10,strict)
