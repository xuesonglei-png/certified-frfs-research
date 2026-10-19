import itertools
import numpy as np
import pytest
from certified_frfs.bagged_fixed_forest import fit_bagged_fixed_forest,average_fixed_forest
from certified_frfs.orthogonal_forest_logistic import orthogonal_scores
from certified_frfs.forest_fuzzy_classifier import forest_box_margins


def test_bootstrap_mean_scores_retain_class_counts_and_exact_forest_box_minimum():
    rng = np.random.default_rng(31); X = rng.uniform(size=(32,3)); y = np.arange(32)%2
    result = fit_bagged_fixed_forest(X,y,list(range(3)),[[0,1],[1,2]],members=3)
    for ids in result['bootstrap_indices']:
        assert len(ids) == len(y) and all(sum(y[ids] == c) == sum(y == c) for c in np.unique(y))
    merged = result['fit']; Q = rng.uniform(size=(4,3))
    np.testing.assert_allclose(orthogonal_scores(Q,merged),np.mean([orthogonal_scores(Q,f) for f in result['members']],axis=0),atol=1e-12)
    labels = np.zeros(4,int); eps = .2
    oracle = forest_box_margins(Q,labels,merged['coef'],merged['intercept'],merged['classes'],merged['edges'],merged['pair_coef'],eps=eps)
    member_bounds = [forest_box_margins(Q,labels,f['coef'],f['intercept'],f['classes'],f['edges'],f['pair_coef'],eps=eps) for f in result['members']]
    assert np.all(oracle >= np.mean(member_bounds,axis=0)-1e-12)
    for i,row in enumerate(Q):
        states = [[max(0,x-eps),min(1,x+eps),np.clip(.5,max(0,x-eps),min(1,x+eps))] for x in row]
        grid = np.array(list(itertools.product(*states))); scores = orthogonal_scores(grid,merged)
        np.testing.assert_allclose(oracle[i],np.min(scores[:,0]-scores[:,1]),atol=1e-12)


def test_averaging_rejects_different_graphs_instead_of_silently_creating_cycles():
    X = np.random.default_rng(42).uniform(size=(20,3)); y = np.arange(20)%2
    result = fit_bagged_fixed_forest(X,y,[0,1,2],[[0,1]],members=2)
    other = {**result['members'][1],'edges':[[0,2]]}
    with pytest.raises(ValueError,match='share forest'): average_fixed_forest([result['members'][0],other])
