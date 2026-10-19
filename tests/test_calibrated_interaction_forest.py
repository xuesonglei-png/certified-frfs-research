import numpy as np
from sklearn.metrics import mutual_info_score
from certified_frfs.calibrated_interaction_forest import _mi,calibrated_interaction_forest,CalibratedForestContext
from certified_frfs.triangular_classifier import triangular_memberships

def test_vectorized_information_matches_independent_sklearn_with_empty_cells():
    rng = np.random.default_rng(433); bins = rng.integers(0,3,(47,5)); y = rng.integers(0,4,47)
    bins[:,2] = 0
    expected = [mutual_info_score(bins[:,a],y) for a in range(5)]
    np.testing.assert_allclose(_mi(bins,y,3,4),expected,atol=1e-12)

def test_xor_interaction_admitted_nuisance_not_forced_into_spanning_tree():
    X = np.array([[u,v,z] for u in (0.,1.) for v in (0.,1.) for z in (0.,1.)]*30)
    y = (X[:,0] != X[:,1]).astype(int)
    tree = calibrated_interaction_forest(X,y)
    assert tree['edges'] == [[0,1]] and len(tree['edges']) < X.shape[1]-1
    assert next(c for c in tree['candidates'] if c['edge'] == [0,1])['maxT_p'] == 1/32

def test_null_identical_coordinates_no_false_synergy_and_context_subforest():
    X = np.column_stack([np.linspace(0,1,30)]*4); y = np.arange(30)%3
    tree = calibrated_interaction_forest(X,y)
    assert not tree['edges']
    context = CalibratedForestContext(X,y)
    assert context.tree([0,2])['edges'] == []
    fit = context.classifier([0,2],1.,1.)
    assert fit['features'] == [0,2] and not fit['edges']

def test_recorded_permutations_replay_correct_global_max_and_each_pvalue():
    rng = np.random.default_rng(71); X = rng.uniform(size=(36,5)); y = np.arange(36)%3
    tree = calibrated_interaction_forest(X,y); bins = triangular_memberships(X).argmax(axis=-1)
    maxima = []
    for indices in tree['permutation_indices']:
        truth = y[indices]; singles = [mutual_info_score(bins[:,a],truth) for a in range(5)]
        scores = [mutual_info_score(3*bins[:,u]+bins[:,v],truth)-singles[u]-singles[v] for u in range(5) for v in range(u+1,5)]
        maxima.append(max(scores))
    np.testing.assert_allclose(tree['null_maxima'],maxima,atol=1e-12)
    for row in tree['candidates']:
        assert row['maxT_p'] == (1+sum(round(v,12) >= row['synergy'] for v in maxima))/32
