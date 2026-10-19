import numpy as np
from certified_frfs.predictive_deletion_support import predictive_supports


def test_predictive_deletion_detects_informative_attribute_without_query_overlap():
    rng = np.random.default_rng(99)
    X = rng.random((60, 3))
    y = (X[:, 0] > .5).astype(int)
    candidates, trace = predictive_supports(X, y, C=1, fractions=(.5,))
    assert trace['ranking'][0] == 0
    assert 0 in candidates[0]['features']
    visits = []
    for fold in trace['folds']:
        assert set(fold['reference_indices']).isdisjoint(fold['validation_indices'])
        visits.extend(fold['validation_indices'])
    assert sorted(visits) == list(range(60))
