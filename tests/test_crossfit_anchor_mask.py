import numpy as np
from certified_frfs.crossfit_anchor_mask import crossfit_anchor_mask


def test_singletons_are_explicitly_unverified_and_all_queries_disjoint():
    rng = np.random.default_rng(14)
    X = rng.random((21, 3))
    y = np.r_[np.zeros(10, int), np.ones(10, int), 2]
    result = crossfit_anchor_mask(X, y, C=1)
    assert result['unverified_count'] == 1
    assert not result['mask'][-1] and result['margins'][-1] is None
    ids = []
    for fold in result['folds']:
        assert set(fold['reference_indices']).isdisjoint(fold['validation_indices'])
        assert 20 in fold['reference_indices']
        ids.extend(fold['validation_indices'])
    assert sorted(ids) == list(range(20))
