import numpy as np
from certified_frfs.crossfit_label_backward import CrossfitLabelLoss


def test_crossfit_preprocess_excludes_query_and_keeps_singletons_in_reference():
    rng = np.random.default_rng(421)
    X = rng.random((17, 4)); y = np.array([0]*9+[1]*7+[2])
    loss = CrossfitLabelLoss(X, y)
    visited = []
    for fold in loss.folds:
        a, b = fold['reference_indices'], fold['validation_indices']
        assert set(a).isdisjoint(b) and set(a)|set(b) == set(range(len(y)))
        assert 16 in a and 16 not in b
        visited.extend(b)
        altered = X.copy(); altered[b] += 1000
        changed = CrossfitLabelLoss(altered, y)
        same = next(r for r in changed.folds if r['validation_indices'] == b)
        np.testing.assert_allclose(fold['gram'], same['gram'], atol=1e-12)
        np.testing.assert_allclose(fold['cross'], same['cross'], atol=1e-12)
    assert sorted(visited) == list(range(16))
    assert np.isfinite(loss([0, 2]))
