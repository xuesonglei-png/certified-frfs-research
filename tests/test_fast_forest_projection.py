import itertools
import numpy as np
import pytest
from certified_frfs.fast_forest_projection import ForestProjectionLoss as fast
from certified_frfs.forest_certified_backward import ForestProjectionLoss as direct


@pytest.mark.parametrize('q', [2, 3])
def test_gram_loss_matches_direct_omission_for_every_raw_subset(q):
    rng = np.random.default_rng(995+q); X = rng.uniform(size=(23, 5)); y = np.arange(23)%q
    fit = {'coef': rng.normal(size=(q, 5, 3)).tolist(), 'intercept': rng.normal(size=q).tolist(),
        'edges': [[0, 1], [1, 2], [2, 3], [3, 4]], 'pair_coef': rng.normal(size=(q, 4, 3, 3)).tolist()}
    a, b = fast(X, y, np.arange(q), fit), direct(X, y, np.arange(q), fit)
    for mask in itertools.product((False, True), repeat=5):
        fs = list(np.flatnonzero(mask))
        np.testing.assert_allclose(a(fs), b(fs), rtol=1e-12, atol=1e-12)
    assert a(list(range(5))) == 0.
