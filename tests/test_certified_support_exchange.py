import numpy as np
from certified_frfs.certified_support_exchange import refine_support_exchange


def test_exchange_preserves_certified_mass_cardinality():
    X = np.array([[.05, .1, .05], [.1, .05, .1], [.15, .2, .15], [.2, .15, .2],
                  [.8, .85, .8], [.85, .8, .85], [.9, .95, .9], [.95, .9, .95]])
    y = np.array([0]*4+[1]*4)
    result = refine_support_exchange(X, y, {'features': [0, 1], 'weights': [np.sqrt(3)/2, np.sqrt(3)/2, 0.]})
    assert result['certificate']['feasible']
    assert len(result['features']) <= 2
    np.testing.assert_allclose(sum(result['weights']), np.sqrt(3))
    assert result['attempts']
