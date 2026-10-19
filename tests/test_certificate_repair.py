import numpy as np

from certified_frfs.certificate_repair import repair_support
from certified_frfs.feasible_transport import weight_certificate


def test_repair_recovers_certified_metric_and_keeps_requested_attributes():
    rng = np.random.default_rng(51)
    X = rng.random((30, 4))
    y = (X[:, 0] > .5).astype(int)
    W = np.zeros((2, 4, 3))
    W[1, 0] = [-1, 0, 1]
    result = repair_support(X, y, W, [1, 2])
    assert {1, 2}.issubset(result['features'])
    assert result['certificate']['feasible']
    v = np.array(result['weights'])
    assert np.isclose(v.sum(), 2)
    assert v.max() <= 1 + 1e-10
    assert weight_certificate(X, y, result['features'], np.ones(4)/2,
                              v[result['features']])['feasible']
    supports = [r['features'] for r in result['repair_trace']]
    assert all(len(b) == len(a) + 1 and set(a).issubset(b)
               for a, b in zip(supports, supports[1:]))

