import numpy as np
from certified_frfs.boundary_fuzzy_support import boundary_weights, boundary_score_supports


def test_boundary_weights_balance_classes_and_prioritize_mistakes():
    y = np.array([0, 0, 0, 1])
    scores = np.array([[6., 0.], [0., 6.], [0., 0.], [0., 6.]])
    weights = boundary_weights(scores, y)
    assert np.isclose(weights[y == 0].sum(), .5)
    assert np.isclose(weights[y == 1].sum(), .5)
    assert weights[1] > weights[2] > weights[0]
    np.testing.assert_allclose(weights, boundary_weights(scores+np.arange(4)[:, None], y))


def test_grouped_selection_preserves_sole_informative_raw_attribute():
    rng = np.random.default_rng(104)
    X = rng.random((40, 4))
    y = (X[:, 0] > .5).astype(int)
    W = np.zeros((2, 4, 3))
    W[1, 0] = [-1, 0, 1]
    rows = boundary_score_supports(X, y, W, [0, 0], fractions=(.5,))
    assert len(rows) == 1 and len(rows[0]['features']) == 2
    assert 0 in rows[0]['features']
