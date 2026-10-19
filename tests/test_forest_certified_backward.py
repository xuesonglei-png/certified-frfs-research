import numpy as np
from certified_frfs.forest_certified_backward import ForestProjectionLoss, forest_certified_backward_path


def example():
    X = np.array([[u, v, .5, .5, .5] for u in (.1, .9) for v in (.1, .9)]*3)
    y = ((X[:, 0] > .5) == (X[:, 1] > .5)).astype(int)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    V = np.zeros((2, 4, 3, 3)); V[1, 0] = np.outer([-1, 0, 1], [-1, 0, 1])*4
    fit = {'feasible': True, 'coef': np.zeros((2, 5, 3)).tolist(), 'intercept': [0., 0.],
           'edges': edges, 'pair_coef': V.tolist()}
    return X, y, fit


def test_projection_ranking_detects_joint_signal_with_zero_main_effect():
    X, y, fit = example(); loss = ForestProjectionLoss(X, y, [0, 1], fit)
    assert loss([0, 1, 3, 4]) < 1e-12
    assert loss([1, 2, 3, 4]) > .1
    np.testing.assert_allclose(loss.center, [.5]*5, atol=1e-12)


def test_backward_removes_unused_feature_and_verifies_geometry():
    X, y, fit = example()
    path = forest_certified_backward_path(X, y, [0, 1], fit, np.zeros((2, 5, 3)),
                                         [0., 0.], pair_penalty=3.)
    last = path['trace'][-1]
    assert last['features'] == [0, 1, 3, 4]
    assert last['certificate']['feasible'] and last['fit']['feasible']
    # Teacher protection is explicitly vacuous in this XOR mechanism example.
    assert path['original_teacher_protected_count'] == 0
