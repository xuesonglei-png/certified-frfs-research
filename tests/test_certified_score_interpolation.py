import numpy as np
from certified_frfs.certified_score_interpolation import certified_step
from certified_frfs.triangular_classifier import triangular_box_margins


def test_exact_largest_step_has_feasible_endpoint_and_infeasible_extension():
    X = np.array([[.1], [.9]])
    y = np.array([0, 1])
    classes = [0, 1]
    W = np.array([[[0., 0., 0.]], [[-1., 0., 1.]]])
    b = np.zeros(2)
    original = triangular_box_margins(X, y, W, b, classes, eps=.01)
    target = -W
    result = certified_step(X, y, W, b, target, b, classes, original)
    assert np.isclose(result['step'], .25)
    assert result['cuts'] > 0
    extended = W+(result['step']+1e-5)*(target-W)
    assert np.any(triangular_box_margins(X, y, extended, b, classes, eps=.01) < .5*original)


def test_feasible_target_accepts_full_segment():
    X = np.array([[.1], [.9]])
    y = np.array([0, 1])
    W = np.array([[[0., 0., 0.]], [[-1., 0., 1.]]])
    original = triangular_box_margins(X, y, W, [0, 0], [0, 1])
    result = certified_step(X, y, W, [0, 0], W*2, [0, 0], [0, 1], original)
    assert result['step'] == 1
