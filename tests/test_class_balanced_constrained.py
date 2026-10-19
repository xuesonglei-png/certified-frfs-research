import numpy as np
from certified_frfs.constrained_fuzzy_logistic import fit_constrained_logistic


def test_class_balanced_loss_keeps_exact_margin_certificate():
    X = np.array([[.1], [.2], [.8], [.9], [.85]])
    y = np.array([0, 0, 0, 1, 1])
    W = np.array([[[0., 0., 0.]], [[-1., 0., 1.]]])
    result = fit_constrained_logistic(X, y, W, [0., 0.], [0, 1], [0], class_balanced=True)
    assert result['feasible'] and result['class_balanced']
