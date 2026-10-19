import numpy as np
from certified_frfs.constrained_fuzzy_logistic import fit_constrained_logistic
from certified_frfs.triangular_classifier import triangular_box_margins


def test_constrained_logistic_deletes_redundant_group_and_preserves_raw_box_margin():
    X = np.array([[.1, .1], [.2, .2], [.8, .8], [.9, .9]])
    y = np.array([0, 0, 1, 1])
    W = np.zeros((2, 2, 3))
    W[1, :, :] = [-.5, 0, .5]
    fitted = fit_constrained_logistic(X, y, W, [0, 0], [0, 1], [0])
    assert fitted['feasible']
    assert np.all(np.array(fitted['coef'])[:, 1] == 0)
    actual = triangular_box_margins(X, y, fitted['coef'], fitted['intercept'], [0, 1])
    assert np.all(actual >= .5*np.array(fitted['original_margins'])-1e-8)
