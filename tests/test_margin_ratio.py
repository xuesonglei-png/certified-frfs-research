import numpy as np
import pytest
from certified_frfs.constrained_fuzzy_logistic import fit_constrained_logistic


def test_explicit_margin_fraction_is_verified_and_reported():
    X = np.array([[.1], [.2], [.8], [.9]])
    y = np.array([0, 0, 1, 1])
    W = np.array([[[0, 0, 0]], [[-1, 0, 1]]])
    result = fit_constrained_logistic(X, y, W, [0, 0], [0, 1], [0], ratio=.25)
    assert result['feasible'] and result['ratio'] == .25
    assert np.all(np.array(result['deployed_margins']) >= .25*np.array(result['original_margins'])-1e-8)
    with pytest.raises(ValueError):
        fit_constrained_logistic(X, y, W, [0, 0], [0, 1], [0], ratio=0)
