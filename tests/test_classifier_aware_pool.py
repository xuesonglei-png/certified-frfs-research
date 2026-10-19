import numpy as np
from certified_frfs.classifier_aware_pool import fit_classifier_aware_pool


def test_classifier_aware_pool_returns_feasible_candidate():
    X = np.array([[.05,.1],[.1,.05],[.15,.2],[.2,.15],
                  [.8,.85],[.85,.8],[.9,.95],[.95,.9]])
    y = np.array([0]*4+[1]*4)
    result = fit_classifier_aware_pool(X, y)
    assert result['selected']['certificate']['feasible']
    assert result['selected']['features'] == np.flatnonzero(np.array(result['selected']['weights']) > 0).tolist()
    assert result['trace']
