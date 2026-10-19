import numpy as np
from scipy.spatial.distance import cdist
from certified_frfs.query_certificate import certify_queries


def test_query_bounds_hold_with_shared_anchor_and_reference_perturbations():
    rng = np.random.default_rng(18)
    A = rng.uniform(.1, .9, (8, 3))
    B = rng.uniform(.1, .9, (5, 3))
    y = np.repeat([0, 1], 4)
    targets = np.array([0, 1, 0, 1, 1])
    result = certify_queries(A, y, B, targets, [0, 2], eps=.03)
    for _ in range(40):
        Z = A+rng.uniform(-.03, .03, A.shape)
        Q = B+rng.uniform(-.03, .03, B.shape)
        distance = cdist(Q[:, [0, 2]], Z[:, [0, 2]], 'cityblock')/3
        same = targets[:, None] == y[None, :]
        margins = np.where(~same, distance, np.inf).min(axis=1)-np.where(same, distance, np.inf).min(axis=1)
        assert np.all(np.array(result['lower_bounds']) <= margins+1e-12)


def test_certified_wrong_label_is_stability_not_correctness():
    A = np.array([[.1], [.9]])
    result = certify_queries(A, [0, 1], np.array([[.11]]), [0], [0], eps=.01)
    assert result['certified'] == [True]
    # No query truth is accepted by the certificate: a wrong prediction can be stable.
    assert result['coverage'] == 1.
