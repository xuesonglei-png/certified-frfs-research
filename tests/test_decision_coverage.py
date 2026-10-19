import numpy as np
from scipy.spatial.distance import cdist
from certified_frfs.decision_coverage import certified_decision_coverage, select_decision_coverage


def test_bounds_lower_than_observed_shared_realization_margins():
    rng = np.random.default_rng(71)
    X = rng.uniform(.1, .9, (9, 3))
    y = np.repeat([0, 1, 2], 3)
    w = np.array([.2, .3, .5])
    result = certified_decision_coverage(X, y, [0, 2], eps=.05, weights=w)
    bound = np.array(result['lower_bounds'])
    for _ in range(30):
        Z = X + rng.uniform(-.05, .05, X.shape)
        distance = cdist(Z[:, [0, 2]]*w[[0, 2]], Z[:, [0, 2]]*w[[0, 2]], 'cityblock')
        np.fill_diagonal(distance, np.inf)
        actual = np.where(y[:, None] != y[None, :], distance, np.inf).min(axis=1) - np.where(
            y[:, None] == y[None, :], distance, np.inf).min(axis=1)
        assert np.all(bound <= actual + 1e-12)


def test_singleton_class_has_no_certified_self_friend():
    X = np.array([[0.], [.1], [.9]])
    result = certified_decision_coverage(X, np.array([0, 0, 1]), [0], eps=0)
    assert result['lower_bounds'][2] is None
    assert result['class_coverage'][1]['fraction'] == 0


def test_selection_reports_vacuous_coverage_and_recomputed_certificate():
    X = np.array([[0., .2], [.1, .3], [.9, .4]])
    y = np.array([0, 0, 1])
    result = select_decision_coverage(X, y, [1, 0], eps=.01)
    assert not result['nonvacuous']
    assert result['certificate'] == certified_decision_coverage(X, y, result['features'], eps=.01)
    best = max(result['trace'], key=lambda row: (row['objective'], -row['k']))
    assert len(result['features']) == best['k']


def test_continuous_margin_utility_matches_class_balanced_formula():
    X = np.array([[.1, .2], [.12, .22], [.8, .2], [.82, .21]])
    y = np.array([0, 0, 1, 1])
    result = select_decision_coverage(X, y, [0, 1], margin_scale=.05, penalty=.02)
    for row in result['trace']:
        k = row['k']
        cert = certified_decision_coverage(X, y, [0, 1][:k])
        utility = np.clip(np.array(cert['lower_bounds'])/(.05*k/2), -1, 1)
        classes = [utility[y == label].mean() for label in [0, 1]]
        expected = .5*np.mean(classes)+.5*min(classes)-.02*k/2
        assert np.isclose(row['objective'], expected)
