import numpy as np

from certified_frfs.selective_support_selector_v2 import _query_scores, _support_pool


def test_certified_predictions_remain_fixed_under_shared_perturbations():
    rng = np.random.default_rng(182)
    A = np.vstack((rng.uniform(0., .12, (8, 4)), rng.uniform(.88, 1., (8, 4))))
    y = np.repeat([0, 1], 8)
    B = np.array([[.08]*4, [.93]*4])
    model = {"features": [0, 1, 2, 3], "weights": [.5]*4}
    result = _query_scores(A, y, B, np.array([0, 1]), model)
    assert result["query_coverage"] == 1.
    assert result["query_certified_BA"] == 1.
    for _ in range(20):
        noisy = _query_scores(np.clip(A+rng.uniform(-.01,.01,A.shape),0,1), y,
                              np.clip(B+rng.uniform(-.01,.01,B.shape),0,1),
                              np.array([0,1]), model)
        assert noisy["predictions"] == result["predictions"]


def test_query_truth_does_not_change_stability_coverage():
    A = np.vstack((np.full((5, 2), .05), np.full((5, 2), .95)))
    y = np.repeat([0, 1], 5)
    B = np.array([[.08, .08], [.93, .93]])
    model = {"features": [0, 1], "weights": [1., 1.]}
    correct = _query_scores(A, y, B, np.array([0,1]), model)
    incorrect = _query_scores(A, y, B, np.array([1,0]), model)
    assert correct["predictions"] == incorrect["predictions"]
    assert correct["query_coverage"] == incorrect["query_coverage"] == 1.
    assert correct["query_certified_BA"] == 1.
    assert incorrect["query_certified_BA"] == 0.


def test_exchange_pool_contains_valid_nonprefix_supports():
    rng = np.random.default_rng(45)
    A = rng.random((30, 6)); y = np.repeat([0,1], 15)
    ranking, supports = _support_pool(A, y, np.ones(6)/np.sqrt(6), "uniform_sqrt", 31)
    assert len(supports) == len(set(supports))
    assert all(tuple(sorted(set(fs))) == fs and len(fs) >= 3 for fs in supports)
    prefixes = {tuple(sorted(ranking[:k])) for k in range(3,7)}
    assert any(fs not in prefixes for fs in supports)
    assert tuple(range(6)) in supports
