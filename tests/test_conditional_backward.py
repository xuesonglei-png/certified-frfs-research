import numpy as np
from certified_frfs.conditional_certified_backward import ReconstructionLoss, conditional_backward_path
from certified_frfs.triangular_classifier import triangular_memberships
from certified_frfs.feasible_transport import weight_certificate


def test_cached_conditional_loss_matches_direct_weighted_ridge():
    rng = np.random.default_rng(73)
    X = rng.random((20, 4)); y = np.repeat([0, 1], [14, 6]); W = rng.normal(size=(2, 4, 3))
    loss = ReconstructionLoss(X, y, W)
    fs = [0, 2]
    design = np.column_stack([np.ones(len(X)), triangular_memberships(X[:, fs]).reshape(len(X), -1)])
    penalty = np.eye(design.shape[1])*.001; penalty[0, 0] = 0
    beta = np.linalg.solve(design.T@(loss.weights[:, None]*design)+penalty, design.T@(loss.weights[:, None]*loss.target))
    expected = loss.weights@np.mean((design@beta-loss.target)**2, axis=1)
    np.testing.assert_allclose(loss(fs), expected, atol=1e-12)


def test_conditional_path_accepts_only_original_certificates_and_ranked_choices():
    rng = np.random.default_rng(41)
    X = rng.random((16, 4)); y = np.repeat([0, 1], 8); W = rng.normal(size=(2, 4, 3))
    path = conditional_backward_path(X, y, W)
    for step in path['trace']:
        fs = step['features']
        assert weight_certificate(X, y, fs, np.ones(4)/2, np.array(step['weights'])[fs])['feasible']
    for ranking in path['rankings']:
        values = [(r['loss'], r['removed']) for r in ranking['choices']]
        assert values == sorted(values)
    assert all(r['accepted'] == r['certificate']['feasible'] for r in path['attempts'])
