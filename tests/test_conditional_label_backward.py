import numpy as np
from certified_frfs.conditional_label_backward import LabelReconstructionLoss, conditional_label_path
from certified_frfs.triangular_classifier import triangular_memberships


def test_label_cost_matches_direct_balanced_ridge_and_label_permutation():
    rng = np.random.default_rng(419)
    X = rng.random((23, 4)); y = np.array([0]*16+[1]*7)
    loss = LabelReconstructionLoss(X, y)
    fs = [0, 2]
    Z = np.column_stack([np.ones(len(X)), triangular_memberships(X[:, fs]).reshape(len(X), -1)])
    target = np.eye(2)[y]-.5
    weights = 1/(2*np.bincount(y)[y])
    P = np.eye(Z.shape[1])*.001; P[0, 0] = 0
    beta = np.linalg.solve(Z.T@(weights[:, None]*Z)+P, Z.T@(weights[:, None]*target))
    expected = weights@np.mean((Z@beta-target)**2, axis=1)
    np.testing.assert_allclose(loss(fs), expected, atol=1e-12)
    np.testing.assert_allclose(loss(fs), LabelReconstructionLoss(X, 1-y)(fs), atol=1e-12)


def test_label_path_accepts_only_original_certificates():
    rng = np.random.default_rng(420)
    X = rng.random((18, 4)); y = (X[:, 0] > .5).astype(int)
    path = conditional_label_path(X, y)
    assert all(r['certificate']['feasible'] for r in path['trace'])
    assert all(r['accepted'] == r['certificate']['feasible'] for r in path['attempts'])
    assert all(len(a['features']) == len(b['features'])+1 for a, b in zip(path['trace'], path['trace'][1:]))
