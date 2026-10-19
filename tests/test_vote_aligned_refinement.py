import numpy as np
from certified_frfs.vote_aligned_refinement import soft_topk_loss


def test_implicit_topk_gradient():
    rng = np.random.default_rng(813)
    C = rng.uniform(0, 1, (7, 12, 4))
    labels = np.arange(12) % 3; truth = np.arange(7) % 3
    w = np.array([.2, .3, .25, .25])
    value, gradient = soft_topk_loss(C, labels, truth, w)
    numeric = np.zeros(4)
    for j in range(4):
        delta = np.zeros(4); delta[j] = 1e-6
        numeric[j] = (soft_topk_loss(C, labels, truth, w+delta)[0]-soft_topk_loss(C, labels, truth, w-delta)[0])/2e-6
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numeric, atol=1e-6, rtol=1e-4)
