import numpy as np
from certified_frfs.joint_group_fuzzy import joint_group_scores


def test_joint_solver_records_attempts_and_rechecks_feasibility():
    rng = np.random.default_rng(22)
    X = rng.random((18, 3))
    y = np.r_[np.zeros(9, int), np.ones(9, int)]
    W = np.zeros((2, 3, 3))
    W[1, 0] = [-1, 0, 1]
    result = joint_group_scores(X, y, W, [0, 0], [0, 1])
    assert result['solver_attempts']
    assert result['feasible']
