import numpy as np
from certified_frfs.joint_group_fuzzy import joint_group_scores


def test_joint_group_model_certifies_margin_and_ranks_zero_information_group_last():
    X = np.array([[.1, .5], [.2, .5], [.8, .5], [.9, .5]])
    y = np.array([0, 0, 1, 1])
    W = np.zeros((2, 2, 3))
    W[1, 0] = [-1, 0, 1]
    result = joint_group_scores(X, y, W, [0, 0], [0, 1])
    assert result['feasible']
    assert result['ranking'] == [0, 1]
    assert np.all(np.array(result['deployed_margins']) >= .5*np.array(result['original_margins'])-1e-8)
