import numpy as np
import pytest
from certified_frfs.core import certificate
from certified_frfs.predictive_repair import repair_to_certificate


def test_restore_informative_coordinate_under_exact_zero_budget():
    X = np.array([[0., 0.], [.1, 0.], [.9, 0.], [1., 0.]])
    y = np.array([0, 0, 1, 1])
    result = repair_to_certificate(X, y, [1, 0], 1, eps=0, eta=0)
    assert result['features'] == [1, 0]
    assert result['restored_count'] == 1
    assert certificate(X, y, result['features'], eps=0)['upper_bound'] == 0


def test_already_feasible_ranked_subset_is_preserved():
    X = np.array([[0., 0.], [.1, 0.], [.9, 0.], [1., 0.]])
    y = np.array([0, 0, 1, 1])
    result = repair_to_certificate(X, y, [0, 1], 1, eps=0, eta=.1)
    assert result['features'] == [0]
    assert result['restored_count'] == 0


def test_duplicate_ranking_rejected():
    with pytest.raises(ValueError):
        repair_to_certificate(np.zeros((4, 2)), np.array([0, 0, 1, 1]), [0, 0], 1)
