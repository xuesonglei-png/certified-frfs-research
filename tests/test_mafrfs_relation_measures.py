"""Hand calculations and scalar loops distinguish fuzzy labels from proxies."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mafrfs_relation_measures import fuzzy_labels_from_full_relation, fuzzy_lower_approximations, fuzzy_dependency, fuzzy_conditional_entropy


def standard_negation(x): return 1-x


def test_fuzzy_memberships_and_class_union_match_hand_calculation_not_crisp_proxy():
    full = np.array([[1., .5], [.5, 1.]])
    classes, labels = fuzzy_labels_from_full_relation(full, [0, 1])
    np.testing.assert_array_equal(classes, [0, 1])
    np.testing.assert_allclose(labels, [[2/3, 1/3], [1/3, 2/3]])
    identity = np.eye(2)
    score = fuzzy_dependency(identity, labels, t_conorm=np.maximum, negation=standard_negation)
    assert score == pytest.approx(2/3)
    # Crisp labels would give dependency 1; summing class lower memberships
    # would also give 1. Both would incorrectly substitute the published vee.
    assert score < 1
    np.testing.assert_allclose(fuzzy_lower_approximations(full, labels, t_conorm=np.maximum, negation=standard_negation), [[.5, 1/3], [1/3, .5]])


def test_scalar_replay_monotonicity_and_fixed_full_labels():
    rng = np.random.default_rng(258); R = rng.uniform(0, 1, (7, 7)); R = (R+R.T)/2; np.fill_diagonal(R, 1)
    _, labels = fuzzy_labels_from_full_relation(R, [0, 0, 0, 1, 1, 2, 2])
    finer = .6*R; np.fill_diagonal(finer, 1)
    scalar = np.array([[min(max(1-finer[i, j], labels[j, c]) for j in range(7)) for c in range(3)] for i in range(7)])
    vector = fuzzy_lower_approximations(finer, labels, t_conorm=np.maximum, negation=standard_negation)
    np.testing.assert_allclose(vector, scalar)
    actual = fuzzy_dependency(finer, labels, t_conorm=np.maximum, negation=standard_negation)
    assert actual == pytest.approx(np.mean([max(row) for row in scalar]))
    assert actual >= fuzzy_dependency(R, labels, t_conorm=np.maximum, negation=standard_negation)
    assert actual == pytest.approx(fuzzy_dependency(finer, labels[:, ::-1], t_conorm=np.maximum, negation=standard_negation))


def test_conditional_entropy_supplied_label_relation_and_log_base():
    R = np.array([[1., .5], [.5, 1.]]); L = np.array([[1., .3], [.3, 1.]])
    assert fuzzy_conditional_entropy(R, L, log_base=2) == pytest.approx(-np.log2(13/15))
    assert fuzzy_conditional_entropy(R, np.ones((2, 2)), log_base=np.e) == 0
    with pytest.raises(ValueError): fuzzy_conditional_entropy(R, L, log_base=1)
    with pytest.raises(ValueError): fuzzy_labels_from_full_relation([[1, 2], [2, 1]], [0, 1])
    with pytest.raises(ValueError): fuzzy_labels_from_full_relation([[1, .2], [.3, 1]], [0, 1])
