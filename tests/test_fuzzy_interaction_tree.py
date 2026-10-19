import numpy as np
from certified_frfs.fuzzy_interaction_tree import interaction_tree


def test_joint_label_information_detects_XOR_without_single_feature_signal():
    # A mechanism example, not a countable benchmark dataset.
    base = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    X = np.tile(base, (5, 1)); y = np.tile([0, 1, 1, 0], 5)
    tree = interaction_tree(X, y)
    assert tree['edges'][0] == [0, 1]
    assert tree['single_MI'] == [0., 0., 0.]
    assert tree['selected'][0]['complementarity'] > .6
    assert len(tree['edges']) == 2
