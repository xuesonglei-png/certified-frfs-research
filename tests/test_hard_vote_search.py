import numpy as np
from certified_frfs.hard_vote_search import refine_hard_vote


def test_small_certified_search_keeps_constraints_and_training_monotonicity():
    X = np.array([[.05, .1], [.1, .05], [.15, .2], [.2, .15],
                  [.8, .85], [.85, .8], [.9, .95], [.95, .9]])
    y = np.array([0]*4+[1]*4)
    initial = {'features': [0, 1], 'weights': [1/np.sqrt(2)]*2}
    result = refine_hard_vote(X, y, initial, steps=2)
    for node in result['trace']:
        assert node['certificate']['feasible']
        np.testing.assert_allclose(sum(node['weights']), np.sqrt(2))
        assert max(node['weights']) <= np.sqrt(2)+1e-8
    assert np.all(np.diff([node['training_utility'] for node in result['trace']]) > 0)
