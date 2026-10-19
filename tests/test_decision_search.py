import numpy as np
import pytest
from certified_frfs.decision_search import select_local_decisions
from certified_frfs.decision_coverage import certified_decision_coverage


def test_local_moves_budget_and_fresh_certificate():
    rng = np.random.default_rng(23)
    X = rng.uniform(.1, .9, (12, 5))
    y = np.repeat([0, 1], 6)
    result = select_local_decisions(X, y, list(range(5)), budget=12)
    assert result['evaluations'] <= 12
    assert len({tuple(r['features']) for r in result['trace']}) == result['evaluations']
    assert result['objective'] >= max(r['objective'] for r in result['trace'][:5])-1e-11
    assert result['certificate'] == certified_decision_coverage(X, y, result['features'])
    assert result == select_local_decisions(X, y, list(range(5)), budget=12)


def test_nominal_accuracy_objective_and_reject_robust_accuracy():
    X = np.array([[0., .2], [.1, .8], [.9, .2], [1., .8]])
    y = np.array([0, 0, 1, 1])
    result = select_local_decisions(X, y, [0, 1], eps=0, accuracy_only=True)
    for row in result['trace']:
        assert np.isclose(row['objective'], row['balanced_coverage']-.02*len(row['features'])/2)
    with pytest.raises(ValueError):
        select_local_decisions(X, y, [0, 1], accuracy_only=True)
