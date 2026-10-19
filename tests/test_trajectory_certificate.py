import numpy as np
import pytest
from certified_frfs.feasible_query_descent import fit_feasible_query
from certified_frfs.trajectory_certificate import certify_capped_trajectory


def test_every_proposal_keeps_cap_mass_and_selected_endpoint_is_certified():
    rng = np.random.default_rng(2201); y = np.repeat([0, 1], 12)
    X = rng.uniform(0, .1, (24, 4))+.7*y[:, None]
    trajectory = fit_feasible_query(X, y, certified=False, weight_cap_ratio=2.)
    fit = certify_capped_trajectory(X, y, trajectory)
    assert fit['selected']['certificate']['feasible']
    assert len(fit['candidates']) <= 13
    assert fit['selected']['objective'] <= fit['candidates'][0]['objective']+1e-12
    for c in fit['candidates']:
        v = np.array(c['weights']); assert np.all(v >= 0) and np.all(v <= 1.+1e-10)
        assert abs(v.sum()-2.) < 1e-10


def test_trajectory_with_changed_query_splits_is_rejected():
    y = np.repeat([0, 1], 12); X = np.repeat((.1+.7*y)[:, None], 4, axis=1)
    trajectory = fit_feasible_query(X, y, certified=False, weight_cap_ratio=2.)
    trajectory['training_splits'] = []
    with pytest.raises(ValueError): certify_capped_trajectory(X, y, trajectory)
