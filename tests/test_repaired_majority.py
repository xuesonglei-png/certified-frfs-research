import numpy as np
from certified_frfs.repaired_majority import repair_majority_metric
from certified_frfs.feasible_transport import weight_certificate


def test_repaired_support_remains_certified_and_records_added_cost():
    rng = np.random.default_rng(994); X = rng.uniform(.1, .9, (16, 3)); y = np.repeat([0, 1], 8)
    X[:, 0] = .1+.8*y; full = np.ones(3)/np.sqrt(3)
    candidate = {'features': [1], 'weights': [0., float(full.sum()), 0.], 'fit': {}, 'config': {}}
    result = repair_majority_metric(X, y, candidate, np.array([.7, .2, .1]), ridge=.01)
    w = np.array(result['weights']); fs = result['features']
    assert weight_certificate(X, y, fs, full, w[fs])['feasible']
    assert abs(w.sum()-full.sum()) < 1e-12
    assert result['repair']['trace'] or result['repair']['restored_budget'] == 0
    if result['repair']['trace']: assert result['repair']['trace'][0]['added_attribute'] == 0


def test_already_certified_candidate_is_preserved_exactly():
    rng = np.random.default_rng(995); X = rng.uniform(.1, .9, (16, 3)); y = np.repeat([0, 1], 8)
    full = np.ones(3)/np.sqrt(3)
    candidate = {'features': [0, 1, 2], 'weights': full.tolist(), 'fit': {}, 'config': {}}
    result = repair_majority_metric(X, y, candidate, np.ones(3)/3, ridge=.01)
    assert result['features'] == candidate['features'] and result['weights'] == candidate['weights']
    assert not result['repair']['trace']
