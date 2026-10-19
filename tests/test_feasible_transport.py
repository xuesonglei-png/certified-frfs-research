import numpy as np
from certified_frfs.feasible_transport import fit_feasible_transport, weight_certificate
from certified_frfs.core import gamma


def table():
    rng = np.random.default_rng(981)
    y = np.repeat([0, 1], 8)
    X = rng.uniform(.1, .9, (16, 3)); X[:, 0] = .1+.8*y
    return X, y, np.ones(3)/3


def test_selection_checks_all_points_and_conserves_mass():
    X, y, w = table()
    v, record = fit_feasible_transport(X, y, [0, 1], w)
    assert len(record['candidates']) == 5
    for r in record['candidates']:
        assert abs(sum(r['weights'])-sum(w)) < 1e-12
    feasible = [r for r in record['candidates'] if r['feasible']]
    if feasible:
        assert record['selected']['training_objective'] == min(r['training_objective'] for r in feasible)
        assert v is not None
    else:
        assert v is None


def test_empty_feasible_set_never_returns_uncertified_weights(monkeypatch):
    import certified_frfs.feasible_transport as module
    monkeypatch.setattr(module, 'weight_certificate', lambda *a, **k: {'feasible': False})
    X, y, w = table()
    v, record = fit_feasible_transport(X, y, [0, 1], w)
    assert v is None and record['selected'] is None


def test_changed_metric_bound_covers_sampled_shared_losses():
    X, y, w = table(); fs = [0, 1]; v = np.array([.7, .3])
    r = weight_certificate(X, y, fs, w, v)
    rng = np.random.default_rng(982); wide = np.array([.7, .3, 0.])
    for _ in range(12):
        Z = np.clip(X+rng.uniform(-.01, .01, X.shape), 0, 1)
        assert gamma(Z, y, weights=w)-gamma(Z, y, fs, weights=wide) <= r['dependency_certificate']['upper_bound']+1e-10


def test_nested_selector_splits_and_feasible_ranking():
    X, y, w = table()
    _, record = fit_feasible_transport(X, y, [0, 1], w, selection='crossfit')
    seen = []
    for split in record['selection_splits']:
        assert set(split['train']).isdisjoint(split['validation'])
        assert set(split['train']) | set(split['validation']) == set(range(len(y)))
        seen.extend(split['validation'])
    assert sorted(seen) == list(range(len(y)))
    feasible = [r for r in record['candidates'] if r['feasible']]
    if feasible:
        assert record['selected']['selection_utility'] == max(r['selection_utility'] for r in feasible)


def test_restoration_records_cost_and_keeps_certificate():
    from certified_frfs.feasible_transport import fit_repaired_transport
    X, y, w = table()
    fs, v, record = fit_repaired_transport(X, y, [1], w, selection='loo')
    assert record['restored_count'] == len(fs)-1
    assert 1 in fs
    assert weight_certificate(X, y, fs, w, v)['feasible']
