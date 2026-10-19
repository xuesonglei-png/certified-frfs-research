import numpy as np
from certified_frfs.decision_stability import stability_ranking


def test_subsamples_are_unique_stratified_and_deterministic():
    rng = np.random.default_rng(20)
    X = rng.uniform(.1, .9, (12, 4))
    y = np.repeat([0, 1], 6)
    result = stability_ranking(X, y, repeats=2)
    assert result == stability_ranking(X, y, repeats=2)
    assert sorted(result['ranking']) == list(range(4))
    counts = np.zeros(4)
    for record in result['subsamples']:
        indices = record['indices']
        assert len(indices) == len(set(indices))
        assert np.unique(y[indices], return_counts=True)[1].tolist() == [5, 5]
        counts[record['features']] += 1
    assert np.allclose(counts/2, result['frequency'])


def test_matched_subsamples_across_objectives():
    X = np.random.default_rng(71).uniform(.1, .9, (8, 3))
    y = np.repeat([0, 1], 4)
    results = [stability_ranking(X, y, repeats=2, mode=m) for m in ['robust', 'nominal', 'relief']]
    assert all([r['indices'] for r in result['subsamples']] ==
               [r['indices'] for r in results[0]['subsamples']] for result in results)
