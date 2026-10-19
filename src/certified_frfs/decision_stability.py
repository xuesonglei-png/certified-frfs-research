"""Training-only stratified subsample aggregation; certificates are recomputed."""
import numpy as np
from .baselines import relieff_scores
from .decision_search import select_local_decisions


def stability_ranking(X, y, *, seed=101, repeats=4, sample_fraction=.75,
                      mode='robust'):
    X, y = np.asarray(X, float), np.asarray(y)
    if mode not in ['robust', 'nominal', 'relief'] or repeats < 1 or not 0 < sample_fraction <= 1:
        raise ValueError('invalid stability settings')
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    frequency, rank_sum = np.zeros(d), np.zeros(d)
    records = []
    for repeat in range(repeats):
        indices = []
        for label in np.unique(y):
            available = np.flatnonzero(y == label)
            count = min(len(available), max(2, int(np.ceil(sample_fraction*len(available)))))
            indices.extend(rng.choice(available, count, replace=False).tolist())
        indices = np.array(sorted(indices))
        scores = relieff_scores(X[indices], y[indices], random_state=seed+repeat)
        ranking = np.argsort(-scores, kind='stable').tolist()
        for position, attribute in enumerate(ranking):
            rank_sum[attribute] += (d-position)/d
        if mode == 'relief':
            features = ranking[:max(1, int(np.ceil(.5*d)))]
        else:
            result = select_local_decisions(X[indices], y[indices], ranking,
                eps=.01 if mode == 'robust' else 0., margin_scale=.05, budget=24)
            features = result['features']
        frequency[features] += 1
        records.append({'indices': indices.tolist(), 'features': features})
    frequency /= repeats
    rank_sum /= repeats
    ranking = np.lexsort((np.arange(d), -rank_sum, -frequency)).tolist()
    return {'ranking': ranking, 'frequency': frequency.tolist(),
            'rank_score': rank_sum.tolist(), 'subsamples': records,
            'scope': 'selection_stability_not_population_or_aggregation_certificate'}
