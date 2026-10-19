"""Certificate-checked deletion from a capped trained metric."""
import numpy as np
from .capped_sparse import capped_sparse_projection
from .query_probability import QueryProbabilityObjective
from .validated_pool import preprocess_fit
from .majority_metric import MajorityTerms
from .feasible_transport import weight_certificate


def certified_cap_backward(X, y, previous, *, tolerance=.01, steps=8, checks=4):
    """Fixed-budget sparse projection edits ranked by training objective.

    Preserve the initial best surrogate tolerance throughout the path. Each
    accepted deletion has a fresh certificate. The shortlist and finite budget
    do not establish reduct minimality or predictive accuracy guarantees.
    """
    if tolerance < 0 or steps < 1 or checks < 1:
        raise ValueError('invalid path budgets')
    X, y = np.asarray(X, float), np.asarray(y); d = X.shape[1]
    A, _ = preprocess_fit(X, X); full = np.ones(d)/np.sqrt(d); mass = full.sum()
    objective = QueryProbabilityObjective(X, y)
    if objective.splits != previous['training_splits']:
        raise ValueError('training split mismatch')
    protected = MajorityTerms(A, y, weights=full).bounds(range(d), allowed_enemies=2) > 1e-10
    threshold = previous['selected']['objective']+tolerance
    current = np.array(previous['compact_selected']['weights'])/mass
    trace = []; evaluated = []
    for iteration in range(steps+1):
        v = current*mass; fs = np.flatnonzero(v > 0).tolist()
        cert = weight_certificate(A, y, fs, full, v[fs], protected=protected)
        score = objective(current)[0]
        if not cert['feasible'] or score > threshold+1e-12:
            raise RuntimeError('accepted path violated its fixed conditions')
        trace.append({'iteration': iteration, 'features': fs, 'weights': v.tolist(),
            'objective': score, 'certificate': cert})
        if iteration == steps or (len(fs)-1)*2/d < 1-1e-14:
            break
        proposals = []
        for removed in fs:
            ids = np.array([a for a in fs if a != removed]); w = np.zeros(d)
            w[ids] = capped_sparse_projection(current[ids], len(ids), 2/d)
            value = objective(w)[0]
            if value <= threshold:
                proposals.append((value, removed, w))
        proposals.sort(key=lambda row:(row[0], row[1])); accepted = None
        for value, removed, w in proposals[:checks]:
            candidate_v = w*mass; support = np.flatnonzero(w > 0).tolist()
            candidate_cert = weight_certificate(A, y, support, full, candidate_v[support], protected=protected)
            evaluated.append({'iteration': iteration, 'removed': removed, 'features': support,
                'weights': candidate_v.tolist(), 'objective': value, 'certificate': candidate_cert})
            if candidate_cert['feasible']:
                accepted = w
                break
        if accepted is None:
            break
        current = accepted
    return {'selected': trace[-1], 'trace': trace, 'evaluated': evaluated,
        'training_splits': objective.splits, 'threshold': threshold, 'tolerance': tolerance,
        'steps': steps, 'checks': checks,
        'scope': 'budgeted_certified_training_deletion_not_minimality_or_test_risk'}
