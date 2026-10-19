"""Joint importance ordered deletions with unchanged metric certification.

Rejected deletions are skipped. No monotonicity, minimum-cardinality or
classification superiority claim follows from this greedy training path.
"""
import numpy as np
from .redistributed_loss import transfer_weights
from .capped_sparse import capped_sparse_projection
from .majority_metric import MajorityTerms
from .feasible_transport import weight_certificate


def certified_backward_path(X, y, ranking):
    X, y = np.asarray(X, float), np.asarray(y)
    d = X.shape[1]
    if sorted(ranking) != list(range(d)):
        raise ValueError('complete unique ranking required')
    mass = np.sqrt(d)
    original = np.ones(d)/mass
    protected = MajorityTerms(X, y, weights=original).bounds(range(d), allowed_enemies=2) > 1e-10
    active = list(range(d))
    full = weight_certificate(X, y, active, original, original, protected=protected)
    if not full['feasible']:
        raise RuntimeError('full metric failed')
    trace = [{'features': active.copy(), 'weights': original.tolist(), 'certificate': full}]
    attempts = []
    for removed in reversed(ranking):
        support = [a for a in active if a != removed]
        if len(support) < int(np.ceil(d/2)):
            break
        weights = capped_sparse_projection(transfer_weights(X, support, original)/mass,
                                            len(support), 2/d)*mass
        cert = weight_certificate(X, y, support, original, weights, protected=protected)
        attempts.append({'removed': int(removed), 'features': support, 'weights': weights.tolist(),
                         'certificate': cert, 'accepted': cert['feasible']})
        if cert['feasible']:
            active = support
            embedded = np.zeros(d)
            embedded[active] = weights
            trace.append({'features': active.copy(), 'weights': embedded.tolist(), 'certificate': cert})
    return {'trace': trace, 'attempts': attempts,
            'scope': 'training_only_exact_certified_greedy_path_not_global_reduct_or_predictive_guarantee'}
