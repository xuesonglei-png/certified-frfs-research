"""Training-only score-reconstruction repair with exact certificate checks.

This heuristic is not an optimal minimum-cardinality solver. Each repair
restores one raw attribute that minimizes balanced teacher-score error;
feasibility is independently recomputed, never inferred from set inclusion.
"""
import numpy as np

from .triangular_classifier import triangular_memberships
from .feasible_transport import weight_certificate
from .redistributed_loss import transfer_weights
from .capped_sparse import capped_sparse_projection
from .majority_metric import MajorityTerms


def repair_support(X, y, coef, features):
    X = np.asarray(X, float)
    y = np.asarray(y)
    coef = np.asarray(coef, float)
    n, d = X.shape
    q = coef.shape[0]
    if coef.shape != (q, d, 3) or len(y) != n:
        raise ValueError('incompatible teacher')
    active = sorted(set(features))
    if not active or len(active) != len(features) or min(active) < 0 or max(active) >= d:
        raise ValueError('invalid support')
    mass = np.sqrt(d)
    original = np.ones(d) / mass
    protected = MajorityTerms(X, y, weights=original, eps=.01).bounds(
        range(d), allowed_enemies=2) > 1e-10
    Z = triangular_memberships(X)
    target = np.einsum('ndk,qdk->nq', Z, coef - coef[[0]])
    sample_weight = np.array([1 / (q * sum(y == label)) for label in y])

    def error(fs):
        design = np.column_stack([np.ones(n), Z[:, fs].reshape(n, -1)])
        penalty = np.eye(design.shape[1]) * .001
        penalty[0, 0] = 0
        beta = np.linalg.solve(design.T @ (sample_weight[:, None] * design) + penalty,
                               design.T @ (sample_weight[:, None] * target))
        return float(sample_weight @ np.mean((design @ beta - target)**2, axis=1))

    trace = []
    while True:
        reduced = capped_sparse_projection(transfer_weights(X, active, original) / mass,
                                            len(active), 2 / d) * mass
        cert = weight_certificate(X, y, active, original, reduced, protected=protected)
        trace.append({'features': active.copy(), 'certificate': cert})
        if cert['feasible']:
            weights = np.zeros(d)
            weights[active] = reduced
            return {'features': active, 'weights': weights.tolist(), 'certificate': cert,
                    'repair_trace': trace, 'origin': 'score_reconstruction_certificate_repair'}
        if len(active) == d:
            raise RuntimeError('full metric unexpectedly failed its certificate')
        choices = [(error(sorted(active + [a])), a) for a in range(d) if a not in active]
        _, restored = min(choices)
        active = sorted(active + [restored])

