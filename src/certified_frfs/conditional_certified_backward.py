"""Re-estimate conditional score reconstruction before certified deletions.

Uses reference-only teacher scores, not held-out labels. The greedy shortlist
does not establish a globally minimal reduct or test prediction advantage.
"""
import numpy as np
from .triangular_classifier import triangular_memberships
from .majority_metric import MajorityTerms
from .redistributed_loss import transfer_weights
from .capped_sparse import capped_sparse_projection
from .feasible_transport import weight_certificate


class ReconstructionLoss:
    def __init__(self, X, y, coef):
        self.Z = triangular_memberships(X).reshape(len(X), -1)
        self.design = np.column_stack([np.ones(len(X)), self.Z])
        coef = np.asarray(coef, float)
        if coef.ndim != 3 or coef.shape[1:] != (X.shape[1], 3):
            raise ValueError('invalid teacher dimensions')
        centered = coef-coef.mean(axis=0, keepdims=True)
        self.target = self.Z@centered.reshape(len(coef), -1).T
        classes, counts = np.unique(y, return_counts=True)
        self.weights = np.array([1/(len(classes)*counts[np.flatnonzero(classes == label)[0]]) for label in y])
        self.gram = self.design.T@(self.weights[:, None]*self.design)
        self.cross = self.design.T@(self.weights[:, None]*self.target)
        self.cache = {}

    def __call__(self, features):
        identity = tuple(features)
        if identity not in self.cache:
            columns = [0]+[1+3*a+m for a in features for m in range(3)]
            penalty = np.eye(len(columns))*.001
            penalty[0, 0] = 0
            beta = np.linalg.solve(self.gram[np.ix_(columns, columns)]+penalty, self.cross[columns])
            residual = self.design[:, columns]@beta-self.target
            self.cache[identity] = float(self.weights@np.mean(residual**2, axis=1))
        return self.cache[identity]


def conditional_backward_path(X, y, coef, *, checks=3):
    X, y = np.asarray(X, float), np.asarray(y)
    if int(checks) != checks or checks < 1:
        raise ValueError('positive integer shortlist required')
    d = X.shape[1]
    mass = np.sqrt(d)
    original = np.ones(d)/mass
    protected = MajorityTerms(X, y, weights=original).bounds(range(d), allowed_enemies=2) > 1e-10
    reconstruction = ReconstructionLoss(X, y, coef)
    active = list(range(d))
    cert = weight_certificate(X, y, active, original, original, protected=protected)
    if not cert['feasible']:
        raise RuntimeError('full metric failed original certificate')
    trace = [{'features': active.copy(), 'weights': original.tolist(), 'certificate': cert,
              'reconstruction_loss': reconstruction(active)}]
    attempts, rankings = [], []
    while len(active) > int(np.ceil(d/2)):
        candidates = [(reconstruction([a for a in active if a != removed]), removed) for removed in active]
        candidates.sort()
        rankings.append({'features': active.copy(), 'choices': [{'removed': int(a), 'loss': float(v)} for v, a in candidates]})
        accepted = None
        for value, removed in candidates[:checks]:
            fs = [a for a in active if a != removed]
            weights = capped_sparse_projection(transfer_weights(X, fs, original)/mass, len(fs), 2/d)*mass
            cert = weight_certificate(X, y, fs, original, weights, protected=protected)
            attempts.append({'removed': int(removed), 'features': fs, 'weights': weights.tolist(),
                             'reconstruction_loss': value, 'certificate': cert, 'accepted': cert['feasible']})
            if cert['feasible']:
                embedded = np.zeros(d)
                embedded[fs] = weights
                accepted = {'features': fs, 'weights': embedded.tolist(), 'certificate': cert,
                            'reconstruction_loss': value}
                break
        if accepted is None:
            break
        trace.append(accepted)
        active = accepted['features']
    return {'trace': trace, 'attempts': attempts, 'rankings': rankings, 'checks': checks,
            'scope': 'adaptive_reference_score_reconstruction_shortlist_exact_FRS_not_global_reduct_or_test_risk'}
