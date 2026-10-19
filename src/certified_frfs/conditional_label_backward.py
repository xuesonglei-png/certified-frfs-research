"""Conditional class-balanced label reconstruction under unchanged FRS checks."""
import numpy as np
from .conditional_certified_backward import ReconstructionLoss
from .triangular_classifier import triangular_memberships
from .majority_metric import MajorityTerms
from .redistributed_loss import transfer_weights
from .capped_sparse import capped_sparse_projection
from .feasible_transport import weight_certificate


class LabelReconstructionLoss(ReconstructionLoss):
    def __init__(self, X, y):
        self.Z = triangular_memberships(X).reshape(len(X), -1)
        self.design = np.column_stack([np.ones(len(X)), self.Z])
        classes, own, counts = np.unique(y, return_inverse=True, return_counts=True)
        self.target = np.eye(len(classes))[own]-1/len(classes)
        self.weights = 1/(len(classes)*counts[own])
        self.gram = self.design.T@(self.weights[:, None]*self.design)
        self.cross = self.design.T@(self.weights[:, None]*self.target)
        self.cache = {}


def conditional_label_path(X, y):
    X, y = np.asarray(X, float), np.asarray(y)
    d = X.shape[1]
    mass = np.sqrt(d)
    original = np.ones(d)/mass
    protected = MajorityTerms(X, y, weights=original).bounds(range(d), allowed_enemies=2) > 1e-10
    reconstruction = LabelReconstructionLoss(X, y)
    active = list(range(d))
    cert = weight_certificate(X, y, active, original, original, protected=protected)
    if not cert['feasible']:
        raise RuntimeError('full metric failed original certificate')
    trace = [{'features': active.copy(), 'weights': original.tolist(), 'certificate': cert,
              'reconstruction_loss': reconstruction(active)}]
    attempts, rankings = [], []
    while len(active) > int(np.ceil(d/2)):
        candidates = sorted((reconstruction([a for a in active if a != removed]), removed) for removed in active)
        rankings.append({'features': active.copy(), 'choices': [{'removed': int(a), 'loss': float(v)} for v, a in candidates]})
        accepted = None
        for value, removed in candidates:
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
    return {'trace': trace, 'attempts': attempts, 'rankings': rankings, 'checks': d,
            'scope': 'balanced_label_reconstruction_exact_FRS_not_global_reduct_or_test_risk'}
