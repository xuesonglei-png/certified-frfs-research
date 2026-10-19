"""Reference-internal predictive reconstruction with fold-local preprocessing."""
import numpy as np
from .nested_teacher_regularized import inner_splits
from .validated_pool import preprocess_fit
from .triangular_classifier import triangular_memberships
from .majority_metric import MajorityTerms
from .redistributed_loss import transfer_weights
from .capped_sparse import capped_sparse_projection
from .feasible_transport import weight_certificate


class CrossfitLabelLoss:
    def __init__(self, raw, y):
        raw, y = np.asarray(raw, float), np.asarray(y)
        self.classes = np.unique(y)
        own = np.argmax(y[:, None] == self.classes, axis=1)
        target = np.eye(len(self.classes))[own]-1/len(self.classes)
        self.folds = []
        self.cache = {}
        for a, b in inner_splits(y, seed=20261011):
            A, B = preprocess_fit(raw[a], raw[b])
            design = np.column_stack([np.ones(len(a)), triangular_memberships(A).reshape(len(a), -1)])
            query = np.column_stack([np.ones(len(b)), triangular_memberships(B).reshape(len(b), -1)])
            counts = np.bincount(own[a], minlength=len(self.classes))
            w = 1/(len(self.classes)*counts[own[a]])
            query_counts = np.bincount(own[b], minlength=len(self.classes))
            query_w = 1/(len(np.unique(y[b]))*query_counts[own[b]])
            self.folds.append({'reference_indices': a.tolist(), 'validation_indices': b.tolist(),
                'gram': design.T@(w[:, None]*design), 'cross': design.T@(w[:, None]*target[a]),
                'query': query, 'target': target[b], 'weights': query_w})

    def __call__(self, features):
        key = tuple(features)
        if key not in self.cache:
            cols = [0]+[1+3*a+m for a in features for m in range(3)]
            P = np.eye(len(cols))*.001; P[0, 0] = 0
            values = []
            for fold in self.folds:
                beta = np.linalg.solve(fold['gram'][np.ix_(cols, cols)]+P, fold['cross'][cols])
                residual = fold['query'][:, cols]@beta-fold['target']
                values.append(float(fold['weights']@np.mean(residual**2, axis=1)))
            self.cache[key] = float(np.mean(values))
        return self.cache[key]


def crossfit_label_path(raw, y):
    A, _ = preprocess_fit(raw, raw)
    y = np.asarray(y); d = A.shape[1]; mass = np.sqrt(d)
    original = np.ones(d)/mass
    protected = MajorityTerms(A, y, weights=original).bounds(range(d), allowed_enemies=2) > 1e-10
    reconstruction = CrossfitLabelLoss(raw, y)
    active = list(range(d))
    cert = weight_certificate(A, y, active, original, original, protected=protected)
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
            weights = capped_sparse_projection(transfer_weights(A, fs, original)/mass, len(fs), 2/d)*mass
            cert = weight_certificate(A, y, fs, original, weights, protected=protected)
            attempts.append({'removed': int(removed), 'features': fs, 'weights': weights.tolist(),
                             'reconstruction_loss': value, 'certificate': cert, 'accepted': cert['feasible']})
            if cert['feasible']:
                embedded = np.zeros(d); embedded[fs] = weights
                accepted = {'features': fs, 'weights': embedded.tolist(), 'certificate': cert,
                            'reconstruction_loss': value}
                break
        if accepted is None:
            break
        trace.append(accepted); active = accepted['features']
    return {'trace': trace, 'attempts': attempts, 'rankings': rankings, 'checks': d,
            'crossfit_splits': [{'reference_indices': r['reference_indices'], 'validation_indices': r['validation_indices']} for r in reconstruction.folds],
            'scope': 'reference_internal_crossfit_selection_surrogate_exact_FRS_not_independent_risk_certificate'}
