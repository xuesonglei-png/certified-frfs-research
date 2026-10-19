"""Reference-internal balanced classification error rather than score MSE."""
import numpy as np
from .crossfit_label_backward import CrossfitLabelLoss
from .validated_pool import preprocess_fit
from .majority_metric import MajorityTerms
from .redistributed_loss import transfer_weights
from .capped_sparse import capped_sparse_projection
from .feasible_transport import weight_certificate


class CrossfitClassificationLoss(CrossfitLabelLoss):
    def __init__(self, raw, y):
        super().__init__(raw, y)
        self.details = {}

    def __call__(self, features):
        key = tuple(features)
        if key not in self.cache:
            cols = [0]+[1+3*a+m for a in features for m in range(3)]
            P = np.eye(len(cols))*.001; P[0, 0] = 0
            errors, mse = [], []
            for fold in self.folds:
                beta = np.linalg.solve(fold['gram'][np.ix_(cols, cols)]+P, fold['cross'][cols])
                scores = fold['query'][:, cols]@beta
                errors.append(float(fold['weights']@(scores.argmax(axis=1) != fold['target'].argmax(axis=1))))
                mse.append(float(fold['weights']@np.mean((scores-fold['target'])**2, axis=1)))
            self.details[key] = {'balanced_error': float(np.mean(errors)), 'MSE': float(np.mean(mse))}
            self.cache[key] = (self.details[key]['balanced_error'], self.details[key]['MSE'])
        return self.cache[key]


def crossfit_classification_path(raw, y):
    A, _ = preprocess_fit(raw, raw)
    y = np.asarray(y); d = A.shape[1]; mass = np.sqrt(d)
    original = np.ones(d)/mass
    protected = MajorityTerms(A, y, weights=original).bounds(range(d), allowed_enemies=2) > 1e-10
    loss = CrossfitClassificationLoss(raw, y)
    active = list(range(d))
    cert = weight_certificate(A, y, active, original, original, protected=protected)
    if not cert['feasible']:
        raise RuntimeError('full metric failed original certificate')
    error, mse = loss(active)
    trace = [{'features': active.copy(), 'weights': original.tolist(), 'certificate': cert,
              'crossfit_balanced_error': error, 'reconstruction_loss': mse}]
    attempts, rankings = [], []
    while len(active) > int(np.ceil(d/2)):
        candidates = sorted((*loss([a for a in active if a != removed]), removed) for removed in active)
        rankings.append({'features': active.copy(), 'choices': [{'removed': int(a), 'balanced_error': e, 'MSE': m} for e, m, a in candidates]})
        accepted = None
        for error, mse, removed in candidates:
            fs = [a for a in active if a != removed]
            weights = capped_sparse_projection(transfer_weights(A, fs, original)/mass, len(fs), 2/d)*mass
            cert = weight_certificate(A, y, fs, original, weights, protected=protected)
            attempts.append({'removed': int(removed), 'features': fs, 'weights': weights.tolist(),
                             'crossfit_balanced_error': error, 'reconstruction_loss': mse,
                             'certificate': cert, 'accepted': cert['feasible']})
            if cert['feasible']:
                embedded = np.zeros(d); embedded[fs] = weights
                accepted = {'features': fs, 'weights': embedded.tolist(), 'certificate': cert,
                            'crossfit_balanced_error': error, 'reconstruction_loss': mse}
                break
        if accepted is None:
            break
        trace.append(accepted); active = accepted['features']
    return {'trace': trace, 'attempts': attempts, 'rankings': rankings, 'checks': d,
            'crossfit_splits': [{'reference_indices': r['reference_indices'], 'validation_indices': r['validation_indices']} for r in loss.folds],
            'scope': 'crossfit_selected_balanced_error_exact_FRS_not_independent_prediction_or_logistic_risk_certificate'}
