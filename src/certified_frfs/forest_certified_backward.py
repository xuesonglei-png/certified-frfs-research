"""Interaction-aware deletion ranking with unchanged geometry/teacher guards.

Fixed-reference projected score loss ranks candidates; it is a surrogate, not
a generalization guarantee. Actual refits must pass both existing certificates.
"""
import numpy as np
from .forest_projection import project_forest_parameters
from .forest_fuzzy_classifier import forest_scores, forest_box_margins
from .triangular_classifier import triangular_box_margins
from .centered_forest_fuzzy_logistic_batched import fit_forest_logistic
from .majority_metric import MajorityTerms
from .redistributed_loss import transfer_weights
from .capped_sparse import capped_sparse_projection
from .feasible_transport import weight_certificate


class ForestProjectionLoss:
    def __init__(self, X, y, classes, full_fit):
        self.X = np.asarray(X, float); y, classes = np.asarray(y), np.asarray(classes)
        self.W = np.asarray(full_fit['coef']); self.bias = np.asarray(full_fit['intercept'])
        self.edges = full_fit.get('edges', [])
        self.V = np.asarray(full_fit.get('pair_coef', np.zeros((len(classes), 0, 3, 3))))
        self.V = self.V.reshape(len(classes), len(self.edges), 3, 3)
        self.weights = np.array([1/(len(classes)*np.sum(y == label)) for label in y])
        if not np.all(np.isin(y, classes)) or not all(np.any(y == c) for c in classes):
            raise ValueError('reference must contain every declared class')
        self.center = self.weights@self.X
        self.target = forest_scores(self.X, self.W, self.bias, self.edges, self.V)
        self.target -= self.target.mean(axis=1, keepdims=True)
        self.cache = {}

    def __call__(self, features):
        key = tuple(features)
        if key not in self.cache:
            projected = project_forest_parameters(self.W, self.bias, self.edges, self.V, features, self.center)
            scores = forest_scores(self.X, projected['coef'], projected['intercept'], projected['edges'], projected['pair_coef'])
            scores -= scores.mean(axis=1, keepdims=True)
            self.cache[key] = float(self.weights@np.mean((scores-self.target)**2, axis=1))
        return self.cache[key]


def forest_certified_backward_path(X, y, classes, full_fit, teacher_W, teacher_bias,
                                   *, pair_penalty, fraction=.8, target_size=None):
    X, y, classes = np.asarray(X, float), np.asarray(y), np.asarray(classes)
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('target fraction must be in (0,1]')
    d = X.shape[1]; mass = np.sqrt(d); original_weights = np.ones(d)/mass
    minimum = max(1, int(np.ceil(fraction*d)))
    if target_size is not None:
        if int(target_size) != target_size or not 1 <= target_size <= d:
            raise ValueError('target size must be an integer in [1,d]')
        minimum = int(target_size)
    loss = ForestProjectionLoss(X, y, classes, full_fit)
    teacher_margins = triangular_box_margins(X, y, teacher_W, teacher_bias, classes)
    full_margins = forest_box_margins(X, y, loss.W, loss.bias, classes, loss.edges, loss.V)
    teacher_protected = teacher_margins > 1e-10
    if np.any(full_margins[teacher_protected] < .5*teacher_margins[teacher_protected]-1e-8) or np.any(full_margins[teacher_protected] <= 1e-10):
        raise ValueError('full model violates original teacher guards')
    protected = MajorityTerms(X, y, weights=original_weights).bounds(range(d), allowed_enemies=2) > 1e-10
    active = list(range(d)); certificate = weight_certificate(X, y, active, original_weights, original_weights, protected=protected)
    if not certificate['feasible']: raise RuntimeError('full geometry certificate failed')
    trace = [{'features': active.copy(), 'weights': original_weights.tolist(),
              'certificate': certificate, 'fit': full_fit, 'projection_loss': loss(active)}]
    attempts = []; rankings = []
    while len(active) > minimum:
        ranked = [(loss([a for a in active if a != removed]), removed) for removed in active]
        ranked.sort(key=lambda r: (round(r[0], 12), r[1]))
        rankings.append({'features': active.copy(), 'choices': [{'removed': int(j), 'loss': cost} for cost, j in ranked]})
        accepted = None
        for cost, removed in ranked:
            fs = [a for a in active if a != removed]
            weights = capped_sparse_projection(transfer_weights(X, fs, original_weights)/mass, len(fs), 2/d)*mass
            certificate = weight_certificate(X, y, fs, original_weights, weights, protected=protected)
            fit = fit_forest_logistic(X, y, teacher_W, teacher_bias, classes, fs,
                loss.edges, C=1., pair_penalty=pair_penalty) if certificate['feasible'] else None
            valid = certificate['feasible'] and fit['feasible']
            attempts.append({'removed': int(removed), 'features': fs, 'weights': weights.tolist(),
                'projection_loss': cost, 'certificate': certificate, 'fit': fit, 'accepted': bool(valid)})
            if valid:
                embedded = np.zeros(d); embedded[fs] = weights
                accepted = {'features': fs, 'weights': embedded.tolist(), 'certificate': certificate,
                            'fit': fit, 'projection_loss': cost}
                break
        if accepted is None: break
        trace.append(accepted); active = accepted['features']
    return {'trace': trace, 'attempts': attempts, 'rankings': rankings,
            'center': loss.center.tolist(), 'original_teacher_protected_count': int(teacher_protected.sum()),
            'fraction': float(fraction), 'ranking_tie_decimals': 12,
            'target_size': minimum,
            'scope': 'interaction_aware_reference_projection_surrogate_joint_verified_FRS_and_teacher_guards_not_test_risk_or_global_reduct'}
