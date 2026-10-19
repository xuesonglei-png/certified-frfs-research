"""Class-normalized teacher uncertainty weights for grouped reconstruction."""
import numpy as np
from scipy.special import softmax
from .triangular_classifier import triangular_memberships


def boundary_weights(scores, y):
    scores = np.asarray(scores, float)
    y = np.asarray(y)
    classes, own = np.unique(y, return_inverse=True)
    if scores.shape != (len(y), len(classes)) or not np.isfinite(scores).all():
        raise ValueError('score columns must follow sorted training classes')
    uncertainty = .05 + (1-softmax(scores, axis=1)[np.arange(len(y)), own])**2
    weights = np.zeros(len(y))
    for c in range(len(classes)):
        mask = own == c
        weights[mask] = uncertainty[mask]/(len(classes)*uncertainty[mask].sum())
    return weights


def boundary_score_supports(X, y, coef, intercept, fractions=(.5, .65, .8)):
    X, y = np.asarray(X, float), np.asarray(y)
    coef, intercept = np.asarray(coef, float), np.asarray(intercept, float)
    n, d = X.shape
    q = coef.shape[0]
    if coef.shape != (q, d, 3) or intercept.shape != (q,) or len(y) != n:
        raise ValueError('invalid fuzzy teacher')
    if not fractions or any(not 0 < f < 1 for f in fractions):
        raise ValueError('strictly compact fractions required')
    Z = triangular_memberships(X)
    scores = np.einsum('ndk,qdk->nq', Z, coef)+intercept
    weights = boundary_weights(scores, y)
    target = scores-scores[:, [0]]

    def error(fs):
        design = np.column_stack([np.ones(n), Z[:, fs].reshape(n, -1)])
        reg = np.eye(design.shape[1])*.001
        reg[0, 0] = 0
        beta = np.linalg.solve(design.T@(weights[:, None]*design)+reg,
                               design.T@(weights[:, None]*target))
        return float(weights@np.mean((design@beta-target)**2, axis=1))

    sizes = {int(np.ceil(d*f)) for f in fractions}
    active, rows = list(range(d)), []
    while len(active) > min(sizes):
        removed, value = min(((a, error([v for v in active if v != a])) for a in active),
                             key=lambda pair: (pair[1], pair[0]))
        active.remove(removed)
        if len(active) in sizes:
            rows.append({'features': active.copy(), 'reconstruction_error': value,
                         'sample_weights': weights.tolist(),
                         'origin': 'class_normalized_boundary_teacher_backward'})
    return rows
