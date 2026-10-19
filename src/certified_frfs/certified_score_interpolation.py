"""Exact feasible segment toward an unconstrained fuzzy score model.

For each protected object, rival and raw-box vertex, the half-margin
constraint is linear along the segment. A cutting-plane loop uses the exact
separable minimizer as a separation oracle. Every cut bounds the step from
above; termination returns an independently verified feasible step.
"""
import numpy as np
from .triangular_classifier import triangular_memberships, triangular_box_margins


def certified_step(X, y, anchor_W, anchor_b, target_W, target_b, classes,
                   teacher_margins, *, eps=.01, ratio=.5):
    X = np.asarray(X, float)
    y = np.asarray(y)
    classes = np.asarray(classes)
    W, b = np.asarray(anchor_W, float), np.asarray(anchor_b, float)
    delta_W = np.asarray(target_W, float) - W
    delta_b = np.asarray(target_b, float) - b
    original = np.asarray(teacher_margins, float)
    protected = original > 1e-10
    if original.shape != y.shape or not 0 < ratio <= 1:
        raise ValueError('invalid teacher margins or ratio')
    anchor_margin = triangular_box_margins(X, y, W, b, classes, eps=eps)
    required = np.maximum(ratio*original, 1e-7)
    if np.any(anchor_margin[protected] < required[protected]-1e-8):
        raise ValueError('anchor is not feasible')
    lo, hi = np.maximum(0, X-eps), np.minimum(1, X+eps)
    points = np.stack([lo, hi, np.clip(.5, lo, hi)], axis=-1)
    members = triangular_memberships(points)
    own = np.argmax(y[:, None] == classes[None, :], axis=1)
    step = 1.
    cuts = 0
    # At each violating point the exact worst raw-box witness supplies a cut.
    for _ in range(1000):
        current_W, current_b = W+step*delta_W, b+step*delta_b
        bounds = []
        for i in np.flatnonzero(protected):
            c = own[i]
            for r in range(len(classes)):
                if r == c:
                    continue
                contrast = current_W[c]-current_W[r]
                values = np.einsum('dk,dpk->dp', contrast, members[i])
                index = values.argmin(axis=1)
                witness = members[i, np.arange(X.shape[1]), index]
                intercept = b[c]-b[r]+np.sum((W[c]-W[r])*witness)
                slope = delta_b[c]-delta_b[r]+np.sum((delta_W[c]-delta_W[r])*witness)
                if intercept+step*slope < required[i]-1e-9:
                    if slope >= 0:
                        raise RuntimeError('invalid separating cut from feasible anchor')
                    bounds.append(max(0., (intercept-required[i])/(-slope)))
        if not bounds:
            verified = triangular_box_margins(X, y, current_W, current_b, classes, eps=eps)
            assert np.all(verified[protected] >= required[protected]-1e-8)
            return {'step': step, 'coef': current_W.tolist(), 'intercept': current_b.tolist(),
                    'deployed_margins': verified.tolist(), 'cuts': cuts,
                    'scope': 'exact_segment_feasibility_not_unconstrained_optimal_classifier'}
        next_step = min(bounds)
        cuts += len(bounds)
        if next_step >= step:
            raise RuntimeError('separation did not decrease step')
        step = next_step
    raise RuntimeError('separation iteration limit')
