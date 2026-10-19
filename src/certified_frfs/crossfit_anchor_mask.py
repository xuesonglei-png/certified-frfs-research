"""Reference-only out-of-fold robust teacher anchors; no population guarantee."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .nested_triangular_stage import inner_splits
from .validated_pool import preprocess_fit
from .triangular_classifier import triangular_memberships, triangular_box_margins
from .linear_retention import linear_parameters


def crossfit_anchor_mask(raw, y, *, C):
    raw, y = np.asarray(raw, float), np.asarray(y)
    if not np.isfinite(C) or C <= 0:
        raise ValueError('invalid C')
    margins = np.full(len(y), np.nan)
    visits = np.zeros(len(y), int)
    folds = []
    for a, b in inner_splits(y):
        A, B = preprocess_fit(raw[a], raw[b])
        model = LogisticRegression(C=C, max_iter=2000, random_state=0).fit(
            triangular_memberships(A).reshape(len(A), -1), y[a])
        W, bias, classes = linear_parameters(model)
        W = W.reshape(len(classes), raw.shape[1], 3)
        values = triangular_box_margins(B, y[b], W, bias, classes, eps=.01)
        margins[b] = values
        visits[b] += 1
        folds.append({'reference_indices': a.tolist(), 'validation_indices': b.tolist(),
                      'coef': W.tolist(), 'intercept': bias.tolist(), 'classes': classes.tolist(),
                      'validation_margins': values.tolist()})
    assert np.all(visits <= 1)
    return {'mask': ((visits == 1) & (margins > 1e-10)).tolist(),
            'margins': [float(v) if np.isfinite(v) else None for v in margins],
            'folds': folds, 'unverified_count': int((visits == 0).sum()),
            'scope': 'crossfitted_reference_anchors_only_not_all_positive_teacher_anchors'}
