"""Reference-only certified supports evaluated by the common fuzzy classifier."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from .validated_pool import preprocess_fit
from .linear_retention import linear_parameters
from .fuzzy_teacher_support import fuzzy_score_supports
from .triangular_classifier import triangular_memberships, triangular_box_margins
from .certificate_repair import repair_support


def fit_stage(raw, y, *, C):
    A, _ = preprocess_fit(raw, raw)
    y = np.asarray(y)
    d = A.shape[1]
    if not np.isfinite(C) or C <= 0:
        raise ValueError('invalid C')

    def train(fs):
        model = LogisticRegression(C=C, max_iter=2000, random_state=0).fit(
            triangular_memberships(A[:, fs]).reshape(len(A), -1), y)
        local, intercept, classes = linear_parameters(model)
        coef = np.zeros((len(classes), d, 3))
        coef[:, fs] = local.reshape(len(classes), len(fs), 3)
        return coef, intercept, classes

    W, b, classes = train(list(range(d)))
    original = triangular_box_margins(A, y, W, b, classes, eps=.01)
    protected = original > 1e-10
    full = repair_support(A, y, W, list(range(d)))
    full_fit = {'feasible': True, 'coef': W.tolist(), 'intercept': b.tolist(),
                'original_margins': original.tolist(), 'deployed_margins': original.tolist(),
                'scope': 'common_fuzzy_logistic_full'}
    full_record = {'candidate': full, 'fit': full_fit, 'fallback': False, 'failure': None}
    policies = {'1.0': full_record}
    cache = {tuple(range(d)): full_record}
    candidates = {len(c['features']): c for c in fuzzy_score_supports(A, y, W, b)}
    for fraction in (.5, .65, .8):
        size = int(np.ceil(fraction*d))
        if size == d:
            policies[str(fraction)] = full_record
            continue
        candidate = repair_support(A, y, W, candidates[size]['features'])
        candidate['initial_features'] = candidates[size]['features']
        identity = tuple(candidate['features'])
        if identity in cache:
            policies[str(fraction)] = cache[identity]
            continue
        coef, intercept, fitted_classes = train(candidate['features'])
        np.testing.assert_array_equal(classes, fitted_classes)
        margins = triangular_box_margins(A, y, coef, intercept, classes, eps=.01)
        feasible = bool(np.all(margins[protected] >= .5*original[protected]-1e-8)
                        and np.all(margins[protected] > 1e-10))
        fit = {'feasible': feasible, 'coef': coef.tolist(), 'intercept': intercept.tolist(),
               'original_margins': original.tolist(), 'deployed_margins': margins.tolist(),
               'status': 'passed' if feasible else 'deployed_margin_rejected',
               'scope': 'common_fuzzy_logistic_screened_not_constrained_optimizer'}
        record = {'candidate': candidate, 'fit': fit, 'fallback': False, 'failure': None}
        if not feasible:
            record = {'candidate': full, 'fit': full_fit, 'fallback': True,
                      'failure': {'requested_candidate': candidate, 'requested_fit': fit}}
        cache[identity] = record
        policies[str(fraction)] = record
    return {'teacher_C': float(C), 'teacher_coef': W.tolist(), 'teacher_intercept': b.tolist(),
            'teacher_classes': classes.tolist(), 'policies': policies, 'd': d,
            'scope': 'independent_reference_only_common_classifier_metric_and_margin_screened'}
