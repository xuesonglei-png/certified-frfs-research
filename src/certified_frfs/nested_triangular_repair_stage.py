"""Independent certified repair stage, retaining the original score LP."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from .validated_pool import preprocess_fit
from .linear_retention import linear_parameters
from .fuzzy_teacher_support import fuzzy_score_supports
from .triangular_classifier import triangular_memberships
from .triangular_robust_refit import fit_triangular_robust
from .certificate_repair import repair_support


def fit_stage(raw, y, *, C):
    A, _ = preprocess_fit(np.asarray(raw, float), np.asarray(raw, float))
    y = np.asarray(y)
    d = A.shape[1]
    if not np.isfinite(C) or C <= 0:
        raise ValueError('invalid C')
    teacher = LogisticRegression(C=C, max_iter=2000, random_state=0).fit(
        triangular_memberships(A).reshape(len(A), -1), y)
    W, b, classes = linear_parameters(teacher)
    W = W.reshape(len(classes), d, 3)
    full = repair_support(A, y, W, list(range(d)))
    full_fit = fit_triangular_robust(A, y, W, b, classes, list(range(d)))
    full_failure = None
    if not full_fit['feasible']:
        full_failure = full_fit
        full_fit = {'feasible': True, 'coef': W.tolist(), 'intercept': b.tolist(),
                    'scope': 'explicit_full_teacher_fallback'}
    full_record = {'candidate': full, 'fit': full_fit,
                   'fallback': full_failure is not None, 'failure': full_failure}
    candidates = {len(c['features']): c for c in fuzzy_score_supports(A, y, W, b)}
    policies = {'1.0': full_record}
    cache = {tuple(range(d)): full_record}
    for fraction in (.5, .65, .8):
        k = int(np.ceil(fraction*d))
        if k == d:
            policies[str(fraction)] = full_record
            continue
        initial = candidates[k]
        candidate = repair_support(A, y, W, initial['features'])
        candidate['initial_features'] = initial['features']
        identity = tuple(candidate['features'])
        if identity in cache:
            policies[str(fraction)] = cache[identity]
            continue
        fitted = fit_triangular_robust(A, y, W, b, classes, candidate['features'])
        record = {'candidate': candidate, 'fit': fitted, 'fallback': False, 'failure': None}
        if not fitted['feasible']:
            record = {'candidate': full, 'fit': full_fit, 'fallback': True,
                      'failure': {'requested_candidate': candidate, 'requested_fit': fitted}}
        cache[identity] = record
        policies[str(fraction)] = record
    return {'teacher_C': float(C), 'teacher_coef': W.tolist(),
            'teacher_intercept': b.tolist(), 'teacher_classes': classes.tolist(),
            'policies': policies, 'd': d,
            'scope': 'independent_reference_only_certificate_repaired_triangular_stage'}

