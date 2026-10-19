"""Fresh grouped constrained models and actual certificate-repaired policies."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .validated_pool import preprocess_fit
from .linear_retention import linear_parameters
from .triangular_classifier import triangular_memberships
from .joint_group_fuzzy import joint_group_scores
from .constrained_fuzzy_logistic import fit_constrained_logistic
from .certificate_repair import repair_support


def fit_stage(raw, y, *, C, class_balanced=False):
    A, _ = preprocess_fit(raw, raw)
    y = np.asarray(y)
    d = A.shape[1]
    if not np.isfinite(C) or C <= 0:
        raise ValueError('invalid C')
    teacher = LogisticRegression(C=C, max_iter=2000, random_state=0).fit(
        triangular_memberships(A).reshape(len(A), -1), y)
    W, b, classes = linear_parameters(teacher)
    W = W.reshape(len(classes), d, 3)
    full_candidate = repair_support(A, y, W, list(range(d)))
    full_fit = fit_constrained_logistic(A, y, W, b, classes, list(range(d)), C=C, class_balanced=class_balanced)
    full_failure = None
    if not full_fit['feasible']:
        full_failure = full_fit
        full_fit = {'feasible': True, 'coef': W.tolist(), 'intercept': b.tolist(),
                    'scope': 'explicit_full_teacher_fallback'}
    full_record = {'candidate': full_candidate, 'fit': full_fit,
                   'fallback': full_failure is not None, 'failure': full_failure}
    joint = joint_group_scores(A, y, W, b, classes, C=C, class_balanced=class_balanced)
    policies = {'1.0': full_record}
    cache = {tuple(range(d)): full_record}
    for fraction in (.5, .65, .8):
        key = str(fraction)
        if not joint['feasible']:
            policies[key] = {'candidate': full_candidate, 'fit': full_fit,
                             'fallback': True, 'failure': {'joint_fit': joint,
                                                          'status': 'joint_group_failed'}}
            continue
        initial = sorted(joint['ranking'][:int(np.ceil(fraction*d))])
        candidate = repair_support(A, y, W, initial)
        candidate['initial_features'] = initial
        identity = tuple(candidate['features'])
        if identity in cache:
            policies[key] = cache[identity]
            continue
        fit = fit_constrained_logistic(A, y, W, b, classes, candidate['features'], C=C, class_balanced=class_balanced)
        record = {'candidate': candidate, 'fit': fit, 'fallback': False, 'failure': None}
        if not fit['feasible']:
            record = {'candidate': full_candidate, 'fit': full_fit, 'fallback': True,
                      'failure': {'requested_candidate': candidate, 'requested_fit': fit}}
        cache[identity] = record
        policies[key] = record
    return {'teacher_C': float(C), 'teacher_coef': W.tolist(), 'teacher_intercept': b.tolist(),
            'teacher_classes': classes.tolist(), 'policies': policies, 'd': d, 'joint_fit': joint,
            'class_balanced': bool(class_balanced),
            'scope': 'independent_reference_only_joint_group_exact_constrained_policies'}
