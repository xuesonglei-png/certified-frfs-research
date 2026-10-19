"""Aligned balanced teacher, grouped objective and certificate-first path.

The classifier certificate is relative to this fresh balanced teacher. It is
not a claim that the ordinary teacher's positive anchors remain protected.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .validated_pool import preprocess_fit
from .linear_retention import linear_parameters
from .triangular_classifier import triangular_memberships
from .joint_group_fuzzy import joint_group_scores
from .constrained_fuzzy_logistic import fit_constrained_logistic
from .certificate_repair import repair_support
from .joint_certified_backward import certified_backward_path


def fit_stage(raw, y, *, C):
    if not np.isfinite(C) or C <= 0:
        raise ValueError('invalid C')
    A, _ = preprocess_fit(raw, raw)
    y = np.asarray(y)
    d = A.shape[1]
    teacher = LogisticRegression(C=C, max_iter=2000, random_state=0,
                                 class_weight='balanced').fit(
        triangular_memberships(A).reshape(len(A), -1), y)
    W, b, classes = linear_parameters(teacher)
    W = W.reshape(len(classes), d, 3)
    full_candidate = repair_support(A, y, W, list(range(d)))
    fitted = fit_constrained_logistic(A, y, W, b, classes, list(range(d)),
                                     C=C, class_balanced=True)
    full_failure = None if fitted['feasible'] else fitted
    if not fitted['feasible']:
        fitted = {'feasible': True, 'coef': W.tolist(), 'intercept': b.tolist(),
                  'scope': 'explicit_full_balanced_teacher_fallback'}
    full = {'candidate': full_candidate, 'fit': fitted,
            'fallback': full_failure is not None, 'failure': full_failure}
    joint = joint_group_scores(A, y, W, b, classes, C=C, class_balanced=True)
    policies, cache = {'1.0': full}, {tuple(range(d)): full}
    path = certified_backward_path(A, y, joint['ranking']) if joint['feasible'] else None
    for fraction in (.5, .65, .8):
        if path is None:
            policies[str(fraction)] = {'candidate': full_candidate, 'fit': fitted,
                'fallback': True, 'failure': {'joint_fit': joint, 'status': 'joint_group_failed'}}
            continue
        chosen = [r for r in path['trace'] if len(r['features']) >= int(np.ceil(fraction*d))][-1]
        fs = chosen['features']
        identity = tuple(fs)
        if identity in cache:
            policies[str(fraction)] = cache[identity]
            continue
        candidate = {**chosen, 'repair_trace': [{'features': fs, 'certificate': chosen['certificate']}],
                     'origin': 'certificate_first_joint_backward'}
        fit = fit_constrained_logistic(A, y, W, b, classes, fs, C=C, class_balanced=True)
        record = {'candidate': candidate, 'fit': fit, 'fallback': False, 'failure': None}
        if not fit['feasible']:
            record = {'candidate': full_candidate, 'fit': fitted, 'fallback': True,
                      'failure': {'requested_candidate': candidate, 'requested_fit': fit}}
        cache[identity] = record
        policies[str(fraction)] = record
    stage = {'teacher_C': float(C), 'teacher_coef': W.tolist(), 'teacher_intercept': b.tolist(),
             'teacher_classes': classes.tolist(), 'teacher_class_balanced': True,
             'class_balanced': True, 'd': d, 'joint_fit': joint, 'policies': policies,
             'support_rule': 'certificate_first_joint_backward',
             'scope': 'fresh_balanced_teacher_ALLpositive_halfmargin_not_ordinary_teacher_anchors'}
    if path is not None:
        stage['backward_path'] = path
    return stage
