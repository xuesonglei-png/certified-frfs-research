"""Align repair's score reconstruction with the balanced grouped predictor.

The supplied stage is a cache of identically fitted reference-only teacher,
full classifier and group model. Certificates still use the original uniform
metric and original teacher margins; only the restoration priority changes.
"""
import copy
import numpy as np
from .validated_pool import preprocess_fit
from .certificate_repair import repair_support
from .constrained_fuzzy_logistic import fit_constrained_logistic


def refit_aligned_stage(raw, y, source):
    stage = copy.deepcopy(source)
    assert stage['class_balanced']
    stage['repair_target'] = 'balanced_joint_scores'
    joint = stage['joint_fit']
    if not joint['feasible']:
        return stage
    A, _ = preprocess_fit(raw, raw)
    d = A.shape[1]
    full = stage['policies']['1.0']
    cache = {tuple(range(d)): full}
    policies = {'1.0': full}
    for fraction in (.5, .65, .8):
        initial = sorted(joint['ranking'][:int(np.ceil(fraction*d))])
        candidate = repair_support(A, y, joint['coef'], initial)
        candidate['initial_features'] = initial
        candidate['repair_target'] = 'balanced_joint_scores'
        identity = tuple(candidate['features'])
        if identity in cache:
            policies[str(fraction)] = cache[identity]
            continue
        fit = fit_constrained_logistic(A, y, stage['teacher_coef'],
            stage['teacher_intercept'], stage['teacher_classes'], candidate['features'],
            C=stage['teacher_C'], class_balanced=True)
        record = {'candidate': candidate, 'fit': fit, 'fallback': False, 'failure': None}
        if not fit['feasible']:
            record = {'candidate': full['candidate'], 'fit': full['fit'], 'fallback': True,
                      'failure': {'requested_candidate': candidate, 'requested_fit': fit}}
        cache[identity] = record
        policies[str(fraction)] = record
    stage['policies'] = policies
    return stage
