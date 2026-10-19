"""Conditional teacher-score path and fresh constrained balanced deployment."""
import copy
import numpy as np
from .validated_pool import preprocess_fit
from .conditional_certified_backward import conditional_backward_path
from .constrained_fuzzy_logistic import fit_constrained_logistic


def refit_conditional_stage(raw, y, source):
    stage = copy.deepcopy(source)
    stage['support_rule'] = 'conditional_teacher_score_certified_backward'
    if not stage['joint_fit']['feasible']:
        return stage
    A, _ = preprocess_fit(raw, raw)
    d = A.shape[1]
    path = conditional_backward_path(A, y, stage['teacher_coef'], checks=d)
    stage['conditional_path'] = path
    full = stage['policies']['1.0']
    policies, cache = {'1.0': full}, {tuple(range(d)): full}
    for fraction in (.5, .65, .8):
        selected = [r for r in path['trace'] if len(r['features']) >= int(np.ceil(fraction*d))][-1]
        fs = selected['features']
        identity = tuple(fs)
        if identity in cache:
            policies[str(fraction)] = cache[identity]
            continue
        candidate = {**selected, 'repair_trace': [{'features': fs, 'certificate': selected['certificate']}],
                     'origin': 'conditional_teacher_score_certified_backward'}
        fit = fit_constrained_logistic(A, y, stage['teacher_coef'], stage['teacher_intercept'],
            stage['teacher_classes'], fs, C=stage['teacher_C'], class_balanced=True)
        record = {'candidate': candidate, 'fit': fit, 'fallback': False, 'failure': None}
        if not fit['feasible']:
            record = {'candidate': full['candidate'], 'fit': full['fit'], 'fallback': True,
                      'failure': {'requested_candidate': candidate, 'requested_fit': fit}}
        cache[identity] = record
        policies[str(fraction)] = record
    stage['policies'] = policies
    return stage
