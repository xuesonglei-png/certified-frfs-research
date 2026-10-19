"""Fresh certified deletion paths and balanced constrained deployment refits."""
import copy
import numpy as np
from .validated_pool import preprocess_fit
from .joint_certified_backward import certified_backward_path
from .constrained_fuzzy_logistic import fit_constrained_logistic


def refit_backward_stage(raw, y, source):
    stage = copy.deepcopy(source)
    stage['support_rule'] = 'certificate_first_joint_backward'
    joint = stage['joint_fit']
    if not joint['feasible']:
        return stage
    A, _ = preprocess_fit(raw, raw)
    d = A.shape[1]
    path = certified_backward_path(A, y, joint['ranking'])
    stage['backward_path'] = path
    full = stage['policies']['1.0']
    policies, cache = {'1.0': full}, {tuple(range(d)): full}
    for fraction in (.5, .65, .8):
        eligible = [r for r in path['trace'] if len(r['features']) >= int(np.ceil(fraction*d))]
        selected = eligible[-1]
        fs = selected['features']
        identity = tuple(fs)
        if identity in cache:
            policies[str(fraction)] = cache[identity]
            continue
        candidate = {**selected, 'repair_trace': [{'features': fs, 'certificate': selected['certificate']}],
                     'origin': 'certificate_first_joint_backward'}
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
