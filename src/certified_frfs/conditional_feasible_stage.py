"""Teacher-only conditional support with certified-path feasibility repair."""
import copy
import numpy as np
from .validated_pool import preprocess_fit
from .conditional_certified_backward import conditional_backward_path
from .constrained_fuzzy_logistic import fit_constrained_logistic


def refit_feasible_stage(raw, y, source):
    stage = copy.deepcopy(source)
    stage['support_rule'] = 'teacher_only_conditional_feasibility_repair'
    A, _ = preprocess_fit(raw, raw)
    d = stage['d']
    path = stage.get('conditional_path')
    if path is None:
        path = conditional_backward_path(A, y, stage['teacher_coef'], checks=d)
    stage['conditional_path'] = path
    full = source['policies']['1.0']
    cache = {tuple(range(d)): full['fit']}
    # Reuse only fits on this exact reference, support and C.
    for record in source['policies'].values():
        cache[tuple(record['candidate']['features'])] = record['fit']
        if record['fallback'] and isinstance(record['failure'], dict) and 'requested_fit' in record['failure']:
            cache[tuple(record['failure']['requested_candidate']['features'])] = record['failure']['requested_fit']
    policies = {'1.0': full}
    for fraction in (.5, .65, .8):
        eligible = [r for r in reversed(path['trace']) if len(r['features']) >= int(np.ceil(fraction*d))]
        attempts = []
        for selected in eligible:
            fs = selected['features']
            if len(fs) == d:
                record = copy.deepcopy(full)
                break
            identity = tuple(fs)
            if identity not in cache:
                cache[identity] = fit_constrained_logistic(A, y, stage['teacher_coef'], stage['teacher_intercept'],
                    stage['teacher_classes'], fs, C=stage['teacher_C'], class_balanced=True)
            fit = cache[identity]
            attempts.append({'features': fs, 'fit': fit})
            if fit['feasible']:
                candidate = {**selected, 'repair_trace': [{'features': fs, 'certificate': selected['certificate']}],
                             'origin': stage['support_rule']}
                record = {'candidate': candidate, 'fit': fit, 'fallback': False, 'failure': None}
                break
        if attempts and not attempts[-1]['fit']['feasible']:
            record['fallback'] = True
            record['failure'] = {'status': 'no_compact_path_fit_verified', 'attempts': attempts}
        record['feasibility_repair'] = attempts
        policies[str(fraction)] = record
    stage['policies'] = policies
    stage['scope'] = 'teacher_only_conditional_FRS_path_smallest_verified_path_fit_not_global_optimum'
    return stage
