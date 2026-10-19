"""Independent policy restoration, scalar margins, objectives and selection."""
import numpy as np
import audit_nested_triangular_repair as original
from audit_nested_balanced_joint import objective
from audit_triangular_classifier_diagnostic import manual_margins
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.conditional_certified_backward import conditional_backward_path

original.original.FOLDER = original.original.ROOT/'results/nested-conditional-feasible'
base_audit = original.audit_stage


def audit_stage(raw, y, query, truth, stage, scores, counts, *, seed=None):
    base_audit(raw, y, query, truth, stage, scores, counts, seed=seed)
    assert stage['support_rule'] == 'teacher_only_conditional_feasibility_repair'
    A, _ = preprocess_fit(raw, raw)
    path = conditional_backward_path(A, y, stage['teacher_coef'], checks=stage['d'])
    assert path == stage['conditional_path']
    classes = np.array(stage['teacher_classes'])
    teacher, _ = manual_margins(A, y, np.array(stage['teacher_coef']),
                                np.array(stage['teacher_intercept']), classes)
    mask = teacher > 1e-10
    seen = set()
    for fraction in (.5, .65, .8):
        record = stage['policies'][str(fraction)]
        eligible = [r for r in reversed(path['trace']) if len(r['features']) >= int(np.ceil(fraction*stage['d']))]
        attempts = record['feasibility_repair']
        for i, attempt in enumerate(attempts):
            assert attempt['features'] == eligible[i]['features']
            fit = attempt['fit']
            if 'coef' in fit:
                actual, _ = manual_margins(A, y, np.array(fit['coef']), np.array(fit['intercept']), classes)
                feasible = bool(np.all(actual[mask] >= .5*teacher[mask]-1e-8) and np.all(actual[mask] > 1e-10))
                assert feasible == fit['feasible']
                np.testing.assert_allclose(actual, fit['deployed_margins'], atol=1e-10)
                if tuple(attempt['features']) not in seen:
                    objective(A, y, stage, fit)
                    seen.add(tuple(attempt['features']))
                    counts['repair_objectives_checked'] = counts.get('repair_objectives_checked', 0)+1
            assert not fit['feasible'] or i == len(attempts)-1
        if attempts and attempts[-1]['fit']['feasible']:
            assert record['candidate']['features'] == attempts[-1]['features']
            assert record['fit'] == attempts[-1]['fit'] and not record['fallback']
        else:
            assert len(record['candidate']['features']) == stage['d']
            assert len(attempts) == len(eligible)-1
        counts['repair_attempts_checked'] = counts.get('repair_attempts_checked', 0)+len(attempts)
    full = stage['policies']['1.0']['fit']
    if 'objective' in full:
        objective(A, y, stage, full)
    counts['conditional_paths_checked'] = counts.get('conditional_paths_checked', 0)+1


original.original.audit_stage = audit_stage

if __name__ == '__main__':
    original.original.main()
