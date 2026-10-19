import numpy as np
from certified_frfs.nested_joint_group_stage import fit_stage
from certified_frfs.conditional_backward_stage import refit_conditional_stage
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_box_margins


def test_conditional_stage_preserves_original_teacher_AND_FRS():
    rng = np.random.default_rng(107)
    X = rng.random((24, 4)); y = (X[:, 0] > .65).astype(int)
    source = fit_stage(X, y, C=1, class_balanced=True)
    stage = refit_conditional_stage(X, y, source)
    assert stage['teacher_coef'] == source['teacher_coef']
    assert stage['policies']['1.0'] == source['policies']['1.0']
    assert 'support_rule' not in source
    A, _ = preprocess_fit(X, X)
    original = triangular_box_margins(A, y, stage['teacher_coef'], stage['teacher_intercept'], stage['teacher_classes'])
    mask = original > 1e-10
    for r in stage['policies'].values():
        assert r['candidate']['certificate']['feasible']
        fit = r['fit']
        actual = triangular_box_margins(A, y, fit['coef'], fit['intercept'], stage['teacher_classes'])
        assert np.all(actual[mask] >= .5*original[mask]-1e-8) and np.all(actual[mask] > 1e-10)
