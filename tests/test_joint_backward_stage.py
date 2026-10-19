import numpy as np
from certified_frfs.nested_joint_group_stage import fit_stage
from certified_frfs.joint_backward_stage import refit_backward_stage
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_box_margins


def test_backward_deployments_preserve_both_original_certificates():
    rng = np.random.default_rng(23)
    X = rng.random((24, 4))
    y = (X[:, 0] > .55).astype(int)
    source = fit_stage(X, y, C=1, class_balanced=True)
    stage = refit_backward_stage(X, y, source)
    assert stage['teacher_coef'] == source['teacher_coef']
    assert stage['policies']['1.0'] == source['policies']['1.0']
    assert 'support_rule' not in source
    A, _ = preprocess_fit(X, X)
    original = triangular_box_margins(A, y, stage['teacher_coef'], stage['teacher_intercept'], stage['teacher_classes'])
    mask = original > 1e-10
    for record in stage['policies'].values():
        assert record['candidate']['certificate']['feasible']
        fit = record['fit']
        actual = triangular_box_margins(A, y, fit['coef'], fit['intercept'], stage['teacher_classes'])
        assert np.all(actual[mask] >= .5*original[mask]-1e-8)
        assert np.all(actual[mask] > 1e-10)
