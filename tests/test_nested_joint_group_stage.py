import numpy as np
from certified_frfs.nested_joint_group_stage import fit_stage
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_box_margins


def test_joint_stage_verifies_actual_deployed_model_and_metric():
    rng = np.random.default_rng(17)
    X = rng.random((24, 4))
    y = (X[:, 0] > .5).astype(int)
    stage = fit_stage(X, y, C=1)
    A, _ = preprocess_fit(X, X)
    original = triangular_box_margins(A, y, stage['teacher_coef'],
                                     stage['teacher_intercept'], stage['teacher_classes'])
    protected = original > 1e-10
    for record in stage['policies'].values():
        assert record['candidate']['certificate']['feasible']
        fit = record['fit']
        margins = triangular_box_margins(A, y, fit['coef'], fit['intercept'], stage['teacher_classes'])
        assert np.all(margins[protected] >= .5*original[protected]-1e-8)
        assert np.all(margins[protected] > 1e-10)
