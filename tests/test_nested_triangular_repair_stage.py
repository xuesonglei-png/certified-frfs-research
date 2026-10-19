import numpy as np
from certified_frfs.nested_triangular_repair_stage import fit_stage
from certified_frfs.triangular_classifier import triangular_box_margins


def test_repaired_stage_checks_actual_certificates_and_margins():
    rng = np.random.default_rng(12)
    X = rng.random((24, 4))
    y = (X[:, 0] > .5).astype(int)
    stage = fit_stage(X, y, C=1)
    from certified_frfs.validated_pool import preprocess_fit
    A, _ = preprocess_fit(X, X)
    original = triangular_box_margins(A, y, stage['teacher_coef'],
                                     stage['teacher_intercept'], stage['teacher_classes'], eps=.01)
    mask = original > 1e-10
    for record in stage['policies'].values():
        assert record['candidate']['certificate']['feasible']
        fit = record['fit']
        margin = triangular_box_margins(A, y, fit['coef'], fit['intercept'],
                                       stage['teacher_classes'], eps=.01)
        assert np.all(margin[mask] >= .5*original[mask] - 1e-8)
