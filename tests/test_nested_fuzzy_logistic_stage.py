import numpy as np
from certified_frfs.nested_fuzzy_logistic_stage import fit_stage
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_box_margins


def test_actual_screened_classifier_retains_both_certificates():
    rng = np.random.default_rng(104)
    X = rng.random((32, 4))
    y = (X[:, 0] > .5).astype(int)
    stage = fit_stage(X, y, C=1)
    A, _ = preprocess_fit(X, X)
    original = np.array(stage['policies']['1.0']['fit']['original_margins'])
    protected = original > 1e-10
    for record in stage['policies'].values():
        assert record['candidate']['certificate']['feasible']
        fit = record['fit']
        actual = triangular_box_margins(A, y, fit['coef'], fit['intercept'],
                                       stage['teacher_classes'], eps=.01)
        assert np.all(actual[protected] >= .5*original[protected]-1e-8)
        if record['fallback']:
            assert len(record['candidate']['features']) == 4
            assert record['failure']['requested_fit']['status'] == 'deployed_margin_rejected'
