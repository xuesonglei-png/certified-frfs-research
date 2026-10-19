import numpy as np
from sklearn.linear_model import LogisticRegression
from certified_frfs.balanced_teacher_backward_stage import fit_stage
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.linear_retention import linear_parameters
from certified_frfs.triangular_classifier import triangular_memberships, triangular_box_margins


def test_balanced_teacher_refit_and_ALLpositive_halfmargin_preservation():
    rng = np.random.default_rng(84)
    X = rng.random((30, 4))
    y = (X[:, 0] > .75).astype(int)
    stage = fit_stage(X, y, C=1)
    A, _ = preprocess_fit(X, X)
    teacher = LogisticRegression(C=1, max_iter=2000, random_state=0,
                                 class_weight='balanced').fit(triangular_memberships(A).reshape(len(A), -1), y)
    W, b, classes = linear_parameters(teacher)
    np.testing.assert_allclose(W.reshape(2, 4, 3), stage['teacher_coef'], atol=1e-12)
    np.testing.assert_allclose(b, stage['teacher_intercept'], atol=1e-12)
    assert stage['teacher_class_balanced']
    original = triangular_box_margins(A, y, stage['teacher_coef'], stage['teacher_intercept'], classes)
    protected = original > 1e-10
    for record in stage['policies'].values():
        assert record['candidate']['certificate']['feasible']
        fit = record['fit']
        actual = triangular_box_margins(A, y, fit['coef'], fit['intercept'], classes)
        assert np.all(actual[protected] >= .5*original[protected]-1e-8)
        assert np.all(actual[protected] > 1e-10)
