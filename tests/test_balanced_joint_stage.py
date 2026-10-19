import numpy as np
from scipy.special import logsumexp
from certified_frfs.nested_joint_group_stage import fit_stage
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_memberships, triangular_box_margins


def test_balanced_group_objective_and_protected_margins():
    rng = np.random.default_rng(17)
    X = rng.random((24, 4))
    y = (X[:, 0] > .7).astype(int)
    stage = fit_stage(X, y, C=1, class_balanced=True)
    A, _ = preprocess_fit(X, X)
    assert stage['class_balanced']
    joint = stage['joint_fit']
    assert joint['feasible'] and joint['class_balanced']
    W, b = np.array(joint['coef']), np.array(joint['intercept'])
    scores = np.einsum('ndk,qdk->nq', triangular_memberships(A), W)+b
    weights = np.array([len(y)/(2*sum(y == label)) for label in y])
    centered = W-W.mean(axis=0, keepdims=True)
    value = weights@(logsumexp(scores, axis=1)-scores[np.arange(len(y)), y])
    value += np.sum(W[1]**2)/2+len(y)*.01*np.linalg.norm(centered, axis=(0, 2)).sum()
    np.testing.assert_allclose(value, joint['objective'], rtol=1e-8, atol=1e-7)
    original = triangular_box_margins(A, y, stage['teacher_coef'],
                                     stage['teacher_intercept'], stage['teacher_classes'])
    mask = original > 1e-10
    for record in stage['policies'].values():
        assert record['candidate']['certificate']['feasible']
        fit = record['fit']
        actual = triangular_box_margins(A, y, fit['coef'], fit['intercept'], stage['teacher_classes'])
        assert np.all(actual[mask] >= .5*original[mask]-1e-8)
        assert np.all(actual[mask] > 1e-10)
        if 'objective' in fit:
            assert fit['class_balanced']
