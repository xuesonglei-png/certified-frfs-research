"""Implicit gradients, neighbor cardinality and soft shared-box lower mass."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from soft_topk_metric_objective import soft_topk_membership, soft_topk_objective, topk_leakage_bound
from certified_frfs.softvote_metric import softvote_terms


def test_membership_conserves_five_neighbors_and_excludes_self():
    rng = np.random.default_rng(105); A = rng.uniform(0, 1, (18, 4))
    D = abs(A[:, None, :]-A[None, :, :]).mean(axis=2)
    mask = ~np.eye(len(A), dtype=bool)
    Q = soft_topk_membership(D, mask)
    np.testing.assert_allclose(Q.sum(axis=1), 5, atol=1e-10, rtol=0)
    assert np.all(Q.diagonal() == 0) and Q.min() >= 0 and Q.max() <= 1
    for i in range(len(A)):
        order = np.argsort(D[i, mask[i]])
        assert np.all(np.diff(Q[i, mask[i]][order]) <= 1e-12)


def test_implicit_gradient_matches_independent_central_difference():
    rng = np.random.default_rng(108); A = rng.uniform(0, 1, (16, 4)); y = np.repeat([0, 1], 8)
    terms = softvote_terms(A, y); w = np.array([.1, .2, .3, .4])
    value, grad = soft_topk_objective(w, terms)
    assert np.isfinite(value)
    for i in range(4):
        step = np.zeros(4); step[i] = 1e-6
        numeric = (soft_topk_objective(w+step, terms)[0]-soft_topk_objective(w-step, terms)[0])/2e-6
        np.testing.assert_allclose(grad[i], numeric, atol=2e-6, rtol=2e-5)


def test_robust_soft_same_class_mass_lower_bounds_actual_shared_realizations():
    rng = np.random.default_rng(109); A = rng.uniform(0, 1, (20, 5)); y = np.repeat([0, 1], 10)
    C, same, mask, _ = softvote_terms(A, y); w = np.ones(5)/5
    adverse = soft_topk_membership(np.einsum('ijd,d->ij', C, w), mask)
    lower = (adverse*same).sum(axis=1)/5
    for _ in range(32):
        Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
        actual = soft_topk_membership(abs(Z[:, None, :]-Z[None, :, :])@w, mask)
        assert np.all((actual*same).sum(axis=1)/5 >= lower-1e-10)


def test_temperature_limit_approaches_actual_five_nearest_neighbors():
    D = np.array([[.01, .04, .07, .11, .15, .4, .65, .8]])
    Q = soft_topk_membership(D, np.ones(D.shape, bool), temperature=.001)
    np.testing.assert_allclose(Q, [[1, 1, 1, 1, 1, 0, 0, 0]], atol=1e-8)


def test_full_neighborhood_and_zero_weight_are_finite_and_visible_errors_raise():
    D = np.arange(6.)[None, :]; mask = np.ones((1, 6), bool); mask[0, 0] = False
    np.testing.assert_array_equal(soft_topk_membership(D, mask), mask.astype(float))
    rng = np.random.default_rng(110); A = rng.uniform(0, 1, (12, 3)); y = np.repeat([0, 1], 6)
    value, grad = soft_topk_objective(np.array([0., .5, .5]), softvote_terms(A, y))
    assert np.isfinite(value) and np.isfinite(grad).all()
    with pytest.raises(ValueError): soft_topk_membership(D, mask, k=6)


def test_gap_cardinality_bound_controls_actual_leakage_and_class_mass_error():
    rng = np.random.default_rng(118)
    for m in (6, 11, 20):
        for temperature in (.02, .05, .2):
            D = np.sort(rng.uniform(0, 1, m))
            Q = soft_topk_membership(D[None, :], np.ones((1, m), bool), temperature=temperature)[0]
            upper = topk_leakage_bound(D[5]-D[4], m, temperature=temperature)
            leakage = Q[5:].sum()
            assert leakage <= upper+1e-10
            labels = rng.integers(0, 3, m)
            for c in range(3):
                soft = Q[labels == c].sum()/5
                hard = np.sum(labels[:5] == c)/5
                assert abs(soft-hard) <= upper/5+1e-10
    assert topk_leakage_bound(.1, 20, temperature=.01) < topk_leakage_bound(.1, 20, temperature=.1)
