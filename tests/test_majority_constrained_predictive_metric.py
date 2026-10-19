"""Fixed training witnesses must survive actual shared perturbations."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from majority_constrained_predictive_metric import anchor_witness_polytope, fit_majority_constrained_target, smooth_objective
from certified_frfs.softvote_metric import softvote_terms


def fixture():
    rng = np.random.default_rng(719)
    y = np.repeat([0, 1], 10)
    A = np.clip(.2+.6*y[:, None]+rng.normal(0, .10, (20, 6)), 0, 1)
    A[:, 0] = .1+.8*y+rng.uniform(-.005, .005, len(y))
    return A, y, list(range(6)), np.ones(6)/np.sqrt(6)


def test_training_witness_constraints_preserve_actual_shared_box_majorities():
    A, y, fs, anchor = fixture()
    target, fit = fit_majority_constrained_target(A, y, fs, anchor)
    assert fit['optimizer_success'] and fit['status'] == 'local_majority_constrained_target'
    assert fit['training_objective_final'] <= fit['training_objective_initial']+1e-8
    assert target[0] > anchor[0]
    np.testing.assert_allclose(target.sum(), np.sqrt(6), atol=1e-10)
    assert target.min() >= 0 and target.max()/np.sqrt(6) <= 2/6+1e-10
    rng = np.random.default_rng(88)
    for _ in range(32):
        Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
        D = abs(Z[:, None, fs]-Z[None, :, fs])@target
        np.fill_diagonal(D, np.inf)
        for row in fit['polytope']['witnesses']:
            i = row['object']; friends = row['friends']; enemies = row['retained_enemies']
            assert len(friends) == 3 and np.all(y[friends] == y[i])
            assert max(D[i, friends]) < min(D[i, enemies])
            nn = np.argsort(D[i], kind='stable')[:5]
            assert np.sum(y[nn] == y[i]) >= 3
    # The target only has a witness guarantee; do not invent dependency or risk.
    assert 'certificate' not in fit


def test_polytope_anchor_is_strictly_feasible_and_targets_are_class_balanced():
    A, y, fs, anchor = fixture()
    C, rhs, info = anchor_witness_polytope(A, y, fs, anchor)
    assert np.all(C@anchor > rhs)
    assert len(info['targets']) == 18
    assert [sum(y[i] == c for i in info['targets']) for c in [0, 1]] == [9, 9]
    assert info['constraint_rows'] == 18*8*3


def test_failed_fit_returns_exact_anchor_with_visible_failure():
    A, y, fs, anchor = fixture()
    target, fit = fit_majority_constrained_target(A, y, fs, anchor, max_iter=1)
    np.testing.assert_array_equal(target, anchor)
    assert not fit['optimizer_success']
    assert fit['status'] == 'constrained_optimizer_failed_anchor_fallback'


def test_unique_simplex_and_invalid_iteration_limit():
    A, y, _, _ = fixture()
    anchor = np.ones(3)*np.sqrt(6)/3
    target, fit = fit_majority_constrained_target(A, y, [0, 1, 2], anchor)
    np.testing.assert_array_equal(target, anchor)
    assert fit['status'] == 'constrained_unique_capped_simplex'
    with pytest.raises(ValueError):
        fit_majority_constrained_target(A, y, list(range(6)), np.ones(6)/np.sqrt(6), max_iter=0)


def test_entropy_is_finite_at_zero_and_gradient_matches_difference_inside_simplex():
    A, y, _, _ = fixture(); terms = softvote_terms(A, y)
    w = np.array([0., .2, .2, .2, .2, .2])
    value, grad = smooth_objective(w, terms)
    assert np.isfinite(value) and np.isfinite(grad).all()
    w = np.arange(1., 7.); w /= w.sum()
    value, grad = smooth_objective(w, terms)
    for i in range(6):
        step = np.zeros(6); step[i] = 1e-6
        numeric = (smooth_objective(w+step, terms)[0]-smooth_objective(w-step, terms)[0])/2e-6
        np.testing.assert_allclose(grad[i], numeric, atol=1e-6)


def test_near_tied_friends_metadata_reconstructs_actual_optimization_matrix():
    # A reduced Iris geometry exposed a different third friend when distances
    # were multiplied after subtraction. Retain that actual numeric counterexample.
    positive = np.array([
        [.44444444444444464, .4761904761904763, .5423728813559322, .5833333333333334],
        [.4999999999999998, .4285714285714286, .6271186440677966, .5416666666666667],
        [.6388888888888888, .4761904761904763, .576271186440678, .5416666666666667],
        [.4722222222222223, .4285714285714286, .5932203389830508, .5833333333333334],
        [.4999999999999998, .4761904761904763, .6101694915254237, .5416666666666667],
    ])
    A = np.vstack([positive, np.zeros((5, 4))]); y = np.repeat([1, 0], 5)
    anchor = np.array([.5017064158981609, .7630237007827618, .7352698833190778]); fs = [1, 2, 3]
    C, _, info = anchor_witness_polytope(A, y, fs, anchor)
    lo, hi = np.maximum(0, A[:, fs]-.01), np.minimum(1, A[:, fs]+.01)
    blocks = []
    for r in info['witnesses']:
        i = r['object']; friends, enemies = r['friends'], r['retained_enemies']
        alternatives = []
        for t in (lo[i], hi[i]):
            near = np.maximum(0, np.maximum(lo[enemies]-t, t-hi[enemies]))
            far = np.maximum(abs(t-lo[friends]), abs(t-hi[friends]))
            alternatives.append(near[:, None, :]-far[None, :, :])
        blocks.append(np.minimum(*alternatives).reshape(-1, 3))
    np.testing.assert_array_equal(C, np.concatenate(blocks))
