import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from shared_graph_dependency_v1 import class_upper, build, memberships


def test_reduced_switching_and_clipping_against_dense_shared_box_grid():
    A = np.array([[.04], [.12], [.2]])
    y = np.array([0, 1, 1]); wf, wr = np.array([3.]), np.array([1.])
    report = class_upper(A, y, wf, wr, 0, eps=.06, seconds=10.)
    lo, hi = np.maximum(0., A-.06), np.minimum(1., A+.06)
    grid_best = -1.
    for vals in itertools.product(*(np.linspace(l, h, 9) for l, h in zip(lo[:, 0], hi[:, 0]))):
        Z = np.array(vals)[:, None]
        loss = (memberships(Z, y, wf)-memberships(Z, y, wr))[0]
        grid_best = max(grid_best, loss)
    assert len(report['selections'][0]['possible_enemies']) == 2
    assert report['solver_bound_available'] and report['upper_bound'] >= grid_best-1e-9
    assert report['observed_signed_loss_lower'] <= report['upper_bound']


def test_saturation_disjunction_and_zero_radius_exactness():
    A = np.array([[0., .2], [.8, .9], [1., .4]])
    y = np.array([0, 1, 1]); wf = np.array([1., 1.]); wr = np.array([2., 0.])
    report = class_upper(A, y, wf, wr, 0, eps=0.)
    loss = (memberships(A, y, wf)-memberships(A, y, wr))[0]
    assert report['selections'][0]['saturation_branch']
    assert report['solver_upper_bound'] == pytest.approx(loss, abs=1e-8)


def test_negative_class_loss_is_retained_before_positive_part():
    A = np.array([[.1], [.2], [.7]])
    y = np.array([0, 0, 1]); wf, wr = np.array([.2]), np.array([.8])
    report = class_upper(A, y, wf, wr, 0, eps=.01)
    assert report['solver_upper_bound'] < -.2
    assert report['upper_bound'] == 0.


def test_relaxation_dominates_exact_small_table_and_random_attacks():
    rng = np.random.default_rng(217)
    for _ in range(3):
        A = rng.uniform(.2, .8, (5, 2)); y = np.array([0, 0, 1, 1, 1])
        wf, wr = rng.uniform(.1, 1.2, (2, 2))
        relaxed = class_upper(A, y, wf, wr, 0, eps=.11, seconds=10.)
        exact = class_upper(A, y, wf, wr, 0, eps=.11, seconds=10., exact_abs=True)
        assert relaxed['solver_upper_bound']+1e-7 >= exact['solver_upper_bound']
        for _ in range(30):
            Z = rng.uniform(A-.11, A+.11)
            value = (memberships(Z, y, wf)-memberships(Z, y, wr))[:2].mean()
            assert value <= relaxed['solver_upper_bound']+1e-7


def test_shared_variables_and_validation():
    A = np.array([[.1], [.2], [.4], [.5]]); y = np.array([0, 0, 1, 1])
    m, _, meta = build(A, y, [1.], [.5], 0)
    assert meta['shared_pair_coordinates'] == 4
    assert class_upper(A, y, [1.], [1.], 0)['upper_bound'] == 0.
    for bad in (np.array([[-.1], [.5]]), np.array([[np.nan], [.5]])):
        with pytest.raises(ValueError):
            class_upper(bad, [0, 1], [1.], [.5], 0)
