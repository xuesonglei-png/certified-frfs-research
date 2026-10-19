import itertools
import numpy as np
from scipy.optimize import linprog
from certified_frfs.signed_forest import signed_forest_maximum
from certified_frfs.coupled_redistributed import coupled_redistributed_bound
from certified_frfs.core import gamma


def test_negative_edges_need_interior_other_object_endpoint():
    r = signed_forest_maximum([0., .2, .5], [0., .8, .5], [(0, 1, 1.), (1, 2, -2.)])
    assert abs(r['maximum']-.5) < 1e-12 and abs(r['witness'][1]-.5) < 1e-12


def test_signed_dp_matches_all_orientation_linear_programs():
    rng = np.random.default_rng(1010)
    for _ in range(15):
        lo = rng.uniform(0, .5, 5); hi = lo+rng.uniform(.05, .5, 5)
        edges = [(0, 1, rng.normal()), (1, 2, rng.normal()), (1, 3, rng.normal()), (3, 4, rng.normal())]
        best = -np.inf
        for signs in itertools.product([-1, 1], repeat=len(edges)):
            coefficients = np.zeros(5); constraints = []
            for sign, (i, j, c) in zip(signs, edges):
                row = np.zeros(5); row[i] = sign; row[j] = -sign
                coefficients += c*row; constraints.append(-row)
            fit = linprog(-coefficients, A_ub=constraints, b_ub=np.zeros(len(edges)), bounds=list(zip(lo, hi)), method='highs')
            if fit.success: best = max(best, -fit.fun)
        assert abs(signed_forest_maximum(lo, hi, edges)['maximum']-best) < 1e-8


def test_coupled_changed_metric_bound_never_weakens_and_covers_samples():
    rng = np.random.default_rng(1011)
    for _ in range(8):
        X = rng.uniform(.1, .9, (10, 3)); y = np.repeat([0, 1], 5); full = np.ones(3)/3
        reduced = rng.dirichlet([1, 1]); wide = np.r_[reduced, 0.]
        r = coupled_redistributed_bound(X, y, [0, 1], full, reduced, eps=.001)
        assert r['upper_bound'] <= r['objectwise_bound']
        for _ in range(12):
            Z = np.clip(X+rng.uniform(-.001, .001, X.shape), 0, 1)
            assert gamma(Z, y, weights=full)-gamma(Z, y, [0, 1], weights=wide) <= r['upper_bound']+1e-10


def test_signed_proxy_is_exact_when_shared_nearest_witness_and_unsaturated():
    # Two opposite-class objects have the same unique enemy in both metrics.
    X = np.array([[.1, .2], [.3, .35]]); y = np.array([0, 1])
    full = np.array([.4, .4]); reduced = np.array([.8])
    result = coupled_redistributed_bound(X, y, [0], full, reduced, eps=.01)
    assert result['stable_unsaturated_count'] == 2
    lo = X-.01; hi = X+.01; exact = -np.inf
    for bits in itertools.product([0, 1], repeat=X.size):
        Z = np.where(np.array(bits).reshape(X.shape), hi, lo)
        exact = max(exact, gamma(Z, y, weights=full)-gamma(Z, y, [0], weights=np.array([.8, 0.])))
    assert abs(result['coupled_signed_bound']-max(0., exact)) < 1e-8
