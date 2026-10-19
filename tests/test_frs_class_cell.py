from itertools import product
import numpy as np
from scipy.optimize import linprog
from frs_class_cell import build_class_cell, evaluate_class_cell, append_class_cell_lp
from certified_frfs.core import positive_region


def test_cell_bounds_shared_tables_for_changed_metric_and_changing_enemies():
    rng = np.random.default_rng(181); A = rng.uniform(0, 1, (15, 3)); y = np.repeat([0, 1, 2], 5)
    full = np.ones(3)/np.sqrt(3); anchor = np.array([.9, np.sqrt(3)-.9])
    C, floors, info = build_class_cell(A, y, [0, 1], anchor, full)
    for _ in range(100):
        w = anchor+rng.uniform(-.02, .02, 2)
        if len(C) and not np.all(C@w >= floors):
            continue
        bound = evaluate_class_cell(info, w)
        for _ in range(3):
            Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
            loss = positive_region(Z, y, weights=full)-positive_region(Z, y, [0, 1], weights=np.r_[w, 0.])
            assert all(loss[y == c].mean() <= bound['class_upper_bounds'][c]+1e-10 for c in (0, 1, 2))


def test_all_shared_corners_match_exact_star_class_decrease_on_stable_case():
    A = np.array([[0., .03], [1., 0.], [.45, .4]]); y = np.array([0, 0, 1]); full = np.ones(2)/np.sqrt(2)
    anchor = np.array([np.sqrt(2)])
    C, rhs, info = build_class_cell(A, y, [1], anchor, full)
    assert all(r['mode'] == 'shared_star' for r in info['classes']) and info['anchor_strictly_feasible']
    assert np.all(C@anchor > rhs)
    lo, hi = np.maximum(0, A-.01), np.minimum(1, A+.01)
    for w in (np.array([1.35]), anchor, np.array([1.48])):
        if not np.all(C@w >= rhs):
            continue
        report = evaluate_class_cell(info, w); maxima = np.zeros(2)
        for bits in product([0, 1], repeat=A.size):
            Z = np.where(np.asarray(bits).reshape(A.shape), hi, lo)
            diff = positive_region(Z, y, weights=full)-positive_region(Z, y, [1], weights=np.r_[0., w])
            maxima = np.maximum(maxima, [diff[y == c].mean() for c in (0, 1)])
        np.testing.assert_allclose(report['class_upper_bounds'], maxima, atol=2e-10, rtol=0)


def test_LP_epigraph_constraints_are_equivalent_to_recomputed_cell_bound():
    A = np.array([[0., .03], [1., 0.], [.45, .4]]); y = np.array([0, 0, 1]); full = np.ones(2)/np.sqrt(2)
    C, rhs, info = build_class_cell(A, y, [1], np.array([np.sqrt(2)]), full); p, n, mass = 1, len(A), np.sqrt(2)
    for w in (1.25, 1.35, np.sqrt(2), 1.55):
        rows, b = [], []
        def append(cols, vals, bound):
            rows.append((cols, vals)); b.append(bound)
        size, extras = append_class_cell_lp(info, mass, p, n, append)
        matrix = np.zeros((len(rows), size))
        for i, (cols, vals) in enumerate(rows):
            matrix[i, cols] = vals
        bounds = [(w/mass, w/mass)]+[(0, 0)]*n+[(abs(w-mass)/mass, abs(w-mass)/mass)]+extras
        fit = linprog(np.zeros(size), A_ub=matrix, b_ub=b, bounds=bounds, method='highs')
        recomputed = evaluate_class_cell(info, np.array([w]))
        raw_feasible = all(v <= info['eta'] for v in recomputed['raw_upper_bounds_with_inflation'])
        assert fit.success == raw_feasible


def test_identity_with_zero_omitted_weights_has_exact_anchor_zero_loss():
    rng = np.random.default_rng(114); A = rng.uniform(0, 1, (12, 3)); y = np.repeat([0, 1], 6)
    full = np.array([.9, .7, 0.]); _, _, info = build_class_cell(A, y, [0, 1], full[:2], full)
    assert info['anchor_identity'] and info['anchor_strictly_feasible']
    np.testing.assert_array_equal(info['signed_anchor_object_bounds'], np.zeros(len(A)))
    assert max(info['anchor_evaluation']['class_upper_bounds']) <= 1e-10
