from collections import Counter
from itertools import product
import numpy as np
from scipy.optimize import linprog
from certified_frfs.core import positive_region
from frs_class_cell import evaluate_class_cell as symmetric_evaluation
from frs_one_sided_cell import build_class_cell, evaluate_class_cell, append_class_cell_lp
from fit_frs_one_sided_hinge import fit_one_sided_cell_target
from audit_frs_one_sided_fit import check_one_sided_fit


def test_one_sided_bound_dominates_symmetric_and_adds_feasible_metrics():
    A = np.r_[np.full((9, 3), .1), np.full((9, 3), .4)]; y = np.repeat([0, 1], 9)
    full = np.ones(3)/np.sqrt(3); _, _, info = build_class_cell(A, y, [0, 1, 2], full, full)
    assert all(r['mode'] == 'signed_Lipschitz' for r in info['classes'])
    w = full+np.array([.2, -.2, 0.])
    new = evaluate_class_cell(info, w); old = symmetric_evaluation(info, w)
    assert new['feasible'] and not old['feasible']
    for r, s in zip(new['raw_upper_bounds_with_inflation'], old['raw_upper_bounds_with_inflation']):
        np.testing.assert_allclose(r-1e-10, (s-1e-10)/2, atol=1e-13)
    assert np.all(np.asarray(new['class_upper_bounds']) <= np.asarray(old['class_upper_bounds']))


def test_random_metrics_and_joint_tables_respect_monotonic_envelope():
    rng = np.random.default_rng(304); A = rng.uniform(0, 1, (15, 3)); y = np.repeat([0, 1, 2], 5)
    full = np.ones(3)/np.sqrt(3); anchor = np.array([.9, np.sqrt(3)-.9])
    C, rhs, info = build_class_cell(A, y, [0, 1], anchor, full)
    for _ in range(80):
        w = rng.uniform(.01, 2, 2)
        if len(C) and not np.all(C@w >= rhs):
            continue
        new, old = evaluate_class_cell(info, w), symmetric_evaluation(info, w)
        assert all(a <= b+1e-13 for a, b in zip(new['class_upper_bounds'], old['class_upper_bounds']))
        for _ in range(3):
            Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
            loss = positive_region(Z, y, weights=full)-positive_region(Z, y, [0, 1], weights=np.r_[w, 0.])
            assert all(loss[y == c].mean() <= new['class_upper_bounds'][c]+1e-10 for c in (0, 1, 2))


def test_all_shared_corners_in_mixed_star_and_unstable_cell():
    A = np.array([[.4, .43], [.44, .47], [.41, .44]]); y = np.array([0, 0, 1])
    full = np.ones(2)/np.sqrt(2); C, rhs, info = build_class_cell(A, y, [0, 1], full, full)
    lo, hi = A-.01, A+.01
    for w in (full, full+np.array([.2, -.2]), full+np.array([-.2, .2])):
        if len(C) and not np.all(C@w >= rhs):
            continue
        report = evaluate_class_cell(info, w)
        for bits in product([0, 1], repeat=A.size):
            Z = np.where(np.array(bits).reshape(A.shape), hi, lo)
            loss = positive_region(Z, y, weights=full)-positive_region(Z, y, weights=w)
            assert all(loss[y == c].mean() <= report['class_upper_bounds'][c]+1e-10 for c in (0, 1))


def test_one_sided_lp_epigraph_is_equivalent_to_actual_bound():
    A = np.r_[np.full((9, 3), .1), np.full((9, 3), .4)]; y = np.repeat([0, 1], 9)
    mass = np.sqrt(3); full = np.ones(3)/mass; p, n = 3, len(A)
    _, _, info = build_class_cell(A, y, [0, 1, 2], full, full)
    for change in (.1, .2, .35, .5):
        w = full+np.array([change, -change, 0.]); rows, b = [], []
        def append(cols, vals, bound):
            rows.append((cols, vals)); b.append(bound)
        size, extras = append_class_cell_lp(info, mass, p, n, append); matrix = np.zeros((len(rows), size))
        for i, (cols, vals) in enumerate(rows):
            matrix[i, cols] = vals
        bounds = [(q/mass, q/mass) for q in w]+[(0., 0.)]*n+[(abs(q-a)/mass, abs(q-a)/mass) for q,a in zip(w, full)]+extras
        fit = linprog(np.zeros(size), A_ub=matrix, b_ub=b, bounds=bounds, method='highs')
        assert fit.success == evaluate_class_cell(info, w)['feasible']


def test_one_sided_optimizer_replays_cell_geometry_and_duals():
    rng = np.random.default_rng(781); A = rng.uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    anchor = np.ones(3)/np.sqrt(3)
    target, fit = fit_one_sided_cell_target(A, y, [0, 1, 2], anchor,
        full_reference=anchor, reference_scope='original_full')
    assert fit['optimizer_success'], fit['status']
    assert fit['class_cell_final']['feasible'] and fit['training_objective_final'] <= fit['training_objective_initial']
    counts = Counter()
    check_one_sided_fit(A, y, {'features': [0, 1, 2], 'anchor': anchor.tolist(), 'target': target.tolist(), 'fit': fit}, counts)
    assert counts['independent_master_primal_dual_replays'] >= 1
