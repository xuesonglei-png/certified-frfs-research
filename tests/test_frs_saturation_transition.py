from collections import Counter
from fractions import Fraction
import copy
import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from frs_saturation_transition_cell import (build_class_cell, evaluate_class_cell,
    append_class_cell_lp, transition_bound)
from independent_frs_saturation_transition import reconstruct_cell, scalar_cell_evaluation, independent_append_cell_lp
from frs_pair_transition_cell import evaluate_class_cell as old_evaluation
from audit_forest_compact_transfer import equivalent


def test_signed_saturation_floor_retains_strict_negative_compensation_and_lp_bounds():
    A = np.array([[0., c] for c in (0., 1.) for _ in range(6)])
    y = np.repeat([0, 1], 6); anchor = np.ones(2)/np.sqrt(2); target = np.array([0., np.sqrt(2)])
    _, _, info = build_class_cell(A, y, [0, 1], anchor, anchor)
    reconstruct_cell(A, y, info, Counter())
    assert not info['selected_objects']
    new, old = evaluate_class_cell(info, target), old_evaluation(info, target)
    equivalent(new, scalar_cell_evaluation(info, target))
    assert max(new['raw_upper_bounds_with_inflation']) < -.25
    assert min(old['raw_upper_bounds_with_inflation']) >= 0
    # Exact saturation identity at a singleton box:anchor memberships1/2,
    # target memberships1,so signed deficit=-1/2,where clipping would give0.
    assert Fraction(1, 2)-min(Fraction(1), Fraction(1)) == -Fraction(1, 2)
    matrices = []
    for function in (append_class_cell_lp, independent_append_cell_lp):
        rows, cols, data, b = [], [], [], []
        def add(c, v, rhs):
            rows.extend([len(b)]*len(c)); cols.extend(c); data.extend(v); b.append(rhs)
        size, extras = function(info, np.sqrt(2), 2, 0, add)
        matrix = coo_matrix((data, (rows, cols)), shape=(len(b), size)).tocsr()
        bounds = [(v/np.sqrt(2), v/np.sqrt(2)) for v in target]
        bounds += [(abs(v-a)/np.sqrt(2), abs(v-a)/np.sqrt(2)) for v, a in zip(target, anchor)]
        fit = linprog(np.zeros(size), A_ub=matrix, b_ub=b, bounds=bounds+extras, method='highs')
        assert fit.success
        assert all(lo < 0 for lo, _ in extras)
        assert min(fit.x[4:]) < -.25
        matrices.append(matrix)
    assert (matrices[0] != matrices[1]).nnz == 0
    broken = copy.deepcopy(info); broken['pair_transition_objects'][0]['saturation_floor'] = -1.
    with pytest.raises(AssertionError):
        reconstruct_cell(A, y, broken, Counter())


def test_new_signed_bound_dominates_old_and_holds_on_shared_box_tables():
    rng = np.random.default_rng(7703); A = rng.uniform(0, 1, (12, 3)); y = np.repeat([0, 1], 6)
    anchor = np.ones(3)/np.sqrt(3); _, _, info = build_class_cell(A, y, [0, 1, 2], anchor, anchor)
    reconstruct_cell(A, y, info, Counter())
    lo, hi = np.maximum(0., A-.01), np.minimum(1., A+.01); comparisons = 0
    for _ in range(15):
        target = rng.dirichlet(np.ones(3))*np.sqrt(3)
        new, old = evaluate_class_cell(info, target), old_evaluation(info, target)
        equivalent(new, scalar_cell_evaluation(info, target))
        assert np.all(np.asarray(new['raw_upper_bounds_with_inflation']) <= np.asarray(old['raw_upper_bounds_with_inflation'])+1e-12)
        for _ in range(12):
            Z = rng.uniform(lo, hi)
            def rho(i, w):
                return min(1., min(float(abs(Z[i]-Z[e])@w) for e in range(len(Z)) if y[e] != y[i]))
            for item in info['pair_transition_objects']:
                i = item['object']
                assert rho(i, anchor)-rho(i, target) <= transition_bound(item, anchor, target)+1e-12
                comparisons += 1
    assert comparisons > 0
