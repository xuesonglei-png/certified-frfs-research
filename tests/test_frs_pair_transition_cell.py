from collections import Counter
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from frs_one_sided_cell import evaluate_class_cell as monotone_evaluation
from frs_pair_transition_cell import build_class_cell, evaluate_class_cell, append_class_cell_lp, transition_bound
from independent_frs_pair_transition_cell import reconstruct_cell, scalar_cell_evaluation
from audit_forest_compact_transfer import equivalent


def test_exact_compensation_mechanism_removes_one_of_two_attributes():
    # Duplicate nearest enemies intentionally rule out a stable-enemy cell.
    # The already known signed pair bound credits increasing coordinate two.
    A = np.array([[x, c] for c in (0., 1.) for x in (0., 0., .1, .1, .2, .2)])
    y = np.repeat([0, 1], 6); anchor = np.ones(2)/np.sqrt(2); target = np.array([0., np.sqrt(2)])
    C, rhs, info = build_class_cell(A, y, [0, 1], anchor, anchor)
    assert not len(C) and not info['selected_objects']
    counts = Counter(); reconstructed, floors = reconstruct_cell(A, y, info, counts)
    assert reconstructed.shape == C.shape and np.array_equal(floors, rhs)
    after = evaluate_class_cell(info, target); equivalent(after, scalar_cell_evaluation(info, target))
    assert after['feasible'] and not monotone_evaluation(info, target)['feasible']
    assert max(after['class_upper_bounds']) <= 1e-9
    # Exact interval geometry in rational coordinates: every first-coordinate
    # distance <= .22 and every second-coordinate enemy distance >= .98.
    # Thus (.5,.5)->(0,1) pair transition <= .5*(.22-.98)<0 throughout
    # the joint box. Positive homogeneous scaling gives actual anchor/target.
    assert F(1, 2)*(F(22, 100)-F(98, 100)) < 0
    # LP epigraph equivalence at the newly admitted endpoint.
    p, n, mass = 2, len(A), np.sqrt(2); rows, cols, values, b = [], [], [], []
    def append(c, v, bound):
        rows.extend([len(b)]*len(c)); cols.extend(c); values.extend(v); b.append(bound)
    size, extras = append_class_cell_lp(info, mass, p, n, append)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(b), size)).tocsr()
    bounds = [(v/mass, v/mass) for v in target]+[(0., 0.)]*n
    bounds += [(abs(v-a)/mass, abs(v-a)/mass) for v, a in zip(target, anchor)]+extras
    result = linprog(np.zeros(size), A_ub=matrix, b_ub=b, bounds=bounds, method='highs')
    assert result.success, result.message


def test_transition_dominates_one_sided_and_bounds_shared_table_loss():
    rng = np.random.default_rng(2304); A = rng.uniform(0, 1, (12, 3)); y = np.repeat([0, 1], 6)
    anchor = np.ones(3)/np.sqrt(3); _, _, info = build_class_cell(A, y, [0, 1, 2], anchor, anchor)
    reconstruct_cell(A, y, info, Counter())
    lo, hi = np.maximum(0., A-.01), np.minimum(1., A+.01)
    checked_pair_losses = 0
    for _ in range(25):
        weights = rng.dirichlet(np.ones(3))*np.sqrt(3)
        bound = evaluate_class_cell(info, weights); equivalent(bound, scalar_cell_evaluation(info, weights))
        old = monotone_evaluation(info, weights)
        assert np.all(np.asarray(bound['raw_upper_bounds_with_inflation']) <= np.asarray(old['raw_upper_bounds_with_inflation'])+1e-12)
        for _ in range(8):
            Z = rng.uniform(lo, hi)
            def memberships(w):
                return np.array([min(1., min(sum(abs(Z[i]-Z[e])*w) for e in range(len(A)) if y[e] != y[i])) for i in range(len(A))])
            loss = memberships(anchor)-memberships(weights)
            # The residual transition theorem holds without a stable-enemy
            # cell; validate it even when the star constraints are unavailable.
            for item in info['pair_transition_objects']:
                assert loss[item['object']] <= transition_bound(item, anchor, weights)+1e-12
                checked_pair_losses += 1
            if bound['cell_constraints_verified']:
                for c, upper in zip(np.unique(y), bound['raw_upper_bounds_with_inflation']):
                    assert loss[y == c].mean() <= upper+1e-12
    assert checked_pair_losses > 0
