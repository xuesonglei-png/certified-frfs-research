from collections import Counter
import numpy as np
from fit_strict_frs_cell_hinge import fit_strict_frs_cell_target
from audit_frs_cell_fit_stored import check_cell_fit


def test_stricter_solver_keeps_same_math_and_actual_gap_requirement():
    rng = np.random.default_rng(781); A = rng.uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    anchor = np.ones(3)/np.sqrt(3)
    target, fit = fit_strict_frs_cell_target(A, y, [0, 1, 2], anchor,
        full_reference=anchor, reference_scope='original_full')
    assert fit['optimizer_success'] and fit['class_cell_final']['feasible']
    assert fit['solver_primal_feasibility_tolerance'] == fit['solver_dual_feasibility_tolerance'] == 1e-9
    assert fit['gap_tolerance'] == 1e-7 and fit['class_cell']['eta'] == .1 and fit['class_cell']['eps'] == .01
    counts = Counter()
    check_cell_fit(A, y, {'features': [0, 1, 2], 'anchor': anchor.tolist(), 'target': target.tolist(), 'fit': fit}, counts)
    assert counts['independent_master_primal_dual_replays'] >= 1
