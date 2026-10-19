from collections import Counter
import copy
import numpy as np
import pytest
from fit_pair_transition_frs_hinge import fit_pair_transition_target
from audit_pair_transition_frs_fit import check_pair_transition_fit


def test_residual_pair_epigraph_nonnegative_duals_and_training_descent():
    rng = np.random.default_rng(781); A = rng.uniform(0, 1, (24, 3)); y = np.repeat([0, 1], 12)
    anchor = np.ones(3)/np.sqrt(3)
    target, fit = fit_pair_transition_target(A, y, [0, 1, 2], anchor,
        full_reference=anchor, reference_scope='original_full')
    assert fit['optimizer_success'], fit['status']
    assert fit['class_cell_final']['feasible']
    assert fit['training_objective_final'] <= fit['training_objective_initial']+1e-10
    assert sum(len(r['enemies']) for r in fit['class_cell']['pair_transition_objects']) > 0
    record = {'features': [0, 1, 2], 'anchor': anchor.tolist(), 'target': target.tolist(), 'fit': fit}
    counts = Counter(); check_pair_transition_fit(A, y, record, counts)
    assert counts['independent_master_primal_dual_replays'] >= 1
    altered = copy.deepcopy(record)
    altered['fit']['class_cell']['pair_transition_objects'][0]['distance_lower'][0][0] += .1
    with pytest.raises(AssertionError):
        check_pair_transition_fit(A, y, altered, Counter())
