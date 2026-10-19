from collections import Counter
import copy
import numpy as np
import pytest
from fit_saturation_transition_frs_hinge import fit_saturation_transition_target
from audit_saturation_transition_frs_fit import check_saturation_transition_fit


def test_negative_finite_epigraph_lower_bounds_in_cccp_and_independent_dual():
    A = np.random.default_rng(781).uniform(0, 1, (24, 3)); y = np.repeat([0, 1], 12)
    anchor = np.ones(3)/np.sqrt(3)
    target, fit = fit_saturation_transition_target(A, y, [0, 1, 2], anchor,
        full_reference=anchor, reference_scope='original_full')
    assert fit['optimizer_success'], fit['status']
    assert fit['class_cell_final']['feasible']
    assert fit['training_objective_final'] <= fit['training_objective_initial']+1e-10
    assert any(r['saturation_floor'] < 0 for r in fit['class_cell']['pair_transition_objects'])
    record = {'features':[0, 1, 2], 'anchor':anchor.tolist(), 'target':target.tolist(), 'fit':fit}
    counts = Counter(); check_saturation_transition_fit(A, y, record, counts)
    assert counts['independent_master_primal_dual_replays'] >= 1
    tampered = copy.deepcopy(record)
    tampered['fit']['class_cell']['pair_transition_objects'][0]['saturation_floor'] -= .1
    with pytest.raises(AssertionError):
        check_saturation_transition_fit(A, y, tampered, Counter())
