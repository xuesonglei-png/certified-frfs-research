from collections import Counter
import copy
import numpy as np
import pytest
from fit_global_transition_capped_frs_hinge import fit_global_transition_capped_target
from audit_global_transition_capped_fit import check_global_transition_capped_fit


def test_global_region_both_initializations_replay_loss_geometry_and_negative_bound_duals():
    A = np.random.default_rng(781).uniform(0, 1, (24, 3)); y = np.repeat([0, 1], 12)
    anchor = np.ones(3)/np.sqrt(3); fs = [0, 1, 2]; counts = Counter()
    endpoint, first = fit_global_transition_capped_target(A, y, fs, anchor,
        full_reference=anchor, reference_scope='original_full')
    warm = .5*anchor+.5*endpoint
    for initial in (None, warm):
        target, fit = (endpoint, first) if initial is None else fit_global_transition_capped_target(A, y, fs, anchor,
            full_reference=anchor, reference_scope='original_full', initialization=initial,
            initialization_scope='synthetic_feasible_warm')
        assert fit['optimizer_success'], fit['status']
        assert fit['class_cell']['selected_objects'] == fit['class_cell']['linear_cell_rows'] == []
        assert fit['class_cell_final']['feasible']
        record = {'features':fs, 'anchor':anchor.tolist(), 'target':target.tolist(), 'fit':fit}
        check_global_transition_capped_fit(A, y, record, counts)
        damaged = copy.deepcopy(record)
        damaged['fit']['class_cell']['classes'][0]['remaining_signed_anchor_sum'] -= .1
        with pytest.raises(AssertionError):
            check_global_transition_capped_fit(A, y, damaged, Counter())
    assert counts['independent_dc_fit_replays'] == 2
    assert counts['independent_master_primal_dual_replays'] >= 2
