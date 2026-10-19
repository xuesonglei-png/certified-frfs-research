from collections import Counter
import copy
import numpy as np
import pytest
from fit_strict_capped_frs_hinge import fit_strict_capped_frs_target
from fit_pair_capped_frs_hinge import fit_pair_capped_frs_target
from fit_pair_latent_frs_hinge import fit_pair_latent_frs_target
from audit_pair_capped_frs_fit import check_pair_capped_fit
from audit_pair_latent_frs_fit import check_pair_latent_fit


def test_bounded_fixed_and_latent_losses_in_pair_cell_have_matched_starts_and_duals():
    rng = np.random.default_rng(781); A = rng.uniform(0, 1, (24, 3)); y = np.repeat([0, 1], 12)
    fs = [0, 1, 2]; anchor = np.ones(3)/np.sqrt(3)
    warm, prior = fit_strict_capped_frs_target(A, y, fs, anchor,
        full_reference=anchor, reference_scope='original_full')
    assert prior['optimizer_success']
    for function, checker in ((fit_pair_capped_frs_target, check_pair_capped_fit),
                              (fit_pair_latent_frs_target, check_pair_latent_fit)):
        for initialization, scope in ((None, 'anchor'), (warm, 'audited_capped_selected_training_endpoint')):
            target, fit = function(A, y, fs, anchor, full_reference=anchor, reference_scope='original_full',
                initialization=initialization, initialization_scope=scope)
            assert fit['optimizer_success'], fit['status']
            assert fit['class_cell_final']['feasible']
            assert fit['training_objective_final'] <= fit['optimization_start_objective']+1e-10
            counts = Counter()
            record = {'features': fs, 'anchor': anchor.tolist(), 'target': target.tolist(), 'fit': fit}
            checker(A, y, record, counts)
            assert counts['independent_master_primal_dual_replays'] >= 1
            forged = copy.deepcopy(record)
            forged['fit']['class_cell']['pair_transition_objects'][0]['distance_upper'][0][0] += .1
            with pytest.raises(AssertionError):
                checker(A, y, forged, Counter())
