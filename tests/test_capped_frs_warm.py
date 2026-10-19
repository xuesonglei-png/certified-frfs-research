from collections import Counter
import numpy as np
import pytest
from fit_frs_one_sided_hinge import fit_one_sided_cell_target
from fit_capped_frs_warm import fit_capped_frs_warm_target
from audit_capped_frs_warm_fit import check_capped_warm_fit


def test_training_only_warm_start_preserves_anchor_cell_and_replays():
    rng = np.random.default_rng(781); A = rng.uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    anchor = np.ones(3)/np.sqrt(3)
    warm, _ = fit_one_sided_cell_target(A, y, [0, 1, 2], anchor, full_reference=anchor, reference_scope='original_full')
    for initialization, scope in [(None, 'anchor'), (warm, 'training_one_sided_unbounded_endpoint')]:
        target, fit = fit_capped_frs_warm_target(A, y, [0, 1, 2], anchor,
            full_reference=anchor, reference_scope='original_full', initialization=initialization, initialization_scope=scope)
        assert fit['optimizer_success'], fit['status']
        assert fit['training_objective_final'] <= fit['optimization_start_objective']+1e-10
        assert fit['training_objective_final'] <= fit['training_objective_initial']+1e-10
        counts = Counter()
        check_capped_warm_fit(A, y, {'features': [0, 1, 2], 'anchor': anchor.tolist(), 'target': target.tolist(), 'fit': fit}, counts)
        assert counts['independent_master_primal_dual_replays'] >= 1


def test_invalid_warm_start_rejected_instead_of_silent_rescaling():
    rng = np.random.default_rng(781); A = rng.uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    anchor = np.ones(3)/np.sqrt(3)
    for invalid in (anchor*1.1, [1., -1., 1.], [np.nan, .5, .5], [np.sqrt(3), 0., 0.]):
        with pytest.raises(ValueError, match='initialization'):
            fit_capped_frs_warm_target(A, y, [0, 1, 2], anchor, full_reference=anchor,
                reference_scope='original_full', initialization=invalid)
