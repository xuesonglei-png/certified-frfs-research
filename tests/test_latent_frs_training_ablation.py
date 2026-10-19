from collections import Counter
import copy
import numpy as np
import pytest
from fit_strict_capped_frs_hinge import fit_strict_capped_frs_target
from latent_frs_training_ablation import fit_block
from audit_latent_frs_training_block import check_block


def test_matched_starts_strict_solver_cells_and_bound_decomposition():
    rng = np.random.default_rng(781)
    A = rng.uniform(0, 1, (24, 3)); y = np.repeat([0, 1], 12)
    anchor = np.ones(3)/np.sqrt(3); fs = [0, 1, 2]
    endpoint, _ = fit_strict_capped_frs_target(A, y, fs, anchor,
        full_reference=anchor, reference_scope='original_full')
    block = fit_block(A, y, fs, anchor, anchor, endpoint)
    counts = Counter(); check_block(A, y, block, counts)
    assert counts['independent_dc_fit_replays'] == 4
    assert counts['matched_training_blocks'] == 1
    for r in block['evaluation']['at_identical_weights'].values():
        assert r['latent_five_bound'] <= r['fixed_three_bound']+1e-10
    for arm in block['arms'].values():
        for r in arm['fits'].values():
            assert r['fit']['optimizer_success'], r['fit']['status']
    # A changed start cannot pass by merely editing a summary.
    forged = copy.deepcopy(block)
    forged['arms']['latent_five']['fits']['capped_endpoint']['fit']['requested_initial_weights'][0] += .01
    with pytest.raises(AssertionError):
        check_block(A, y, forged, Counter())
    forged = copy.deepcopy(block)
    forged['evaluation']['selected_training_bound_decrease_vs_fixed_three_pp'] += 1
    with pytest.raises(AssertionError):
        check_block(A, y, forged, Counter())
