from collections import Counter
import copy
import numpy as np
import pytest
from frs_coordinate_minimum import solve_diagnostic
from fit_pair_capped_frs_hinge import fit_pair_capped_frs_target
from sparse_global_frs_training_ablation import fit_block, ARMS, STARTS
from audit_sparse_global_training_block import audit_block


def test_matched_three_arms_three_common_starts_with_independent_fits_and_no_validation_api():
    A = np.random.default_rng(781).uniform(0, 1, (24, 3)); y = np.repeat([0, 1], 12)
    fs = [0, 1, 2]; anchor = np.ones(3)/np.sqrt(3)
    diagnostic = solve_diagnostic(A, y, fs, anchor)
    warm, _ = fit_pair_capped_frs_target(A, y, fs, anchor, full_reference=anchor, reference_scope='original_full')
    block = fit_block(A, y, fs, anchor, warm, diagnostic)
    counts = Counter(); audit_block(A, y, block, diagnostic, counts)
    assert counts['independent_dc_fit_replays'] == 9 and counts['target_optimizer_failures'] == 0
    assert all(tuple(block['arms'][a]['fits']) == STARTS for a in ARMS)
    assert not block['evaluation']['classification_scored']
    altered = copy.deepcopy(block); altered['starts']['common_zero'][0] += .1
    with pytest.raises(AssertionError):
        audit_block(A, y, altered, diagnostic, Counter())


def test_zero_candidate_start_has_same_budget_for_each_arm_and_cannot_be_hidden():
    A = np.array([[x, c] for c in (0., 1.) for x in (0., 0., .1, .1, .2, .2)])
    y = np.repeat([0, 1], 6); anchor = np.ones(2)/np.sqrt(2)
    diagnostic = solve_diagnostic(A, y, [0, 1], anchor)
    block = fit_block(A, y, [0, 1], anchor, anchor, diagnostic)
    assert block['common_zero_feature'] == 0
    assert block['starts']['common_zero'][0] == 0
    for arm in ARMS:
        assert block['arms'][arm]['fits']['common_zero']['fit']['requested_initial_weights'] == block['starts']['common_zero']
    counts = Counter(); audit_block(A, y, block, diagnostic, counts)
    assert counts['independent_dc_fit_replays'] == 9
