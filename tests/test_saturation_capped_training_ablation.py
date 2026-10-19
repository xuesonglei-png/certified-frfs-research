from collections import Counter
import copy
import numpy as np
import pytest
from saturation_capped_training_ablation import fit_block, ARMS, STARTS
from audit_pair_capped_frs_fit import check_pair_capped_fit
from audit_saturation_capped_frs_fit import check_saturation_capped_fit
from fit_pair_transition_frs_hinge import fit_pair_transition_target


def test_both_arms_both_starts_share_anchor_pool_regularizer_and_replay_duals():
    A = np.random.default_rng(781).uniform(0, 1, (24, 3)); y = np.repeat([0, 1], 12)
    anchor = np.ones(3)/np.sqrt(3)
    raw, _ = fit_pair_transition_target(A, y, [0, 1, 2], anchor,
        full_reference=anchor, reference_scope='original_full')
    warm = .8*anchor+.2*raw
    block = fit_block(A, y, [0, 1, 2], anchor, anchor, warm)
    counts = Counter()
    for arm, checker in zip(ARMS, (check_pair_capped_fit, check_saturation_capped_fit)):
        assert set(block['arms'][arm]['fits']) == set(STARTS)
        for start in STARTS:
            r = block['arms'][arm]['fits'][start]; checker(A, y, r, counts)
            assert r['fit']['optimizer_success']
            assert r['fit']['training_objective_final'] <= r['fit']['optimization_start_objective']+1e-10
            assert r['fit']['requested_initial_weights'] == (anchor if start == 'anchor' else warm).tolist()
            assert r['fit']['training_witnesses'] == block['arms'][ARMS[0]]['fits']['anchor']['fit']['training_witnesses']
        selected = min(STARTS, key=lambda s:(block['arms'][arm]['fits'][s]['fit']['training_objective_final'], STARTS.index(s)))
        assert block['arms'][arm]['selected_start'] == selected
    assert counts['independent_dc_fit_replays'] == 4
    tampered = copy.deepcopy(block['arms'][ARMS[1]]['fits']['capped_endpoint'])
    tampered['fit']['class_cell']['pair_transition_objects'][0]['anchor_membership_upper'] -= .1
    with pytest.raises(AssertionError):
        check_saturation_capped_fit(A, y, tampered, Counter())
