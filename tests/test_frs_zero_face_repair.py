from collections import Counter
import copy
import numpy as np
import pytest
from frs_coordinate_minimum import solve_diagnostic
from soft_topk_predictive_metric import anchor_witness_polytope
from frs_pair_transition_cell import build_class_cell
from frs_zero_face_repair import repair_seed
from audit_frs_zero_face_repair import check_seed


def fixture():
    A = np.array([[x, c] for c in (0., 1.) for x in (0., 0., .1, .1, .2, .2)])
    y = np.repeat([0, 1], 6); anchor = np.ones(2)/np.sqrt(2)
    diag = solve_diagnostic(A, y, [0, 1], anchor); info = diag['geometry']
    C, rhs, _ = anchor_witness_polytope(A, y, [0, 1], anchor)
    SC, SR, _ = build_class_cell(A, y, [0, 1], anchor, anchor)
    return info, C, rhs, SC, SR, diag


def test_projection_keeps_zero_and_original_constraints_without_rescaling_or_relaxation():
    info, C, rhs, SC, SR, diag = fixture()
    old = copy.deepcopy(diag['regimes']['pair_transition']['coordinates'][0])
    old['candidate_weights'][1] -= .05  # Hypothetical invalid mass; not a real-data LP artifact.
    old['accepted_individual_zero_witness'] = False
    out = repair_seed(info, C, rhs, SC, SR, 'pair_transition', old)
    assert out['repaired_accepted'] and out['target'][0] == 0
    assert abs(sum(out['target'])-info['mass']) < 1e-8
    counts = Counter(); check_seed(info, C, rhs, SC, SR, 'pair_transition', old, out, counts)
    assert counts['independent_zero_repair_LP_duals'] == 2
    corrupted = copy.deepcopy(out); corrupted['conservative_buffer'] = 0.
    with pytest.raises(AssertionError):
        check_seed(info, C, rhs, SC, SR, 'pair_transition', old, corrupted, Counter())


def test_infeasible_zero_face_is_not_reported_as_a_repaired_certificate():
    info, C, rhs, SC, SR, diag = fixture()
    row = {'success':True, 'coordinate':1, 'feature':1, 'exact_zero_candidate':True,
        'accepted_individual_zero_witness':False, 'candidate_weights':[np.sqrt(2), 0.]}
    out = repair_seed(info, C, rhs, SC, SR, 'pair_transition', row)
    assert not out['repaired_accepted'] and out['target'] is None
    assert not out['margin_master']['success']
    check_seed(info, C, rhs, SC, SR, 'pair_transition', row, out, Counter())
