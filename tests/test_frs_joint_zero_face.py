import copy
from collections import Counter
import numpy as np
import pytest
from frs_coordinate_minimum import geometry
from frs_joint_zero_face import eliminate, solve_face
from audit_frs_joint_zero_face import check_elimination, check_face


def test_simultaneous_zeros_stay_zero_and_pass_original_constraints_and_duals():
    A = np.array([[x,x,c] for c in (0.,1.) for x in (0.,0.,.1,.1,.2,.2)])
    y = np.repeat([0,1],6); anchor = np.ones(3)/np.sqrt(3)
    info,C,rhs,SC,SR = geometry(A,y,[0,1,2],anchor)
    result = eliminate(info,C,rhs,SC,SR,'pair_transition')
    assert result['exact_zero_indices'] == [0,1] and result['exact_support_features'] == [2]
    counts = Counter(); check_elimination(info,C,rhs,SC,SR,'pair_transition',result,counts)
    assert counts['accepted_joint_zero_faces'] == 2
    assert counts['independent_zero_repair_LP_duals'] >= 4
    corrupt = copy.deepcopy(result); corrupt['steps'][1]['zero_indices'] = [1]
    with pytest.raises(AssertionError):
        check_elimination(info,C,rhs,SC,SR,'pair_transition',corrupt,Counter())


def test_failed_joint_deletion_does_not_replace_a_feasible_previous_point():
    A = np.array([[x,c] for c in (0.,1.) for x in (0.,0.,.1,.1,.2,.2)])
    y = np.repeat([0,1],6); anchor = np.ones(2)/np.sqrt(2)
    info,C,rhs,SC,SR = geometry(A,y,[0,1],anchor)
    out = solve_face(info,C,rhs,SC,SR,'pair_transition',[0,1],anchor)
    assert not out['accepted'] and out['target'] is None
    check_face(info,C,rhs,SC,SR,'pair_transition',[0,1],anchor.tolist(),out,Counter())
