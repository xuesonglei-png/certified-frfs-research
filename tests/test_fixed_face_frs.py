import copy
from collections import Counter
import numpy as np
import pytest
from fit_pair_capped_frs_hinge import fit_pair_capped_frs_target
from fit_fixed_face_frs import fit_fixed_face_frs_target
from audit_fixed_face_frs_fit import check_fixed_face_fit
from frs_pair_transition_metric_repair import repair_cell_metric
from soft_topk_predictive_metric import anchor_witness_polytope
from frs_pair_transition_cell import build_class_cell


def fixture():
    A=np.array([[x,c] for c in (0.,1.) for x in (0.,0.,.1,.1,.2,.2)])
    return A,np.repeat([0,1],6),np.ones(2)/np.sqrt(2)


def test_fixed_face_can_start_above_original_objective_without_restoring_deleted_features():
    A,y,anchor=fixture();seed=np.array([0.,np.sqrt(2)])
    for eps in (0.,.01):
        point,fit=fit_fixed_face_frs_target(A,y,[0,1],anchor,full_reference=anchor,
            reference_scope='original_full',initialization=seed,fixed_zero_indices=[0],optimization_eps=eps)
        assert fit['requested_initial_objective'] > fit['training_objective_initial']
        assert point[0]==0. and fit['optimization_start_weights']==seed.tolist()
        assert fit['training_objective_final'] <= fit['optimization_start_objective']
        assert fit['class_cell']['anchor']==anchor.tolist()
        record={'features':[0,1],'anchor':anchor.tolist(),'target':point.tolist(),'fit':fit}
        counts=Counter();check_fixed_face_fit(A,y,record,counts)
        assert counts['independent_master_primal_dual_replays']>0
        broken=copy.deepcopy(record);broken['fit']['fixed_zero_indices']=[1]
        with pytest.raises(AssertionError):
            check_fixed_face_fit(A,y,broken,Counter())


def test_face_seed_repair_preserves_zeros_after_actual_constraint_violation():
    A,y,anchor=fixture();seed=np.array([0.,np.sqrt(2)])
    C,rhs,_=anchor_witness_polytope(A,y,[0,1],anchor)
    SC,SR,cell=build_class_cell(A,y,[0,1],anchor,anchor)
    C=np.vstack((C,SC));rhs=np.r_[rhs,SR]
    # Deliberately violate linear/class constraints on the same face.
    candidate=np.zeros(2)
    point,trace,blend=repair_cell_metric(C,rhs,cell,candidate,seed)
    assert trace and blend>0. and point[0]==0. and np.all(C@point>=rhs)


def test_empty_face_robust_control_recovers_original_weights_and_objective():
    A,y,anchor=fixture();kw={'full_reference':anchor,'reference_scope':'original_full'}
    old,oldfit=fit_pair_capped_frs_target(A,y,[0,1],anchor,**kw)
    new,newfit=fit_fixed_face_frs_target(A,y,[0,1],anchor,**kw)
    np.testing.assert_array_equal(old,new)
    assert newfit['training_objective_final']==oldfit['training_objective_final']
    assert newfit['class_cell']==oldfit['class_cell']
    with pytest.raises(ValueError):
        fit_fixed_face_frs_target(A,y,[0,1],anchor,fixed_zero_indices=[0],**kw)
