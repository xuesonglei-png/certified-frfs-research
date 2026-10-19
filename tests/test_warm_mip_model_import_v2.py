import numpy as np
import highspy
import pytest
from scipy.sparse import csr_matrix
from warm_majority_mip_solver_v2 import import_verified_model


def instance(coefficient):
    matrix=csr_matrix([[1.,coefficient],[0.,1.]])
    lp=highspy.HighsLp();lp.num_col_=2;lp.num_row_=2
    lp.col_cost_=[1.,2.];lp.col_lower_=[0.,0.];lp.col_upper_=[1.,1.]
    lp.row_lower_=[-np.inf,-np.inf];lp.row_upper_=[1.,1.]
    lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_=matrix.indptr;lp.a_matrix_.index_=matrix.indices;lp.a_matrix_.value_=matrix.data
    lp.integrality_=[highspy.HighsVarType.kContinuous,highspy.HighsVarType.kInteger]
    h=highspy.Highs();h.setOptionValue('output_flag',False)
    return h,lp,matrix


def test_roundoff_warning_requires_explicit_matrix_and_bounds_roundtrip():
    h,lp,matrix=instance(4e-16);record=import_verified_model(h,lp,matrix)
    assert record['status']=='HighsStatus.kWarning'
    assert record['matrix_roundtrip_max_abs_difference']==4e-16
    assert record['changed_nonzero_coefficients']==1 and record['cost_bounds_integrality_unchanged']


def test_larger_discard_warning_is_rejected_and_ordinary_import_unchanged():
    h,lp,matrix=instance(2e-10)
    with pytest.raises(RuntimeError,match='exceeds explicit roundoff'):import_verified_model(h,lp,matrix)
    h,lp,matrix=instance(.02);record=import_verified_model(h,lp,matrix)
    assert record['status']=='HighsStatus.kOk' and record['matrix_roundtrip_max_abs_difference']==0.
