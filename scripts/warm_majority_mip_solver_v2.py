"""Verified HiGHS import including ONLY roundoff-level coefficient warnings.

The failed original solver is preserved. A backend warning is not success unless
costs/bounds/integrality and the complete sparse matrix survive roundtrip, except
coefficient differences<=1e-12. Original certificate and primal replay remain
mandatory; a solver-reported bound is not an original-model optimality proof.
"""
import numpy as np
import highspy
from scipy.sparse import csr_matrix,csc_matrix,vstack
from scipy.optimize import OptimizeResult
from warm_majority_mip_solver import warm_start_values


def import_verified_model(h,lp,expected_matrix):
    status=h.passModel(lp)
    if status not in (highspy.HighsStatus.kOk,highspy.HighsStatus.kWarning):
        raise RuntimeError(f'HiGHS model import failed: {status}')
    imported=h.getLp()
    assert imported.num_col_==lp.num_col_ and imported.num_row_==lp.num_row_
    for key in ('col_cost_','col_lower_','col_upper_','row_lower_','row_upper_'):
        np.testing.assert_array_equal(np.asarray(getattr(imported,key)),np.asarray(getattr(lp,key)),err_msg=key)
    assert list(imported.integrality_)==list(lp.integrality_)
    a=imported.a_matrix_;shape=expected_matrix.shape
    if a.format_==highspy.MatrixFormat.kRowwise:
        matrix=csr_matrix((a.value_,a.index_,a.start_),shape=shape)
    elif a.format_==highspy.MatrixFormat.kColwise:
        matrix=csc_matrix((a.value_,a.index_,a.start_),shape=shape).tocsr()
    else:raise RuntimeError('unsupported backend sparse matrix layout')
    difference=(expected_matrix-matrix).tocsr();difference.eliminate_zeros()
    maximum=0. if not difference.nnz else float(abs(difference.data).max())
    if maximum>1e-12:
        raise RuntimeError(f'backend coefficient change {maximum} exceeds explicit roundoff-only import allowance')
    return {'status':str(status),'matrix_roundtrip_max_abs_difference':maximum,
        'changed_nonzero_coefficients':difference.nnz,'roundoff_import_allowance':1e-12,
        'cost_bounds_integrality_unchanged':True,'rows':shape[0],'columns':shape[1],
        'original_model_optimality_not_proven_from_backend_bound':True}


def solve_warm_mip(A,fs,seed,seed_report,declarations,objective,matrix,b,eq,lower,upper,integrality,time_limit):
    M=vstack((matrix,eq),format='csr')
    lp=highspy.HighsLp();lp.num_col_=len(objective);lp.num_row_=M.shape[0]
    lp.col_cost_=objective;lp.col_lower_=np.asarray(lower);lp.col_upper_=np.asarray(upper)
    lp.row_lower_=np.r_[np.full(len(b),-np.inf),1.];lp.row_upper_=np.r_[b,1.]
    lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_=M.indptr;lp.a_matrix_.index_=M.indices;lp.a_matrix_.value_=M.data
    lp.integrality_=[highspy.HighsVarType.kInteger if value else highspy.HighsVarType.kContinuous for value in integrality]
    h=highspy.Highs();h.setOptionValue('output_flag',False)
    for k,v in {'time_limit':float(time_limit),'mip_rel_gap':.01,'threads':1,'random_seed':0}.items():
        assert h.setOptionValue(k,v)==highspy.HighsStatus.kOk
    imported=import_verified_model(h,lp,M)
    ids,vals=warm_start_values(A,fs,seed,seed_report,declarations)
    admitted=h.setSolution(len(ids),ids,vals);assert admitted==highspy.HighsStatus.kOk
    run_status=h.run();info=h.getInfo();state=h.getModelStatus();solution=h.getSolution()
    feasible=solution.value_valid and info.primal_solution_status==highspy.SolutionStatus.kSolutionStatusFeasible
    x=np.asarray(solution.col_value) if feasible else None
    status=0 if state==highspy.HighsModelStatus.kOptimal else 1 if state in (
        highspy.HighsModelStatus.kTimeLimit,highspy.HighsModelStatus.kIterationLimit,
        highspy.HighsModelStatus.kSolutionLimit,highspy.HighsModelStatus.kObjectiveBound) else 2
    result=OptimizeResult(x=x,fun=None if x is None else float(objective@x),status=status,message=h.modelStatusToString(state),
        mip_gap=None if x is None else info.mip_gap,mip_dual_bound=info.mip_dual_bound)
    record={'solver':'highspy','version':h.version(),'threads':1,'random_seed':0,'model_import':imported,
        'seed_supplied_before_run':True,'partial_seed_indices':ids.tolist(),'partial_seed_values':vals.tolist(),
        'set_solution_status':str(admitted),'run_status':str(run_status),'model_status':str(state),
        'seed_is_known_feasible_partial_assignment_not_test_label_tuning':True}
    return result,record
