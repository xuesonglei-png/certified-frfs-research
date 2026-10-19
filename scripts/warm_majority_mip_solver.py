"""Sparse known-feasible start for the unchanged direct-majority MIP."""
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.sparse import vstack
import highspy


def warm_start_values(A,fs,seed,seed_report,declarations):
    p,n,mass=len(fs),len(A),float(np.sqrt(A.shape[1]));v=np.asarray(seed)/mass
    values={j:float(x) for j,x in enumerate(v)}
    for i,(row,decl) in enumerate(zip(seed_report['rows'],declarations,strict=True)):
        success=row['qualified'];values[p+i]=float(success)
        friends=set(row['three_friends']) if success else set()
        enemies=set(row['two_allowed_enemies']) if success else set()
        values[decl['threshold_index']]=0. if not success else max(float(abs(A[f,fs]-A[i,fs])@v) for f in friends)
        for f,j in zip(decl['friends'],decl['friend_selection_indices'],strict=True):values[j]=float(f in friends)
        for e,j in zip(decl['enemies'],decl['enemy_exception_indices'],strict=True):values[j]=float(e in enemies)
    ids=np.asarray(sorted(values),dtype=np.int32);vals=np.asarray([values[j] for j in ids])
    return ids,vals


def solve_warm_mip(A,fs,seed,seed_report,declarations,objective,matrix,b,eq,lower,upper,integrality,time_limit):
    M=vstack((matrix,eq),format='csr');row_lower=np.r_[np.full(len(b),-np.inf),1.];row_upper=np.r_[b,1.]
    lp=highspy.HighsLp();lp.num_col_=len(objective);lp.num_row_=M.shape[0]
    lp.col_cost_=objective;lp.col_lower_=np.asarray(lower);lp.col_upper_=np.asarray(upper)
    lp.row_lower_=row_lower;lp.row_upper_=row_upper
    lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_=M.indptr;lp.a_matrix_.index_=M.indices;lp.a_matrix_.value_=M.data
    lp.integrality_=[highspy.HighsVarType.kInteger if value else highspy.HighsVarType.kContinuous for value in integrality]
    h=highspy.Highs();h.setOptionValue('output_flag',False)
    for k,v in {'time_limit':float(time_limit),'mip_rel_gap':.01,'threads':1,'random_seed':0}.items():
        assert h.setOptionValue(k,v)==highspy.HighsStatus.kOk
    assert h.passModel(lp)==highspy.HighsStatus.kOk
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
    record={'solver':'highspy','version':h.version(),'threads':1,'random_seed':0,
        'seed_supplied_before_run':True,'partial_seed_indices':ids.tolist(),'partial_seed_values':vals.tolist(),
        'set_solution_status':str(admitted),'run_status':str(run_status),'model_status':str(state),
        'seed_is_known_feasible_partial_assignment_not_test_label_tuning':True}
    return result,record
