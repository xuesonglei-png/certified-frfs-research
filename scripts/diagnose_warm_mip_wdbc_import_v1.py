"""Replay failed WDBC model import only; never solve, fit or score predictions."""
import json
import time
from pathlib import Path
import numpy as np
import highspy
from scipy.sparse import vstack
from threadpoolctl import threadpool_limits
import fit_warm_majority_face_milp as target
from run_plurality_certificate_development_v1 import ROOT,PARENTS,inputs,digest
from continue_soft_topk_knn_pilot import verify

OUT=ROOT/'research/warm-wdbc-model-import-v1.json'
FREEZE=ROOT/'research/warm-wdbc-model-import-v1-source-freeze.json'


def main():
    threadpool_limits(1);assert not OUT.exists() and not FREEZE.exists()
    failure=json.loads((ROOT/'results/warm-majority-frs-development-v2-chain-status.json').read_text())
    assert failure['phase']=='failed'
    parents={n:verify(p) for n,p in PARENTS.items()}
    paths=[ROOT/'scripts/diagnose_warm_mip_wdbc_import_v1.py',ROOT/'scripts/fit_warm_majority_face_milp.py',
        ROOT/'scripts/warm_majority_mip_solver.py',ROOT/'research/warm-majority-frs-development-v2-source-freeze.json']
    FREEZE.write_text(json.dumps({'created_unix':time.time(),'fingerprints':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        'failed_status':failure,'scope':'wdbc_same_model_backend_import_only_no_solve'},indent=2),encoding='utf-8')
    source,_,A,y,_,_,hashes=inputs('wdbc',{'parents':parents})
    records=[]
    class ImportedOnly(Exception):pass
    def inspect(A,fs,seed,seed_report,declarations,objective,matrix,b,eq,lower,upper,integrality,time_limit):
        M=vstack((matrix,eq),format='csr')
        lp=highspy.HighsLp();lp.num_col_=len(objective);lp.num_row_=M.shape[0]
        lp.col_cost_=objective;lp.col_lower_=np.asarray(lower);lp.col_upper_=np.asarray(upper)
        lp.row_lower_=np.r_[np.full(len(b),-np.inf),1.];lp.row_upper_=np.r_[b,1.]
        lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_=M.indptr;lp.a_matrix_.index_=M.indices;lp.a_matrix_.value_=M.data
        lp.integrality_=[highspy.HighsVarType.kInteger if value else highspy.HighsVarType.kContinuous for value in integrality]
        h=highspy.Highs();h.setOptionValue('output_flag',True);h.setOptionValue('threads',1)
        result=h.passModel(lp)
        records.append({'backend':h.version(),'pass_model_status':str(result),'rows':M.shape[0],
            'columns':M.shape[1],'nonzeros_input':len(M.data),'minimum_nonzero_absolute_coefficient':float(abs(M.data[M.data!=0]).min()),
            'input_coefficients_abs_at_most_1e9':int(np.sum(abs(M.data)<=1e-9)),
            'input_data_all_finite':bool(np.isfinite(M.data).all()),
            'objective_all_finite':bool(np.isfinite(objective).all()),
            'no_nan_bounds':bool(not np.isnan(lower).any() and not np.isnan(upper).any()),
            'inconsistent_column_bounds':int(np.sum(np.asarray(lower)>np.asarray(upper))),
            'imported_rows':h.getNumRow(),'imported_columns':h.getNumCol(),
            'nonzeros_imported':len(h.getLp().a_matrix_.value_),
            'no_solver_run_or_prediction':True})
        raise ImportedOnly
    target.solve_warm_mip=inspect
    try:
        target.fit_warm_majority_face_milp(A,y,source['original_support'],source['original_anchor'],
            full_reference=source['full_reference'],seed=source['original_anchor'],time_limit=120.)
    except ImportedOnly:pass
    assert len(records)==1
    out={'dataset':'wdbc','source_hashes':hashes,'records':records,'freeze_sha256':digest(FREEZE),
        'fitting_performed':False,'predictions_scored':False,'original_failed_status_preserved':True,
        'classification_superiority_established':False,'goal_achieved':False}
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':main()
