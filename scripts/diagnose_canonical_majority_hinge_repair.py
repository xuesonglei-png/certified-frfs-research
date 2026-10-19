"""Two failed training fits, canonical V2, independent objectives and duals."""
import json
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from fit_robust_majority_hinge_v2 import fit_majority_hinge_target
from audit_majority_hinge_fit_cached import check_fit


def main():
    threadpool_limits(1);freeze=json.loads((ROOT/'research/majority-hinge-canonical-repair-source-freeze.json').read_text())
    for rel,sha in freeze['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    path=ROOT/'results/majority-hinge-inner-all15/glass.json';source=json.loads(path.read_text());X,y=load_dataset('glass')
    raw=X[source['reference_indices']];labels=y[source['reference_indices']];reports=[];counts=Counter()
    for bi,li in ((1,5),(2,3)):
        block=source['inner'][bi];old=block['stage']['learners'][li];assert not old['fit']['optimizer_success']
        ca=np.array(block['reference_indices']);A,_=preprocess_fit(raw[ca],raw[ca])
        target,fit=fit_majority_hinge_target(A,labels[ca],old['features'],old['anchor'],constrained=old['fit']['constrained'])
        fresh={'features':old['features'],'anchor':old['anchor'],'target':target.tolist(),'fit':fit,'family':old['family']}
        check_fit(A,labels[ca],fresh,counts)
        reports.append({'inner_fold':bi,'source_learner':li,'source_failed_status':old['fit']['status'],
            'source_last_trace':old['fit']['trace'][-1],'v2_fit':fresh,
            'validation_or_outer_evaluations_performed':False,'independent_training_replay_passed':True})
        print('Glass training-only fold',bi+1,'canonical repair',fit['status'],'success',fit['optimizer_success'],flush=True)
    report={'reports':reports,'counts':dict(counts),'fingerprints':freeze['fingerprints'],'source_v1_sha256':digest(path),
        'original_v1_failure_preserved':True,'validation_or_outer_evaluations_performed':False,'goal_achieved':False,
        'scope':'training_only_actual_coordinate_repair_diagnosis_no_accuracy_or_prediction_claim'}
    (ROOT/'results/majority-hinge-canonical-repair-diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(dict(counts),flush=True)


if __name__=='__main__':main()
