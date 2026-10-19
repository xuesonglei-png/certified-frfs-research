from collections import Counter
import numpy as np
from fit_robust_majority_hinge import fit_majority_hinge_target
from audit_majority_hinge_fit import check_fit
from majority_hinge_stage import PredictiveMetricStage,GRID
from audit_predictive_target_knn_pilot import check_stage


def test_numeric_dual_and_scalar_faces_replay_random_nontrivial_optimum():
    rng=np.random.default_rng(51);A=rng.uniform(0,1,(18,3));y=np.repeat([0,1],9);anchor=np.ones(3)/np.sqrt(3)
    target,fit=fit_majority_hinge_target(A,y,[0,1,2],anchor);counts=Counter()
    check_fit(A,y,{'features':[0,1,2],'anchor':anchor.tolist(),'target':target.tolist(),'fit':fit},counts)
    assert counts['independent_master_primal_dual_replays']==1 and counts['target_optimizer_failures']==0


def test_both_families_preserve_original_guards_and_independent_polytope():
    rng=np.random.default_rng(67);A=np.vstack([rng.uniform(.02,.12,(9,2)),rng.uniform(.78,.91,(9,2))]);y=np.repeat([0,1],9)
    base={'features':[0,1],'weights':(np.ones(2)/np.sqrt(2)).tolist(),'origin':'synthetic_full'}
    stage=PredictiveMetricStage(A,y);metrics=[stage.metric(base,i,certified=True) for i in range(len(GRID))];export=stage.export();counts=Counter()
    check_stage(A,y,{'learners':[],'certificate_checks':export['certificate_checks']},counts)
    for r in export['learners']:check_fit(A,y,r,counts)
    assert all(export['certificate_checks'][m['deployed_guard_id']]['accepted'] for m in metrics)
    assert {r['family'] for r in export['learners']}=={'majority_hinge_unconstrained','majority_hinge_constrained'}
