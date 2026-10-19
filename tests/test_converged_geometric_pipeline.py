from collections import Counter
import numpy as np
from geometric_vote_evaluation import evaluate,final_evaluation
from independent_geometric_evaluation import independent_evaluate,replay_final
from converged_crossgroup_stage import PredictiveMetricStage,GRID
from audit_converged_crossgroup_stage import check_stage,check_metric
from audit_forest_compact_transfer import equivalent


def test_ties_metrics_certificate_and_shared_noise_independently_replay():
    A=np.array([[.01,.03],[.02,.02],[.03,.01],[.8,.7],[.81,.72],[.79,.73],
                [.49,.5],[.5,.49],[.51,.51]])
    y=np.array([8,8,8,1,1,1,7,7,7]);B=np.array([[.02,.02],[.8,.72],[.5,.5]])
    truth=np.array([8,1,7]);metric={'features':[0,1],'weights':[.3,1.1]}
    equivalent(evaluate(A,y,B,truth,metric),independent_evaluate(A,y,B,truth,metric))
    final={'metric':metric,**final_evaluation(A,y,B,truth,metric)};counts=Counter()
    replay_final(A,y,B,truth,final,counts)
    assert counts['independent_noise_predictions']==16
    assert final['query_certificate']['certified']==[True,True,True]


def test_empty_prior_class_recoding_independent_replay():
    A=np.zeros((6,2));y=np.array([8,8,1,1,7,7]);B=np.zeros((3,2));truth=np.array([8,1,7])
    metric={'features':[],'weights':[]};record={'metric':metric,**final_evaluation(A,y,B,truth,metric)}
    replay_final(A,y,B,truth,record,Counter());assert record['evaluation']['predictions']==[8]*3
    recoded=np.array([100 if c==8 else 0 if c==1 else 3 for c in y])
    assert evaluate(A,recoded,B,np.array([100,0,3]),metric)['predictions']==[100]*3


def test_uniform1000_targets_guards_and_interpolations_independently_replay():
    rng=np.random.default_rng(67)
    A=np.vstack([rng.uniform(.02,.12,(9,2)),rng.uniform(.78,.91,(9,2))]);y=np.repeat([0,1],9)
    base={'features':[0,1],'weights':(np.ones(2)/np.sqrt(2)).tolist(),'origin':'smoke_full'}
    stage=PredictiveMetricStage(A,y)
    metrics=[stage.metric(base,i,certified=True) for i in range(len(GRID))];export=stage.export()
    counts=Counter();check_stage(A,y,export,counts)
    for i,metric in enumerate(metrics):check_metric(metric,base,export,'proposed',i,counts)
    assert all(r['fit']['max_iter']==1000 for r in export['learners'])
    assert counts['independent_target_objective_replays']==2
