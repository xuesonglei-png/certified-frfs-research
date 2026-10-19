"""Shared cache, binary budget and independently replayed reference provenance."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from thinned_atomic_classifier_stage import REFINED,PruningStage
from independent_thinned_atomic_pruning import check_thinned_atomic_pruning as check_pruning
from run_multiscale_certified_metric_knn_pool import evaluate


def test_all_weighted_arms_share_fit_and_independent_finite_replay():
    A=np.zeros((16,6)); y=np.repeat([0,1],8); A[:,0]=.1+.8*y
    base={'features':list(range(6)),'weights':(np.ones(6)/np.sqrt(6)).tolist()}
    stage=PruningStage(A,y)
    for method in REFINED:
        unchanged=stage.metric(base,0,require_original=method=='proposed')
        refined=stage.metric(base,1,require_original=method=='proposed')
        assert unchanged['features']==base['features'] and 0 in refined['features'] and len(refined['features'])==3
        assert refined['pruning_id']==0 and refined['guard_reference']=='original_full'
    assert len(stage.records)==1
    r=stage.records[0]; assert check_pruning(A,y,r['initial_features'],r['initial_weights'],r['fit'])==len(r['fit']['checks'])
    control={'features':[1,2],'weights':[np.sqrt(6)/2]*2}
    stage.metric(control,1); r=stage.records[1]
    assert r['fit']['guard_reference']=='own_anchor'
    check_pruning(A,y,r['initial_features'],r['initial_weights'],r['fit'])


def test_scoring_changed_query_truth_cannot_change_pruning_weights():
    A=np.zeros((16,6)); y=np.repeat([0,1],8); A[:,0]=.1+.8*y
    B=A[[0,8]].copy(); base={'features':list(range(6)),'weights':(np.ones(6)/np.sqrt(6)).tolist()}
    stage=PruningStage(A,y); before=stage.metric(base,1,require_original=True)
    one=evaluate(A,y,B,np.array([0,1]),before['features'],before['weights'])
    two=evaluate(A,y,B,np.array([1,0]),before['features'],before['weights'])
    assert one['metrics']!=two['metrics'] and one['predictions']==two['predictions']
    assert stage.metric(base,1,require_original=True)==before and len(stage.records)==1
