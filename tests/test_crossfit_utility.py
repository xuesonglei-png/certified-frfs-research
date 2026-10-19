import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from certified_frfs.crossfit_utility import CrossfitUtilityWorkspace
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.utility_search import utility_path


def test_crossfit_matches_independent_classifiers_and_fold_local_scaling():
    rng=np.random.default_rng(901);X=rng.normal(size=(30,4));X[0,0]=500
    X[1,2]=np.nan;y=np.repeat([0,1,2],10);fs=[0,2]
    ws=CrossfitUtilityWorkspace(X,y);scores=[]
    for index,s in enumerate(ws.splits):
        tr=np.array(s['train']);va=np.array(s['validation'])
        assert not np.intersect1d(tr,va).size
        A,B=preprocess_fit(X[tr],X[va]);w=np.array(s['weights'])
        rng=np.random.default_rng(144+index)
        Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
        Q=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
        values=[]
        for reference,query in [(A,B),(Z,Q)]:
            model=KNeighborsClassifier(n_neighbors=5,metric='manhattan',algorithm='brute')
            model.fit(reference[:,fs]*w[fs],y[tr])
            values.append(balanced_accuracy_score(y[va],model.predict(query[:,fs]*w[fs])))
        scores.append(values)
    result=ws.evaluate(fs)
    np.testing.assert_allclose([result['clean_LOO'],result['shared_noise_LOO']],np.mean(scores,axis=0),atol=1e-12)


def test_singletons_remain_reference_only_and_invalid_families_fail():
    X=np.arange(36).reshape(12,3);y=np.array([0]*5+[1]*6+[2])
    ws=CrossfitUtilityWorkspace(X,y)
    for s in ws.splits:
        assert 11 in s['train'] and 11 not in s['validation']
    assert np.isfinite(ws.evaluate([0])['utility'])
    with pytest.raises(ValueError):CrossfitUtilityWorkspace(X,y,family='typo')


def test_crossfit_ranking_keeps_dual_certificates():
    rng=np.random.default_rng(902);y=np.repeat([0,1],12)
    X=np.column_stack([.1+.8*y,rng.uniform(.1,.9,(24,3))]);w=np.array([1.,.05,.05,.05])
    ws=CrossfitUtilityWorkspace(X,y)
    result=utility_path(X,y,w,dependency_eta=.1,shared_loss=True,ranking_workspace=ws)
    assert result['candidates'][-1]==[0]
    assert result['ranking_splits']==ws.splits
    for step in result['trace']:
        chosen=next((a for a in step['certification_attempts'] if a['feasible']),None)
        if chosen is not None:assert chosen['dependency_certificate']['upper_bound']<=.1
