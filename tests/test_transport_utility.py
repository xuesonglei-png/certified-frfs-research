import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from certified_frfs.transport_utility import CrossfitTransportUtility


def test_transport_scores_match_independent_query_classifier():
    rng=np.random.default_rng(964);X=rng.uniform(0,1,(30,4));y=np.repeat([0,1,2],10);fs=[0,2]
    ws=CrossfitTransportUtility(X,y);actual=ws.evaluate(fs);scores=[]
    for index,((A,B,yt,yv,w),record) in enumerate(zip(ws.folds,ws.fit_records[tuple(fs)])):
        v=np.array(record['weights']);rng=np.random.default_rng(144+index)
        Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1);Q=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
        values=[]
        for ref,query in [(A,B),(Z,Q)]:
            m=KNeighborsClassifier(n_neighbors=5,metric='manhattan',algorithm='brute')
            m.fit(ref[:,fs]*v,yt);values.append(balanced_accuracy_score(yv,m.predict(query[:,fs]*v)))
        scores.append(values)
        assert record['fit']['regularized_objective_after']<=record['fit']['regularized_objective_before']+1e-12
    np.testing.assert_allclose([actual['clean_LOO'],actual['shared_noise_LOO']],np.mean(scores,axis=0),atol=1e-12)
    assert ws.evaluate(fs) is actual
