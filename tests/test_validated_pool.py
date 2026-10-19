import numpy as np
from certified_frfs.validated_pool import select_validated_pool


def test_holdout_partition_final_metric_and_fresh_certificate():
    rng=np.random.default_rng(5)
    y=np.repeat([0,1],20)
    X=rng.uniform(0,.2,(40,3));X[y==1,0]+=.7
    result=select_validated_pool(X,y,seed=31)
    ii=result['discovery_indices'];jj=result['validation_indices']
    assert not set(ii)&set(jj)
    assert sorted(ii+jj)==list(range(40))
    assert np.isclose(np.linalg.norm(result['weights']),1)
    assert result['certificate']['upper_bound']<=.1+1e-10
    assert result['features']
    assert set(result['chosen']['features'])<=set(result['features'])


def test_preprocessing_fits_only_discovery(monkeypatch):
    import certified_frfs.validated_pool as module
    original=module.preprocess_fit; calls=[]
    def spy(reference,query):
        calls.append((len(reference),len(query)))
        return original(reference,query)
    monkeypatch.setattr(module,'preprocess_fit',spy)
    rng=np.random.default_rng(8);X=rng.random((40,3));y=np.repeat([0,1],20)
    module.select_validated_pool(X,y,method='all',seed=31)
    assert calls==[(28,12),(40,40)]


def test_joint_validation_refits_reference_and_preserves_nominal_score(monkeypatch):
    import certified_frfs.validated_pool as module
    rng=np.random.default_rng(10);A=rng.random((20,3));B=rng.random((8,3))
    y=np.repeat([0,1],10);truth=np.repeat([0,1],4);w=np.ones(3)/np.sqrt(3)
    original=module.KNeighborsClassifier.fit;seen=[]
    def spy(self,X,labels):
        seen.append(np.array(X));return original(self,X,labels)
    monkeypatch.setattr(module.KNeighborsClassifier,'fit',spy)
    joint=module.validation_scores(A,y,B,truth,[0,1],w,44,joint=True)
    assert len(seen)==4 and all(not np.array_equal(seen[0],row) for row in seen[1:])
    plain=module.validation_scores(A,y,B,truth,[0,1],w,44,joint=False)
    assert joint['clean']==plain['clean']
