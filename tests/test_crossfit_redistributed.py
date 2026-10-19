import numpy as np
from certified_frfs.redistributed_search import CrossfitRedistributedUtilityWorkspace,redistributed_path
from certified_frfs.redistributed_loss import transfer_weights
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score


def test_crossfit_transfer_uses_disjoint_folds_and_native_subset_weights():
    rng=np.random.default_rng(930);X=rng.uniform(0,1,(30,4));y=np.repeat([0,1,2],10);w=np.ones(4)/2
    ws=CrossfitRedistributedUtilityWorkspace(X,y);s=ws.evaluate([0,2])
    assert s['score_scope'].startswith('adaptive_training')
    assert len(s['reduced_weights'])==2
    for split in ws.splits:assert not set(split['train'])&set(split['validation'])
    values=[]
    for index,(A,B,yt,yv,wf) in enumerate(ws.folds):
        wr=transfer_weights(A,[0,2],wf);rng=np.random.default_rng(144+index)
        Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1);Q=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
        scores=[]
        for ref,query in [(A,B),(Z,Q)]:
            model=KNeighborsClassifier(n_neighbors=min(5,len(yt)),metric='manhattan',algorithm='brute')
            model.fit(ref[:,[0,2]]*wr,yt);scores.append(balanced_accuracy_score(yv,model.predict(query[:,[0,2]]*wr)))
        values.append(scores)
    np.testing.assert_allclose([s['clean_LOO'],s['shared_noise_LOO']],np.mean(values,axis=0),atol=1e-12)


def test_softvote_fits_only_subfold_reference_labels(monkeypatch):
    calls=[]
    def fit(A,y):
        calls.append((A.copy(),y.copy()));return np.ones(A.shape[1])/np.sqrt(A.shape[1])
    monkeypatch.setattr('certified_frfs.redistributed_search.learn_softvote_weights',fit)
    X=np.arange(120).reshape(30,4);y=np.repeat([0,1,2],10)
    ws=CrossfitRedistributedUtilityWorkspace(X,y,family='softvote')
    assert len(calls)==3
    for (_,labels),split in zip(calls,ws.splits):
        np.testing.assert_array_equal(labels,y[split['train']])
        assert len(labels)<len(y)


def test_crossfit_workspace_can_rank_certified_path():
    y=np.repeat([0,1],12);X=np.column_stack([.1+.8*y,np.linspace(.1,.9,24),np.linspace(.9,.1,24)]);w=np.ones(3)/np.sqrt(3)
    ws=CrossfitRedistributedUtilityWorkspace(X,y);path=redistributed_path(X,y,w,ranking_workspace=ws)
    assert path['ranking_scope'].startswith('adaptive_training')
    assert path['candidates']
