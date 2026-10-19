import numpy as np
from certified_frfs.redistributed_search import redistributed_path,redistributed_candidates,RedistributedUtilityWorkspace
from certified_frfs.redistributed_loss import transfer_weights,redistributed_loss_bound
from certified_frfs.validated_pool import validation_scores
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score


def test_duplicate_path_can_delete_redundant_attribute_and_keep_certificate():
    t=np.r_[np.linspace(.05,.15,8),np.linspace(.85,.95,8)];X=np.column_stack([t,t]);y=np.repeat([0,1],8);w=np.ones(2)/np.sqrt(2)
    path=redistributed_path(X,y,w,dependency_eta=.1)
    assert path['candidates'][-1]==[0] or path['candidates'][-1]==[1]
    for fs in path['candidates']:
        wr=transfer_weights(X,fs,w)
        assert redistributed_loss_bound(X,y,fs,w,wr)['upper_bound']<=.1


def test_subset_scores_use_transferred_weights_and_preserve_mass():
    rng=np.random.default_rng(922);X=rng.uniform(0,1,(20,4));y=np.repeat([0,1],10);w=np.ones(4)/2
    ws=RedistributedUtilityWorkspace(X,y,w);score=ws.evaluate([0,1]);assert np.allclose(np.sum(score['reduced_weights']),np.sum(w))
    assert score['score_scope'].startswith('training')
    predictions=[];fs=[0,1];wr=transfer_weights(X,fs,w)
    for i in range(len(y)):
        ids=np.arange(len(y))!=i
        model=KNeighborsClassifier(n_neighbors=5,metric='manhattan',algorithm='brute')
        model.fit(X[ids][:,fs]*wr,y[ids]);predictions.append(model.predict(X[[i]][:,fs]*wr)[0])
    assert abs(score['clean_LOO']-balanced_accuracy_score(y,predictions))<1e-12


def test_candidates_have_reduced_weight_records():
    X=np.tile([[.1,.2,.3],[.9,.8,.7]],(8,1));y=np.repeat([0,1],8);w=np.ones(3)/np.sqrt(3)
    rows,path=redistributed_candidates(X,y,w);assert len(rows)==6
    for row in rows:
        assert len(row['reduced_weights'])==len(row['features']) or len(row['features'])==3
