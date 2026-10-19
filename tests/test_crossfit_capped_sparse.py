import numpy as np
import certified_frfs.crossfit_capped_sparse as module


def test_query_disjoint_and_complete_fold_reference(monkeypatch):
    X=np.column_stack([np.arange(13),np.arange(13)**2]).astype(float)
    y=np.array([0]*6+[1]*6+[2])
    calls=[]
    def trajectory(table,labels,**kwargs):
        calls.append((table[:,0].astype(int).tolist(),labels.tolist(),kwargs['steps']))
        return {'table':table}
    monkeypatch.setattr(module,'fit_feasible_query',trajectory)
    monkeypatch.setattr(module,'certify_capped_trajectory',lambda table,labels,fit:fit)
    monkeypatch.setattr(module,'sparse_trajectory_extension',lambda table,labels,fit:{'compact_selected':{
        'features':[0,1],'weights':(np.ones(2)/np.sqrt(2)).tolist(),'certificate':{'feasible':True}}})
    monkeypatch.setattr(module,'weight_certificate',lambda *args,**kwargs:{'feasible':True})
    def scores(A,labels,B,truth,fs,w,seed,**kwargs):
        assert len(A)==len(labels) and len(B)==len(truth)
        assert labels[-1]==2
        return {'clean':.5,'noise01':.5}
    monkeypatch.setattr(module,'validation_scores',scores)
    fit=module.fit_crossfit_capped_sparse(X,y)
    assert fit['chosen_recipe']=='full'
    assert len(calls)==6
    for recipe in fit['summary']:
        assert len(recipe['folds'])==3
        query_ids=[]
        for row in recipe['folds']:
            assert set(row['train_indices']).isdisjoint(row['query_indices'])
            assert set(row['train_indices']).union(row['query_indices'])==set(range(13))
            assert 12 in row['train_indices']
            query_ids.extend(row['query_indices'])
        assert sorted(query_ids)==list(range(12))
    for index,call in enumerate(calls):
        assert call[0]==fit['summary'][1]['folds'][index//2]['train_indices']
