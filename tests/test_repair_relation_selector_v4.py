import numpy as np
from repair_relation_selector_v4 import local_soft_relation,repaired,choose,FAMILIES,FRACTIONS
import repair_relation_selector_v4 as module
from certified_frfs.softvote_metric import softvote_terms,softvote_objective


def test_soft_gradient_and_capped_fit():
    X=np.random.default_rng(8).random((24,4)); y=np.repeat([0,1],12)
    terms=softvote_terms(X,y,eps=.01);w=np.array([.21,.24,.26,.29])
    loss,grad=softvote_objective(w,*terms,temperature=.12,reg=.02)
    for a in range(4):
        step=np.eye(4)[a]*1e-6
        numerical=(softvote_objective(w+step,*terms,temperature=.12,reg=.02)[0]-
                   softvote_objective(w-step,*terms,temperature=.12,reg=.02)[0])/2e-6
        assert np.isclose(grad[a],numerical,atol=1e-6)
    result,record=local_soft_relation(X,y)
    assert np.isclose(result.sum(),2.) and result.max()<=1.+1e-9
    assert record['objective_after']<=record['objective_before']+1e-10


def test_restoration_checks_every_count_and_explicit_fallback(monkeypatch):
    A=np.zeros((8,6));y=np.repeat([0,1],4);w=np.ones(6)/np.sqrt(6)
    calls=[]
    def candidate(A,y,w,fs,count):
        calls.append(count)
        return {'features':fs,'weights':w[:count].tolist(),'joint_feasible':count==5}
    monkeypatch.setattr(module,'_candidate',candidate)
    selected,trace=repaired(A,y,w,list(range(6)),3,{})
    assert calls==[3,4,5] and selected['restored_count']==2 and len(trace)==3
    def fallback(A,y,w,fs,count):
        return {'features':fs,'weights':w[:count].tolist(),
                'joint_feasible':count==6 and np.allclose(w,np.ones(6)/np.sqrt(6))}
    monkeypatch.setattr(module,'_candidate',fallback)
    selected,trace=repaired(A,y,w*np.array([1.1,1,1,1,1,.9]),list(range(6)),3,{})
    assert selected['uniform_fallback'] and len(trace)==5


def test_choice_uses_fold_aggregates_not_absolute_support_ids():
    blocks=[]
    for fold in range(3):
        families={}
        for family in FAMILIES:
            families[family]={'candidates':[{'score':{'clean_BA':.8,'query_certified_BA':.7,
                  'balanced_logloss':.4},'model':{'features':list(range(3+i))}}
                  for i in range(len(FRACTIONS))]}
        blocks.append({'families':families})
    options,selected=choose(blocks)
    assert selected['family']=='uniform_sqrt' and selected['recipe_index']==0
