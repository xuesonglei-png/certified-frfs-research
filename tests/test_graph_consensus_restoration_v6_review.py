"""Behavioral guard-admission tests without solving the expensive certifier."""
import numpy as np
import repair_relation_selector_v4 as restoration


def test_restoration_stops_at_first_feasible_support_and_reuses_cache(monkeypatch):
    calls=[]
    def guard(A,y,weights,features,count):
        calls.append(tuple(features))
        return {'features':features,'weights':weights[features].tolist(),'joint_feasible':count>=3}
    monkeypatch.setattr(restoration,'_candidate',guard)
    A=np.zeros((4,4));y=np.array([0,0,1,1]);ref=np.ones(4)/2
    cache={};rank=[3,1,2,0]
    selected,trace=restoration.repaired(A,y,ref,rank,2,cache)
    assert selected['features']==[1,2,3] and selected['restored_count']==1
    assert [r['joint_feasible'] for r in trace]==[False,True]
    restoration.repaired(A,y,ref,rank,2,cache)
    assert calls==[(1,3),(1,2,3)]


def test_full_learned_guard_failure_has_explicit_uniform_fallback(monkeypatch):
    ref=np.array([.7,.6,.4,.3]);calls=[]
    def guard(A,y,weights,features,count):
        feasible=bool(count==4 and np.array_equal(weights,np.ones(4)/2))
        calls.append((list(features),feasible))
        return {'features':features,'weights':weights[features].tolist(),'joint_feasible':feasible}
    monkeypatch.setattr(restoration,'_candidate',guard)
    selected,trace=restoration.repaired(np.zeros((4,4)),np.array([0,0,1,1]),ref,[3,1,2,0],2,{})
    assert selected['features']==[0,1,2,3] and selected['uniform_fallback']
    assert selected['restored_count']==2
    assert [feasible for fs,feasible in calls]==[False,False,False,True]
    assert len(trace)==4
