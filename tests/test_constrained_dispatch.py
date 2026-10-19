import numpy as np
import sys
from pathlib import Path
from certified_frfs.crossfit_search import select_crossfit_queries
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_decision_iteration import subset


def test_constraint_is_active_and_full_fallback_is_feasible():
    X=np.random.default_rng(4).uniform(.1,.9,(12,3)); y=np.repeat([0,1],6)
    result=select_crossfit_queries(X,y,mode='robust',constraint_tolerance=0.,budget=8)
    assert result['constraint_selection'] is not None
    assert result['constraint_selection']['eligible'] >= 1
    assert result['constraint_selection']['selected_predictive_score'] >= result['constraint_selection']['threshold']-1e-12


def test_runner_dispatches_constrained_modes_and_saves_traces(monkeypatch):
    calls=[]
    def fake_search(raw,y,**kwargs):
        calls.append(kwargs)
        return {'features':[1], 'constraint_selection':{'threshold':.8}}
    monkeypatch.setattr('certified_frfs.crossfit_search.select_crossfit_queries',fake_search)
    X=np.zeros((8,3)); y=np.repeat([0,1],4)
    for mode,name in [('robust','decision_robust_constrained'),('nominal','decision_nominal_constrained'),
                      ('accuracy','decision_compression_constrained')]:
        cache={}
        fs=subset(X,y,name,{'rank':'mi','constraint_tolerance':.01,'penalty':.02},101,cache,raw=X)
        assert fs==[1]
        assert calls[-1]['mode']==mode
        assert calls[-1]['constraint_tolerance']==.01
        assert len(cache)==1
