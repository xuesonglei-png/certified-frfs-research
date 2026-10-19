import numpy as np
from certified_frfs.scenario_repair import solve_scenario_repair
from certified_frfs.core import gamma


def test_feasible_scenario_constraints_recomputed():
    X=np.array([[0.,.2],[.1,.1],[.8,.9],[1.,.7]]);y=np.array([0,0,1,1]);full=np.ones(2)/np.sqrt(2)
    initial=np.array([.3,full.sum()-.3])
    fit=solve_scenario_repair([X],y,full,initial,[0,1])
    assert fit['success'];v=np.array(fit['weights'])
    assert abs(v.sum()-full.sum())<1e-10
    assert gamma(X,y,weights=full)-gamma(X,y,weights=v)<=.09+1e-10


def test_cap_makes_too_small_support_infeasible():
    X=np.random.default_rng(4).random((6,4));y=np.repeat([0,1],3);full=np.ones(4)/2
    fit=solve_scenario_repair([X],y,full,np.array([2.,0,0,0]),[0])
    assert not fit['success'] and fit['weights'] is None
