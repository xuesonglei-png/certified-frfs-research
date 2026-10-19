import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shared_graph_dependency_v5 import class_upper, memberships, build
from shared_graph_dependency_v4 import class_upper as previous_upper
from test_shared_graph_dependency_v2 import enemy_cuts


def test_tight_positive_constants_both_targets_preserve_attacks_and_exact_sign_bounds():
    rng=np.random.default_rng(939)
    for positive in (False,True):
        for _ in range(4):
            A=rng.uniform(.2,.8,(5,2));y=np.array([0,0,1,1,1]);wf,wr=rng.uniform(.1,1.2,(2,2))
            obj,cuts=enemy_cuts(A,y,wf,wr,.08)
            report=class_upper(A,y,wf,wr,0,eps=.08,positive=positive,object_upper=obj,enemy_upper=cuts,seconds=10.)
            old=previous_upper(A,y,wf,wr,0,eps=.08,positive=positive,object_upper=obj,enemy_upper=cuts,seconds=10.)
            exact=class_upper(A,y,wf,wr,0,eps=.08,positive=positive,exact_abs=True,seconds=10.)
            assert report['solver_upper_bound']<=old['solver_upper_bound']+1e-7
            assert report['solver_upper_bound']+1e-7>=exact['solver_upper_bound']
            for _ in range(20):
                Z=rng.uniform(A-.08,A+.08)
                delta=(memberships(Z,y,wf)-memberships(Z,y,wr))[:2]
                actual=(np.maximum(0.,delta) if positive else delta).mean()
                assert actual<=report['solver_upper_bound']+1e-7


def test_zero_radius_positive_branch_matches_positive_loss_without_cancellation():
    A=np.array([[.1,.6],[.6,.1],[.2,.8],[.8,.2]]);y=np.array([0,0,1,1])
    wf,wr=np.array([1.,.2]),np.array([.2,1.])
    report=class_upper(A,y,wf,wr,0,eps=0.)
    delta=(memberships(A,y,wf)-memberships(A,y,wr))[:2]
    assert report['solver_upper_bound']==pytest.approx(np.maximum(0.,delta).mean(),abs=1e-8)


def test_positive_big_m_uses_membership_lower_bounds():
    A=np.array([[.1],[.4],[.6]]);y=np.array([0,1,1])
    _,_,meta=build(A,y,[1.],[.8],0,eps=.01)
    r=meta['selections'][0]
    assert r['full_lower']>.2 and r['positive_branch_M']<1.
