import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from certificate_barrier_support_v1 import search,metric_weights


def fake(feasible):
    loss=0. if feasible else .2
    cert={'dependency_certificate':{'upper_bound':loss},'required':[1],'retained_counts':[1]}
    return {'certificate':cert,'family_certificate':cert,
            'classwise':{'bounds':[loss]},'family_classwise':{'bounds':[loss]},
            'joint_feasible':feasible}


def test_crosses_failed_intermediate_and_never_returns_failed_point():
    # {2,3} is two swaps from {0,1}; every one-swap intermediate fails.
    proxy=lambda fs:{'proxy_key':[0,0,0],'training_LOO_BA':.7}
    out=search(4,[0,1],proxy,lambda fs:fake(fs==(2,3)),depths=2,width=4,per_depth=8)
    assert out['selected']['features']==[2,3]
    assert out['selected']['depth']==2
    parent=out['selected']['parent']
    assert not next(r for r in out['checks'] if r['features']==parent)['candidate']['joint_feasible']


def test_budget_and_unavailable_are_explicit():
    proxy=lambda fs:{'proxy_key':[0,0,0],'training_LOO_BA':.5}
    out=search(8,[0,1,2,3],proxy,lambda fs:fake(False),depths=2,width=4,per_depth=3)
    assert out['selected'] is None and out['exact_checks']<=7
    assert len({tuple(r['features']) for r in out['checks']})==out['exact_checks']


def test_exact_half_cap_forces_unique_metric():
    A=np.arange(24,dtype=float).reshape(6,4)/24
    w=metric_weights(A,[0,3],np.array([.2,.4,.6,.8]))
    np.testing.assert_allclose(w,[1.,1.],atol=1e-14)


def test_cannot_search_below_half_cap():
    with pytest.raises(ValueError):search(8,[0,1,2],None,None)
