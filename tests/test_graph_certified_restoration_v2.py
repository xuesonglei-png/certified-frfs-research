import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from graph_certified_restoration_v2 import GuardCache


def c(*,legacy=False,majority=True,positive_bounds=(.2,.2)):
    dep={'dependency_certificate':{'upper_bound':.2,'redistributed_object_bounds':[.2]*8}}
    cert={'majority_feasible':majority,'dependency_certificate':dep['dependency_certificate']}
    return {'features':[0,1],'weights':[1.,1.],'certificate':cert,
      'family_certificate':cert,'classwise':{'bounds':list(positive_bounds)},
      'family_classwise':{'bounds':list(positive_bounds)},'joint_feasible':legacy}


def cache(pos=(.2,.2), signed=(.11,.11), *, status=0, upper=.05):
    A=np.arange(32,dtype=float).reshape(8,4)/32;y=np.repeat([0,1],4);calls=[]
    def sf(*args,**kwargs):
        return {'class_certificate':{'upper_bounds':list(signed)},
          'signed_object_report':{'signed_object_upper_bounds':[.11]*8,'objects':[]}}
    def gf(*args,**kwargs):
        calls.append(kwargs.get('positive'));return {'solver_status':status,
          'solver_bound_available':status in (0,1),'upper_bound':upper}
    return GuardCache(A,y,signed_fn=sf,graph_fn=gf),calls


def test_positive_arm_does_not_accept_from_signed_only():
    g,calls=cache(pos=(.2,.2),signed=(.09,.09),upper=.2)
    out=g.assess(c(positive_bounds=(.2,.2)),np.ones(4)/2,mode='graph_positive')
    assert not out['accepted'] and all(calls)


def test_positive_graph_uses_positive_object_bounds_and_accepts_only_if_all_pass():
    g,calls=cache(pos=(.11,.11),signed=(.11,.11),upper=.05)
    out=g.assess(c(positive_bounds=(.11,.11)),np.ones(4)/2,mode='graph_positive')
    assert out['accepted'] and calls==[True,True]
    assert out['certificate_scope'].startswith('same_table_class_mean_positive')


def test_status_two_is_never_usable():
    g,calls=cache(pos=(.11,.11),signed=(.11,.11),status=2,upper=0.)
    out=g.assess(c(positive_bounds=(.11,.11)),np.ones(4)/2,mode='graph_positive')
    assert not out['accepted'] and np.allclose(out['guards'][0]['new_class_bounds'],[.11,.11])


def test_majority_rejection_precedes_both_graph_targets():
    g,calls=cache()
    out=g.assess(c(majority=False),np.ones(4)/2,mode='graph_positive')
    assert not out['accepted'] and calls==[]
