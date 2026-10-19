import numpy as np
import pytest
from atomic_postmetric_deletion import DistanceProfiles,fit_atomic_deletion


def table():
    x=np.r_[np.linspace(.05,.1,6),np.linspace(.9,.95,6)]
    return np.c_[x,x,np.full(12,.5)],np.repeat([0,1],6)


def test_duplicate_distance_receiver_deletes_a_positive_donor_and_keeps_guards():
    from audit_simplex_source_budget_diagnostic import independent_loo
    X,y=table(); full=np.ones(3)/np.sqrt(3)
    result=fit_atomic_deletion(X,y,[0,1,2],full,0,1.)
    assert result['selected_id']==0
    point=result['checks'][0]
    assert point['feature']==1 and point['features']==[1,2]
    assert point['mean_pair_distance_distortion']==0
    assert point['weights']==pytest.approx([2/np.sqrt(3),1/np.sqrt(3)])
    assert point['eligible'] and point['certificate']['feasible']
    assert point['class_certificate']['feasible_all_classes']
    assert independent_loo(X,y,[1,2],point['weights'])==1.
    assert sum(point['weights'])==pytest.approx(np.sqrt(3))
    assert not result['checks'][1]['accepted']


def test_receiver_profiles_are_label_free_and_follow_distances_not_values():
    X,y=table(); X[:,1]=1-X[:,0]
    g=DistanceProfiles(X)
    choices=g.receivers([0,1,2],0)
    assert choices[0]['feature']==1
    assert choices[0]['mean_pair_distance_distortion']==pytest.approx(0,abs=1e-15)
    result=fit_atomic_deletion(X,y,[0,1,2],np.ones(3)/np.sqrt(3),0,1.,geometry=g)
    assert result['checks'][0]['eligible']


def test_stale_profiles_and_zero_or_invalid_donors_are_rejected():
    X,y=table(); geometry=DistanceProfiles(X); changed=X.copy();changed[0,0]+=.01
    with pytest.raises(ValueError): fit_atomic_deletion(changed,y,[0,1,2],np.ones(3)/np.sqrt(3),0,1.,geometry=geometry)
    with pytest.raises(ValueError): fit_atomic_deletion(X,y,[0,1,2],[0,np.sqrt(3),0],0,1.)
    with pytest.raises(ValueError): fit_atomic_deletion(X,y,[0,1,2],np.ones(3)/np.sqrt(3),3,1.)
