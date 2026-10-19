import numpy as np
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate
from certified_frfs.majority_metric import MajorityTerms
from classwise_star_dependency import classwise_star_dependency


def test_half_attributes_newly_certified_with_original_majority_and_loss_thresholds():
    A=np.vstack([np.tile([0,.03],(4,1)),np.tile([1,0],(4,1)),
                 np.array([[.45,.4],[.45,.46],[.45,.52],[.45,.58]])])
    y=np.r_[np.zeros(8,int),np.ones(4,int)];full=np.ones(2)/np.sqrt(2);wr=np.array([np.sqrt(2)])
    old=weight_certificate(A,y,[1],full,wr);old_cls=classwise_dependency_certificate(y,old)
    new=classwise_star_dependency(A,y,[1],full,wr,base_dependency=old['dependency_certificate'])
    assert old['majority_feasible'] and not old_cls['feasible_all_classes']
    assert new['class_certificate']['feasible_all_classes'] and new['global_upper_bound']<=.1
    assert old['required']==old['retained_counts']==[8,4]
    assert np.all(MajorityTerms(A,y,weights=full).bounds([0,1],allowed_enemies=2)>1e-10)
    assert np.all(MajorityTerms(A[:,[1]],y,weights=wr).bounds([0],allowed_enemies=2)>1e-10)
    for draw in range(48):
        rng=np.random.default_rng(20911+draw);Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
        for fs,w in (([0,1],full),([1],wr)):
            D=np.zeros((len(A),len(A)))
            for a,value in zip(fs,w):D+=abs(Z[:,a,None]*value-Z[None,:,a]*value)
            np.fill_diagonal(D,np.inf);nn=np.argsort(D,axis=1,kind='stable')[:,:5]
            pred=np.array([max((0,1),key=lambda c:sum(y[j]==c for j in ids)) for ids in nn])
            assert np.array_equal(pred,y)
