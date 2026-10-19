import numpy as np
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.flexible_frs_certificate import flexible_weight_certificate
from certified_frfs.majority_metric import MajorityTerms

def test_same_original_mask_requirements_dependency_and_original_feasible_contract():
    rng = np.random.default_rng(888); y = np.repeat([0,1,2],9)
    X = np.column_stack([.1+.4*y,rng.uniform(.1,.9,(27,3))]); full = np.array([1.,.03,.03,.03]); fs = [0,1,2]
    reduced = full[fs]; protected = MajorityTerms(X,y,weights=full).bounds(range(4),allowed_enemies=2) > 1e-10
    old = weight_certificate(X,y,fs,full,reduced,protected=protected)
    new = flexible_weight_certificate(X,y,fs,full,reduced,protected=protected)
    assert old['required'] == new['required'] and old['dependency_certificate'] == new['dependency_certificate']
    assert all(n >= o for n,o in zip(new['retained_counts'],old['retained_counts']))
    assert old['feasible'] and new['feasible']

def test_sparse_class_no_repeated_friend_slots_can_create_new_protection():
    X = np.array([[.1,0.],[.1,1.],[.8,0.],[.8,.3],[.8,.6],[.8,1.]])
    y = np.array([0,0,1,1,1,1]); full = np.array([1.,.01])
    protected = np.array([True,True,False,False,False,False])
    # An explicitly supplied impossible protection requirement is never
    # admitted by padding same-class reference slots into three friends.
    result = flexible_weight_certificate(X,y,[0],full,np.array([1.]),protected=protected)
    assert result['required'][0] == 2 and result['retained_counts'][0] == 0 and not result['feasible']
