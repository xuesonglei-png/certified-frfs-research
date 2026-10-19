import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from certified_frfs.utility_search import UtilityWorkspace,utility_path
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.adaptive_witness import adaptive_bounds
from certified_frfs.core import certificate


def test_utility_matches_explicit_leave_one_out_classifier():
    rng=np.random.default_rng(103);X=rng.uniform(.1,.9,(18,4));y=np.repeat([0,1,2],6);w=np.array([.2,.3,.4,.5]);fs=[0,2]
    result=UtilityWorkspace(X,y,w).evaluate(fs)
    Z=np.clip(X+np.random.default_rng(101).uniform(-.01,.01,X.shape),0,1)
    for key,table in [('clean_LOO',X),('shared_noise_LOO',Z)]:
        preds=[]
        for i in range(len(y)):
            ids=np.arange(len(y))!=i
            model=KNeighborsClassifier(n_neighbors=5,metric='manhattan',algorithm='brute')
            model.fit(table[ids][:,fs]*w[fs],y[ids]);preds.append(model.predict(table[[i]][:,fs]*w[fs])[0])
        assert abs(result[key]-balanced_accuracy_score(y,preds))<1e-12


def test_certified_path_is_feasible_and_keeps_informative_attribute():
    rng=np.random.default_rng(104);y=np.repeat([0,1],12)
    X=np.column_stack([.1+.8*y,rng.uniform(.1,.9,(24,3))]);w=np.array([1.,.05,.05,.05])
    result=utility_path(X,y,w);fixed=MajorityTerms(X,y,weights=w)
    protected=fixed.bounds(range(4),allowed_enemies=2)>1e-10
    assert result['candidates'][-1]==[0]
    for fs in result['candidates']:
        bounds=np.maximum(fixed.bounds(fs,allowed_enemies=2),adaptive_bounds(X,y,fs,w))
        for label,required in zip(np.unique(y),result['required']):
            assert ((bounds>1e-10)&protected&(y==label)).sum()>=required


def test_unconstrained_search_chooses_highest_utility_move():
    rng=np.random.default_rng(105);X=rng.uniform(.1,.9,(24,5));y=np.repeat([0,1,2],8);w=np.ones(5)/np.sqrt(5)
    result=utility_path(X,y,w,certified=False)
    assert len(result['candidates'][-1])==1
    for step in result['trace']:
        assert step['accepted']==step['ranked_moves'][0]['features']
        assert len(step['certification_attempts'])==1


def test_each_move_is_best_among_certificate_feasible_deletions():
    rng=np.random.default_rng(106);X=rng.uniform(.1,.9,(24,5));y=np.repeat([0,1,2],8);w=np.ones(5)/np.sqrt(5)
    result=utility_path(X,y,w);fixed=MajorityTerms(X,y,weights=w)
    p=fixed.bounds(range(5),allowed_enemies=2)>1e-10
    for step in result['trace']:
        feasible=[]
        for move in step['ranked_moves']:
            fs=move['features'];bounds=np.maximum(fixed.bounds(fs,allowed_enemies=2),adaptive_bounds(X,y,fs,w))
            counts=[int(((bounds>1e-10)&p&(y==label)).sum()) for label in np.unique(y)]
            if all(c>=r for c,r in zip(counts,result['required'])):feasible.append(move)
        assert step['accepted']==(feasible[0]['features'] if feasible else None)


def test_joint_constraint_certifies_every_path_subset():
    rng=np.random.default_rng(107);y=np.repeat([0,1],12)
    X=np.column_stack([.1+.8*y,rng.uniform(.1,.9,(24,3))]);w=np.array([1.,.05,.05,.05])
    path=utility_path(X,y,w,dependency_eta=.1)
    assert path['candidates'][-1]==[0]
    for fs in path['candidates']:
        assert certificate(X,y,fs,weights=w,eps=.01,method='hybrid')['upper_bound']<=.1


def test_zero_loss_tolerance_stops_before_deleting_positive_contribution():
    rng=np.random.default_rng(108);y=np.repeat([0,1],12)
    X=np.column_stack([.1+.6*y,rng.uniform(.1,.9,24)]);w=np.array([.5,.2])
    path=utility_path(X,y,w,dependency_eta=0)
    assert path['candidates']==[[0,1]]


def test_refined_dual_search_preserves_both_constraints():
    from certified_frfs.shared_loss import shared_loss_bound
    rng=np.random.default_rng(109);y=np.repeat([0,1],12)
    X=np.column_stack([.1+.7*y,rng.uniform(.1,.9,(24,3))]);w=np.array([.8,.1,.1,.1])
    path=utility_path(X,y,w,dependency_eta=.1,shared_loss=True)
    for fs in path['candidates']:
        assert shared_loss_bound(X,y,fs,w)['upper_bound']<=.1
    assert path['shared_loss']
