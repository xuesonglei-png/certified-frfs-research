from itertools import product
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from certified_frfs.plurality_query import certify_plurality_queries
from certified_frfs.knn_query import certify_knn_queries


def test_plurality_without_majority_is_certified_at_all_shared_corners():
    A=np.array([[.11],[.14],[.18],[.23],[.32],[.9]])
    y=np.array([0,0,1,2,3,1]);B=np.array([[.05]])
    assert not certify_knn_queries(A,y,B,[0],[0],k=5,eps=.002,allowed_enemies=2)['certified'][0]
    result=certify_plurality_queries(A,y,B,[0],[0],k=5,eps=.002)
    assert result['certified']==[True]
    for signs in product([-1,1],repeat=7):
        Z=A+.002*np.array(signs[:6])[:,None];Q=B+.002*signs[-1]
        pred=KNeighborsClassifier(n_neighbors=5,metric='manhattan').fit(Z,y).predict(Q)
        assert pred.tolist()==[0]


def test_nominal_tie_free_bounds_recover_exact_votes_and_reject_vote_ties():
    rng=np.random.default_rng(101);A=rng.random((18,3));y=np.repeat([0,1,2],6);B=rng.random((8,3))
    D=abs(B[:,None,:]-A[None,:,:]).sum(axis=2);neighbors=np.argsort(D,axis=1)[:,:5]
    votes=np.array([[np.sum(y[js]==c) for c in range(3)] for js in neighbors])
    targets=votes.argmax(axis=1)
    result=certify_plurality_queries(A,y,B,targets,range(3),eps=0)
    for i,row in enumerate(result['records']):
        assert row['vote_lower']==votes[i].tolist()==row['vote_upper']
        assert result['certified'][i]==(np.sum(votes[i]==votes[i].max())==1)


def test_LOO_excludes_anchor_and_cannot_certify_wrong_target():
    A=np.array([[.1],[.12],[.15],[.8],[.85],[.9]])
    y=np.array([2,0,0,1,1,1])
    result=certify_plurality_queries(A,y,A[:1],[2],[0],eps=0,k=3,excluded_reference=[0])
    assert result['certified']==[False]
