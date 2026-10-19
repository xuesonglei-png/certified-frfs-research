from itertools import product
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from certified_frfs.knn_query import certify_knn_queries


def test_query_certificate_holds_on_every_shared_box_corner():
    A=np.array([[.1],[.15],[.2],[.75],[.8],[.85]])
    y=np.repeat([0,1],3);B=np.array([[.17],[.79]])
    cert=certify_knn_queries(A,y,B,[0,1],[0],k=3,eps=.03,weights=[.8])
    assert cert['certified']==[True,True]
    for signs in product([-1,1],repeat=8):
        Z=A+.03*np.array(signs[:6])[:,None]
        Q=B+.03*np.array(signs[6:])[:,None]
        pred=KNeighborsClassifier(n_neighbors=3,metric='manhattan').fit(Z*.8,y).predict(Q*.8)
        assert pred.tolist()==[0,1]


def test_wrong_target_can_be_certified():
    A=np.array([[.1],[.15],[.2],[.75],[.8],[.85]])
    # This query may in fact have truth label1: stability never uses truth.
    cert=certify_knn_queries(A,np.repeat([0,1],3),[[.17]],[0],[0],k=3,eps=.01)
    assert cert['certified']==[True]


def test_multiclass_plurality_is_not_assumed_strict_majority():
    A=np.array([[.1],[.2],[.3],[.4],[.5]])
    cert=certify_knn_queries(A,[0,0,1,1,2],[[.15]],[0],[0],k=5,eps=0)
    assert cert['certified']==[False]


def test_query_enemy_budget_preserves_majority_at_shared_corners():
    A=np.array([[.2],[.3],[.4],[.11],[.12],[.9]])
    y=np.array([0,0,0,1,1,1]);B=np.array([[.1]])
    assert certify_knn_queries(A,y,B,[0],[0],k=5,eps=.005)['certified']==[False]
    cert=certify_knn_queries(A,y,B,[0],[0],k=5,eps=.005,allowed_enemies=2)
    assert cert['certified']==[True]
    for signs in product([-1,1],repeat=7):
        Z=A+.005*np.array(signs[:6])[:,None];Q=B+.005*signs[6]
        assert KNeighborsClassifier(n_neighbors=5,metric='manhattan').fit(Z,y).predict(Q)[0]==0
