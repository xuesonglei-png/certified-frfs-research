import numpy as np
from scipy.spatial.distance import cdist
from itertools import product
from certified_frfs.knn_decision import certified_knn_coverage


def test_k5_bound_is_safe_under_shared_perturbations():
    rng=np.random.default_rng(5); X=rng.uniform(.1,.9,(12,3)); y=np.repeat([0,1],6)
    result=certified_knn_coverage(X,y,[0,1],k=5,eps=.03)
    lower=np.array(result['lower_bounds'])
    for _ in range(30):
        Z=X+rng.uniform(-.03,.03,X.shape)
        distance=cdist(Z[:,[0,1]],Z[:,[0,1]],'cityblock')
        for i in range(len(Z)):
            friends=np.sort(distance[i][(y==y[i])&(np.arange(len(y))!=i)])[:3]
            enemies=np.sort(distance[i][y!=y[i]])
            actual=enemies[0]-friends[-1]
            assert lower[i] <= actual+1e-12


def test_three_target_friends_are_required_for_k5():
    X=np.array([[0.1],[0.11],[0.12],[0.8],[0.81],[0.82]])
    result=certified_knn_coverage(X,[0,0,0,1,1,1],[0],k=5,eps=0)
    assert result['majority_q']==3
    assert result['balanced_coverage']==0


def test_separated_classes_have_certified_k5_majorities():
    X=np.array([[.1],[.11],[.12],[.13],[.8],[.81],[.82],[.83]])
    result=certified_knn_coverage(X,np.repeat([0,1],4),[0],k=5,eps=.01)
    assert result['balanced_coverage']==1.


def test_certified_labels_hold_for_every_corner_of_small_shared_box():
    X=np.array([[.1],[.14],[.18],[.22],[.72],[.76],[.8],[.84]])
    y=np.repeat([0,1],4)
    cert=certified_knn_coverage(X,y,[0],k=5,eps=.04)
    assert len(cert['certified_indices'])==8
    for signs in product([-1,1],repeat=8):
        Z=X+.04*np.asarray(signs)[:,None]
        distances=cdist(Z,Z,'cityblock')
        np.fill_diagonal(distances,np.inf)
        neighbors=np.argsort(distances,axis=1,kind='stable')[:,:5]
        for i in cert['certified_indices']:
            assert np.sum(y[neighbors[i]]==y[i]) >=3


def test_weighted_certificate_predicts_same_metric_and_bounds_margin():
    rng=np.random.default_rng(19)
    X=rng.uniform(.1,.9,(12,3)); y=np.repeat([0,1],6)
    w=np.array([.9,.3,.1]);w/=np.linalg.norm(w)
    result=certified_knn_coverage(X,y,[0,1],k=5,eps=.03,weights=w)
    lower=np.array(result['lower_bounds'])
    for _ in range(40):
        Z=X+rng.uniform(-.03,.03,X.shape)
        distance=cdist(Z[:,[0,1]]*w[[0,1]],Z[:,[0,1]]*w[[0,1]],'cityblock')
        for i in range(len(Z)):
            friends=np.sort(distance[i][(y==y[i])&(np.arange(len(y))!=i)])[:3]
            actual=distance[i][y!=y[i]].min()-friends[-1]
            assert lower[i]<=actual+1e-12
