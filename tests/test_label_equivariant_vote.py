import numpy as np
from scipy.spatial.distance import cdist
from label_equivariant_vote import predict,independent_predict


def test_tie_rule_equivariant_to_class_recoding_and_geometric_choice():
    A=np.array([[.1],[.11],[.8],[.81],[.5]]);B=np.array([[0.]])
    y=np.array([1,1,0,0,2]);D=cdist(B,A,'cityblock')
    report=predict(D,y);assert report['predictions']==[1] and report['tied_queries']==[True]
    recode={0:19,1:77,2:3};other=np.array([recode[c] for c in y])
    assert predict(D,other)['predictions']==[77]
    assert report['predictions']==independent_predict(A,y,B,[0],[1.])


def test_strict_three_vote_majorities_remain_unchanged_under_shared_noise():
    rng=np.random.default_rng(13);A=np.array([[.01],[.02],[.03],[.7],[.8]])
    y=np.array([9,9,9,0,0]);B=np.array([[.02]])
    for _ in range(24):
        ref=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
        query=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
        report=predict(cdist(query,ref,'cityblock'),y)
        assert report['predictions']==[9] and report['tied_queries']==[False]


def test_exact_geometric_ties_use_reference_identity_not_label_code():
    D=np.ones((1,5));y=np.array([8,8,1,1,7]);report=predict(D,y)
    assert report['predictions']==[8]
    recode={8:100,1:0,7:1}
    assert predict(D,np.array([recode[c] for c in y]))['predictions']==[100]
