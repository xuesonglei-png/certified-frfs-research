import numpy as np
from certified_frfs.ldfrs import local_influence,fdm_rank_relations


def test_hand_computed_density_without_self():
    X=np.array([[0.],[.1],[.9],[1.]]);y=np.array([0,0,1,1])
    result=local_influence(X,y,k=1)
    assert np.allclose(result['global_density'],2/1.1)
    assert np.allclose(result['class_density'],2/1.1)
    assert np.allclose(result['influence'],1)
    assert np.allclose(result['local_density_factor'],1)


def test_spy_and_boundary_downweighting():
    X=np.array([[0.],[.05],[.9],[.95],[1.],[.06]])
    y=np.array([0,0,1,1,1,1])
    result=local_influence(X,y,k=1,threshold=1.5)
    assert result['local_density_factor'][-1]>1.5
    assert result['influence'][-1]==0
    assert np.all(result['influence']<=1+1e-12)


def test_relevance_indispensability_crisp_signal():
    y=np.array([0,0,1,1]);D=(y[:,None]==y).astype(float)
    result=fdm_rank_relations([np.ones((4,4)),D],D)
    assert result['ranking']==[1,0]
    assert np.isclose(result['trace'][0]['candidates'][1]['score'],2)


def test_published_toy_exposes_distinct_density_factor_neighbors():
    X=np.array([[.7,.9,.4,.6],[.6,.9,.3,.7],[.6,.5,.3,.5],
                [.3,.5,.2,.2],[.5,.8,.4,.6],[.4,.5,.6,.3],[.4,.4,.1,.2]])
    y=np.array([1,1,0,0,1,0,1])
    inclusive=local_influence(X,y,k=3,density_self=True)
    mixed=local_influence(X,y,k=3,density_self=True,factor_self=False)
    assert np.allclose(inclusive['global_density'],[2.86,2.91,2.27,2.50,2.81,2.18,2.50],atol=.005)
    assert np.allclose(inclusive['class_density'],[2.86,2.91,2.16,2.15,2.81,2.18,1.66],atol=.005)
    assert 1.48<inclusive['local_density_factor'][-1]<1.49
    assert 1.72<mixed['local_density_factor'][-1]<1.73
    assert inclusive['influence'][-1]>0
    assert mixed['influence'][-1]==0
