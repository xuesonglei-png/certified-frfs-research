import numpy as np
from certified_frfs.graph_recent import dependency, entropy, graph_select, select_fgs


def test_dependency_beta_one_and_constants():
    y=np.array([0,0,1,1])
    assert dependency(np.eye(4),y,1)==1
    assert dependency(np.ones((4,4)),y,1)==0
    result=select_fgs(np.zeros((4,3)),y,beta=1)
    assert result['features']==[0,1,2]


def test_entropy_crisp_decision():
    y=np.array([0,0,1,1])
    assert entropy((y[:,None]==y).astype(float))==1


def test_kruskal_interaction_and_weak_edge():
    result=graph_select([.2,.3,.4], [(-.5,0,1),(.1,1,2),(.6,0,2)])
    assert result['features']==[0,1,2]
    assert result['groups']==[[0,1],[2]]
    assert result['interactive_features']==[0,1]


def test_xor_interaction_survives():
    X=np.tile(np.array([[0,0],[0,1],[1,0],[1,1]],float),(3,1))
    y=(X[:,0]!=X[:,1]).astype(int)
    result=select_fgs(X,y,beta=1,xi=1)
    assert result['features']==[0,1]
    assert result['interactive_features']==[0,1]
    assert np.isclose(result['tree'][0][0],-1)


def test_published_graph_example_with_declared_ties():
    # Wan et al. Example 1, p217. Ties permit different valid MSTs.
    pairs=[(0,1),(0,2),(0,3),(0,4),(1,2),(1,4),
           (2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]
    weights=[-.1,.4,.3,.7,.3,.6,.1,.5,.6,.6,.8,-.3]
    result=graph_select([.3,.2,.5,.2,.3,.1],[(w,i,j) for w,(i,j) in zip(weights,pairs)])
    assert len(result['tree'])==5
    assert np.isclose(sum(w for w,_,_ in result['tree']),.5)
    assert result['interactive_features']==[4,5]
    assert result['groups']==[[0,3],[1],[2,4,5]]
    assert result['features']==[0,1,2,4,5]


def test_original_dependency_constraint_filters_and_full_fallback():
    from certified_frfs.graph_recent import refine_group_candidates
    from certified_frfs.core import certificate
    X=np.array([[0,.1],[.05,.15],[.1,.2],[.9,.8],[.95,.85],[1,.9]])
    y=np.array([0,0,0,1,1,1])
    base={'features':[0],'groups':[[0],[1]]}
    result=refine_group_candidates(X,y,base,dependency_eta=0,penalty=.02,k=3)
    assert result['features']==[0,1]
    assert result['loss_certificate']['upper_bound']==0
    assert any(not row['feasible'] for row in result['trace'])
    for row in result['trace']:
        independent=certificate(X,y,row['features'],eps=.01)
        assert independent==row['loss_certificate']
        assert row['feasible']==(independent['upper_bound']<=1e-10)
