import numpy as np
from expanded_fgs_grid import expanded_paths
from audit_expanded_fgs_comparison import paths,graph


def test_independent_set_graph_matches_published_small_example():
    pairs=[(0,1),(0,2),(0,3),(0,4),(1,2),(1,4),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]
    weights=[-.1,.4,.3,.7,.3,.6,.1,.5,.6,.6,.8,-.3]
    result=graph([.3,.2,.5,.2,.3,.1],[(w,i,j) for w,(i,j) in zip(weights,pairs)])
    assert result['features']==[0,1,2,4,5] and result['groups']==[[0,3],[1],[2,4,5]]
    assert result['interactive_features']==[4,5] and len(result['tree'])==5


def test_independent_numeric_and_set_graph_replay_all120_configurations():
    A=np.array([[0,.1,1,.5],[.2,.3,1,.6],[.8,.9,1,.1],[1,.6,1,.4],
        [.5,.7,1,.9],[.6,.2,1,.8],[.3,.8,1,.3],[.9,.5,1,.2]],float)
    y=np.array([0,0,1,1,2,2,0,1])
    assert paths(A,y)==expanded_paths(A,y)
