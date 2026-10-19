import numpy as np
from certified_frfs.graph_recent import certified_backward_candidates, combined_certified_candidates


def test_backward_pool_keeps_full_and_only_feasible_candidates():
    X=np.array([[0,.1],[.05,.15],[.1,.2],[.9,.8],[.95,.85],[1,.9]])
    y=np.array([0,0,0,1,1,1])
    result=certified_backward_candidates(X,y,[0,1],eps=.01,eta=0)
    assert [0,1] in result['candidates']
    assert all(row['feasible'] is False or row['loss_certificate']['upper_bound']<=1e-10
               for row in result['trace'])


def test_combined_pool_deduplicates():
    X=np.array([[0,.1],[.05,.15],[.1,.2],[.9,.8],[.95,.85],[1,.9]])
    y=np.array([0,0,0,1,1,1])
    base={'features':[0,1],'groups':[[0],[1]],'cd':[1,1]}
    result=combined_certified_candidates(X,y,base,[0,1],eta=0,weights=np.ones(2)/np.sqrt(2))
    assert len(result['candidates'])==len({tuple(x) for x in result['candidates']})
