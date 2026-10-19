import numpy as np
from certified_frfs.crossfit_search import select_crossfit_queries


def test_reference_only_scaling_disjoint_queries_and_budget():
    X=np.random.default_rng(17).uniform(.1,.9,(16,4))
    X[0,0]=10
    y=np.repeat([0,1],8)
    result=select_crossfit_queries(X,y,budget=12)
    assert result['evaluations']<=12
    assert len({tuple(r['features']) for r in result['trace']})==result['evaluations']
    for record in result['preprocessing']:
        tr,va=record['reference_indices'],record['query_indices']
        assert not set(tr)&set(va)
        assert set(tr)|set(va)==set(range(len(X)))
        assert np.allclose(record['data_min'],X[tr].min(axis=0))
        assert np.allclose(record['data_max'],X[tr].max(axis=0))
    assert result==select_crossfit_queries(X,y,budget=12)


def test_objective_modes_use_same_internal_splits():
    X=np.random.default_rng(31).uniform(.1,.9,(12,3))
    y=np.repeat([0,1],6)
    a=select_crossfit_queries(X,y,mode='robust',budget=7)
    b=select_crossfit_queries(X,y,mode='accuracy',budget=7)
    assert a['preprocessing']==b['preprocessing']
    assert a['ranking']==b['ranking']
    assert all(not f['certificate_evaluated'] for row in b['trace'] for f in row['fold_diagnostics'])


def test_constrained_modes_share_path_and_satisfy_empirical_threshold():
    X=np.random.default_rng(12).uniform(.1,.9,(16,5))
    y=np.repeat([0,1],8)
    results=[select_crossfit_queries(X,y,mode=mode,budget=12,constraint_tolerance=.01)
             for mode in ['robust','nominal','accuracy']]
    paths=[[row['features'] for row in result['trace']] for result in results]
    assert paths[0]==paths[1]==paths[2]
    for result in results:
        selection=result['constraint_selection']
        assert selection['selected_predictive_score'] >= selection['threshold']-1e-12
        assert list(range(5)) in [r['features'] for r in result['trace']]
        assert selection['eligible'] >= 1
