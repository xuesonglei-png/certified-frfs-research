import numpy as np
from certified_frfs.graph_recent import select_fgs
from certified_frfs.recent_budget import fgs_budget_path
from expanded_fgs_grid import CONFIGS,LEGACY,expanded_paths


def test_declared_superset_has_exact_unique_parameters_and_keeps_legacy_order():
    assert len(CONFIGS)==120
    pairs=[(r['beta'],r['xi']) for r in CONFIGS]
    assert len(set(pairs))==120 and all((r['beta'],r['xi']) in pairs for r in LEGACY)
    assert sorted(LEGACY,key=lambda r:(r['beta'],r['xi']))==list(LEGACY)


def test_every_broad_configuration_matches_literal_numerical_selector():
    A=np.array([[0,0,1,.2],[.1,.3,1,.2],[.4,.2,1,.6],[.7,.6,1,.3],
        [.9,.8,1,.9],[1,1,1,.8],[.3,.9,1,.1],[.8,.1,1,.4]],float)
    y=np.array([0,0,1,1,2,2,1,0])
    outputs=expanded_paths(A,y)
    for i,r in enumerate(outputs):
        assert r['index']==i and r['config']==CONFIGS[i]
        literal=fgs_budget_path(A,y,**r['config'])
        assert r['path']==literal
        assert r['path']['native']==select_fgs(A,y,**r['config'])
