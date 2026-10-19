import numpy as np
from certified_frfs.recent_budget import fgs_budget_path,sfrip_budget_path
from certified_frfs.recent import select_sfrip

def test_continuing_sfrip_preserves_native_stopping_and_complete_order():
    rng=np.random.default_rng(15);X=rng.random((18,4));y=np.arange(18)%3
    for confidence in (.6,.8):
        native=select_sfrip(X,y,beta=3,confidence=confidence)['features']
        path=sfrip_budget_path(X,y,beta=3,confidence=confidence)
        assert native==path['native']['features']==path['ranking'][:len(native)]
        assert sorted(path['ranking'])==list(range(4))

def test_fgs_budget_adaptation_preserves_all_native_members_first():
    rng=np.random.default_rng(7);X=rng.random((18,4));y=np.arange(18)%3
    path=fgs_budget_path(X,y,beta=.8,xi=1.)
    native=path['native']['features'];assert set(path['ranking'][:len(native)])==set(native)
    assert sorted(path['ranking'])==list(range(4))
