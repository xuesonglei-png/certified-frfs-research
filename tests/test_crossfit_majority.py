import numpy as np
from certified_frfs.crossfit_majority import recipe_candidates
from certified_frfs.majority_metric import MajorityTerms


def test_recipe_returns_only_classwise_retention_feasible_subsets():
    rng=np.random.default_rng(71);y=np.repeat([0,1],12)
    X=rng.uniform(.1,.3,(24,3));X[y==1,0]+=.6;w=np.ones(3)/np.sqrt(3)
    candidates,paths=recipe_candidates(X,y,w,'protected',include_margin=True);workspace=MajorityTerms(X,y,weights=w)
    protected=workspace.bounds(range(3),allowed_enemies=2)>1e-10
    assert len(candidates)==24 and len(paths)==4
    for row in candidates:
        positive=workspace.bounds(row['features'],allowed_enemies=2)>1e-10
        for label in np.unique(y):
            group=(y==label)&protected
            assert np.sum(positive&group)>=np.ceil(.9*group.sum()-1e-12)
