import numpy as np
from certified_frfs.feasible_query_descent import fit_feasible_query
from certified_frfs.validated_capped_sparse import truncate_trajectory


def test_truncated_long_trajectory_matches_independent_three_steps():
    rng=np.random.default_rng(90);X=rng.random((24,4));y=np.repeat([0,1],12)
    long=fit_feasible_query(X,y,certified=False,weight_cap_ratio=2,steps=10)
    short=fit_feasible_query(X,y,certified=False,weight_cap_ratio=2,steps=3)
    truncated=truncate_trajectory(long,3)
    assert truncated['trace']==short['trace']
    assert truncated['training_splits']==short['training_splits']
    assert truncated['features']==short['features']
    assert np.array_equal(truncated['weights'],short['weights'])
