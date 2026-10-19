import numpy as np
from majority_hinge_soft_bound_comparison import compare


def test_small_same_class_neighbor_gap_does_not_obstruct_semantic_certificate():
    rng=np.random.default_rng(671)
    A=np.vstack([rng.uniform(.05,.051,(9,2)),rng.uniform(.85,.851,(9,2))]);y=np.repeat([0,1],9)
    comparison=compare(A,y,np.array([.5,.5]));means=comparison['class_balanced_training_bounds']
    assert means['hinge_error_bound']==0
    assert means['soft_gap_error_bound']>.9 and means['soft_exact_leakage_error_bound']>.25
    assert all(row['independent_scalar_soft_replayed'] for row in comparison['rows'])
