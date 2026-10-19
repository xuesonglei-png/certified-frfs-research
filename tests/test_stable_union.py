import numpy as np
import pytest
from certified_frfs.stable_union import union_features,certify_union


def test_union_deduplicates_and_validates():
    assert union_features([[0,2],[1,2]],3)==[0,1,2]
    with pytest.raises(ValueError):union_features([[]],3)
    with pytest.raises(ValueError):union_features([[3]],3)


def test_union_restores_signal_under_fresh_certificate():
    rng=np.random.default_rng(7);y=np.repeat([0,1],8)
    X=np.column_stack([.1+.8*y,rng.uniform(.2,.8,16)])
    result=certify_union(X,y,[1],np.array([1.,.05]))
    assert 0 in result['features']
    assert all(c>=r for c,r in zip(result['retained_counts'],result['required']))
