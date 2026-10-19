import numpy as np
import pytest
from exact_count_anchor_v2 import exact_count_policies, independent_exact_count_policies, ORIGIN
from run_multiscale_certified_metric_knn_pool import policy_metrics


def fixture(d):
    A = np.random.default_rng(27).uniform(size=(16,d))
    block = {'pool':[], 'recipes':[], 'paths':{
        method:[{'ranking':list(reversed(range(d))), 'native':{'features':[0]}}]
        for method in ('fgs2023','sfrip2025')}}
    return A, block


def test_preserves_every_feasible_legacy_anchor_and_independent_replay():
    for d in (1,2,5,8):
        A, block = fixture(d)
        for k in range((d+1)//2,d+1):
            new = exact_count_policies(A,block,k)
            old = policy_metrics(A,block,count=k)
            assert new == {key:old[key] for key in new}
            assert new == independent_exact_count_policies(A,block,k)


def test_unique_minimal_feasible_anchor_and_all_supports():
    for d in (5,8,60):
        A, block = fixture(d)
        for k in range(1,(d+1)//2):
            result = exact_count_policies(A,block,k)
            assert result == independent_exact_count_policies(A,block,k)
            for key in ('fgs_matched','sfrip_matched'):
                row = result[key][0]
                assert row['origin'] == ORIGIN
                assert len(row['features']) == k
                np.testing.assert_allclose(row['weights'],np.sqrt(d)/k)
                np.testing.assert_allclose(sum(row['weights']),np.sqrt(d))
                assert np.max(np.array(row['weights'])/np.sqrt(d)) <= 1/k+1e-14


def test_reject_invalid_count_and_nonpermutation():
    A, block = fixture(5)
    for k in (0,6,1.5,True):
        with pytest.raises(ValueError): exact_count_policies(A,block,k)
    block['paths']['fgs2023'][0]['ranking'] = [0,0,1,2,3]
    with pytest.raises(ValueError): exact_count_policies(A,block,2)
