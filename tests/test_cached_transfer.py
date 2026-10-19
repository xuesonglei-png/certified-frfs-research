import itertools
import numpy as np
import pytest
from certified_frfs.cached_transfer import CachedTransfer
from certified_frfs.redistributed_loss import transfer_weights


def test_all_subsets_agree_with_uncached_rule_including_constants():
    rng=np.random.default_rng(940);X=rng.uniform(0,1,(32,6));X[:,1]=.2;X[:,3]=X[:,0];X[:,5]=1-X[:,0]
    w=rng.uniform(0,1,6);w[2]=0;cache=CachedTransfer(X,w)
    for size in range(1,7):
        for fs in itertools.combinations(range(6),size):
            actual=cache.weights(fs)
            np.testing.assert_allclose(actual,transfer_weights(X,fs,w),rtol=0,atol=1e-12)
            assert abs(actual.sum()-w.sum())<1e-12


def test_all_constant_and_unsorted_retained_attributes():
    X=np.ones((12,4));w=np.array([.1,.2,.3,.4]);cache=CachedTransfer(X,w)
    np.testing.assert_allclose(cache.weights([3,1]),transfer_weights(X,[3,1],w),rtol=0,atol=1e-12)
    for fs in [[],[1,1],[4],[-1],[1.5]]:
        with pytest.raises(ValueError):cache.weights(fs)


def test_cache_does_not_alias_weights_or_training_table():
    rng=np.random.default_rng(941);X=rng.normal(size=(24,3));w=np.ones(3);cache=CachedTransfer(X,w)
    expected=cache.weights([1]);X[:]=0;w[:]=0
    np.testing.assert_array_equal(cache.weights([1]),expected)
