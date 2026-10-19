import itertools
import numpy as np
from certified_frfs.cached_transfer_utility import CachedCrossfitTransferUtility
from certified_frfs.redistributed_search import CrossfitRedistributedUtilityWorkspace


def test_cached_score_agrees_on_all_small_random_subsets():
    rng=np.random.default_rng(950);X=rng.uniform(0,1,(30,5));y=np.repeat([0,1,2],10)
    uncached=CrossfitRedistributedUtilityWorkspace(X,y)
    cached=CachedCrossfitTransferUtility(X,y)
    assert uncached.splits==cached.splits
    for size in range(1,6):
        for fs in itertools.combinations(range(5),size):
            a,b=uncached.evaluate(fs),cached.evaluate(fs)
            for key in ['utility','clean_LOO','shared_noise_LOO']:
                assert abs(a[key]-b[key])<1e-12
            np.testing.assert_allclose(a['reduced_weights'],b['reduced_weights'],rtol=0,atol=1e-12)


def test_softvote_and_missing_values_fit_within_subfolds():
    rng=np.random.default_rng(951);X=rng.uniform(0,1,(24,4));X[0,0]=np.nan;y=np.repeat([0,1],12)
    uncached=CrossfitRedistributedUtilityWorkspace(X,y,family='softvote')
    cached=CachedCrossfitTransferUtility(X,y,family='softvote')
    assert uncached.splits==cached.splits
    for fs in [[0],[1,3],[0,1,2,3]]:
        assert abs(uncached.evaluate(fs)['utility']-cached.evaluate(fs)['utility'])<1e-12
