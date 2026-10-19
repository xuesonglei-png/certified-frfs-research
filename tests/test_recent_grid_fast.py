import numpy as np
import pytest
from certified_frfs.recent import sfrip_dependency_from_distances, select_sfrip
from certified_frfs.recent_budget import fgs_budget_path, sfrip_budget_path
from certified_frfs.recent_grid_fast import sfrip_dependency_fast, sfrip_paths_grid, fgs_paths_grid


@pytest.mark.parametrize('labels', [[0]*7, [0]*6+[1], [0, 1, 2, 0, 1, 2, 3]])
def test_vectorized_sfrip_matches_tied_distances_rare_classes_and_no_enemies(labels):
    y = np.array(labels)
    X = np.random.default_rng(619).integers(0, 3, size=(7, 3))/2
    D = np.sqrt(((X[:, None]-X[None])**2).sum(axis=-1))
    for beta in (2., 3., 5.):
        for confidence in (.6, .8, 1.):
            for k in (1, 10):
                expected = sfrip_dependency_from_distances(D, y, beta=beta, confidence=confidence, k=k)
                actual = sfrip_dependency_fast(D, y, beta=beta, confidence=confidence, k=k)
                assert expected == actual


def test_batched_paths_reproduce_every_published_native_and_adapted_choice():
    X = np.random.default_rng(67).integers(0, 5, size=(19, 4))/4
    y = np.arange(19)%3
    fgs_configs = [{'beta': b, 'xi': x} for b in (.6, .8, .9) for x in (.5, 1., 2.)]
    for config, fast in zip(fgs_configs, fgs_paths_grid(X, y, fgs_configs)):
        assert fast == fgs_budget_path(X, y, **config)
    sfrip_configs = [{'beta': b, 'confidence': c} for b in (2., 3., 5.) for c in (.6, .8)]
    for config, fast in zip(sfrip_configs, sfrip_paths_grid(X, y, sfrip_configs)):
        assert fast == sfrip_budget_path(X, y, **config)
        assert fast['native']['features'] == select_sfrip(X, y, **config)['features']
