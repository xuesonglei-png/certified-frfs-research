import numpy as np
import pytest
from certified_frfs.forest_robust_logloss import fit_robust_forest_logloss
from certified_frfs.forest_robust_logloss_rival_epigraph import fit_robust_forest_logloss as rival_fit


@pytest.mark.parametrize('radius', [0., .02])
def test_rival_only_epigraph_preserves_objective_and_exact_separation(radius):
    X = np.random.default_rng(639).uniform(size=(24, 2))
    y = np.arange(24) % 3
    original = fit_robust_forest_logloss(X, y, [0, 1], [[0, 1]], eps=radius)
    rival = rival_fit(X, y, [0, 1], [[0, 1]], eps=radius)
    assert original['usable'] and rival['usable']
    assert rival['epigraph_layout'] == 'rival_only_equivalent'
    np.testing.assert_allclose(rival['exact_envelope_objective'], original['exact_envelope_objective'], atol=2e-7)
    assert rival['exact_envelope_objective']-rival['relaxation_objective'] <= rival['separation_tol']+1e-8
    own = np.searchsorted(rival['classes'], y)
    np.testing.assert_allclose(np.array(rival['upper'])[np.arange(len(X)), own], 0., atol=1e-8)
    assert original['eps'] == rival['eps'] and original['C'] == rival['C']
