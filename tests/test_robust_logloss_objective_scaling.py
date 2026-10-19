import numpy as np
from certified_frfs.forest_robust_logloss import fit_robust_forest_logloss
from certified_frfs.forest_robust_logloss_scaled import fit_robust_forest_logloss as scaled_fit


def test_positive_objective_scaling_preserves_normalized_optimum_and_separation():
    X = np.random.default_rng(639).uniform(size=(24,2)); y = np.arange(24)%3
    original = fit_robust_forest_logloss(X,y,[0,1],[[0,1]],eps=.02)
    scaled = scaled_fit(X,y,[0,1],[[0,1]],eps=.02)
    assert original['usable'] and scaled['usable'] and scaled['objective_scale'] == len(X)
    np.testing.assert_allclose(scaled['exact_envelope_objective'],original['exact_envelope_objective'],atol=2e-7)
    assert scaled['exact_envelope_objective']-scaled['relaxation_objective'] <= scaled['separation_tol']+1e-8
    assert original['eps'] == scaled['eps'] and original['C'] == scaled['C']
