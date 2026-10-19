import numpy as np
from certified_frfs.precise_balanced_logistic import fit_precise_balanced_logistic
from certified_frfs.logistic_stationarity_refinement import refine_balanced_logistic

def test_binary_and_multiclass_numeric_refinement_decreases_gradient_without_changing_objective():
    rng = np.random.default_rng(809); X = rng.normal(size=(60,9))
    for q in (2,4):
        y = np.arange(60)%q; fit = fit_precise_balanced_logistic(X,y,.1)
        coef = fit['coef'].copy(); coef[1] += 1e-4
        result = refine_balanced_logistic(X,y,coef,fit['intercept'],.1)
        assert result['refinement_accepted'] and result['optimizer_converged']
        assert result['gradient_inf'] < result['initial_gradient_inf']
        assert result['objective'] <= result['initial_objective']+1e-12
        np.testing.assert_allclose(X@result['coef'].T+result['intercept'],X@fit['coef'].T+fit['intercept'],atol=1e-5,rtol=1e-5)
