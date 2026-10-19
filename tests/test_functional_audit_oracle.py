import numpy as np
from scripts.functional_spline_audit_oracle import bases,refit_functional
from certified_frfs.functional_spline_forest import functional_contrast_basis,fit_functional_spline_forest


def test_independent_gauss_basis_matches_analytic_integral_whitening_and_refit():
    knots = np.array([[0,.1,.4,.9,1],[0,.2,.45,.6,1],[0,.3,.6,.8,1]])
    basis,_,_ = bases(knots)
    for a in range(3): np.testing.assert_allclose(basis[a],functional_contrast_basis(knots[a]),atol=1e-12)
    rng = np.random.default_rng(778); A = rng.uniform(size=(30,3)); y = np.array([0,1,2]*10)
    fit = fit_functional_spline_forest(A,y,[0,1,2],[[0,1],[1,2]],knots=knots,C=.01)
    diagnostic = refit_functional(A,y,fit)
    assert diagnostic['objective_gap'] < 1e-10 and diagnostic['gradient_inf'] < 2e-6
