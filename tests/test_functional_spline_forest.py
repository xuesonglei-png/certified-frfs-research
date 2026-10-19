import numpy as np
from scipy.integrate import quad
from certified_frfs.functional_spline_forest import (hat_integral_gram,functional_contrast_basis,fit_functional_spline_forest)
from certified_frfs.spline_forest import spline_memberships,spline_scores,spline_box_margins


def test_analytic_hat_gram_equals_quadrature_and_basis_functions_have_zero_mean_unit_inner_products():
    knots = np.array([0.,.1,.45,.7,1.]); H = hat_integral_gram(knots)
    for i in range(5):
        for j in range(5):
            def integrand(x):
                z = spline_memberships(np.array([[x]]),knots)[0,0]
                return z[i]*z[j]
            exact = quad(integrand,0,1,points=knots[1:-1],epsabs=1e-12)[0]
            np.testing.assert_allclose(H[i,j],exact,atol=1e-12)
    basis = functional_contrast_basis(knots)
    np.testing.assert_allclose((H@np.ones(5))@basis,0,atol=1e-12)
    np.testing.assert_allclose(basis.T@H@basis,np.eye(4),atol=1e-12)


def test_refinement_preserves_function_norm_scores_and_exact_boxes_but_nodal_penalty_changes():
    coarse,fine = np.array([0.,.5,1.]),np.array([0.,.25,.5,.75,1.])
    Hc,Hf = hat_integral_gram(coarse),hat_integral_gram(fine)
    P = spline_memberships(fine[:,None],coarse)[:,0]
    np.testing.assert_allclose(P.T@Hf@P,Hc,atol=1e-12)
    linear_c,linear_f = coarse-.5,fine-.5
    np.testing.assert_allclose(linear_c@Hc@linear_c,1/12,atol=1e-12)
    np.testing.assert_allclose(linear_f@Hf@linear_f,1/12,atol=1e-12)
    assert np.isclose(linear_c@linear_c,.5) and np.isclose(linear_f@linear_f,.625)
    rng = np.random.default_rng(861); X = rng.uniform(size=(30,3)); y = np.array([0,1,2]*10)
    fit = fit_functional_spline_forest(X,y,[0,1,2],[[0,1],[1,2]],knots=coarse,C=.03)
    W,V = np.array(fit['coef']),np.array(fit['pair_coef'])
    Wfine = np.einsum('sk,qdk->qds',P,W); Vfine = np.einsum('sk,qekl,tl->qest',P,V,P)
    refined = {**fit,'knots':np.tile(fine,(3,1)).tolist(),'coef':Wfine.tolist(),'pair_coef':Vfine.tolist()}
    np.testing.assert_allclose(spline_scores(X,refined),spline_scores(X,fit),atol=1e-10)
    np.testing.assert_allclose(spline_box_margins(X,y,refined,eps=.2),spline_box_margins(X,y,fit,eps=.2),atol=1e-10)
    for c in range(3):
        for a in range(3): np.testing.assert_allclose(W[c,a]@Hc@W[c,a],Wfine[c,a]@Hf@Wfine[c,a],atol=1e-10)
        for e in range(2):
            coarse_norm = np.trace(V[c,e].T@Hc@V[c,e]@Hc)
            fine_norm = np.trace(Vfine[c,e].T@Hf@Vfine[c,e]@Hf)
            np.testing.assert_allclose(coarse_norm,fine_norm,atol=1e-10)


def test_fitted_functional_scores_are_raw_mean_centered_and_pairs_have_zero_integrated_marginals():
    rng = np.random.default_rng(754); X = rng.uniform(size=(24,4)); y = np.array([0,1]*12)
    fit = fit_functional_spline_forest(X,y,[0,1,3],[[0,1],[1,3]],C=.01)
    mean = hat_integral_gram(np.array(fit['knots'][0]))@np.ones(5)
    np.testing.assert_allclose(np.array(fit['coef'])@mean,0,atol=1e-10)
    V = np.array(fit['pair_coef'])
    np.testing.assert_allclose(V@mean,0,atol=1e-10)
    np.testing.assert_allclose(np.einsum('k,qekl->qel',mean,V),0,atol=1e-10)
    np.testing.assert_allclose(np.array(fit['coef'])[:,2],0,atol=1e-12)


def test_full_score_raw_domain_integral_equals_sum_of_orthogonal_unary_and_pair_norms():
    import itertools
    rng = np.random.default_rng(930); X = rng.uniform(size=(24,3)); y = np.array([0,1]*12)
    fit = fit_functional_spline_forest(X,y,[0,1,2],[[0,1],[1,2]],knots=[0,.5,1.],C=.01)
    H = hat_integral_gram(np.array(fit['knots'][0])); W,V = np.array(fit['coef']),np.array(fit['pair_coef'])
    # Two-point Gauss integration on each knot cell exactly integrates the
    # squared multi-affine score, without relying on its ANOVA decomposition.
    nodes,weights = np.polynomial.legendre.leggauss(2); locations = []; masses = []
    for lo,hi in [(0.,.5),(.5,1.)]:
        locations.extend(((lo+hi)/2+nodes*(hi-lo)/2).tolist()); masses.extend((weights*(hi-lo)/2).tolist())
    assignments = list(itertools.product(range(4),repeat=3))
    grid = np.array([[locations[j] for j in state] for state in assignments])
    measure = np.array([np.prod([masses[j] for j in state]) for state in assignments])
    scores = spline_scores(grid,fit)-np.array(fit['intercept'])
    for c in range(2):
        decomposed = sum(W[c,a]@H@W[c,a] for a in range(3))+sum(np.trace(V[c,e].T@H@V[c,e]@H) for e in range(2))
        np.testing.assert_allclose(measure@(scores[:,c]**2),decomposed,atol=1e-10)
