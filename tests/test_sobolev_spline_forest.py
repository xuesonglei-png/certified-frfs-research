import numpy as np
from sklearn.linear_model import LogisticRegression
from certified_frfs.precise_balanced_logistic import fit_precise_balanced_logistic
from certified_frfs.sobolev_spline_forest import hat_stiffness,sobolev_basis,fit_sobolev_forest
from certified_frfs.functional_spline_forest import hat_integral_gram
from certified_frfs.spline_forest import spline_memberships,spline_scores,spline_box_margins


def test_precise_solver_matches_independent_binary_and_multiclass_convex_fits():
    rng = np.random.default_rng(721); X = rng.normal(size=(50,7))
    for q in (2,3):
        y = np.arange(50)%q; fit = fit_precise_balanced_logistic(X,y,C=.1)
        reference = LogisticRegression(C=.1,class_weight='balanced',tol=1e-11,max_iter=10000).fit(X,y)
        expected = reference.decision_function(X)
        if q == 2: expected = np.column_stack([np.zeros(len(X)),expected])
        actual = X@fit['coef'].T+fit['intercept']
        np.testing.assert_allclose(actual-actual[:,:1],expected-expected[:,:1],atol=1e-5,rtol=1e-5)
        assert fit['optimizer_converged'] and fit['gradient_inf'] <= 1e-8


def test_stiffness_refinement_and_sobolev_energy_basis():
    coarse,fine = np.array([0.,.5,1.]),np.array([0.,.25,.5,.75,1.])
    P = spline_memberships(fine[:,None],coarse)[:,0]
    np.testing.assert_allclose(P.T@hat_stiffness(fine)@P,hat_stiffness(coarse),atol=1e-12)
    for alpha in (0.,.01,.1):
        H = hat_integral_gram(fine); R = H+alpha*hat_stiffness(fine); B = sobolev_basis(fine,alpha)
        np.testing.assert_allclose(B.T@R@B,np.eye(4),atol=1e-12)
        np.testing.assert_allclose((H@np.ones(5))@B,0,atol=1e-12)


def test_refined_fixed_scores_pair_norms_and_attained_boxes_unchanged():
    rng = np.random.default_rng(866); X = rng.uniform(size=(25,3)); y = np.arange(25)%3
    fit = fit_sobolev_forest(X,y,[0,1,2],[[0,1],[1,2]],C=.1,alpha=.01)
    coarse,fine = np.array([0.,.5,1.]),np.array([0.,.25,.5,.75,1.]); P = spline_memberships(fine[:,None],coarse)[:,0]
    W,V = np.array(fit['coef']),np.array(fit['pair_coef']); Wf = np.einsum('sk,qdk->qds',P,W); Vf = np.einsum('sk,qekl,tl->qest',P,V,P)
    refined = {**fit,'knots':np.tile(fine,(3,1)).tolist(),'coef':Wf.tolist(),'pair_coef':Vf.tolist()}
    np.testing.assert_allclose(spline_scores(X,fit),spline_scores(X,refined),atol=1e-10)
    np.testing.assert_allclose(spline_box_margins(X,y,fit,eps=.2),spline_box_margins(X,y,refined,eps=.2),atol=1e-10)
    Rc = hat_integral_gram(coarse)+.01*hat_stiffness(coarse); Rf = hat_integral_gram(fine)+.01*hat_stiffness(fine)
    for c in range(3):
        for e in range(2):
            np.testing.assert_allclose(np.trace(V[c,e].T@Rc@V[c,e]@Rc),np.trace(Vf[c,e].T@Rf@Vf[c,e]@Rf),atol=1e-10)
    assert fit['optimizer_converged']
