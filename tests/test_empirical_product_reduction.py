import itertools
import numpy as np
import pytest
from certified_frfs.empirical_product_reduction import marginal_product_anova,feasible_count_refinement
from certified_frfs.forest_anova_reduction import omission_energy
from certified_frfs.spline_forest import fit_spline_forest,spline_scores,spline_box_margins,spline_memberships


def test_empirical_product_energy_equals_exhaustive_conditional_projection_integral():
    X = np.array([[0.,.2,1.],[.1,.6,.5],[.8,1.,0.],[1.,.4,.7]])
    y = np.array([0,1,0,1]); weights = np.array([.1,.2,.3,.4])
    fit = fit_spline_forest(X,y,[0,1,2],[[0,1],[1,2]],knots=[0,.5,1],C=.1)
    teacher,energy,law = marginal_product_anova(fit,X,weights)
    np.testing.assert_allclose(spline_scores(X,fit),spline_scores(X,teacher),atol=1e-10)
    np.testing.assert_allclose(spline_box_margins(X,y,fit,eps=.2),spline_box_margins(X,y,teacher,eps=.2),atol=1e-10)
    assignments = list(itertools.product(range(4),repeat=3))
    grid = np.array([[X[state[a],a] for a in range(3)] for state in assignments])
    masses = np.array([np.prod([weights[i] for i in state]) for state in assignments])
    full = spline_scores(grid,teacher); Z = spline_memberships(grid,teacher['knots']); W = np.array(teacher['coef']); V = np.array(teacher['pair_coef'])
    mu = np.array(law['marginal_means'])
    np.testing.assert_allclose(np.einsum('qdk,dk->qd',W,mu),0,atol=1e-10)
    for e,(u,v) in enumerate(teacher['edges']):
        np.testing.assert_allclose(V[:,e]@mu[v],0,atol=1e-10)
        np.testing.assert_allclose(np.einsum('k,qkl->ql',mu[u],V[:,e]),0,atol=1e-10)
    for size in range(4):
        for fs in itertools.combinations(range(3),size):
            projected = np.tile(teacher['intercept'],(len(grid),1))
            for a in fs: projected += Z[:,a]@W[:,a].T
            for e,(u,v) in enumerate(teacher['edges']):
                if u in fs and v in fs: projected += np.einsum('nk,qkl,nl->nq',Z[:,u],V[:,e],Z[:,v])
            residual = full-projected; residual -= residual.mean(axis=1,keepdims=True)
            np.testing.assert_allclose(masses@np.mean(residual**2,axis=1),omission_energy(energy,fs),atol=1e-10)
            # Direct averaging over omitted empirical coordinates verifies the
            # conditional expectation without using zero-marginal arguments.
            for state,expected in zip(assignments,projected):
                ids = [j for j,t in enumerate(assignments) if all(t[a] == state[a] for a in fs)]
                conditional = masses[ids]/masses[ids].sum()
                np.testing.assert_allclose(conditional@full[ids],expected,atol=1e-10)


def test_feasible_same_count_rejects_global_DP_and_keeps_or_improves_fallback():
    energy = {'unary_energy':[1.,4.,3.,2.],'edges':[[0,1],[1,2],[2,3]],'pair_energy':[0.,1.,0.]}
    def certificate(fs): return {'features':fs,'certificate':{'feasible':0 in fs}}
    result = feasible_count_refinement(energy,[0,3],certificate,rounds=2)
    assert result['features'] == [0,1] and len(result['features']) == 2
    assert result['attempts'][0]['source'] == 'global_surrogate_DP' and not result['attempts'][0]['accepted']
    assert result['final_loss'] < result['initial_loss']
    assert all(not a['accepted'] or a['candidate']['certificate']['feasible'] for a in result['attempts'])
    with pytest.raises(ValueError): feasible_count_refinement(energy,[1,2],certificate)


def test_degenerate_marginal_coordinates_require_no_matrix_inverse():
    X = np.tile([.3,.7],(6,1)); y = np.array([0,1]*3)
    fit = fit_spline_forest(X,y,[0,1],[[0,1]],C=.1)
    _,energy,_ = marginal_product_anova(fit,X,np.ones(6)/6)
    assert energy['total_energy'] < 1e-10
