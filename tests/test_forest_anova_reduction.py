import itertools
import numpy as np
import pytest
from certified_frfs.forest_anova_reduction import (functional_score_energies,projected_support_fit,omission_energy,forest_energy_optimal_support,canonical_raw_anova)
from certified_frfs.functional_spline_forest import fit_functional_spline_forest
from certified_frfs.spline_forest import spline_scores


@pytest.mark.parametrize('edges',[[[0,1],[1,2],[1,3],[3,4]],[[0,1],[2,3]],[]])
def test_forest_cardinality_dp_equals_exhaustive_all_subsets_for_every_count(edges):
    rng = np.random.default_rng(54+len(edges)); unary = rng.uniform(size=6); pair = rng.uniform(size=len(edges))
    for count in range(7):
        result = forest_energy_optimal_support(unary,edges,pair,count)
        energies = [(sum(unary[a] for a in fs)+sum(pair[e] for e,(u,v) in enumerate(edges) if u in fs and v in fs),fs)
            for fs in itertools.combinations(range(6),count)]
        best = max(v for v,_ in energies)
        np.testing.assert_allclose(result['retained_energy'],best,atol=1e-12)
        assert len(result['features']) == count
        assert result['features'] in [list(fs) for value,fs in energies if abs(value-best) < 1e-12]


def test_conditional_projection_error_equals_exact_sum_omission_energy_for_all_supports():
    rng = np.random.default_rng(753); X = rng.uniform(size=(30,3)); y = np.array([0,1,2]*10)
    fit = fit_functional_spline_forest(X,y,[0,1,2],[[0,1],[1,2]],knots=[0,.5,1],C=.01)
    energies = functional_score_energies(fit)
    # Tensor Gauss rule integrates squared multi-affine difference exactly.
    nodes,weights = np.polynomial.legendre.leggauss(2); locations = []; masses = []
    for lo,hi in [(0.,.5),(.5,1.)]:
        locations.extend(((lo+hi)/2+nodes*(hi-lo)/2).tolist()); masses.extend((weights*(hi-lo)/2).tolist())
    states = list(itertools.product(range(4),repeat=3))
    grid = np.array([[locations[j] for j in state] for state in states]); measure = np.array([np.prod([masses[j] for j in state]) for state in states])
    full = spline_scores(grid,fit); full -= full.mean(axis=1,keepdims=True)
    for bits in itertools.product([0,1],repeat=3):
        fs = [a for a in range(3) if bits[a]]; projected = projected_support_fit(fit,fs)
        reduced = spline_scores(grid,projected); reduced -= reduced.mean(axis=1,keepdims=True)
        exact = float(measure@np.mean((full-reduced)**2,axis=1))
        np.testing.assert_allclose(omission_energy(energies,fs),exact,atol=1e-10)
    common = {**fit,'coef':(np.array(fit['coef'])+rng.normal(size=(1,3,3))).tolist(),
        'pair_coef':(np.array(fit['pair_coef'])+rng.normal(size=(1,2,3,3))).tolist()}
    shifted = functional_score_energies(common)
    np.testing.assert_allclose(shifted['unary_energy'],energies['unary_energy'],atol=1e-10)
    np.testing.assert_allclose(shifted['pair_energy'],energies['pair_energy'],atol=1e-10)


def test_non_anova_model_and_invalid_budgets_are_rejected():
    with pytest.raises(ValueError): forest_energy_optimal_support([1,2],[],[],3)
    with pytest.raises(ValueError): forest_energy_optimal_support([1,-1],[],[],1)
    fit = {'coef':np.ones((2,2,3)).tolist(),'pair_coef':[],'edges':[],'classes':[0,1],
        'knots':[[0,.5,1],[0,.5,1]]}
    fit['coef'][1][0][0] = 5.
    with pytest.raises(ValueError): functional_score_energies(fit)


def test_raw_anova_conversion_preserves_arbitrary_scores_and_exact_box_margins():
    from certified_frfs.spline_forest import spline_box_margins
    rng = np.random.default_rng(727); q,d,k = 3,4,5
    fit = {'coef':rng.normal(size=(q,d,k)).tolist(),'pair_coef':rng.normal(size=(q,d-1,k,k)).tolist(),
        'edges':[[0,1],[1,2],[2,3]],'classes':[0,1,2],'intercept':rng.normal(size=q).tolist(),
        'knots':np.tile([0.,.2,.4,.8,1.],(d,1)).tolist(),'features':list(range(d))}
    converted = canonical_raw_anova(fit); X = rng.uniform(size=(30,d)); y = np.array([0,1,2]*10)
    np.testing.assert_allclose(spline_scores(X,converted),spline_scores(X,fit),atol=1e-11)
    np.testing.assert_allclose(spline_box_margins(X,y,converted,eps=.2),spline_box_margins(X,y,fit,eps=.2),atol=1e-11)
    energies = functional_score_energies(converted); assert energies['total_energy'] >= 0
