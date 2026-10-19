import itertools
import numpy as np
import pytest
from certified_frfs.spline_forest import (spline_memberships,spline_scores,fit_spline_forest,spline_box_margins)
from certified_frfs.triangular_classifier import triangular_memberships
from certified_frfs.orthogonal_forest_logistic import fit_orthogonal_forest,orthogonal_scores
from certified_frfs.forest_fuzzy_classifier import forest_box_margins


def test_partition_of_unity_coordinate_specific_knots_and_three_knot_equivalence():
    X = np.random.default_rng(442).uniform(size=(50,4)); knots = np.array([[0,.2,.4,.8,1],[0,.1,.3,.6,1],[0,.3,.7,.9,1],[0,.4,.5,.6,1]])
    Z = spline_memberships(X,knots); np.testing.assert_allclose(Z.sum(axis=-1),1.,atol=1e-12)
    assert np.all(Z >= 0)
    for a in range(4):
        points = np.full((5,4),.5); points[:,a] = knots[a]
        np.testing.assert_allclose(spline_memberships(points,knots)[:,a],np.eye(5),atol=1e-12)
    np.testing.assert_allclose(spline_memberships(X,[0,.5,1]),triangular_memberships(X),atol=1e-12)


def test_three_knot_refits_scores_and_exact_boxes_equal_original_implementation():
    rng = np.random.default_rng(642); X = rng.uniform(size=(40,4)); y = np.array([0,1,2,0]*10); edges = [[0,1],[1,2],[1,3]]
    old = fit_orthogonal_forest(X,y,[0,1,2,3],edges,C=.3,pair_penalty=3.)
    new = fit_spline_forest(X,y,[0,1,2,3],edges,knots=[0,.5,1],C=.3,pair_penalty=3.)
    np.testing.assert_allclose(spline_scores(X,new),orthogonal_scores(X,old),atol=1e-7,rtol=1e-7)
    oldmargin = forest_box_margins(X,y,old['coef'],old['intercept'],old['classes'],edges,old['pair_coef'],eps=.2)
    np.testing.assert_allclose(spline_box_margins(X,y,new,eps=.2),oldmargin,atol=1e-7,rtol=1e-7)


@pytest.mark.parametrize('k',[2,5,7])
def test_exact_variable_knot_forest_minimum_equals_exhaustive_cells_and_witnesses(k):
    rng = np.random.default_rng(186+k); q,d = 3,3
    knots = np.sort(np.column_stack([np.zeros(d),rng.uniform(.05,.95,size=(d,k-2)),np.ones(d)]),axis=1)
    fit = {'knots':knots.tolist(),'coef':rng.normal(size=(q,d,k)).tolist(),'pair_coef':rng.normal(size=(q,2,k,k)).tolist(),
        'intercept':rng.normal(size=q).tolist(),'edges':[[0,1],[1,2]],'classes':[0,1,2]}
    X = rng.uniform(size=(3,d)); y = np.array([0,1,2]); eps = .35
    result = spline_box_margins(X,y,fit,eps=eps,witnesses=True)
    for i in range(3):
        lo,hi = np.maximum(0,X[i]-eps),np.minimum(1,X[i]+eps)
        grid = np.array(list(itertools.product(*[[lo[a],hi[a]]+[t for t in knots[a,1:-1] if lo[a] <= t <= hi[a]] for a in range(d)])))
        scores = spline_scores(grid,fit)
        exact = min(np.min(scores[:,y[i]]-scores[:,r]) for r in range(q) if r != y[i])
        np.testing.assert_allclose(result['margins'][i],exact,atol=1e-10)
    witness = np.array(result['witness_raw']); assert np.max(abs(witness-X)) <= eps+1e-12
    scores = spline_scores(witness,fit); rivals = result['rivals']
    np.testing.assert_allclose(scores[np.arange(3),y]-scores[np.arange(3),rivals],result['margins'],atol=1e-10)


def test_invalid_partitions_cycles_and_empty_support():
    X = np.random.default_rng(231).uniform(size=(12,3)); y = np.array([0,1]*6)
    for knots in ([0,.5,.5,1],[.1,.5,1],[0,.7,.2,1]):
        with pytest.raises(ValueError): spline_memberships(X,knots)
    with pytest.raises(ValueError): fit_spline_forest(X,y,[0,1,2],[[0,1],[1,2],[0,2]])
    empty = fit_spline_forest(X,y,[],[])
    np.testing.assert_allclose(spline_scores(X,empty),0)
    np.testing.assert_allclose(spline_box_margins(X,y,empty),0)


def test_five_knot_projection_gram_equals_all_direct_raw_substitution_losses():
    from certified_frfs.spline_geometry_stage import SplineProjectionLoss
    rng = np.random.default_rng(748); X = rng.uniform(size=(25,5)); y = np.array([0,1,2,0,1]*5)
    fit = fit_spline_forest(X,y,list(range(5)),[[0,1],[1,2],[2,3],[3,4]])
    loss = SplineProjectionLoss(X,y,fit); q = len(fit['classes'])
    weights = np.array([1/(q*sum(y == c)) for c in y])
    target = spline_scores(X,fit); target -= target.mean(axis=1,keepdims=True)
    for bits in itertools.product([0,1],repeat=5):
        fs = [a for a in range(5) if bits[a]]; substituted = X.copy(); omitted = [a for a in range(5) if not bits[a]]
        substituted[:,omitted] = loss.center[omitted]
        values = spline_scores(substituted,fit); values -= values.mean(axis=1,keepdims=True)
        expected = float(weights @ np.mean((values-target)**2,axis=1))
        np.testing.assert_allclose(loss(fs),expected,atol=1e-11)


def test_spline_stage_preserves_original_reference_certificate():
    from certified_frfs.spline_geometry_stage import SplineGeometryStageContext
    from certified_frfs.feasible_transport import weight_certificate
    rng = np.random.default_rng(641); X = rng.uniform(size=(20,6)); y = np.array([0,1]*10)
    context = SplineGeometryStageContext(X,y); stage = context.fit_policy(fraction=.5)
    np.testing.assert_allclose(context.original_weights,np.ones(6)/np.sqrt(6))
    for point in stage['path']['trace']:
        fs = point['features']; weights = np.array(point['weights'])[fs]
        assert point['certificate'] == weight_certificate(context.A,y,fs,context.original_weights,weights,protected=context.protected)
        assert point['certificate']['feasible']
    assert stage['fit']['features'] == stage['candidate']['features']


def test_nonuniform_five_knot_boxes_equal_independent_marginal_lp_on_larger_tree():
    from scripts.spline_forest_audit_oracle import lp_margins_first,leaf_margins
    rng = np.random.default_rng(849); q,d,k = 3,9,5
    knots = np.sort(np.column_stack([np.zeros(d),rng.uniform(.05,.95,(d,3)),np.ones(d)]),axis=1)
    fit = {'knots':knots.tolist(),'coef':rng.normal(size=(q,d,k)).tolist(),
        'pair_coef':rng.normal(size=(q,d-1,k,k)).tolist(),'intercept':rng.normal(size=q).tolist(),
        'edges':[[a,a+1] for a in range(d-1)],'classes':[0,1,2]}
    X = rng.uniform(size=(2,d)); y = np.array([0,2]); eps = .32
    result = spline_box_margins(X,y,fit,eps=eps,witnesses=True)
    independent,_,_ = leaf_margins(X,y,fit,eps)
    np.testing.assert_allclose(result['pair_margins'],independent,atol=1e-10)
    for rival,minimum in lp_margins_first(X,y,fit,eps).items():
        np.testing.assert_allclose(minimum,result['pair_margins'][0,rival],atol=1e-9)
