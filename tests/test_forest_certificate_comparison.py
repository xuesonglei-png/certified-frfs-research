import itertools
import numpy as np
import pytest
from scipy.optimize import linprog
from certified_frfs.forest_certificate_comparison import compare_forest_certificates
from certified_frfs.orthogonal_forest_logistic import orthogonal_scores
from test_forest_lipschitz_metric import model


def local_polytope_minimum(X,fit,own,rival,eps):
    """Independent LP; tree local polytope equals its discrete marginal hull."""
    d = len(X); edges = fit['edges']; count = 3*d+9*len(edges)
    lo,hi = np.maximum(0,X-eps),np.minimum(1,X+eps)
    points = np.column_stack([lo,hi,np.clip(.5,lo,hi)])
    membership = np.stack([np.maximum(1-2*points,0),1-abs(2*points-1),np.maximum(2*points-1,0)],axis=-1)
    W = np.array(fit['coef']); V = np.array(fit['pair_coef']); b = np.array(fit['intercept'])
    objective = np.zeros(count); objective[:3*d] = np.einsum('dsk,dk->ds',membership,W[own]-W[rival]).ravel()
    matrix = []; rhs = []
    for a in range(d):
        row = np.zeros(count); row[3*a:3*a+3] = 1; matrix.append(row); rhs.append(1)
    for e,(u,v) in enumerate(edges):
        start = 3*d+9*e
        objective[start:start+9] = np.einsum('sk,kl,tl->st',membership[u],V[own,e]-V[rival,e],membership[v]).ravel()
        for k in range(3):
            row = np.zeros(count); row[start+3*k:start+3*k+3] = 1; row[3*u+k] = -1
            matrix.append(row); rhs.append(0)
            row = np.zeros(count); row[start+k:start+9:3] = 1; row[3*v+k] = -1
            matrix.append(row); rhs.append(0)
    solved = linprog(objective,A_eq=np.array(matrix),b_eq=np.array(rhs),bounds=(0,None),method='highs')
    assert solved.success
    return float(solved.fun+b[own]-b[rival])


@pytest.mark.parametrize('q,d',[(2,3),(3,12)])
def test_exact_margins_equal_independent_tree_marginal_lp_and_dominate_conservative_bounds(q,d):
    fit = model(q,d,982); rng = np.random.default_rng(723); X = rng.uniform(size=(5,d)); y = np.arange(5)%q
    result = compare_forest_certificates(X,y,fit,eps=.2)
    exact = result['exact']
    for i,label in enumerate(y):
        minima = [local_polytope_minimum(X[i],fit,label,r,.2) for r in range(q) if r != label]
        np.testing.assert_allclose(exact['margins'][i],min(minima),atol=1e-10)
        z = np.array(exact['witness_raw'][i]); assert np.max(abs(z-X[i])) <= .2+1e-12
        scores = orthogonal_scores(z[None],fit)[0]
        np.testing.assert_allclose(scores[label]-scores[exact['rivals'][i]],exact['margins'][i],atol=1e-10)
    assert np.all(result['independent_termwise_margins'] <= exact['margins']+1e-10)
    assert np.all(result['lipschitz_margins'] <= exact['margins']+1e-10)


def test_box_crossing_membership_knots_matches_exhaustive_raw_assignments():
    fit = model(3,3,746); X = np.array([[.45,.55,.1]]); y = np.array([1]); eps = .2
    result = compare_forest_certificates(X,y,fit,eps=eps)
    lo,hi = np.maximum(0,X[0]-eps),np.minimum(1,X[0]+eps)
    points = np.column_stack([lo,hi,np.clip(.5,lo,hi)])
    grid = np.array(list(itertools.product(*points))); scores = orthogonal_scores(grid,fit)
    exact = min(np.min(scores[:,1]-scores[:,r]) for r in (0,2))
    np.testing.assert_allclose(result['exact']['margins'],exact,atol=1e-12)
