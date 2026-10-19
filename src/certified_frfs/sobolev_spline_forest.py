"""Refinement-invariant raw Sobolev penalty for fuzzy spline forests.

Unary L2+alpha derivative energy; pure-pair tensor norm includes both
partial derivatives and the mixed derivative. Established spline machinery,
not a novelty or test-accuracy claim. Exact box oracle remains unchanged.
"""
import numpy as np
from scipy.linalg import helmert
from .functional_spline_forest import hat_integral_gram
from .spline_forest import validate_knots,validate_edges,spline_memberships
from .precise_balanced_logistic import fit_precise_balanced_logistic


def hat_stiffness(knots):
    knots = np.asarray(knots,float); validate_knots(knots,1); K = np.zeros((len(knots),len(knots)))
    for j,h in enumerate(np.diff(knots)):
        K[j,j] += 1/h; K[j+1,j+1] += 1/h; K[j,j+1] -= 1/h; K[j+1,j] -= 1/h
    return K


def sobolev_basis(knots,alpha):
    if not np.isfinite(alpha) or alpha < 0: raise ValueError('nonnegative smoothness required')
    H = hat_integral_gram(knots); mean = H@np.ones(len(knots)); R = H+alpha*hat_stiffness(knots)
    raw = helmert(len(knots),full=False).T; centered = raw-mean@raw
    L = np.linalg.cholesky(centered.T@R@centered)
    return np.linalg.solve(L,centered.T).T


def fit_sobolev_forest(X,y,features,edges,*,C=1.,alpha=.01,pair_penalty=1.,knots=(0.,.5,1.)):
    X,y = np.asarray(X,float),np.asarray(y); fs = list(features)
    if X.ndim != 2 or y.shape != (len(X),) or len(set(fs)) != len(fs) or any(int(a) != a or not 0 <= a < X.shape[1] for a in fs):
        raise ValueError('invalid support/reference')
    if not np.isfinite(pair_penalty) or pair_penalty <= 0: raise ValueError('positive pair penalty required')
    knots = validate_knots(knots,X.shape[1]); d,k = knots.shape; width = k-1
    basis = np.stack([sobolev_basis(t,alpha) for t in knots]); contrasts = np.einsum('ndk,dki->ndi',spline_memberships(X,knots),basis)
    edges = [e for e in validate_edges(edges,d) if all(a in fs for a in e)]
    blocks = [contrasts[:,fs].reshape(len(X),-1)]
    blocks += [(contrasts[:,u,:,None]*contrasts[:,v,None,:]).reshape(len(X),width**2)/np.sqrt(pair_penalty) for u,v in edges]
    model = fit_precise_balanced_logistic(np.column_stack(blocks),y,C=C)
    q = len(model['classes']); W,V = np.zeros((q,d,k)),np.zeros((q,len(edges),k,k))
    W[:,fs] = np.einsum('qdi,dki->qdk',model['coef'][:,:width*len(fs)].reshape(q,len(fs),width),basis[fs])
    for e,(u,v) in enumerate(edges):
        start = width*len(fs)+e*width**2
        block = model['coef'][:,start:start+width**2].reshape(q,width,width)/np.sqrt(pair_penalty)
        V[:,e] = np.einsum('ki,qij,lj->qkl',basis[u],block,basis[v])
    return {'knots':knots.tolist(),'coef':W.tolist(),'pair_coef':V.tolist(),'intercept':model['intercept'].tolist(),
        'classes':model['classes'].tolist(),'features':fs,'edges':edges,'C':float(C),'pair_penalty':float(pair_penalty),'alpha':float(alpha),
        'class_balanced':True,'teacher_guards_enforced':False,
        **{k:v for k,v in model.items() if k.startswith('optimizer') or k in ('gradient_inf','objective','objective_strength')},
        'penalty_basis':'raw_centered_Sobolev_tensor_product_L2_plus_alpha_stiffness',
        'scope':'smoothness_regularized_uncertified_classifier_with_exact_fixed_model_box_oracle'}
