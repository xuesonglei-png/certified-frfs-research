"""Comparable box certificates for the same fixed fitted forest classifier.

Exact margins, independently minimized unary/edge terms, and contrast-specific
global coordinate Lipschitz bounds. None changes the classifier prediction.
"""
import numpy as np
from .forest_fuzzy_classifier import forest_box_margins
from .orthogonal_forest_logistic import orthogonal_scores
from .triangular_classifier import triangular_memberships
from .forest_lipschitz_metric import forest_contrast_lipschitz


def compare_forest_certificates(X,y,fit,*,eps=.01):
    X,y = np.asarray(X,float),np.asarray(y); q = len(fit['classes']); d = X.shape[1]
    W = np.array(fit['coef']); b = np.array(fit['intercept']); edges = fit['edges']
    V = np.array(fit['pair_coef']).reshape(q,len(edges),3,3)
    exact = forest_box_margins(X,y,W,b,fit['classes'],edges,V,eps=eps,witnesses=True)
    classes = np.array(fit['classes']); own = np.argmax(y[:,None] == classes[None],axis=1)
    lo,hi = np.maximum(0,X-eps),np.minimum(1,X+eps)
    points = np.stack([lo,hi,np.clip(.5,lo,hi)],axis=-1); Z = triangular_memberships(points)
    unary = np.einsum('nqdk,ndsk->nqds',W[own,None]-W[None],Z)
    independent = b[own,None]-b[None]+unary.min(axis=-1).sum(axis=-1)
    for e,(u,v) in enumerate(edges):
        potential = np.einsum('nqkl,nsk,ntl->nqst',V[own,e][:,None]-V[None,:,e],Z[:,u],Z[:,v])
        independent += potential.min(axis=(-1,-2))
    sensitivity = forest_contrast_lipschitz(fit); ell = np.zeros((q,q,d))
    for pair,constants in zip(sensitivity['class_index_pairs'],sensitivity['contrast_constants']):
        c,r = pair; ell[c,r] = ell[r,c] = constants
    scores = orthogonal_scores(X,fit)
    radii = np.maximum(X-lo,hi-X)
    lipschitz = scores[np.arange(len(X)),own,None]-scores-np.einsum('nqd,nd->nq',ell[own],radii)
    independent[np.arange(len(X)),own] = np.inf; lipschitz[np.arange(len(X)),own] = np.inf
    return {'exact':exact,'independent_termwise_margins':independent.min(axis=1),
        'lipschitz_margins':lipschitz.min(axis=1),
        'scope':'same_fixed_forest_numeric_shared_raw_box_exact_vs_conservative_certification_not_population_accuracy'}
