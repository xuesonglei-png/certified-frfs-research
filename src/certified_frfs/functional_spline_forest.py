"""Raw-domain L2-normalized fuzzy score basis, invariant to knot refinement.

Nodal Euclidean penalties change even for unchanged functions when knots are
refined. This basis makes unary/pure-pair coefficient norms equal integrated
function norms on the declared uniform raw cube. No accuracy guarantee.
"""
import numpy as np
from scipy.linalg import helmert
from sklearn.linear_model import LogisticRegression
from .spline_forest import validate_knots,validate_edges,spline_memberships
from .linear_retention import linear_parameters


def hat_integral_gram(knots):
    knots = np.asarray(knots,float)
    if knots.ndim != 1: raise ValueError('one coordinate knot vector required')
    validate_knots(knots,1); widths = np.diff(knots); k = len(knots); H = np.zeros((k,k))
    for j,h in enumerate(widths):
        H[j,j] += h/3; H[j+1,j+1] += h/3
        H[j,j+1] = H[j+1,j] = h/6
    return H


def functional_contrast_basis(knots):
    knots = np.asarray(knots,float); H = hat_integral_gram(knots); k = len(knots)
    mean = H@np.ones(k); raw = helmert(k,full=False).T
    centered = raw-np.ones((k,1))@(mean@raw)[None]
    chol = np.linalg.cholesky(centered.T@H@centered)
    basis = np.linalg.solve(chol,centered.T).T
    return basis


def fit_functional_spline_forest(X,y,features,edges,*,knots=(0.,.25,.5,.75,1.),C=1.,pair_penalty=1.):
    X,y = np.asarray(X,float),np.asarray(y); fs = list(features)
    if X.ndim != 2 or y.shape != (len(X),) or len(set(fs)) != len(fs) or any(int(a) != a or not 0 <= a < X.shape[1] for a in fs):
        raise ValueError('invalid reference/support')
    if not np.isfinite(C) or C <= 0 or not np.isfinite(pair_penalty) or pair_penalty <= 0:
        raise ValueError('positive C and pair penalty required')
    knots = validate_knots(knots,X.shape[1]); d,k = knots.shape; width = k-1
    basis = np.stack([functional_contrast_basis(t) for t in knots])
    Z = spline_memberships(X,knots); contrasts = np.einsum('ndk,dki->ndi',Z,basis)
    edges = [edge for edge in validate_edges(edges,d) if all(a in fs for a in edge)]
    classes = np.unique(y); q = len(classes)
    if q < 2: raise ValueError('at least two training classes required')
    W,V,bias = np.zeros((q,d,k)),np.zeros((q,len(edges),k,k)),np.zeros(q)
    if fs:
        blocks = [contrasts[:,fs].reshape(len(X),-1)]
        blocks += [(contrasts[:,u,:,None]*contrasts[:,v,None,:]).reshape(len(X),width**2)/np.sqrt(pair_penalty) for u,v in edges]
        design = np.column_stack(blocks)
        model = LogisticRegression(C=C,class_weight='balanced',max_iter=4000,tol=1e-8,random_state=0).fit(design,y)
        coef,bias,classes = linear_parameters(model)
        W[:,fs] = np.einsum('qdi,dki->qdk',coef[:,:width*len(fs)].reshape(q,len(fs),width),basis[fs])
        for e,(u,v) in enumerate(edges):
            start = width*len(fs)+e*width**2
            block = coef[:,start:start+width**2].reshape(q,width,width)/np.sqrt(pair_penalty)
            V[:,e] = np.einsum('ki,qij,lj->qkl',basis[u],block,basis[v])
        W -= W[:1].copy(); V -= V[:1].copy(); bias -= bias[:1].copy()
        iterations = int(model.n_iter_.max()); converged = iterations < 4000
    else: iterations,converged = 0,True
    return {'knots':knots.tolist(),'coef':W.tolist(),'pair_coef':V.tolist(),'intercept':bias.tolist(),
        'features':fs,'edges':edges,'classes':classes.tolist(),'C':float(C),'pair_penalty':float(pair_penalty),
        'class_balanced':True,'optimizer_converged':converged,'optimizer_iterations':iterations,
        'teacher_guards_enforced':False,'penalty_basis':'raw_uniform_L2_centered_unary_and_pure_pair',
        'scope':'uniform_raw_domain_functional_regularization_not_population_accuracy_or_novelty_claim'}
