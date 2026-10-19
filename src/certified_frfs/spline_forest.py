"""Piecewise-linear fuzzy partitions and exact shared-coordinate forest boxes.

Each coordinate has its own strictly increasing knots spanning [0,1]. Every
class contrast is multi-affine within each raw-coordinate knot cell. Forest
min-sum over endpoints/interior knots attains the true finite-box minimum.
This extends representation capacity, not generic graphical-model novelty.
"""
import numpy as np
from scipy.linalg import helmert
from sklearn.linear_model import LogisticRegression
from .linear_retention import linear_parameters


def validate_knots(knots,d):
    knots = np.asarray(knots,float)
    if knots.ndim == 1: knots = np.tile(knots,(d,1))
    if knots.ndim != 2 or knots.shape[0] != d or knots.shape[1] < 2 or not np.isfinite(knots).all():
        raise ValueError('finite coordinate-specific knots required')
    if np.any(knots[:,0] != 0) or np.any(knots[:,-1] != 1) or np.any(np.diff(knots,axis=1) <= 0):
        raise ValueError('strictly increasing knots must span [0,1]')
    return knots


def spline_memberships(X,knots):
    X = np.asarray(X,float)
    if X.ndim < 1 or X.shape[-1] < 1 or not np.isfinite(X).all() or np.any((X < 0)|(X > 1)):
        raise ValueError('finite normalized coordinates required')
    knots = validate_knots(knots,X.shape[-1]); k = knots.shape[1]
    Z = np.zeros(X.shape+(k,))
    for j in range(k):
        left = np.ones_like(X) if j == 0 else (X-knots[:,j-1])/(knots[:,j]-knots[:,j-1])
        right = np.ones_like(X) if j == k-1 else (knots[:,j+1]-X)/(knots[:,j+1]-knots[:,j])
        Z[...,j] = np.clip(np.minimum(left,right),0,1)
    return Z


def validate_edges(edges,d):
    parents = list(range(d)); output = []
    def root(a):
        while parents[a] != a:
            parents[a] = parents[parents[a]]; a = parents[a]
        return a
    for edge in edges:
        if len(edge) != 2 or any(int(a) != a for a in edge): raise ValueError('integer canonical edges required')
        u,v = map(int,edge)
        if not 0 <= u < v < d or [u,v] in output: raise ValueError('unique canonical edges required')
        ru,rv = root(u),root(v)
        if ru == rv: raise ValueError('interaction graph must be a forest')
        parents[ru] = rv; output.append([u,v])
    return output


def parameters(X,fit):
    X = np.asarray(X,float)
    if X.ndim != 2: raise ValueError('raw matrix required')
    knots = validate_knots(fit['knots'],X.shape[1]); Z = spline_memberships(X,knots)
    W,b = np.asarray(fit['coef'],float),np.asarray(fit['intercept'],float)
    q,d,k = len(b),X.shape[1],knots.shape[1]; edges = validate_edges(fit['edges'],d)
    V = np.asarray(fit['pair_coef'],float).reshape(q,len(edges),k,k)
    if q < 2 or W.shape != (q,d,k) or b.shape != (q,) or not all(np.isfinite(a).all() for a in (W,b,V)):
        raise ValueError('invalid spline forest coefficients')
    return X,knots,Z,W,b,edges,V


def spline_scores(X,fit):
    X,knots,Z,W,b,edges,V = parameters(X,fit)
    scores = np.einsum('ndk,qdk->nq',Z,W)+b
    for e,(u,v) in enumerate(edges): scores += np.einsum('nk,qkl,nl->nq',Z[:,u],V[:,e],Z[:,v])
    return scores


def fit_spline_forest(X,y,features,edges,*,knots=(0.,.25,.5,.75,1.),C=1.,pair_penalty=1.):
    X,y = np.asarray(X,float),np.asarray(y); fs = list(features)
    if X.ndim != 2 or y.shape != (len(X),) or len(set(fs)) != len(fs) or any(int(a) != a or not 0 <= a < X.shape[1] for a in fs):
        raise ValueError('invalid training reference/support')
    if not np.isfinite(C) or C <= 0 or not np.isfinite(pair_penalty) or pair_penalty <= 0:
        raise ValueError('positive regularization required')
    knots = validate_knots(knots,X.shape[1]); k = knots.shape[1]; basis = helmert(k,full=False).T
    Z = spline_memberships(X,knots); contrasts = Z@basis
    edges = [edge for edge in validate_edges(edges,X.shape[1]) if all(a in fs for a in edge)]
    classes = np.unique(y); q,d = len(classes),X.shape[1]
    if q < 2: raise ValueError('at least two classes required')
    W,V,bias = np.zeros((q,d,k)),np.zeros((q,len(edges),k,k)),np.zeros(q)
    if fs:
        width = k-1; blocks = [contrasts[:,fs].reshape(len(X),-1)]
        blocks += [(contrasts[:,u,:,None]*contrasts[:,v,None,:]).reshape(len(X),width**2)/np.sqrt(pair_penalty) for u,v in edges]
        design = np.column_stack(blocks)
        fitted = LogisticRegression(C=C,class_weight='balanced',max_iter=4000,tol=1e-8,random_state=0).fit(design,y)
        coef,bias,classes = linear_parameters(fitted)
        W[:,fs] = np.einsum('qdi,ki->qdk',coef[:,:width*len(fs)].reshape(q,len(fs),width),basis)
        for e in range(len(edges)):
            start = width*len(fs)+e*width**2
            block = coef[:,start:start+width**2].reshape(q,width,width)/np.sqrt(pair_penalty)
            V[:,e] = np.einsum('ki,qij,lj->qkl',basis,block,basis)
        W -= W[:1].copy(); V -= V[:1].copy(); bias -= bias[:1].copy()
        iterations = int(fitted.n_iter_.max()); converged = iterations < 4000
    else: iterations,converged = 0,True
    return {'knots':knots.tolist(),'coef':W.tolist(),'pair_coef':V.tolist(),'intercept':bias.tolist(),
        'features':fs,'edges':edges,'classes':classes.tolist(),'C':float(C),'pair_penalty':float(pair_penalty),
        'class_balanced':True,'optimizer_converged':converged,'optimizer_iterations':iterations,
        'teacher_guards_enforced':False,'scope':'piecewise_linear_pure_interaction_common_classifier_no_population_guarantee'}


def spline_box_margins(X,y,fit,*,eps=.01,witnesses=False):
    X,knots,Z,W,b,edges,V = parameters(X,fit); y = np.asarray(y); classes = np.array(fit['classes'])
    n,d = X.shape; q,k = len(b),knots.shape[1]
    if y.shape != (n,) or classes.shape != (q,) or len(np.unique(classes)) != q or not np.all(np.isin(y,classes)):
        raise ValueError('invalid query labels/classes')
    if not np.isfinite(eps) or eps < 0: raise ValueError('nonnegative finite radius required')
    own = np.argmax(y[:,None] == classes[None],axis=1)
    lo,hi = np.maximum(0,X-eps),np.minimum(1,X+eps)
    points = np.concatenate([lo[:,:,None],hi[:,:,None],np.clip(knots[None,:,1:-1],lo[:,:,None],hi[:,:,None])],axis=-1)
    M = spline_memberships(points.transpose(0,2,1),knots).transpose(0,2,1,3)
    cost = np.einsum('nqdk,ndsk->nqds',W[own,None]-W[None],M)
    adjacency = [[] for _ in range(d)]; potentials = []
    for e,(u,v) in enumerate(edges):
        potentials.append(np.einsum('nqkl,nsk,ntl->nqst',V[own,e][:,None]-V[None,:,e],M[:,u],M[:,v]))
        adjacency[u].append((v,e)); adjacency[v].append((u,e))
    parent = np.full(d,-2,int); pe = np.full(d,-1,int); order,roots = [],[]
    for start in range(d):
        if parent[start] != -2: continue
        roots.append(start); parent[start] = -1; stack = [start]
        while stack:
            node = stack.pop(); order.append(node)
            for neighbor,e in adjacency[node]:
                if neighbor == parent[node]: continue
                parent[neighbor] = node; pe[neighbor] = e; stack.append(neighbor)
    choices = {}
    for node in reversed(order):
        p = parent[node]
        if p < 0: continue
        e = pe[node]; potential = potentials[e] if edges[e][0] == p else potentials[e].swapaxes(-1,-2)
        values = cost[:,:,node,None,:]+potential
        choices[node] = values.argmin(axis=-1); cost[:,:,p] += values.min(axis=-1)
    pair = b[own,None]-b[None]; states = np.zeros((n,q,d),int)
    for root in roots:
        pair += cost[:,:,root].min(axis=-1); states[:,:,root] = cost[:,:,root].argmin(axis=-1)
    pair[np.arange(n),own] = np.inf; margins = pair.min(axis=1)
    if not witnesses: return margins
    for node in order:
        p = parent[node]
        if p >= 0: states[:,:,node] = np.take_along_axis(choices[node],states[:,:,p,None],axis=-1)[...,0]
    rivals = pair.argmin(axis=1); selected = states[np.arange(n),rivals]
    return {'margins':margins,'pair_margins':pair,'rivals':rivals,
        'witness_raw':np.take_along_axis(points,selected[:,:,None],axis=-1)[...,0]}
