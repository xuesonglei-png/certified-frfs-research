"""Shared-anchor convex witness mixture bounds for changed metrics."""
import numpy as np
from scipy.optimize import linprog
from .core import _inputs, interval_bounds, _interval_distances, _distance
from .redistributed_loss import redistributed_loss_bound


def mixture_envelope(grids, *, seconds=2.):
    """Choose witness simplex weights, then recompute a valid dual bound.

    Each grid is K x P coordinate envelope values at every piecewise knot.
    For ANY returned simplex lambda, sum_a max_p lambda.G_a[:,p] bounds
    max_anchor min_witness sum_a G_a. Solver optimality is not required.
    """
    if not grids or seconds <= 0:
        raise ValueError('nonempty grids and positive time budget required')
    grids = [np.asarray(g, float) for g in grids]; k = grids[0].shape[0]; d = len(grids)
    if k < 1 or any(g.ndim != 2 or g.shape[0] != k or not g.shape[1] or not np.isfinite(g).all() for g in grids):
        raise ValueError('finite coordinate grids with equal witness count required')
    constraints = []
    for a,g in enumerate(grids):
        block = np.zeros((g.shape[1], k+d)); block[:,:k] = g.T; block[:,k+a] = -1
        constraints.append(block)
    fit = linprog(np.r_[np.zeros(k), np.ones(d)], A_ub=np.vstack(constraints),
        b_ub=np.zeros(sum(g.shape[1] for g in grids)), A_eq=np.array([np.r_[np.ones(k),np.zeros(d)]]),
        b_eq=[1.], bounds=[(0.,1.)]*k+[(None,None)]*d, method='highs', options={'time_limit':seconds})
    def bound(lam): return float(sum(np.max(lam@g) for g in grids))
    # Include every vertex: never weaker than the old single-witness envelope.
    vertices = np.eye(k); best = min(vertices, key=bound).copy(); value = bound(best)
    if fit.x is not None and np.isfinite(fit.x[:k]).all():
        lam = np.maximum(fit.x[:k],0.)
        if lam.sum()>0:
            lam /= lam.sum(); candidate = bound(lam)
            if candidate < value: best,value = lam,candidate
    return {'upper':value,'lambda':best.tolist(),'solver_status':int(fit.status),
        'solver_success':bool(fit.success),'single_witness_upper':min(bound(v) for v in vertices)}


def coordinate_envelopes(lo,hi,i,j,witnesses,wf,wr):
    grids=[];knots=[]
    for a in range(lo.shape[1]):
        points=[lo[i,a],hi[i,a],lo[j,a],hi[j,a],(lo[j,a]+hi[j,a])/2]
        points.extend(((lo[k,a]+hi[k,a])/2) for k in witnesses)
        points=np.unique(np.clip(points,lo[i,a],hi[i,a]));rows=[]
        for k in witnesses:
            far=np.maximum(abs(points-lo[k,a]),abs(points-hi[k,a]))
            near=np.maximum(0.,np.maximum(lo[j,a]-points,points-hi[j,a]))
            if k==j:
                coefficient=wf[a]-wr[a]
                value=coefficient*(far if coefficient>=0 else near)
            else:value=wf[a]*far-wr[a]*near
            rows.append(value)
        knots.append(points.tolist());grids.append(np.array(rows))
    return grids,knots


def witness_mixture_loss_bound(X,y,features,w_full,w_reduced,*,eps=.01,witness_pool=8,seconds=2.):
    X,y,fs,wf=_inputs(X,y,features,w_full);wr=np.asarray(w_reduced,float)
    if wr.shape!=(len(fs),) or np.any(wr<0) or not np.isfinite(wr).all():raise ValueError('invalid reduced weights')
    old=redistributed_loss_bound(X,y,fs,wf,wr,eps=eps,witness_pool=witness_pool)
    if old['raw_redistributed_bound']==0:
        return {**old,'mixture_checks':[],'mixture_raw':0.,'scope':'shared_anchor_convex_witness_mixture_finite_dependency_not_test_risk'}
    lo,hi=interval_bounds(X,eps);wide=np.zeros(X.shape[1]);wide[fs]=wr
    lb,ub=_interval_distances(lo,hi,fs,wide);dist=_distance(X,np.arange(X.shape[1]),wf)
    values=[];checks=[]
    for i in range(len(y)):
        if old['redistributed_object_bounds'][i]==0:
            values.append(0.)
            continue
        enemies=np.flatnonzero(y!=y[i]);possible=enemies[lb[i,enemies]<=np.min(ub[i,enemies])+1e-12]
        witnesses=enemies[np.lexsort((enemies,dist[i,enemies]))][:witness_pool]
        row=[]
        for j in possible:
            if lb[i,j]>=1:
                row.append(0.)
                continue
            grids,knots=coordinate_envelopes(lo,hi,i,int(j),witnesses,wf,wide)
            fit=mixture_envelope(grids,seconds=seconds)
            bound=max(0.,min(1-lb[i,j],fit['upper']))
            row.append(bound);checks.append({'anchor':i,'enemy':int(j),'witnesses':witnesses.tolist(),
                'fit':fit,'bound':bound,'knots':knots})
        values.append(min(float(max(row)),old['redistributed_object_bounds'][i]))
    raw=float(np.mean(values));upper=min(old['upper_bound'],float(np.clip(raw+1e-10*(1+abs(raw)),0,1)))
    return {**old,'upper_bound':upper,'mixture_raw':raw,'mixture_object_bounds':values,'mixture_checks':checks,
        'scope':'shared_anchor_convex_witness_mixture_finite_dependency_not_test_risk'}
