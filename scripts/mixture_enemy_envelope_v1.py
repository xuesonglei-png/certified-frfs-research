"""Convex-mixture shared-box enemy envelope, exact coordinate epigraph LP.

For a possible reduced enemy j and full-enemy witnesses K, the original
distance minimum is <= sum_k lambda_k d_full(i,k) for every simplex lambda.
Consequently min_lambda sup_box [sum lambda*d_full - d_reduced(i,j)] is an
upper bound. It is no larger (in exact arithmetic) than the best single
witness bound because simplex vertices recover those witnesses.

This is a finite-reference envelope. It is not a new population risk theorem
or evidence of held-out classification improvement.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def coordinate_affine_rows(il, ih, jl, jh, kl, kh, wf, wr, shared_index=None):
    """Return affine lambda rows whose max is the scalar shared-box supremum.

    Other witness distances use their far endpoints. Between j's interval
    boundaries each remaining expression is convex in the shared query
    coordinate, so its maximum occurs at a segment endpoint. If j is itself
    a full witness, its location MUST be shared, including the nearest point.
    """
    kl, kh = np.asarray(kl,float), np.asarray(kh,float)
    points = sorted(set((float(il),float(ih),float(np.clip(jl,il,ih)),float(np.clip(jh,il,ih)))))
    rows, constants = [], []
    for t in points:
        far = np.maximum(abs(t-kl),abs(t-kh))
        near_j = max(0.,jl-t,t-jh)
        if shared_index is None:
            rows.append(wf*far); constants.append(-wr*near_j)
        else:
            for distance in sorted(set((abs(t-jl),abs(t-jh),near_j))):
                coeff=wf*far.copy();coeff[shared_index]=wf*distance
                rows.append(coeff);constants.append(-wr*distance)
    return np.asarray(rows),np.asarray(constants)


def solve_mixture_enemy_bound(lo_i, hi_i, lo_j, hi_j, lo_k, hi_k, full_weights,
                              reduced_weights, *, shared_index=None):
    lo_i,hi_i,lo_j,hi_j=map(lambda x:np.asarray(x,float),(lo_i,hi_i,lo_j,hi_j))
    lo_k,hi_k,wf,wr=map(lambda x:np.asarray(x,float),(lo_k,hi_k,full_weights,reduced_weights))
    d=len(lo_i)
    if (lo_i.shape!=(d,) or hi_i.shape!=(d,) or lo_j.shape!=(d,) or hi_j.shape!=(d,)
            or lo_k.ndim!=2 or lo_k.shape!=hi_k.shape or lo_k.shape[1]!=d
            or not len(lo_k) or wf.shape!=(d,) or wr.shape!=(d,)
            or any(not np.isfinite(x).all() for x in (lo_i,hi_i,lo_j,hi_j,lo_k,hi_k,wf,wr))
            or np.any(wf<0) or np.any(wr<0) or np.any(lo_i>hi_i) or np.any(lo_j>hi_j)
            or np.any(lo_k>hi_k)
            or (shared_index is not None and (int(shared_index)!=shared_index or not 0<=shared_index<len(lo_k)))):
        raise ValueError('invalid shared-box mixture input')
    if shared_index is not None:
        if not np.array_equal(lo_k[shared_index],lo_j) or not np.array_equal(hi_k[shared_index],hi_j):
            raise ValueError('shared enemy must have identical bounds')
    q=len(lo_k);groups=[];rows=[];cols=[];data=[];b=[]
    for a in range(d):
        coefficients, constants=coordinate_affine_rows(lo_i[a],hi_i[a],lo_j[a],hi_j[a],
            lo_k[:,a],hi_k[:,a],wf[a],wr[a],shared_index)
        groups.append((coefficients,constants))
        for coefficient,constant in zip(coefficients,constants,strict=True):
            rid=len(b);rows.extend([rid]*(q+1));cols.extend(list(range(q))+[q+a])
            data.extend(coefficient.tolist()+[-1.]);b.append(-constant)
    matrix=coo_matrix((data,(rows,cols)),shape=(len(b),q+d)).tocsr()
    eq=np.zeros((1,q+d));eq[0,:q]=1.
    result=linprog(np.r_[np.zeros(q),np.ones(d)],A_ub=matrix,b_ub=b,A_eq=eq,b_eq=[1.],
        bounds=[(0.,1.)]*q+[(None,None)]*d,method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'time_limit':30.})
    # Any explicit simplex mixture produces a valid bound, even without an
    # optimal solver result. Recompute its envelope instead of trusting fun.
    single=[sum(float(np.max(coef[:,k]+constant)) for coef,constant in groups) for k in range(q)]
    best=int(np.argmin(single));vertex=np.eye(q)[best]
    weights=vertex.copy()
    if result.success and np.isfinite(result.x[:q]).all():
        candidate=np.maximum(result.x[:q],0.);candidate/=candidate.sum()
        value=sum(float(np.max(coef@candidate+constant)) for coef,constant in groups)
        if value<single[best]:weights=candidate
    value=sum(float(np.max(coef@weights+constant)) for coef,constant in groups)
    return {'upper_bound_raw':value,'best_single_bound':float(single[best]),
            'mixture_weights':weights.tolist(),'solver_success':bool(result.success),
            'solver_status':int(result.status),'solver_message':result.message,
            'coordinate_upper_bounds':[float(np.max(coef@weights+constant)) for coef,constant in groups],
            'scope':'explicit_simplex_shared_box_envelope_not_global_loss_or_classifier_risk'}
