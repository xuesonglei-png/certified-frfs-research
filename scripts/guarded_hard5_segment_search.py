"""Finite hard5 segment refinement with full-reference admission and votes."""
import numpy as np
from segment_hard5_candidates import segment_candidates
from hardaware_guarded_soft5_descent import hard5_LOO_BA
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate


def proposal_sample(y, *, limit=96, seed=20261071):
    y=np.asarray(y)
    if y.ndim != 1 or int(limit)!=limit or limit<7: raise ValueError('invalid sample budget')
    if len(y)<=limit: return np.arange(len(y))
    rng=np.random.default_rng(seed); keep=[]; remainder=[]
    for c in np.unique(y):
        group=rng.permutation(np.flatnonzero(y==c)); first=min(2,len(group))
        keep.extend(group[:first].tolist()); remainder.extend(group[first:].tolist())
    if len(keep)>limit: raise ValueError('proposal budget cannot retain class representatives')
    keep.extend(rng.permutation(remainder)[:limit-len(keep)].tolist())
    return np.array(sorted(keep),int)


def fit_guarded_segment(A,y,features,anchor,target,*,max_candidates=16,proposal_limit=96):
    A,y=np.asarray(A,float),np.asarray(y); fs=list(features)
    anchor,target=np.asarray(anchor,float),np.asarray(target,float)
    if (A.ndim!=2 or len(A)<7 or y.shape!=(len(A),) or not fs or fs!=sorted(set(fs))
            or any(int(a)!=a or not 0<=a<A.shape[1] for a in fs)
            or anchor.shape!=(len(fs),) or target.shape!=anchor.shape or not np.isfinite(A).all()
            or A.min()<0 or A.max()>1 or not np.isfinite(anchor).all() or not np.isfinite(target).all()
            or anchor.min()<0 or target.min()<0): raise ValueError('invalid training support')
    mass=np.sqrt(A.shape[1]); full=np.ones(A.shape[1])/mass
    if abs(anchor.sum()-mass)>1e-8 or abs(target.sum()-mass)>1e-8: raise ValueError('fixed original mass required')
    def original_guard(w):
        cert=weight_certificate(A,y,fs,full,w); classes=classwise_dependency_certificate(y,cert)
        return {'certificate':cert,'class_certificate':classes,
                'accepted':bool(cert['feasible'] and classes['feasible_all_classes'])}
    original=original_guard(anchor); mode='original_full' if original['accepted'] else 'own_anchor'
    def guard(w):
        if mode=='original_full': return original_guard(w)
        cert=weight_certificate(A[:,fs],y,list(range(len(fs))),anchor,w)
        classes=classwise_dependency_certificate(y,cert)
        return {'certificate':cert,'class_certificate':classes,
                'accepted':bool(cert['feasible'] and classes['feasible_all_classes'])}
    subset=proposal_sample(y,limit=proposal_limit)
    proposals,info=segment_candidates(A[subset][:,fs],y[subset],anchor,target,max_candidates=max_candidates)
    checks=[]
    for point in proposals:
        w=(1-point['t'])*anchor+point['t']*target
        checked=original if point['t']==0 and mode=='original_full' else guard(w)
        checks.append({**point,'weights':w.tolist(),'hard_full_training_LOO_BA':hard5_LOO_BA(A[:,fs],y,w),**checked})
    accepted=[i for i,p in enumerate(checks) if p['accepted']]
    if not accepted: raise RuntimeError('anchor failed its own admission')
    # Keep the input target on hard-accuracy ties whenever it is admitted;
    # all actual full-reference scores, including t=1, have been evaluated.
    selected=min(accepted,key=lambda i:(-round(checks[i]['hard_full_training_LOO_BA'],12),abs(1-checks[i]['t']),i))
    fixed=[i for i in accepted if checks[i]['origin']=='fixed_grid']
    fixed_best=max(checks[i]['hard_full_training_LOO_BA'] for i in fixed)
    result=np.array(checks[selected]['weights'])
    return result,{'features':fs,'mass':float(mass),'status':'finite_guarded_hard5_segment_search',
        'guard_reference':mode,'original_initial_guard':original,'proposal_indices':subset.tolist(),
        'proposal_limit':proposal_limit,'proposal_seed':20261071,'proposal_info':info,'checks':checks,
        'selected':selected,'final_hard_LOO_BA':checks[selected]['hard_full_training_LOO_BA'],
        'fixed_best_hard_LOO_BA':fixed_best,'input_target_hard_LOO_BA':checks[6]['hard_full_training_LOO_BA'],
        'input_target_accepted':checks[6]['accepted'],'weights':result.tolist(),
        'scope':'finite_full_training_hard_vote_and_anchor_guard_search_not_test_risk_or_continuous_optimum'}
