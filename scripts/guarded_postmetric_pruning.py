"""Prune learned metric coordinates with unchanged original-reference guards.

Exact positive weights are never called irrelevant. Every removal preserves
original mass by rescaling retained weights, then checks fresh full-reference
FRS admission and actual training hard5NN BA. No query data enter this API.
"""
import numpy as np
from hardaware_guarded_soft5_descent import hard5_LOO_BA
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate


def fit_postmetric_pruning(A,y,features,weights,*,fraction=.5,pool=4,max_checks=48,require_original=True):
    A,y=np.asarray(A,float),np.asarray(y); fs=list(features); w=np.asarray(weights,float)
    if (A.ndim!=2 or len(A)<7 or y.shape!=(len(A),) or not fs or fs!=sorted(set(fs))
            or any(int(a)!=a or not 0<=a<A.shape[1] for a in fs) or w.shape!=(len(fs),)
            or not np.isfinite(A).all() or A.min()<0 or A.max()>1 or not np.isfinite(w).all() or w.min()<0
            or not 0<fraction<=1 or int(pool)!=pool or pool<1 or int(max_checks)!=max_checks or max_checks<1):
        raise ValueError('invalid training support or pruning budget')
    mass=np.sqrt(A.shape[1]); full=np.ones(A.shape[1])/mass
    if abs(w.sum()-mass)>1e-8: raise ValueError('original metric mass required')
    initial_fs=fs.copy(); initial_w=w.copy()
    def original_guard(candidate_fs,candidate_w):
        cert=weight_certificate(A,y,candidate_fs,full,candidate_w)
        classes=classwise_dependency_certificate(y,cert)
        return {'certificate':cert,'class_certificate':classes,
                'accepted':bool(cert['feasible'] and classes['feasible_all_classes'])}
    original=original_guard(fs,w)
    if require_original and not original['accepted']: raise ValueError('input metric lacks original-full admission')
    mode='original_full' if original['accepted'] else 'own_anchor'
    def guard(candidate_fs,candidate_w):
        if mode=='original_full': return original_guard(candidate_fs,candidate_w)
        local=[initial_fs.index(a) for a in candidate_fs]
        cert=weight_certificate(A[:,initial_fs],y,local,initial_w,candidate_w)
        classes=classwise_dependency_certificate(y,cert)
        return {'certificate':cert,'class_certificate':classes,
                'accepted':bool(cert['feasible'] and classes['feasible_all_classes'])}
    initial_guard=original if mode=='original_full' else guard(fs,w)
    if not initial_guard['accepted']: raise RuntimeError('anchor failed its own admission')
    initial_ba=hard5_LOO_BA(A[:,fs],y,w); ba=initial_ba
    target=max(1,int(np.ceil(fraction*len(initial_fs)))); checks=[]; chosen_ids=[]
    status='target_reached'
    while len(fs)>target:
        if len(checks)>=max_checks: status='guard_budget_exhausted'; break
        importance=w*np.ptp(A[:,fs],axis=0)
        order=sorted(range(len(fs)),key=lambda i:(importance[i],w[i],fs[i]))[:pool]
        candidates=[]
        for drop in order:
            if len(checks)>=max_checks: break
            keep=[i for i in range(len(fs)) if i!=drop]; candidate_fs=[fs[i] for i in keep]
            candidate_w=w[keep].copy()
            if candidate_w.sum()<=0: continue
            candidate_w*=mass/candidate_w.sum()
            checked=guard(candidate_fs,candidate_w)
            actual=hard5_LOO_BA(A[:,candidate_fs],y,candidate_w)
            eligible=checked['accepted'] and actual>=ba-1e-12
            point={'id':len(checks),'features_before':fs.copy(),'weights_before':w.tolist(),
                'BA_before':ba,'removed_feature':fs[drop],'features':candidate_fs,'weights':candidate_w.tolist(),
                'hard_training_LOO_BA':actual,'nonworsening_training_BA':bool(actual>=ba-1e-12),
                'eligible':bool(eligible),**checked}
            checks.append(point)
            if eligible: candidates.append(point)
        if not candidates:
            status='guard_budget_exhausted' if len(checks)>=max_checks else 'no_acceptable_pooled_deletion'
            break
        best=min(candidates,key=lambda p:(-round(p['hard_training_LOO_BA'],12),p['removed_feature']))
        chosen_ids.append(best['id']); fs=best['features'].copy(); w=np.array(best['weights']); ba=best['hard_training_LOO_BA']
    endpoint=initial_guard if not chosen_ids else {k:checks[chosen_ids[-1]][k] for k in ('certificate','class_certificate','accepted')}
    return fs,w,{'status':status,'guard_reference':mode,'original_initial_guard':original,
        'initial_guard':initial_guard,
        'fraction':fraction,'target_count':target,'pool':pool,'max_checks':max_checks,'checks':checks,'chosen_ids':chosen_ids,
        'initial_features':initial_fs,'initial_weights':initial_w.tolist(),'initial_hard_LOO_BA':initial_ba,
        'features':fs,'weights':w.tolist(),'final_hard_LOO_BA':ba,'endpoint_guard':endpoint,
        'scope':'finite_training_nonworsening_pooled_metric_pruning_not_irreducibility_optimum_or_query_risk'}
