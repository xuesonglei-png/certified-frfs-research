"""Finite rescale/atomic endpoint deletion path with a fixed guard reference.

All weighted arms can use the same pool/budget. Proposed must retain the
original-full reference; an initially inadmissible control can use its fixed
initial own metric as reference. No reference reset after accepted deletions.
"""
import numpy as np
from atomic_postmetric_deletion import DistanceProfiles
from hardaware_guarded_soft5_descent import hard5_LOO_BA
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate
from certified_frfs.majority_metric import MajorityTerms


def point_key(point):
    return (-round(point['hard_training_LOO_BA'],12),
            round(point['class_certificate']['max_upper_bound'],12),
            point['removed_feature'],point['kind'],point['receiver'] if point['receiver'] is not None else -1)


def fit_atomic_pruning(A,y,features,weights,*,fraction=.5,pool=4,max_checks=48,require_original=True):
    A,y=np.asarray(A,float),np.asarray(y); fs=list(features); w=np.asarray(weights,float)
    geometry=DistanceProfiles(A)
    if (y.shape!=(len(A),) or not fs or fs!=sorted(set(fs))
        or any(int(a)!=a or not 0<=a<A.shape[1] for a in fs) or w.shape!=(len(fs),)
        or not np.isfinite(w).all() or w.min()<0 or not 0<fraction<=1
        or int(pool)!=pool or pool<1 or int(max_checks)!=max_checks or max_checks<1):
        raise ValueError('invalid training support or pruning budget')
    mass=np.sqrt(A.shape[1]); full=np.ones(A.shape[1])/mass
    if abs(w.sum()-mass)>1e-8: raise ValueError('original metric mass required')
    initial_fs,initial_w=fs.copy(),w.copy()
    protected=MajorityTerms(A,y,weights=full).bounds(range(A.shape[1]),allowed_enemies=2)>1e-10
    def original_guard(cfs,cw):
        cert=weight_certificate(A,y,cfs,full,cw,protected=protected)
        cls=classwise_dependency_certificate(y,cert)
        return {'certificate':cert,'class_certificate':cls,'accepted':bool(cert['feasible'] and cls['feasible_all_classes'])}
    original=original_guard(fs,w)
    if require_original and not original['accepted']: raise ValueError('input metric lacks original-full admission')
    mode='original_full' if original['accepted'] else 'own_anchor'
    own_protected=MajorityTerms(A[:,initial_fs],y,weights=initial_w).bounds(
        range(len(initial_fs)),allowed_enemies=2)>1e-10 if mode=='own_anchor' else None
    def guard(cfs,cw):
        if mode=='original_full': return original_guard(cfs,cw)
        local=[initial_fs.index(a) for a in cfs]
        cert=weight_certificate(A[:,initial_fs],y,local,initial_w,cw,protected=own_protected)
        cls=classwise_dependency_certificate(y,cert)
        return {'certificate':cert,'class_certificate':cls,'accepted':bool(cert['feasible'] and cls['feasible_all_classes'])}
    initial_guard=original if mode=='original_full' else guard(fs,w)
    if not initial_guard['accepted']: raise RuntimeError('fixed initial reference failed its own guard')
    initial_ba=hard5_LOO_BA(A[:,fs],y,w); ba=initial_ba; endpoint=initial_guard
    target=max(1,int(np.ceil(fraction*len(fs)))); checks=[]; chosen=[]; status='target_reached'
    while len(fs)>target:
        if len(checks)>=max_checks: status='guard_budget_exhausted';break
        importance=w*np.ptp(A[:,fs],axis=0)
        donors=sorted(range(len(fs)),key=lambda i:(importance[i],w[i],fs[i]))[:pool]
        eligible=[]
        for drop in donors:
            donor=fs[drop]; keep=[i for i in range(len(fs)) if i!=drop]; cfs=[fs[i] for i in keep]
            remaining=w[keep].copy(); proposals=[]
            if remaining.sum()>0:
                proposals.append(('rescale',None,None,remaining*mass/remaining.sum()))
            if w[drop]>0:
                for receiver in geometry.receivers(fs,donor,pool=2):
                    cw=remaining.copy(); cw[cfs.index(receiver['feature'])]+=w[drop]
                    proposals.append(('atomic',receiver['feature'],receiver['mean_pair_distance_distortion'],cw))
            for kind,receiver,distortion,cw in proposals:
                if len(checks)>=max_checks: break
                checked=guard(cfs,cw); actual=hard5_LOO_BA(A[:,cfs],y,cw)
                nonworse=bool(actual>=ba-1e-12)
                point={'id':len(checks),'features_before':fs.copy(),'weights_before':w.tolist(),'BA_before':ba,
                    'removed_feature':donor,'kind':kind,'receiver':receiver,'mean_pair_distance_distortion':distortion,
                    'features':cfs.copy(),'weights':cw.tolist(),'hard_training_LOO_BA':actual,
                    'nonworsening_training_BA':nonworse,'eligible':bool(checked['accepted'] and nonworse),**checked}
                checks.append(point)
                if point['eligible']: eligible.append(point)
            if len(checks)>=max_checks: break
        if not eligible:
            status='guard_budget_exhausted' if len(checks)>=max_checks else 'no_acceptable_pooled_deletion';break
        best=min(eligible,key=point_key);chosen.append(best['id']);fs=best['features'].copy();w=np.array(best['weights']);ba=best['hard_training_LOO_BA']
        endpoint={k:best[k] for k in ('certificate','class_certificate','accepted')}
    return fs,w,{'status':status,'guard_reference':mode,'original_initial_guard':original,
        'initial_guard':initial_guard,'initial_features':initial_fs,'initial_weights':initial_w.tolist(),
        'initial_hard_LOO_BA':initial_ba,'features':fs,'weights':w.tolist(),'final_hard_LOO_BA':ba,
        'fraction':fraction,'target_count':target,'pool':pool,'receiver_pool':2,'max_checks':max_checks,
        'checks':checks,'chosen_ids':chosen,'endpoint_guard':endpoint,
        'criterion':'hard_training_BA_then_classwise_dependency_upper_then_feature_kind_receiver',
        'scope':'finite_fixed_reference_rescale_atomic_path_not_minimum_reduct_or_query_risk'}
