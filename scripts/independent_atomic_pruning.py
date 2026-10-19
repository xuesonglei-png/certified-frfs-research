"""Scalar proposal/profile/vote reconstruction for the entire atomic path."""
import numpy as np
from certified_frfs.feasible_transport import weight_certificate
from audit_atomic_postmetric_deletion import receiver_scores
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from audit_simplex_source_budget_diagnostic import independent_loo
from audit_forest_compact_transfer import equivalent


def check_atomic_pruning(A,y,initial_fs,initial_w,fit):
    initial_fs=list(initial_fs);initial_w=np.array(initial_w,float);mass=np.sqrt(A.shape[1]);full=np.ones(A.shape[1])/mass
    assert fit['initial_features']==initial_fs;equivalent(fit['initial_weights'],initial_w.tolist())
    assert fit['receiver_pool']==2 and fit['criterion']=='hard_training_BA_then_classwise_dependency_upper_then_feature_kind_receiver'
    protected=independent_protected(A,y,full)
    def original(fs,w):
        cert=weight_certificate(A,y,fs,full,w,protected=protected);cls=independent_class_certificate(y,cert)
        return {'certificate':cert,'class_certificate':cls,'accepted':bool(cert['feasible'] and cls['feasible_all_classes'])}
    original_initial=original(initial_fs,initial_w);equivalent(original_initial,fit['original_initial_guard'])
    mode='original_full' if original_initial['accepted'] else 'own_anchor';assert fit['guard_reference']==mode
    own=independent_protected(A[:,initial_fs],y,initial_w) if mode=='own_anchor' else None
    def guard(fs,w):
        if mode=='original_full':return original(fs,w)
        local=[initial_fs.index(a) for a in fs]
        cert=weight_certificate(A[:,initial_fs],y,local,initial_w,w,protected=own);cls=independent_class_certificate(y,cert)
        return {'certificate':cert,'class_certificate':cls,'accepted':bool(cert['feasible'] and cls['feasible_all_classes'])}
    fs=initial_fs.copy();w=initial_w.copy();ba=independent_loo(A,y,fs,w);endpoint=guard(fs,w)
    equivalent(endpoint,fit['initial_guard']);assert endpoint['accepted'];equivalent(ba,fit['initial_hard_LOO_BA'])
    target=max(1,int(np.ceil(fit['fraction']*len(fs))));assert target==fit['target_count']
    checks=fit['checks'];cursor=0;chosen=[];cache={};status='target_reached'
    while len(fs)>target:
        if cursor>=fit['max_checks']:status='guard_budget_exhausted';break
        importance=[w[i]*(max(A[:,a])-min(A[:,a])) for i,a in enumerate(fs)]
        donors=sorted(range(len(fs)),key=lambda i:(importance[i],w[i],fs[i]))[:fit['pool']]
        eligible=[]
        for drop in donors:
            donor=fs[drop];keep=[i for i in range(len(fs)) if i!=drop];cfs=[fs[i] for i in keep];remaining=w[keep].copy();proposals=[]
            if remaining.sum()>0:proposals.append(('rescale',None,None,remaining*mass/remaining.sum()))
            if w[drop]>0:
                for receiver in receiver_scores(A,fs,donor,cache):
                    cw=remaining.copy();cw[cfs.index(receiver['feature'])]+=w[drop]
                    proposals.append(('atomic',receiver['feature'],receiver['mean_pair_distance_distortion'],cw))
            for kind,receiver,distortion,cw in proposals:
                if cursor>=fit['max_checks']:break
                point=checks[cursor];assert point['id']==cursor and point['features_before']==fs and point['features']==cfs
                assert point['removed_feature']==donor and point['kind']==kind and point['receiver']==receiver
                equivalent(point['mean_pair_distance_distortion'],distortion);equivalent(point['weights_before'],w.tolist());equivalent(point['weights'],cw.tolist())
                equivalent(point['BA_before'],ba);fresh=guard(cfs,cw)
                for key in ('certificate','class_certificate','accepted'):equivalent(point[key],fresh[key])
                actual=independent_loo(A,y,cfs,cw);equivalent(point['hard_training_LOO_BA'],actual)
                nonworse=bool(actual>=ba-1e-12);assert point['nonworsening_training_BA']==nonworse
                assert point['eligible']==(fresh['accepted'] and nonworse)
                if point['eligible']:eligible.append(point)
                cursor+=1
            if cursor>=fit['max_checks']:break
        if not eligible:
            status='guard_budget_exhausted' if cursor>=fit['max_checks'] else 'no_acceptable_pooled_deletion';break
        best=min(eligible,key=lambda p:(-round(p['hard_training_LOO_BA'],12),round(p['class_certificate']['max_upper_bound'],12),
            p['removed_feature'],p['kind'],p['receiver'] if p['receiver'] is not None else -1))
        chosen.append(best['id']);fs=best['features'].copy();w=np.array(best['weights']);ba=best['hard_training_LOO_BA']
        endpoint={k:best[k] for k in ('certificate','class_certificate','accepted')}
    assert cursor==len(checks) and chosen==fit['chosen_ids'] and fit['features']==fs and fit['status']==status
    equivalent(fit['weights'],w.tolist());equivalent(fit['final_hard_LOO_BA'],ba);equivalent(fit['endpoint_guard'],endpoint)
    assert endpoint['accepted'] and ba>=fit['initial_hard_LOO_BA']-1e-12 and cursor<=fit['max_checks']
    return cursor
