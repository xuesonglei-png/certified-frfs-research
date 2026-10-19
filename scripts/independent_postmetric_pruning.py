"""Independent finite pruning replay for original-full and own-anchor arms."""
import numpy as np
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from audit_simplex_source_budget_diagnostic import independent_loo
from audit_forest_compact_transfer import equivalent


def check_pruning(A,y,initial_fs,initial_w,fit):
    initial_fs=list(initial_fs); initial_w=np.asarray(initial_w,float); mass=np.sqrt(A.shape[1])
    assert fit['fraction']==.5 and fit['pool']==4 and fit['max_checks']==48
    assert fit['initial_features']==initial_fs; equivalent(fit['initial_weights'],initial_w.tolist())
    full=np.ones(A.shape[1])/mass
    full_protected=independent_protected(A,y,full)
    def original(fs,w):
        cert=weight_certificate(A,y,fs,full,w,protected=full_protected)
        cls=independent_class_certificate(y,cert)
        return {'certificate':cert,'class_certificate':cls,'accepted':bool(cert['feasible'] and cls['feasible_all_classes'])}
    original_initial=original(initial_fs,initial_w)
    equivalent(original_initial,fit['original_initial_guard'])
    mode='original_full' if original_initial['accepted'] else 'own_anchor'; assert fit['guard_reference']==mode
    own_protected=independent_protected(A[:,initial_fs],y,initial_w) if mode=='own_anchor' else None
    def guard(fs,w):
        if mode=='original_full': return original(fs,w)
        local=[initial_fs.index(a) for a in fs]
        cert=weight_certificate(A[:,initial_fs],y,local,initial_w,w,protected=own_protected)
        cls=independent_class_certificate(y,cert)
        return {'certificate':cert,'class_certificate':cls,'accepted':bool(cert['feasible'] and cls['feasible_all_classes'])}
    fs=initial_fs.copy(); w=initial_w.copy(); ba=independent_loo(A,y,fs,w)
    endpoint=guard(fs,w); equivalent(endpoint,fit['initial_guard']); assert endpoint['accepted']
    equivalent(ba,fit['initial_hard_LOO_BA']); assert fit['target_count']==max(1,int(np.ceil(.5*len(fs))))
    cursor=0; chosen=[]; checks=fit['checks']
    while cursor<len(checks):
        importance=[w[i]*(max(A[:,a])-min(A[:,a])) for i,a in enumerate(fs)]
        order=sorted(range(len(fs)),key=lambda i:(importance[i],w[i],fs[i]))[:4]; batch=[]
        for drop in order:
            if cursor>=48: break
            keep=[i for i in range(len(fs)) if i!=drop]; cfs=[fs[i] for i in keep]; cw=w[keep].copy()
            if cw.sum()<=0: continue
            cw*=mass/cw.sum(); point=checks[cursor]
            assert point['id']==cursor and point['features_before']==fs and point['features']==cfs
            assert point['removed_feature']==fs[drop]
            equivalent(point['weights_before'],w.tolist()); equivalent(point['weights'],cw.tolist())
            equivalent(point['BA_before'],ba); checked=guard(cfs,cw)
            for key in ('certificate','class_certificate','accepted'): equivalent(checked[key],point[key])
            actual=independent_loo(A,y,cfs,cw); equivalent(actual,point['hard_training_LOO_BA'])
            nonworse=bool(actual>=ba-1e-12)
            assert point['nonworsening_training_BA']==nonworse and point['eligible']==(checked['accepted'] and nonworse)
            batch.append(point); cursor+=1
        eligible=[p for p in batch if p['eligible']]
        if not eligible: assert cursor==len(checks); break
        best=min(eligible,key=lambda p:(-round(p['hard_training_LOO_BA'],12),p['removed_feature']))
        chosen.append(best['id']); fs=best['features'].copy(); w=np.array(best['weights']); ba=best['hard_training_LOO_BA']
        endpoint={k:best[k] for k in ('certificate','class_certificate','accepted')}
    assert fit['chosen_ids']==chosen and fit['features']==fs
    equivalent(fit['weights'],w.tolist()); equivalent(fit['final_hard_LOO_BA'],ba)
    equivalent(fit['endpoint_guard'],endpoint); assert endpoint['accepted'] and ba>=fit['initial_hard_LOO_BA']-1e-12
    expected='target_reached' if len(fs)<=fit['target_count'] else ('guard_budget_exhausted' if len(checks)>=48 else 'no_acceptable_pooled_deletion')
    assert fit['status']==expected and len(checks)<=48
    return len(checks)
