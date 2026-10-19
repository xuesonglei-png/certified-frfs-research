"""Whole-table dependence envelope using bounded witness-mixture refinement.

Every possible reduced enemy remains in the maximum. A fixed per-object LP
budget refines the largest current upper bounds; all others retain the valid
single-witness bound. Thus solver/budget stops cannot invalidate the bound.
"""
import numpy as np
from certified_frfs.redistributed_loss import redistributed_loss_bound
from certified_frfs.core import interval_bounds
from mixture_enemy_envelope_v1 import solve_mixture_enemy_bound


def mixture_dependency_certificate(A,y,features,full_weights,reduced_weights,*,eps=.01,witness_pool=8,pair_budget=2):
    A,y=np.asarray(A,float),np.asarray(y);fs=np.asarray(features,int)
    wf=np.asarray(full_weights,float);reduced=np.asarray(reduced_weights,float)
    if int(pair_budget)!=pair_budget or pair_budget<0 or eps!=.01:
        raise ValueError('invalid unchanged-radius mixture budget')
    original=redistributed_loss_bound(A,y,fs,wf,reduced,eps=eps,witness_pool=witness_pool)
    n,d=A.shape;wr=np.zeros(d);wr[fs]=reduced
    if original['raw_redistributed_bound']==0.:
        return {'upper_bound':0.,'object_bounds':[0.]*n,'original_upper_bound':0.,'pair_records':[],
                'LPs':0,'scope':'same_finite_shared_box_dependency_loss_not_classifier_risk'}
    lo,hi=interval_bounds(A,eps);bounds=[];records=[]
    for i in range(n):
        enemies=np.flatnonzero(y!=y[i])
        lower=np.maximum(0.,np.maximum(lo[enemies]-hi[i],lo[i]-hi[enemies]))@wr
        upper=np.maximum(abs(lo[i]-hi[enemies]),abs(hi[i]-lo[enemies]))@wr
        possible=enemies[lower<=upper.min()+1e-12]
        distance=np.sum(abs(A[enemies]-A[i])*wf,axis=1)
        witnesses=enemies[np.lexsort((enemies,distance))][:witness_pool]
        jl,jh=lo[possible][:,None,:],hi[possible][:,None,:]
        kl,kh=lo[witnesses][None,:,:],hi[witnesses][None,:,:]
        def at(t):return wf*np.maximum(abs(t-kl),abs(t-kh))-wr*np.maximum(0.,np.maximum(jl-t,t-jh))
        terms=np.maximum(at(lo[i]),at(hi[i]))
        for point in (jl,jh,kl,kh):terms=np.maximum(terms,at(np.clip(point,lo[i],hi[i])))
        coefficient=wf-wr;near=np.maximum(0.,np.maximum(lo[i]-jh,jl-hi[i]))
        far=np.maximum(abs(lo[i]-jh),abs(hi[i]-jl))
        terms=np.where((possible[:,None]==witnesses[None,:])[:,:,None],
            coefficient*np.where(coefficient>=0,far,near),terms)
        signed=terms.sum(axis=2).min(axis=1)
        candidate_lower=np.array([lower[np.flatnonzero(enemies==j)[0]] for j in possible])
        old=np.maximum(0.,np.minimum(1-candidate_lower,signed));updated=old.copy()
        done=set()
        for attempt in range(min(int(pair_budget),len(possible))):
            pending=[j for j in range(len(possible)) if j not in done]
            # Once a previously refined enemy is worst, more refinements of
            # smaller enemies cannot improve this object's maximum.
            index=max(pending,key=lambda j:(updated[j],-int(possible[j])))
            if done and max(updated[list(done)])>=updated[index]:break
            j=int(possible[index]);shared=int(np.flatnonzero(witnesses==j)[0]) if j in witnesses else None
            result=solve_mixture_enemy_bound(lo[i],hi[i],lo[j],hi[j],lo[witnesses],hi[witnesses],wf,wr,shared_index=shared)
            np.testing.assert_allclose(result['best_single_bound'],signed[index],atol=1e-10,rtol=1e-9)
            updated[index]=max(0.,min(1-candidate_lower[index],signed[index],result['upper_bound_raw']))
            records.append({'object':i,'enemy':j,'witnesses':witnesses.tolist(),'result':result,
                'old_enemy_bound':float(old[index]),'new_enemy_bound':float(updated[index])})
            done.add(index)
        # The old production bound may use omitted-coordinate separation and
        # is independently valid; keep the tighter of two valid envelopes.
        bounds.append(min(float(updated.max()),float(original['redistributed_object_bounds'][i])))
    raw=float(np.mean(bounds));upper=float(np.clip(raw+1e-10*(1+abs(raw)),0.,1.))
    classes=np.unique(y);class_upper=[]
    for c in classes:
        value=float(np.mean(np.asarray(bounds)[y==c]));class_upper.append(min(1.,value+1e-10*(1+abs(value))))
    return {'upper_bound':upper,'raw_upper_bound':raw,'object_bounds':bounds,'class_upper_bounds':class_upper,
        'classes':classes.tolist(),'original_upper_bound':original['upper_bound'],
        'original_object_bounds':original['redistributed_object_bounds'],'pair_records':records,'LPs':len(records),
        'eps':eps,'witness_pool':witness_pool,'pair_budget':pair_budget,
        'scope':'same_finite_shared_box_dependency_loss_not_classifier_risk'}
