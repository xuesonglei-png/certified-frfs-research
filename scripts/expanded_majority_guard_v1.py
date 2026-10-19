"""Expanded friend-witness sufficient certificate with original protected sets.

This reuses the existing order-statistic certificate principle. It is a tighter
admission procedure, not a claim of a new theorem or population accuracy bound.
"""
import copy
import numpy as np
from scipy.spatial.distance import cdist
from certified_frfs.core import interval_bounds
from certified_frfs.margin import triplet_interval_min
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.relation_support_selector_v1 import _candidate


def expanded_bounds(A,y,weights,*,eps=.01,n_friends=15):
    A=np.asarray(A,float);y=np.asarray(y);w=np.asarray(weights,float)
    if n_friends<3 or A.ndim!=2 or w.shape!=(A.shape[1],) or y.shape!=(len(A),):
        raise ValueError('invalid expanded majority inputs')
    if not np.isfinite(A).all() or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('invalid table/weights')
    lo,hi=interval_bounds(A,eps);D=cdist(A*w,A*w,'cityblock');np.fill_diagonal(D,np.inf)
    bounds=np.full(len(A),-np.inf)
    for i,label in enumerate(y):
        friends=np.flatnonzero((y==label)&(np.arange(len(A))!=i))
        enemies=np.flatnonzero(y!=label)
        if len(friends)<3:continue
        if len(enemies)<=2:bounds[i]=np.inf;continue
        friends=friends[np.lexsort((friends,D[i,friends]))][:n_friends]
        terms=np.zeros((len(enemies),len(friends)))
        for a in range(A.shape[1]):
            terms+=w[a]*triplet_interval_min(lo[i,a],hi[i,a],
                lo[enemies,a,None],hi[enemies,a,None],lo[friends,a][None,:],hi[friends,a][None,:])
        # At least three target friends precede each nonexceptional enemy.
        third=np.partition(terms,-3,axis=1)[:,-3]
        raw=np.partition(third,2)[2]
        bounds[i]=raw-1e-10*(1+abs(raw))
    return bounds


class ExpandedGuard:
    def __init__(self,A,y,reference):
        self.A,self.y,self.reference=A,y,np.asarray(reference)
        d=A.shape[1];uniform=np.ones(d)/np.sqrt(d)
        self.protected={}
        for name,w in (('certificate',uniform),('family_certificate',self.reference)):
            self.protected[name]=MajorityTerms(A,y,weights=w,eps=.01).bounds(range(d),allowed_enemies=2)>1e-10
        self.cache={}

    def candidate(self,fs):
        key=tuple(fs)
        if key in self.cache:return self.cache[key]
        original=_candidate(self.A,self.y,self.reference,fs,len(fs))
        candidate=copy.deepcopy(original)
        candidate['legacy_joint_feasible']=original['joint_feasible']
        candidate['expanded_guard_used']=False
        # If another unchanged guard already fails, retain the original rejection.
        dependency_ok=all(original[k]['dependency_certificate']['upper_bound']<=.1
            for k in ('certificate','family_certificate'))
        class_ok=original['classwise']['feasible'] and original['family_classwise']['feasible']
        if not original['joint_feasible'] and dependency_ok and class_ok:
            bounds=expanded_bounds(self.A[:,fs],self.y,candidate['weights'])
            flags=bounds>1e-10;candidate['expanded_guard_used']=True
            candidate['legacy_certificates']={k:copy.deepcopy(original[k]) for k in ('certificate','family_certificate')}
            candidate['expanded_majority_bounds']=[float(v) if np.isfinite(v) else None for v in bounds]
            for name in ('certificate','family_certificate'):
                protected=self.protected[name]
                groups=[(self.y==c)&protected for c in np.unique(self.y)]
                required=[int(np.ceil(.9*g.sum()-1e-12)) for g in groups]
                retained=[int((flags&g).sum()) for g in groups]
                cert=candidate[name];assert cert['required']==required
                cert['retained_counts']=retained;cert['majority_feasible']=all(c>=r for c,r in zip(retained,required))
                cert['feasible']=cert['majority_feasible'] and cert['dependency_certificate']['upper_bound']<=.1
            candidate['joint_feasible']=all(candidate[name]['feasible'] for name in ('certificate','family_certificate'))
        self.cache[key]=candidate
        return candidate
