"""Training-only, budgeted support search ordered by certificate violations.

Uncertified intermediate nodes may be expanded; only actually certified
endpoints are returned.  A nominal proxy licenses no certificate or prediction
claim.  This is a heuristic search integration, not a new feasibility theorem.
"""
from __future__ import annotations
import numpy as np
from certified_frfs.capped_sparse import capped_sparse_projection
from certified_frfs.redistributed_loss import transfer_weights


def metric_weights(A, features, reference):
    fs = list(features); d = A.shape[1]; mass = float(np.sum(reference))
    if len(fs) == d:
        return np.asarray(reference).copy()
    v = transfer_weights(A, fs, reference)
    return capped_sparse_projection(v/mass, len(fs), 2/d)*mass


def violation(candidate):
    """Both original and learned-reference conditions, normalized by contract."""
    parts = []
    for certkey, classkey in (('certificate', 'classwise'),
                              ('family_certificate', 'family_classwise')):
        cert = candidate[certkey]
        parts.append(max(0., cert['dependency_certificate']['upper_bound']/.1-1))
        parts.extend(max(0., (v+1e-10*(1+abs(v)))/.1-1)
                     for v in candidate[classkey]['bounds'])
        parts.extend(max(0., (r-c)/max(1,r)) for r,c in
                     zip(cert['required'],cert['retained_counts'],strict=True))
    return (max(parts, default=0.), sum(parts))


class NominalProxy:
    def __init__(self, A, y, reference, protected):
        self.A = np.asarray(A); self.y = np.asarray(y)
        self.reference = np.asarray(reference); self.protected = np.asarray(protected,bool)
        self.components = np.abs(self.A.T[:,:,None]-self.A.T[:,None,:])
        self.cross = self.y[:,None] != self.y[None,:]
        self.classes = np.unique(y)
        full = np.einsum('d,dij->ij',self.reference,self.components)
        self.full_gamma = np.minimum(1.,np.min(np.where(self.cross,full,np.inf),axis=1))
        self.cache = {}

    def __call__(self, features):
        fs = tuple(features)
        if fs in self.cache: return self.cache[fs]
        w = metric_weights(self.A,fs,self.reference)
        D = np.einsum('d,dij->ij',w,self.components[list(fs)])
        gamma = np.minimum(1.,np.min(np.where(self.cross,D,np.inf),axis=1))
        loss = np.maximum(0.,self.full_gamma-gamma)
        classloss = [float(loss[self.y==c].mean()) for c in self.classes]
        np.fill_diagonal(D,np.inf)
        nn = np.argsort(D,axis=1,kind='stable')[:,:min(5,len(self.y)-1)]
        votes = np.stack([(self.y[nn]==c).sum(axis=1) for c in self.classes],axis=1)
        pred = self.classes[votes.argmax(axis=1)]; correct=pred==self.y
        deficits=[]
        for c in self.classes:
            group=(self.y==c)&self.protected
            required=int(np.ceil(.9*sum(group)-1e-12))
            deficits.append(max(0.,(required-sum(correct&group))/max(1,required)))
        parts=[max(0.,v/.1-1) for v in classloss]+deficits
        ba=float(np.mean([correct[self.y==c].mean() for c in self.classes]))
        out={'nominal_class_loss':classloss,'nominal_protected_deficits':deficits,
             'training_LOO_BA':ba,'proxy_key':[max(parts,default=0.),sum(parts),-ba]}
        self.cache[fs]=out
        return out


def search(d, seed_support, proxy, evaluate, *, depths=2, width=4, per_depth=8):
    """At most 1+depths*per_depth exact evaluations; no feasible-node premise."""
    seed=tuple(sorted(seed_support))
    if len(seed)<int(np.ceil(d/2)) or len(set(seed))!=len(seed) or not all(0<=a<d for a in seed):
        raise ValueError('support must respect unchanged half-support cap')
    if min(depths,width,per_depth)<1: raise ValueError('positive fixed search budgets required')
    seen={}; proposals=[]
    def check(fs,depth,parent):
        c=evaluate(fs); p=proxy(fs)
        row={'features':list(fs),'depth':depth,'parent':None if parent is None else list(parent),
             'candidate':c,'proxy':p,'exact_violation':list(violation(c))}
        seen[fs]=row
        return row
    check(seed,0,None); beam=[seed]
    for depth in range(1,depths+1):
        pending={}
        for parent in beam:
            for outgoing in parent:
                for incoming in sorted(set(range(d))-set(parent)):
                    fs=tuple(sorted((set(parent)-{outgoing})|{incoming}))
                    if fs not in seen and fs not in pending: pending[fs]=parent
        if not pending: break
        ranked=sorted(pending,key=lambda fs:(*proxy(fs)['proxy_key'],fs))
        proposals.append({'depth':depth,'parents':[list(x) for x in beam],
                          'screened':[{'features':list(fs),'parent':list(pending[fs]),
                                       'proxy':proxy(fs)} for fs in ranked],
                          'exact_shortlist':[list(fs) for fs in ranked[:per_depth]]})
        for fs in ranked[:per_depth]: check(fs,depth,pending[fs])
        # Retain failed nodes too. This may temporarily worsen the violation;
        # no local monotonicity or optimality claim is made.
        beam=sorted(seen,key=lambda fs:(*seen[fs]['exact_violation'],
                        -seen[fs]['proxy']['training_LOO_BA'],fs))[:width]
    feasible=[fs for fs,row in seen.items() if row['candidate']['joint_feasible']]
    chosen=max(feasible,key=lambda fs:(seen[fs]['proxy']['training_LOO_BA'],
                                      tuple(-a for a in fs))) if feasible else None
    return {'seed_support':list(seed),'depths':depths,'width':width,'per_depth':per_depth,
            'checks':list(seen.values()),'proposals':proposals,
            'selected':None if chosen is None else seen[chosen],
            'exact_checks':len(seen),'proxy_checks':len({tuple(r['features']) for p in proposals for r in p['screened']}|{seed}),
            'scope':'training_nominal_screen_original_dual_certified_endpoint_not_test_risk'}
