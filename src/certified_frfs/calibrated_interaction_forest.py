"""Permutation-calibrated interaction forest rather than forced d-1 edges.

The max-statistic test controls familywise admission under the GLOBAL
exchangeable null Y independent of X. No strong partial-null error control,
causal interaction claim, novel permutation theorem or accuracy guarantee.
"""
import numpy as np
from .triangular_classifier import triangular_memberships
from .geometry_only_forest_stage import GeometryStageContext


def _mi(codes,labels,states,q):
    n,m = codes.shape
    identifiers = ((np.arange(m)[None,:]*states+codes)*q+labels[:,None]).ravel()
    counts = np.bincount(identifiers,minlength=m*states*q).reshape(m,states,q).astype(float)
    left,right = counts.sum(axis=2,keepdims=True),counts.sum(axis=1,keepdims=True)
    denominator = left*right; positive = counts > 0
    ratio = np.ones_like(counts); np.divide(counts*n,denominator,out=ratio,where=positive)
    return np.sum(np.where(positive,counts*np.log(ratio),0),axis=(1,2))/n


def calibrated_interaction_forest(X,y,*,permutations=31,alpha=.05,seed=20261019):
    X,y = np.asarray(X,float),np.asarray(y)
    if X.ndim != 2 or y.shape != (len(X),) or X.shape[1] < 1 or not np.isfinite(X).all():
        raise ValueError('finite reference table required')
    if not isinstance(permutations,int) or permutations < 1 or not 0 < alpha < 1:
        raise ValueError('positive permutations and valid alpha required')
    classes,labels = np.unique(y,return_inverse=True); q = len(classes)
    if q < 2: raise ValueError('multiple classes required')
    bins = triangular_memberships(X).argmax(axis=-1); d = X.shape[1]
    edges = [(u,v) for u in range(d) for v in range(u+1,d)]
    if not edges:
        return {'edges':[],'candidates':[],'null_maxima':[],'permutation_indices':[],
            'single_MI':_mi(bins,labels,3,q).tolist(),'seed':seed,'permutations':permutations,'alpha':alpha,
            'scope':'no_pair_terms_for_one_feature'}
    pairs = np.column_stack([3*bins[:,u]+bins[:,v] for u,v in edges])
    us,vs = np.array(edges).T
    def statistics(truth):
        single = _mi(bins,truth,3,q); joint = _mi(pairs,truth,9,q)
        # Apply the identical quantization to observed and null statistics;
        # this prevents arithmetic-level ties from changing Monte Carlo ranks.
        return np.round(joint-single[us]-single[vs],12),single,joint
    observed,single,joint = statistics(labels); rng = np.random.default_rng(seed)
    indices = [rng.permutation(len(X)) for _ in range(permutations)]
    maxima = [float(max(statistics(labels[order])[0])) for order in indices]
    pvalues = (1+np.sum(np.array(maxima)[:,None] >= observed[None,:],axis=0))/(permutations+1)
    candidates = [{'edge':list(edge),'synergy':float(observed[i]),'joint_MI':float(joint[i]),
        'maxT_p':float(pvalues[i]),'eligible':bool(observed[i] > 0 and pvalues[i] <= alpha)} for i,edge in enumerate(edges)]
    candidates.sort(key=lambda row:(-row['synergy'],row['edge']))
    parent = list(range(d)); selected = []
    def root(a):
        while parent[a] != a: a = parent[a]
        return a
    for row in candidates:
        if not row['eligible']: continue
        u,v = row['edge']; a,b = root(u),root(v)
        if a != b: parent[a] = b; selected.append(row['edge'])
    return {'edges':selected,'candidates':candidates,'single_MI':single.tolist(),'null_maxima':maxima,
        'permutation_indices':[a.tolist() for a in indices],'seed':seed,'permutations':permutations,'alpha':alpha,
        'score_decimal_places':12,
        'scope':'training_only_global_null_calibrated_synergy_forest_not_causal_or_partial_null_FWER'}


class CalibratedForestContext(GeometryStageContext):
    def __init__(self,raw,y,*,seed=20261019):
        super().__init__(raw,y)
        self.calibration = calibrated_interaction_forest(self.A,self.y,seed=seed)

    def tree(self,features):
        fs = tuple(sorted(features))
        if fs not in self.trees:
            edges = [e for e in self.calibration['edges'] if all(a in fs for a in e)]
            self.trees[fs] = {'features':list(fs),'edges':edges,'local_tree':None,
                'scope':'induced_subforest_of_common_full_training_calibration'}
        return self.trees[fs]
