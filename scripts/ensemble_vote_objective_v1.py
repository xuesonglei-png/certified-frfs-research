"""Nominal vote loss across prespecified training-only reference partitions.

This reduces dependence on one neighborhood sample; it neither guarantees
generalization nor cross-fits preprocessing. Deployment certification is separate.
Both compared losses must use identical partitions and anchor-calibrated tau.
"""
import numpy as np
from certified_frfs.softvote_metric import softvote_terms
from density_calibrated_soft5 import calibrate,objective as own_mass
from soft_plurality_objective_v1 import plurality_objective

SEEDS=(20261071,20261072,20261073)


def partition_terms(A,y,seed):
    A,y=np.asarray(A,float),np.asarray(y)
    if A.ndim!=2 or y.shape!=(len(A),) or not np.isfinite(A).all():
        raise ValueError('finite training table required')
    rng=np.random.default_rng(seed);group=np.empty(len(y),int)
    for label in np.unique(y):
        ids=rng.permutation(np.flatnonzero(y==label));group[ids]=np.arange(len(ids))%3
    C,same,mask,labels=softvote_terms(A,y,eps=0.)
    queries=np.flatnonzero(np.array([np.sum(y==label)>1 for label in y]))
    mask=mask & (group[queries,None]!=group[None,:]);same=same & mask
    keep=same.any(axis=1) & (mask.sum(axis=1)>=5)
    return (C[keep],same[keep],mask[keep],labels[keep]),{
        'groups':3,'seed':int(seed),'group_assignments':group.tolist(),
        'query_indices':queries[keep].tolist(),
        'excluded_query_indices':sorted(set(range(len(y)))-set(queries[keep].tolist())),
        'geometry':'enclosing_training_minmax_not_crossfitted_preprocessing'}


def build_ensemble(A,y,anchor,*,seeds=SEEDS):
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('distinct partition seeds required')
    terms=[];temperatures=[];records=[]
    for seed in seeds:
        part,info=partition_terms(A,y,seed);tau,calibration=calibrate(part,anchor)
        terms.append(part);temperatures.append(tau);records.append({'partition':info,'calibration':calibration})
    return terms,temperatures,records


def ensemble_objective(w,terms,reference_labels,temperatures,*,family='plurality',reg=.01):
    if family not in ('own_mass','plurality') or len(terms)!=len(temperatures) or not terms:
        raise ValueError('matching partitions and known loss required')
    values=[];gradients=[]
    for part,tau in zip(terms,temperatures,strict=True):
        if family=='own_mass':value,gradient=own_mass(w,part,tau,reg=reg)
        else:value,gradient=plurality_objective(w,part,reference_labels,tau,margin=.1,smoothing=.05,reg=reg)
        values.append(value);gradients.append(gradient)
    # Each valid partition includes the same entropy once; averaging keeps .01,
    # rather than inadvertently summing to .03. Empty partitions contribute zero
    # just as the single-partition objective, and must be explicitly reported.
    return float(np.mean(values)),np.mean(gradients,axis=0)
