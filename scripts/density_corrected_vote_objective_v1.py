"""Inverse class-coverage soft-neighborhood prototype; no prediction guarantee.

Both density-corrected and matched-uniform targets exclude queries missing ANY
reference class with positive full leave-one-out population. Calibration is common
and unweighted. Fixed-radius inverse-coverage votes estimate full-reference votes;
the fitted radius is sample-dependent, so no unbiased fitted-risk claim follows.
"""
import numpy as np
from scipy.special import expit,log_expit,logsumexp
from ensemble_vote_objective_v1 import SEEDS,partition_terms
from density_calibrated_soft5 import calibrate


def balanced_weighted_membership(D,mask,coverage,temperature,*,return_slope=False):
    D,mask,R=np.asarray(D,float),np.asarray(mask,bool),np.asarray(coverage,float)
    if D.shape!=mask.shape or D.shape!=R.shape or not np.isfinite(D).all() or not np.isfinite(R).all() or np.any(R<0):
        raise ValueError('matching finite tables and nonnegative coverage required')
    if not np.isfinite(temperature) or temperature<=0 or np.any(R[~mask]!=0.):
        raise ValueError('positive temperature and zero excluded coverage required')
    mass=R.sum(axis=1)
    if np.any(mass<5-1e-12):raise ValueError('available weighted reference mass must be at least5')
    Q=mask.astype(float);S=np.zeros_like(D);active=mass>5+1e-12
    if not active.any():return (Q,S) if return_slope else Q
    dist,allowed,weights=D[active],mask[active],R[active]
    order=np.argsort(np.where(allowed,dist,np.inf),axis=1,kind='stable')
    ordered_weights=np.take_along_axis(weights,order,axis=1)
    cumulative=np.cumsum(ordered_weights,axis=1)
    boundary=np.argmax(cumulative>=5,axis=1)
    inside=np.zeros_like(allowed)
    for i,end in enumerate(boundary):inside[i,order[i,:end+1]]=True
    outside=allowed & ~inside;surplus=(weights*inside).sum(axis=1)-5
    assert np.all(surplus>=0.)
    log_surplus=np.full(len(dist),-np.inf);positive=surplus>0
    log_surplus[positive]=np.log(surplus[positive])
    log_weights=np.full(weights.shape,-np.inf);nonzero=weights>0;log_weights[nonzero]=np.log(weights[nonzero])
    center=dist[np.arange(len(dist)),order[np.arange(len(dist)),boundary]]
    Z=-(dist-center[:,None])/temperature
    low=-np.max(np.where(allowed,Z,-np.inf),axis=1)-40
    high=-np.min(np.where(allowed,Z,np.inf),axis=1)+40
    radius=np.zeros(len(dist));done=np.zeros(len(dist),bool)
    for _ in range(200):
        z=Z+radius[:,None]
        lout=np.where(outside,log_weights+log_expit(z),-np.inf)
        ldef=np.where(inside,log_weights+log_expit(-z),-np.inf)
        norm_out=logsumexp(lout,axis=1);norm_def=logsumexp(ldef,axis=1)
        out_plus_surplus=np.logaddexp(norm_out,log_surplus)
        residual=out_plus_surplus-norm_def
        done|=abs(residual)<1e-11
        if done.all():
            Q[active]=np.where(allowed,expit(z),0.)
            S[active]=np.where(allowed,expit(z)*expit(-z),0.)
            assert np.max(abs((Q*R).sum(axis=1)-5))<1e-9
            return (Q,S) if return_slope else Q
        low=np.where((residual<0)&~done,radius,low)
        high=np.where((residual>0)&~done,radius,high)
        derivative=(np.exp(lout-out_plus_surplus[:,None])*expit(-z)).sum(axis=1)
        derivative+=(np.exp(ldef-norm_def[:,None])*expit(z)).sum(axis=1)
        newton=radius-residual/np.maximum(derivative,1e-14)
        usable=(newton>low)&(newton<high)&np.isfinite(newton)
        radius=np.where(done,radius,np.where(usable,newton,(low+high)/2))
    raise RuntimeError('weighted log-balanced neighborhood radius did not converge')


def covered_partition(A,y,anchor,seed):
    terms,group=partition_terms(A,y,seed);C,same,mask,labels=terms;y=np.asarray(y)
    classes=np.unique(y);membership=(y[:,None]==classes[None,:]).astype(float)
    population=np.array([(y==c).sum() for c in classes])[None,:]-(labels[:,None]==classes[None,:])
    sampled=mask.astype(float)@membership
    covered=np.all((population==0)|(sampled>0),axis=1)
    rejected=[group['query_indices'][i] for i in np.flatnonzero(~covered)]
    C,same,mask,labels,population,sampled=C[covered],same[covered],mask[covered],labels[covered],population[covered],sampled[covered]
    ratios=np.divide(population,sampled,out=np.zeros_like(sampled),where=sampled>0)
    R=(ratios@membership.T)*mask
    np.testing.assert_allclose(R.sum(axis=1),len(y)-1,atol=1e-12,rtol=0)
    filtered=(C,same,mask,labels);tau,calibration=calibrate(filtered,anchor)
    record={'seed':int(seed),'group_assignments':group['group_assignments'],
        'query_indices':[i for i,keep in zip(group['query_indices'],covered,strict=True) if keep],
        'no_own_or_five_references':group['excluded_query_indices'],
        'missing_reference_class_queries':rejected,'class_population':population.tolist(),
        'class_sampled':sampled.tolist(),'classes':classes.tolist(),'calibration':calibration,
        'scope':'conditional_full_class_coverage_training_only_not_unbiased_fitted_radius'}
    return filtered,R,tau,record


def build_density_corrected(A,y,anchor,*,seeds=SEEDS):
    if not seeds or len(seeds)!=len(set(seeds)):raise ValueError('distinct nonempty seeds required')
    parts=[covered_partition(A,y,anchor,seed) for seed in seeds]
    return [r[0] for r in parts],[r[1] for r in parts],[r[2] for r in parts],[r[3] for r in parts]


def objective(w,parts,coverages,y,temperatures,*,family='plurality',correction=True,reg=.01):
    if family not in ('own_mass','plurality') or not parts or len(parts)!=len(coverages) or len(parts)!=len(temperatures):
        raise ValueError('matching nonempty partitions and known objective required')
    w,y=np.asarray(w,float),np.asarray(y);classes=np.unique(y)
    membership=(y[:,None]==classes[None,:]).astype(float);total=0.;gradient=np.zeros_like(w)
    for terms,coverage,tau in zip(parts,coverages,temperatures,strict=True):
        C,same,mask,labels=terms
        if not len(labels):continue
        R=coverage if correction else mask.astype(float)
        D=np.einsum('ijd,d->ij',C,w);Q,S=balanced_weighted_membership(D,mask,R,tau,return_slope=True)
        votes=(Q*R)@membership/5;weighted_slope=S*R
        center=np.einsum('ij,ijd->id',weighted_slope,C)/np.maximum(weighted_slope.sum(axis=1)[:,None],np.finfo(float).tiny)
        slopes=weighted_slope@membership
        derivatives=(slopes[:,:,None]*center[:,None,:]-np.einsum('ij,jc,ijd->icd',weighted_slope,membership,C))/(5*tau)
        own=np.searchsorted(classes,labels);alpha=np.array([1/(len(np.unique(labels))*sum(labels==c)) for c in labels])
        if family=='own_mass':
            true=votes[np.arange(len(labels)),own]
            value=-alpha@np.log(true+1e-12)
            grad=-np.einsum('i,id->d',alpha/(true+1e-12),derivatives[np.arange(len(labels)),own])
        else:
            logits=(.1+votes-votes[np.arange(len(labels)),own,None])/.05
            logits[np.arange(len(labels)),own]=-np.inf
            norm=logsumexp(np.column_stack((np.zeros(len(labels)),logits)),axis=1)
            probabilities=np.exp(logits-norm[:,None]);value=.05*(alpha@norm)
            grad=np.einsum('i,ic,icd->d',alpha,probabilities,derivatives-derivatives[np.arange(len(labels)),own,None,:])
        log=np.log(len(w)*(w+1e-12));value+=reg*np.sum(w*log);grad+=reg*(log+w/(w+1e-12))
        total+=value/len(parts);gradient+=grad/len(parts)
    return float(total),gradient
