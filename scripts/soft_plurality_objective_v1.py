"""Nominal multiclass vote-aligned fuzzy neighborhood prototype.

Unlike -log(true-class neighbor mass), the loss compares that class with each
rival class, allowing 2-vs-1-vs-1-vs-1 plurality. Distances/grouping and the
anchor-calibrated soft5 temperature are training-only. This is a differentiable
surrogate, not a hard5 error bound or proof of held-out superiority. Robust
FRS admission would remain a separate unchanged certificate.
"""
import numpy as np
from scipy.special import logsumexp
from crossgroup_soft5_terms import crossgroup_terms
from density_calibrated_soft5 import calibrate,balanced_membership
from soft_topk_metric_objective import topk_leakage_bound


def terms_and_temperature(A,y,anchor_normalized):
    terms,info=crossgroup_terms(A,y,eps=0.)
    temperature,calibration=calibrate(terms,np.asarray(anchor_normalized,float))
    return terms,temperature,{'groups':info,'calibration':calibration,
        'scope':'nominal_training_only_class_plurality_not_robust_or_population_risk'}


def plurality_objective(w,terms,reference_labels,temperature,*,margin=.1,smoothing=.05,reg=.01):
    C,same,mask,labels=terms;w=np.asarray(w,float);reference_labels=np.asarray(reference_labels)
    if (w.ndim!=1 or w.shape!=(C.shape[-1],) or not np.isfinite(w).all() or np.any(w<0)
        or reference_labels.shape!=(C.shape[1],) or not np.isfinite(temperature) or temperature<=0
        or margin<0 or smoothing<=0 or reg<0):raise ValueError('invalid nominal plurality objective input')
    if not len(labels):return 0.,np.zeros_like(w)
    classes=np.unique(reference_labels)
    membership=(reference_labels[:,None]==classes[None,:]).astype(float)
    D=np.einsum('ijd,d->ij',C,w)
    Q,S=balanced_membership(D,mask,temperature,return_slope=True)
    votes=Q@membership/5
    weighted_C=np.einsum('ij,ijd->id',S,C)/np.maximum(S.sum(axis=1)[:,None],np.finfo(float).tiny)
    slopes=S@membership
    gradient_votes=(slopes[:,:,None]*weighted_C[:,None,:]-np.einsum('ij,jc,ijd->icd',S,membership,C))/(5*temperature)
    own=np.searchsorted(classes,labels);alpha=np.array([1/(len(np.unique(labels))*sum(labels==label)) for label in labels])
    gaps=margin+votes-votes[np.arange(len(labels)),own,None]
    logits=gaps/smoothing;logits[np.arange(len(labels)),own]=-np.inf
    normalizer=logsumexp(np.column_stack((np.zeros(len(labels)),logits)),axis=1)
    rival_coefficients=np.exp(logits-normalizer[:,None])
    gradient=np.einsum('i,ic,icd->d',alpha,rival_coefficients,
        gradient_votes-gradient_votes[np.arange(len(labels)),own,None,:])
    log=np.log(len(w)*(w+1e-12))
    value=float(smoothing*(alpha@normalizer)+reg*np.sum(w*log))
    gradient+=reg*(log+w/(w+1e-12))
    return value,gradient


def soft_to_hard_plurality_certificate(distances,reference_labels,predicted_class,temperature):
    """Conditional nominal theorem for a supplied distance row, not learned risk.

    If the soft-vote winner beats each rival by >2*T/5 where T is the proved
    gap/cardinality leakage bound, it also wins the hard5 strict plurality.
    This is not the project's shared-box majority certificate.
    """
    d=np.asarray(distances,float);y=np.asarray(reference_labels)
    if d.ndim!=1 or y.shape!=d.shape or len(d)<5 or not np.isfinite(d).all():raise ValueError('valid distance row required')
    classes=np.unique(y)
    if predicted_class not in classes:raise ValueError('predicted class missing from references')
    q=balanced_membership(d[None,:],np.ones((1,len(d)),bool),temperature)[0]
    vote={c:float(q[y==c].sum()/5) for c in classes}
    gap=0. if len(d)==5 else float(np.sort(d)[5]-np.sort(d)[4])
    leakage=topk_leakage_bound(gap,len(d),temperature=temperature)
    rivals=[vote[c] for c in classes if c!=predicted_class]
    separation=np.inf if not rivals else vote[predicted_class]-max(rivals)
    certified=separation>2*leakage/5
    return {'certified':bool(certified),'soft_plurality_gap':float(separation),
        'hard_neighborhood_gap':gap,'leakage_bound':leakage,'required_margin':2*leakage/5,
        'scope':'conditional_nominal_soft_to_hard_strict_plurality_not_shared_box_or_generalization'}
