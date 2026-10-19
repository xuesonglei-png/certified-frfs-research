"""Exact box margins for a fixed centered linear classifier projection.

The teacher is fixed. Omitted measurements are replaced by a fixed reference
center. This certificate does not apply to a newly refitted classifier.
"""
import numpy as np


def linear_parameters(estimator):
    coef=np.asarray(estimator.coef_,float)
    intercept=np.asarray(estimator.intercept_,float)
    if len(estimator.classes_)==2 and coef.shape[0]==1:
        # Binary decision_function>0 selects classes_[1]; convert to scores
        # whose argmax agrees away from exact zero ties.
        coef=np.vstack([np.zeros_like(coef[0]),coef[0]])
        intercept=np.r_[0.,intercept[0]]
    return coef,intercept,np.asarray(estimator.classes_)


def projected_parameters(coef,intercept,features,center):
    coef=np.asarray(coef,float);intercept=np.asarray(intercept,float);center=np.asarray(center,float)
    fs=np.asarray(list(features),int);d=coef.shape[1]
    if coef.ndim!=2 or intercept.shape!=(len(coef),) or center.shape!=(d,) or len(set(fs))!=len(fs) or np.any(fs<0) or np.any(fs>=d):
        raise ValueError('invalid linear parameters or support')
    if not all(np.isfinite(v).all() for v in (coef,intercept,center)):
        raise ValueError('nonfinite linear parameters')
    omitted=np.ones(d,bool);omitted[fs]=False
    reduced=coef.copy();reduced[:,omitted]=0.
    shifted=intercept+coef[:,omitted]@center[omitted]
    return reduced,shifted


def box_class_margins(X,y,coef,intercept,classes,*,eps=.01):
    X=np.asarray(X,float);y=np.asarray(y);coef=np.asarray(coef,float)
    intercept=np.asarray(intercept,float);classes=np.asarray(classes)
    if X.ndim!=2 or coef.ndim!=2 or coef.shape!=(len(classes),X.shape[1]) or intercept.shape!=(len(classes),) or y.shape!=(len(X),) or len(classes)<2:
        raise ValueError('invalid table or class scores')
    if len(np.unique(classes))!=len(classes) or not np.all(np.isin(y,classes)) or not np.isfinite(eps) or eps<0:
        raise ValueError('invalid classes or radius')
    if not all(np.isfinite(v).all() for v in (X,coef,intercept)) or np.any(X<0) or np.any(X>1):
        raise ValueError('finite normalized table required')
    own=np.argmax(y[:,None]==classes[None,:],axis=1)
    delta=coef[own,None,:]-coef[None,:,:]
    offset=intercept[own,None]-intercept[None,:]
    lo=np.maximum(0.,X-eps);hi=np.minimum(1.,X+eps)
    # A linear function achieves its minimum at the sign-selected vertex.
    pair=offset+np.sum(np.where(delta>=0,delta*lo[:,None,:],delta*hi[:,None,:]),axis=2)
    pair[np.arange(len(X)),own]=np.inf
    return pair.min(axis=1)


def linear_retention_certificate(X,y,coef,intercept,classes,features,center,*,eps=.01,retention=.9,margin_ratio=.5):
    if not 0<retention<=1 or not 0<margin_ratio<=1:
        raise ValueError('invalid retention or margin ratio')
    initial=box_class_margins(X,y,coef,intercept,classes,eps=eps)
    rc,rb=projected_parameters(coef,intercept,features,center)
    reduced=box_class_margins(X,y,rc,rb,classes,eps=eps)
    protected=initial>1e-10
    retained=protected & (reduced>1e-10) & (reduced>=margin_ratio*initial-1e-12)
    groups=[(np.asarray(y)==q)&protected for q in classes]
    required=[int(np.ceil(retention*g.sum()-1e-12)) for g in groups]
    counts=[int((g&retained).sum()) for g in groups]
    return {'feasible':all(c>=r for c,r in zip(counts,required)),
            'required':required,'retained_counts':counts,'protected_count':int(protected.sum()),
            'initial_margins':initial.tolist(),'reduced_margins':reduced.tolist(),
            'retained':retained.tolist(),'eps':eps,'margin_ratio':margin_ratio,'retention':retention,
            'scope':'exact_fixed_linear_classifier_box_margins_on_reference_anchors_not_refit_or_test_risk'}
