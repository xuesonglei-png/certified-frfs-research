"""Additive triangular fuzzy scores with exact shared-source box margins."""
import numpy as np

def triangular_memberships(X):
    X=np.asarray(X,float)
    if not np.isfinite(X).all() or np.any(X<0) or np.any(X>1):raise ValueError('finite normalized inputs required')
    return np.stack([np.maximum(0,1-2*X),1-abs(2*X-1),np.maximum(0,2*X-1)],axis=-1)

def triangular_box_margins(X,y,W,b,classes,*,eps=.01,independent_memberships=False):
    X=np.asarray(X,float);y=np.asarray(y);W=np.asarray(W,float);b=np.asarray(b,float);classes=np.asarray(classes)
    if X.ndim!=2 or W.shape!=(len(classes),X.shape[1],3) or b.shape!=(len(classes),) or y.shape!=(len(X),):raise ValueError('invalid dimensions')
    if len(classes)<2 or len(np.unique(classes))!=len(classes) or not np.all(np.isin(y,classes)):raise ValueError('invalid classes')
    if not np.isfinite(eps) or eps<0 or not np.isfinite(W).all() or not np.isfinite(b).all():raise ValueError('invalid radius or scores')
    triangular_memberships(X)
    lo=np.maximum(0,X-eps);hi=np.minimum(1,X+eps);knot=np.clip(.5,lo,hi)
    points=np.stack([lo,hi,knot],axis=-1);memberships=triangular_memberships(points)
    own=np.argmax(y[:,None]==classes[None,:],axis=1);difference=W[own,None]-W[None]
    if independent_memberships:
        lower,upper=memberships.min(axis=2),memberships.max(axis=2)
        terms=np.where(difference>=0,difference*lower[:,None],difference*upper[:,None]).sum(axis=-1)
    else:
        terms=np.einsum('nqdk,ndpk->nqdp',difference,memberships).min(axis=-1)
    pair=terms.sum(axis=-1)+b[own,None]-b[None]
    pair[np.arange(len(X)),own]=np.inf
    return pair.min(axis=1)
