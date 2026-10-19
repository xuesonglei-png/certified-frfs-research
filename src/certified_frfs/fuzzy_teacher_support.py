"""Reference-only raw-attribute supports from triangular fuzzy teacher scores."""
import numpy as np
from .triangular_classifier import triangular_memberships

def fuzzy_score_supports(X,y,coef,intercept,fractions=(.5,.65,.8)):
    X=np.asarray(X,float);y=np.asarray(y);coef=np.asarray(coef,float);intercept=np.asarray(intercept,float)
    n,d=X.shape;q=coef.shape[0]
    if coef.shape!=(q,d,3) or intercept.shape!=(q,) or len(y)!=n:raise ValueError('invalid fuzzy teacher')
    Z=triangular_memberships(X)
    target=np.einsum('ndk,qdk->nq',Z,coef);target-=target[:,[0]]
    weights=np.array([1/(q*sum(y==c)) for c in y])
    def error(fs):
        design=np.column_stack([np.ones(n),Z[:,fs].reshape(n,-1)])
        reg=np.eye(design.shape[1])*.001;reg[0,0]=0
        beta=np.linalg.solve(design.T@(weights[:,None]*design)+reg,design.T@(weights[:,None]*target))
        residual=design@beta-target
        return float(weights@np.mean(residual*residual,axis=1))
    sizes={int(np.ceil(d*f)) for f in fractions};active=list(range(d));rows=[]
    while len(active)>min(sizes):
        choice=min(((a,error([v for v in active if v!=a])) for a in active),key=lambda t:(t[1],t[0]));active.remove(choice[0])
        if len(active) in sizes:rows.append({'features':active.copy(),'reconstruction_error':choice[1],'origin':'triangular_teacher_score_backward'})
    return rows
