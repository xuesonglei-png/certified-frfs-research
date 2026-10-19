"""Reference-only backward score reconstruction, followed by certification."""
import numpy as np
from .redistributed_loss import transfer_weights
from .capped_sparse import capped_sparse_projection

def reconstruction_supports(X,y,W,b,fractions=(.5,.65,.8)):
    X=np.asarray(X,float); y=np.asarray(y); W=np.asarray(W,float); b=np.asarray(b,float)
    n,d=X.shape
    if len(y)!=n or W.ndim!=2 or W.shape[1]!=d or len(b)!=len(W) or len(W)<2:
        raise ValueError('invalid teacher dimensions')
    if any(not .5<=f<=1 for f in fractions):raise ValueError('fraction outside cap feasibility')
    target=X@(W[1:]-W[0]).T+(b[1:]-b[0])
    classes=np.unique(y); alpha=np.array([1/(len(classes)*sum(y==c)) for c in y])
    def error(fs):
        Z=np.column_stack([np.ones(n),X[:,fs]])
        regularizer=np.eye(len(fs)+1)*.001;regularizer[0,0]=0
        coef=np.linalg.solve(Z.T@(alpha[:,None]*Z)+regularizer,Z.T@(alpha[:,None]*target))
        residual=Z@coef-target
        return float(alpha@np.mean(residual**2,axis=1))
    sizes={int(np.ceil(d*f)) for f in fractions}; fs=list(range(d));rows=[]
    while len(fs)>min(sizes):
        a,value=min(((a,error([v for v in fs if v!=a])) for a in fs),key=lambda z:(z[1],z[0]))
        fs.remove(a)
        if len(fs) in sizes:
            full=np.ones(d)/np.sqrt(d);mass=full.sum()
            transferred=transfer_weights(X,fs,full)/mass
            retained=capped_sparse_projection(transferred,len(fs),2/d)*mass
            wide=np.zeros(d);wide[fs]=retained
            rows.append({'features':fs.copy(),'weights':wide.tolist(),'objective':None,
                         'reconstruction_error':value,'origin':'reference_teacher_score_backward'})
    return rows
