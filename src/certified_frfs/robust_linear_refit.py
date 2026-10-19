"""LP refit with explicit deployed robust margins; no test-risk guarantee."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from .linear_retention import box_class_margins


def fit_robust_linear(X,y,teacher_coef,teacher_intercept,classes,features,*,eps=.01,ratio=.5):
    X=np.asarray(X,float);y=np.asarray(y);classes=np.asarray(classes)
    fs=np.asarray(features,int);k=len(fs);q=len(classes);n=len(y)
    if not k or len(set(fs))!=k or np.any(fs<0) or np.any(fs>=X.shape[1]):
        raise ValueError('invalid support')
    original=box_class_margins(X,y,teacher_coef,teacher_intercept,classes,eps=eps)
    protected=original>1e-10
    own=np.argmax(y[:,None]==classes[None,:],axis=1)
    # Variables W, intercept, absolute W, per-anchor soft hinge slack.
    nw=q*k;nb=q;nt=nw;nv=2*nw+nb+n;to=nw+nb;so=2*nw+nb
    pairs=[(i,r) for i in range(n) for r in range(q) if r!=own[i]]
    rows=2*nw+len(pairs)+int(protected.sum())*(q-1)
    A=lil_matrix((rows,nv));b=np.zeros(rows);row=0
    for j in range(nw):
        A[row,j]=1;A[row,to+j]=-1;row+=1
        A[row,j]=-1;A[row,to+j]=-1;row+=1
    for i,r in pairs:
        c=own[i]
        for hard in ([False,True] if protected[i] else [False]):
            for a in range(k):
                A[row,c*k+a]=-X[i,fs[a]];A[row,r*k+a]=X[i,fs[a]]
                A[row,to+c*k+a]=eps;A[row,to+r*k+a]=eps
            A[row,nw+c]=-1;A[row,nw+r]=1
            if hard:
                b[row]=-max(ratio*original[i],1e-7)
            else:
                A[row,so+i]=-1;b[row]=-1
            row+=1
    objective=np.zeros(nv);objective[to:so]=.01
    for i in range(n):objective[so+i]=1/(q*sum(y==y[i]))
    bounds=[(None,None)]*(nw+nb)+[(0,None)]*(nt+n)
    # Fix one class score as gauge; avoids unbounded common intercept shifts.
    for j in range(k):bounds[j]=(0,0)
    bounds[nw]=(0,0)
    result=linprog(objective,A_ub=A.tocsr(),b_ub=b,bounds=bounds,method='highs',options={'time_limit':10})
    if not result.success:return {'feasible':False,'status':int(result.status),'message':result.message}
    W=np.zeros_like(np.asarray(teacher_coef,float));W[:,fs]=result.x[:nw].reshape(q,k)
    intercept=result.x[nw:nw+nb]
    margin=box_class_margins(X,y,W,intercept,classes,eps=eps)
    verified=bool(np.all(margin[protected]>=ratio*original[protected]-1e-8) and np.all(margin[protected]>1e-10))
    return {'feasible':verified,'status':int(result.status),'coef':W.tolist(),'intercept':intercept.tolist(),
            'original_margins':original.tolist(),'deployed_margins':margin.tolist(),
            'protected_count':int(protected.sum()),'objective':float(result.fun),
            'scope':'conservative_symmetric_box_LP_exact_clipped_margin_check_reference_only'}
