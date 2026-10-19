"""Linear refit using exact clipped-box pairwise margin constraints."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from .linear_retention import box_class_margins

def fit_exact_box_linear(X,y,W_teacher,b_teacher,classes,features,*,eps=.01,ratio=.5):
    X=np.asarray(X,float);y=np.asarray(y);classes=np.asarray(classes);fs=np.asarray(features,int)
    original=box_class_margins(X,y,W_teacher,b_teacher,classes,eps=eps)
    if not 0<ratio<=1 or not len(fs) or len(set(fs))!=len(fs) or np.any(fs<0) or np.any(fs>=X.shape[1]):raise ValueError('invalid support or ratio')
    q=len(classes);k=len(fs);n=len(y);own=np.argmax(y[:,None]==classes[None,:],axis=1)
    classpairs=[(c,r) for c in range(q) for r in range(c+1,q)];pairids={p:i for i,p in enumerate(classpairs)}
    nw=q*k;wo=nw+q;uo=wo+nw;so=uo+len(classpairs)*k;nv=so+n
    protected=original>1e-10;pairs=[(i,r) for i in range(n) for r in range(q) if r!=own[i]]
    nrows=2*nw+2*len(classpairs)*k+len(pairs)+int(protected.sum())*(q-1)
    A=lil_matrix((nrows,nv));bound=np.zeros(nrows);row=0
    for j in range(nw):
        A[row,j]=1;A[row,wo+j]=-1;row+=1
        A[row,j]=-1;A[row,wo+j]=-1;row+=1
    for pi,(c,r) in enumerate(classpairs):
        for a in range(k):
            for sign in (1,-1):
                A[row,c*k+a]=sign;A[row,r*k+a]=-sign;A[row,uo+pi*k+a]=-1;row+=1
    lo=np.maximum(0,X[:,fs]-eps);hi=np.minimum(1,X[:,fs]+eps);mid=(hi+lo)/2;rad=(hi-lo)/2
    for i,r in pairs:
        c=own[i];pi=pairids[tuple(sorted((c,r)))]
        for hard in ([False,True] if protected[i] else [False]):
            for a in range(k):
                A[row,c*k+a]=-mid[i,a];A[row,r*k+a]=mid[i,a];A[row,uo+pi*k+a]=rad[i,a]
            A[row,nw+c]=-1;A[row,nw+r]=1
            if hard:bound[row]=-max(ratio*original[i],1e-7)
            else:A[row,so+i]=-1;bound[row]=-1
            row+=1
    objective=np.zeros(nv);objective[wo:uo]=.01
    for i in range(n):objective[so+i]=1/(q*sum(y==y[i]))
    limits=[(None,None)]*(nw+q)+[(0,None)]*(nv-nw-q)
    for j in range(k):limits[j]=(0,0)
    limits[nw]=(0,0)
    result=linprog(objective,A_ub=A.tocsr(),b_ub=bound,bounds=limits,method='highs',options={'time_limit':10})
    if not result.success:return {'feasible':False,'status':int(result.status),'message':result.message}
    W=np.zeros_like(np.asarray(W_teacher,float));W[:,fs]=result.x[:nw].reshape(q,k);b=result.x[nw:nw+q]
    margin=box_class_margins(X,y,W,b,classes,eps=eps)
    verified=bool(np.all(margin[protected]>=ratio*original[protected]-1e-8) and np.all(margin[protected]>1e-10))
    return {'feasible':verified,'coef':W.tolist(),'intercept':b.tolist(),'status':int(result.status),
        'original_margins':original.tolist(),'deployed_margins':margin.tolist(),'protected_count':int(protected.sum()),
        'objective':float(result.fun),'scope':'exact_clipped_pair_box_LP_reference_margin_not_population_risk'}
