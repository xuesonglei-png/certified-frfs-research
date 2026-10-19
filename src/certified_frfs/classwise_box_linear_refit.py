"""Classwise retained-anchor MILP for deployed exact-box margins."""
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
from .linear_retention import box_class_margins

def fit_classwise_box_linear(X,y,W_teacher,b_teacher,classes,features,*,eps=.01,ratio=.5,distillation=1.,retention=.9):
    X=np.asarray(X,float);y=np.asarray(y);classes=np.asarray(classes);fs=np.asarray(features,int)
    if not 0<retention<=1:raise ValueError('invalid retention')
    if not np.isfinite(distillation) or distillation < 0:raise ValueError('invalid distillation')
    original=box_class_margins(X,y,W_teacher,b_teacher,classes,eps=eps)
    if not 0<ratio<=1 or not len(fs) or len(set(fs))!=len(fs) or np.any(fs<0) or np.any(fs>=X.shape[1]):raise ValueError('invalid support or ratio')
    q=len(classes);k=len(fs);n=len(y);own=np.argmax(y[:,None]==classes[None,:],axis=1)
    classpairs=[(c,r) for c in range(q) for r in range(c+1,q)];pairids={p:i for i,p in enumerate(classpairs)}
    nw=q*k;wo=nw+q;uo=wo+nw;so=uo+len(classpairs)*k;do=so+n;nv=do+n*(q-1)
    protected=original>1e-10
    groups=[(y==c)&protected for c in classes]
    required=[int(np.ceil(retention*g.sum()-1e-12)) for g in groups]
    zo=nv;nv+=n
    coefficient_bound=max(1.,2*float(np.max(abs(np.asarray(W_teacher)-np.asarray(W_teacher)[0]))))
    intercept_bound=max(1.,2*float(np.max(abs(np.asarray(b_teacher)-np.asarray(b_teacher)[0]))))
    pairs=[(i,r) for i in range(n) for r in range(q) if r!=own[i]]
    nrows=2*nw+2*len(classpairs)*k+len(pairs)+int(protected.sum())*(q-1)+2*n*(q-1)+sum(g.any() for g in groups)
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
            if hard:
                threshold=max(ratio*original[i],1e-7)
                bigM=2*coefficient_bound*hi[i].sum()+2*intercept_bound+threshold
                A[row,zo+i]=bigM;bound[row]=bigM-threshold
            else:A[row,so+i]=-1;bound[row]=-1
            row+=1
    for group,minimum in zip(groups,required):
        if group.any():
            A[row,zo+np.flatnonzero(group)]=-1;bound[row]=-minimum;row+=1
    target=X@(np.asarray(W_teacher)-np.asarray(W_teacher)[0]).T+(np.asarray(b_teacher)-np.asarray(b_teacher)[0])
    for i in range(n):
        for c in range(1,q):
            for sign in (1,-1):
                for a in range(k):
                    A[row,c*k+a]=sign*X[i,fs[a]]
                    A[row,a]=-sign*X[i,fs[a]]
                A[row,nw+c]=sign;A[row,nw]=-sign
                A[row,do+i*(q-1)+c-1]=-1
                bound[row]=sign*target[i,c];row+=1
    assert row == nrows
    objective=np.zeros(nv);objective[wo:uo]=.01
    for i in range(n):
        mass=1/(q*sum(y==y[i]));objective[so+i]=mass
        objective[do+i*(q-1):do+(i+1)*(q-1)]=distillation*mass/(q-1)
    limits=[(-coefficient_bound,coefficient_bound)]*nw+[(-intercept_bound,intercept_bound)]*q
    limits += [(0,coefficient_bound)]*nw+[(0,2*coefficient_bound)]*(len(classpairs)*k)
    limits += [(0,None)]*(n+n*(q-1))+[(0,1) if protected[i] else (0,0) for i in range(n)]
    for j in range(k):limits[j]=(0,0)
    limits[nw]=(0,0)
    lower=[-np.inf if a is None else a for a,b in limits];upper=[np.inf if b is None else b for a,b in limits]
    integrality=np.zeros(nv,int);integrality[zo:]=1
    result=milp(objective,integrality=integrality,bounds=Bounds(lower,upper),
                constraints=LinearConstraint(A.tocsr(),-np.inf,bound),options={'time_limit':10,'mip_rel_gap':0})
    if not result.success:return {'feasible':False,'status':int(result.status),'message':result.message}
    W=np.zeros_like(np.asarray(W_teacher,float));W[:,fs]=result.x[:nw].reshape(q,k);b=result.x[nw:nw+q]
    margin=box_class_margins(X,y,W,b,classes,eps=eps)
    retained=(margin>=ratio*original-1e-8)&(margin>1e-10)&protected
    counts=[int((retained&g).sum()) for g in groups]
    verified=all(c>=r for c,r in zip(counts,required))
    return {'feasible':verified,'coef':W.tolist(),'intercept':b.tolist(),'status':int(result.status),
        'original_margins':original.tolist(),'deployed_margins':margin.tolist(),'protected_count':int(protected.sum()),
        'objective':float(result.fun),'distillation':distillation,'retention':retention,'required':required,'retained_counts':counts,
        'coefficient_bound':coefficient_bound,'intercept_bound':intercept_bound,
        'scope':'classwise_fraction_of_reference_teacher_anchors_not_all_anchors_or_population_risk'}
