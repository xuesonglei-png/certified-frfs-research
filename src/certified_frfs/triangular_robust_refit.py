"""Exact shared-raw-box LP on grouped triangular membership scores."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from .triangular_classifier import triangular_memberships,triangular_box_margins

def fit_triangular_robust(X,y,W_teacher,b_teacher,classes,features,*,eps=.01,ratio=.5):
    X=np.asarray(X,float);y=np.asarray(y);classes=np.asarray(classes);fs=np.asarray(features,int)
    original=triangular_box_margins(X,y,W_teacher,b_teacher,classes,eps=eps)
    if not 0<ratio<=1 or not len(fs) or len(set(fs))!=len(fs) or np.any(fs<0) or np.any(fs>=X.shape[1]):raise ValueError('invalid support or ratio')
    n,d=X.shape;q=len(classes);k=len(fs);nw=q*k*3;bo=nw;ao=bo+q;to=ao+nw
    own=np.argmax(y[:,None]==classes[None,:],axis=1);pairs=[(i,r) for i in range(n) for r in range(q) if r!=own[i]]
    so=to+len(pairs)*k;do=so+n;nv=do+n*(q-1);protected=original>1e-10
    rows=2*nw+len(pairs)*(3*k+1)+int(protected.sum())*(q-1)+2*n*(q-1)
    A=lil_matrix((rows,nv));bound=np.zeros(rows);row=0
    for j in range(nw):
        for sign in (1,-1):A[row,j]=sign;A[row,ao+j]=-1;row+=1
    lo=np.maximum(0,X[:,fs]-eps);hi=np.minimum(1,X[:,fs]+eps)
    points=np.stack([lo,hi,np.clip(.5,lo,hi)],axis=-1);members=triangular_memberships(points)
    for pi,(i,r) in enumerate(pairs):
        c=own[i]
        for a in range(k):
            for p in range(3):
                for m in range(3):
                    A[row,(c*k+a)*3+m]=-members[i,a,p,m]
                    A[row,(r*k+a)*3+m]=members[i,a,p,m]
                A[row,to+pi*k+a]=1;row+=1
        for hard in ([False,True] if protected[i] else [False]):
            A[row,bo+c]=-1;A[row,bo+r]=1
            A[row,to+pi*k:to+(pi+1)*k]=-1
            if hard:bound[row]=-max(ratio*original[i],1e-7)
            else:A[row,so+i]=-1;bound[row]=-1
            row+=1
    clean=triangular_memberships(X[:,fs]);full=triangular_memberships(X)
    target=np.einsum('ndk,qdk->nq',full,np.asarray(W_teacher)-np.asarray(W_teacher)[0])+np.asarray(b_teacher)-np.asarray(b_teacher)[0]
    for i in range(n):
        for c in range(1,q):
            for sign in (1,-1):
                for a in range(k):
                    for m in range(3):
                        A[row,(c*k+a)*3+m]=sign*clean[i,a,m]
                        A[row,a*3+m]=-sign*clean[i,a,m]
                A[row,bo+c]=sign;A[row,bo]=-sign
                A[row,do+i*(q-1)+c-1]=-1;bound[row]=sign*target[i,c];row+=1
    assert row==rows
    objective=np.zeros(nv);objective[ao:to]=.01
    for i in range(n):
        balance=1/(q*sum(y==y[i]));objective[so+i]=balance
        objective[do+i*(q-1):do+(i+1)*(q-1)]=balance/(q-1)
    limits=[(None,None)]*(nw+q)+[(0,None)]*nw+[(None,None)]*(len(pairs)*k)+[(0,None)]*(n+n*(q-1))
    for j in range(k*3):limits[j]=(0,0)
    limits[bo]=(0,0)
    result=linprog(objective,A_ub=A.tocsr(),b_ub=bound,bounds=limits,method='highs',options={'time_limit':10})
    if not result.success:return {'feasible':False,'status':int(result.status),'message':result.message}
    W=np.zeros_like(np.asarray(W_teacher,float));W[:,fs]=result.x[:nw].reshape(q,k,3);b=result.x[bo:bo+q]
    margins=triangular_box_margins(X,y,W,b,classes,eps=eps)
    verified=bool(np.all(margins[protected]>=ratio*original[protected]-1e-8) and np.all(margins[protected]>1e-10))
    return {'feasible':verified,'coef':W.tolist(),'intercept':b.tolist(),'classes':classes.tolist(),
        'objective':float(result.fun),'status':int(result.status),'protected_count':int(protected.sum()),
        'original_margins':original.tolist(),'deployed_margins':margins.tolist(),
        'scope':'shared_raw_triangular_exact_LP_all_positive_reference_anchors_not_population_risk'}
