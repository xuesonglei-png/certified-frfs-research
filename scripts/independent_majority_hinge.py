"""Scalar box/witness oracle and full epigraph LP for small independent QA."""
import numpy as np
from scipy.optimize import linprog


def coefficient(A,i,e,f,eps):
    values=[]
    for a in range(A.shape[1]):
        ilo,ihi=max(0,A[i,a]-eps),min(1,A[i,a]+eps)
        elo,ehi=max(0,A[e,a]-eps),min(1,A[e,a]+eps)
        flo,fhi=max(0,A[f,a]-eps),min(1,A[f,a]+eps)
        endpoints=[]
        for t in (ilo,ihi):
            near=max(0,elo-t,t-ehi);far=max(abs(t-flo),abs(t-fhi))
            endpoints.append(near-far)
        values.append(min(endpoints))
    return np.array(values)


def independent_oracle(A,w,info):
    A,w=np.asarray(A,float),np.asarray(w,float);losses=[];faces=[]
    margin,eps=info['margin'],info['eps']
    for row in info['witnesses']:
        i=row['query'];constant=row['constant_loss'];gradient=np.zeros(len(w));intercept=0.;chosen=[];friends=[]
        if constant is not None:value=constant;intercept=constant
        else:
            enemies=[]
            for j,e in enumerate(row['enemies']):
                C=[coefficient(A,i,e,f,eps) for f in row['friends']]
                contrasts=[float(c@w) for c in C]
                worst=min(range(3),key=lambda f:(contrasts[f],f))
                enemies.append({'id':e,'index':j,'hinge':max(0.,1-contrasts[worst]/margin),'friend':worst,'C':C[worst]})
            selected=sorted(enemies,key=lambda r:(-r['hinge'],r['id']))[:3]
            value=sum(r['hinge'] for r in selected)/3
            chosen=[r['index'] for r in selected];friends=[r['friend'] for r in selected]
            for r in selected:
                if r['hinge']>0:gradient-=r['C']/(3*margin);intercept+=1/3
        losses.append(value);faces.append({'query':i,'gradient':gradient.tolist(),'intercept':intercept,
            'enemy_indices':chosen,'worst_friend_indices':friends})
    return np.array(losses),faces


def independent_full_lp(A,anchor,info,*,reg=.01,C=None,rhs=None):
    """Full top3-CVaR epigraph, distinct from cut-plane master construction."""
    A,anchor=np.asarray(A,float),np.asarray(anchor,float);p=len(anchor)
    eligible=[r for r in info['witnesses'] if r['constant_loss'] is None];offset=p+len(eligible)
    excess={};counter=offset
    for row in eligible:
        for e in row['enemies']:excess[(row['query'],e)]=counter;counter+=1
    uoffset=counter;size=counter+p;c=np.zeros(size);c[uoffset:]=reg
    alpha=np.array(info['alpha']);constant=sum(alpha[r['query']]*r['constant_loss'] for r in info['witnesses'] if r['constant_loss'] is not None)
    matrix=[];bounds=[]
    for j,row in enumerate(eligible):
        i=row['query'];t=p+j;c[t]=alpha[i]
        for e in row['enemies']:
            v=excess[(i,e)];c[v]=alpha[i]/3
            for f in row['friends']:
                constraint=np.zeros(size);constraint[:p]=-coefficient(A,i,e,f,info['eps'])/info['margin']
                constraint[t]=-1;constraint[v]=-1;matrix.append(constraint);bounds.append(-1.)
    for j in range(p):
        positive=np.zeros(size);positive[j]=1;positive[uoffset+j]=-1;matrix.append(positive);bounds.append(anchor[j])
        negative=np.zeros(size);negative[j]=-1;negative[uoffset+j]=-1;matrix.append(negative);bounds.append(-anchor[j])
    if C is not None:
        for row,value in zip(C,rhs):
            constraint=np.zeros(size);constraint[:p]=-row;matrix.append(constraint);bounds.append(-value)
    equality=np.zeros((1,size));equality[0,:p]=1
    fit=linprog(c,A_ub=np.array(matrix),b_ub=np.array(bounds),A_eq=equality,b_eq=[1],
                bounds=[(0,1)]*p+[(0,None)]*(size-p),method='highs')
    assert fit.success;return {'objective':float(fit.fun+constant),'weights':fit.x[:p].tolist()}
