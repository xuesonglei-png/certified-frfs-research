"""Convex top-three enemy hinge bound for robust strict-majority 5NN.

For fixed three same-class friends, each enemy has the largest hinge over
its friend contrasts. Misclassification requires at least three enemies
with hinge >=1. Their three largest hinges therefore upper-bound the error.
The bound holds for every joint reference/query box realization, but concerns
the supplied finite crossgroup training table, not population test risk.
"""
import numpy as np
from scipy.spatial.distance import cdist
from crossgroup_soft5_terms import assignments


class RobustMajorityHinge:
    def __init__(self,A,y,anchor,*,margin=.02,eps=.01):
        A,y,anchor=np.asarray(A,float),np.asarray(y),np.asarray(anchor,float)
        if (A.ndim!=2 or len(A)<7 or y.shape!=(len(A),) or not np.isfinite(A).all()
            or A.min()<0 or A.max()>1 or anchor.shape!=(A.shape[1],) or not np.isfinite(anchor).all()
            or anchor.min()<0 or abs(anchor.sum()-1)>1e-8 or not np.isfinite(margin) or margin<=0
            or not np.isfinite(eps) or not 0<=eps<=1):raise ValueError('invalid finite robust majority table')
        self.A,self.y,self.anchor=A.copy(),y.copy(),anchor.copy()
        self.margin,self.eps=margin,eps;self.lo=np.maximum(0,A-eps);self.hi=np.minimum(1,A+eps)
        self.group=assignments(y);self.rows=[];self.cache={};classes=np.unique(y)
        self.alpha=np.array([1/(len(classes)*sum(y==c)) for c in y])
        D=cdist(A*anchor,A*anchor,'cityblock')
        for i in range(len(A)):
            refs=np.flatnonzero(self.group!=self.group[i]);friends=refs[y[refs]==y[i]];enemies=refs[y[refs]!=y[i]]
            friends=friends[np.lexsort((friends,D[i,friends]))][:3]
            constant=1. if len(refs)<5 or len(friends)<3 else 0. if len(enemies)<3 else None
            self.rows.append({'query':i,'references':refs.tolist(),'friends':friends.tolist(),'enemies':enemies.tolist(),
                'constant_loss':constant,'kind':'insufficient_reference_or_friends' if constant==1 else
                'at_most_two_enemies' if constant==0 else 'top_three_enemy_hinge'})

    def coefficients(self,i):
        if i not in self.cache:
            row=self.rows[i];f,e=np.array(row['friends']),np.array(row['enemies']);values=[]
            for t in (self.lo[i],self.hi[i]):
                near=np.maximum(0,np.maximum(self.lo[e]-t,t-self.hi[e]))
                far=np.maximum(abs(t-self.lo[f]),abs(t-self.hi[f]))
                values.append(near[:,None,:]-far[None,:,:])
            self.cache[i]=np.minimum(*values)
        return self.cache[i]

    def oracle(self,w):
        w=np.asarray(w,float);loss=[];faces=[]
        for i,row in enumerate(self.rows):
            if row['constant_loss'] is not None:
                value=row['constant_loss'];gradient=np.zeros(len(w));intercept=value;selected=[];worst=[]
            else:
                C=self.coefficients(i);contrast=C@w;friend=contrast.argmin(axis=1)
                hinge=np.maximum(0,1-contrast[np.arange(len(contrast)),friend]/self.margin)
                selected=np.lexsort((np.array(row['enemies']),-hinge))[:3].tolist()
                worst=[int(friend[j]) for j in selected];active=[j for j in selected if hinge[j]>0]
                gradient=-sum((C[j,friend[j]] for j in active),start=np.zeros(len(w)))/(3*self.margin)
                intercept=len(active)/3;value=float(hinge[selected].mean())
            loss.append(value);faces.append({'query':i,'gradient':gradient.tolist(),'intercept':intercept,
                'enemy_indices':selected,'worst_friend_indices':worst})
        return np.array(loss),faces

    def objective(self,w,*,reg=.01):
        losses,_=self.oracle(w)
        return float(self.alpha@losses+reg*abs(np.asarray(w)-self.anchor).sum())

    def export(self):
        return {'margin':self.margin,'eps':self.eps,'group_assignments':self.group.tolist(),
                'alpha':self.alpha.tolist(),'witnesses':self.rows,
                'bound':'balanced_crossgroup_error_for_every_joint_box_realization_le_top3_hinge_mean',
                'scope':'fixed_training_witness_bound_not_generalization_or_population_risk'}
