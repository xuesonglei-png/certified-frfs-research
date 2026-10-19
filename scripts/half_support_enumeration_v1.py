"""Sound nominal witness rejection before exhaustive finite-recipe checks.

Pruning only rejects measured nominal violations with a numerical buffer.
Nominal top-five ties are widened conservatively before bounding possible
same-class majority. Failure of a sufficient certifier is not robust
mathematical infeasibility of the underlying problem.
"""
import numpy as np
from certificate_barrier_support_v1 import metric_weights


def possible_nominal_majority(D,y,*,tie_buffer=1e-10):
    D=np.asarray(D,float).copy();y=np.asarray(y)
    if D.shape!=(len(y),len(y)) or len(y)<6:raise ValueError('square distances with >=6 objects required')
    np.fill_diagonal(D,np.inf)
    kth=np.partition(D,4,axis=1)[:,4]
    tol=tie_buffer*(1+abs(kth))
    strict=D<kth[:,None]-tol[:,None]
    tied=(D>=kth[:,None]-tol[:,None])&(D<=kth[:,None]+tol[:,None])
    same=y[:,None]==y[None,:];np.fill_diagonal(same,False)
    maxfriends=(strict&same).sum(axis=1)+np.minimum((tied&same).sum(axis=1),5-strict.sum(axis=1))
    return maxfriends>=3,maxfriends


def reasons_of(record,*,eta=.1,buffer=1e-9):
    reasons=[]
    if record['positive_global_loss']>eta+buffer:reasons.append('nominal_positive_global_dependency')
    if any(v>eta+buffer for v in record['positive_class_loss']):reasons.append('nominal_positive_class_dependency')
    if any(v>eta+buffer for v in record['signed_class_loss']):reasons.append('nominal_signed_class_dependency')
    if any(c<r for c,r in zip(record['possible_majority_counts'],record['required_counts'],strict=True)):
        reasons.append('nominal_tie_conservative_original_protected_majority_shortfall')
    return reasons


class NominalWitnesses:
    def __init__(self,A,y,ref,protected):
        self.A=np.asarray(A,float);self.y=np.asarray(y);self.ref=np.asarray(ref,float)
        self.protected=np.asarray(protected,bool);self.classes=np.unique(y)
        self.components=abs(self.A.T[:,:,None]-self.A.T[:,None,:])
        self.cross=self.y[:,None]!=self.y[None,:]
        D=np.einsum('d,dij->ij',self.ref,self.components)
        self.gamma=np.minimum(1.,np.min(np.where(self.cross,D,np.inf),axis=1))
        self.required=[int(np.ceil(.9*sum((self.y==c)&self.protected)-1e-12)) for c in self.classes]

    def __call__(self,features):
        fs=list(features);w=metric_weights(self.A,fs,self.ref)
        D=np.einsum('d,dij->ij',w,self.components[fs])
        gamma=np.minimum(1.,np.min(np.where(self.cross,D,np.inf),axis=1))
        signed=self.gamma-gamma;positive=np.maximum(0.,signed)
        possible,maxfriends=possible_nominal_majority(D,self.y)
        np.fill_diagonal(D,np.inf);nn=np.argsort(D,axis=1,kind='stable')[:,:5]
        counts=np.column_stack([np.sum(self.y[nn]==c,axis=1) for c in self.classes])
        correct=self.classes[counts.argmax(axis=1)]==self.y
        out={'weights':w.tolist(),'positive_global_loss':float(positive.mean()),
            'positive_class_loss':[float(positive[self.y==c].mean()) for c in self.classes],
            'signed_class_loss':[float(signed[self.y==c].mean()) for c in self.classes],
            'possible_majority_counts':[int(sum(possible&(self.y==c)&self.protected)) for c in self.classes],
            'required_counts':self.required,'max_sameclass_in_nominal_top5':maxfriends.tolist(),
            'training_LOO_BA':float(np.mean([correct[self.y==c].mean() for c in self.classes]))}
        out['pruning_reasons']=reasons_of(out)
        return out
