"""Replay all common-fuzzy inner and outer classifier fits independently."""
import numpy as np
from sklearn.linear_model import LogisticRegression
import audit_common_fold_recent as audit
from certified_frfs.linear_retention import linear_parameters

def memberships(X):
    return np.stack([np.maximum(0,1-2*X),1-abs(2*X-1),np.maximum(0,2*X-1)],axis=-1)

def independent_fit(A,y,fs,C):
    classes,counts=np.unique(y,return_counts=True)
    if fs:
        model=LogisticRegression(C=C,max_iter=2000,random_state=0).fit(memberships(A[:,fs]).reshape(len(A),-1),y)
        coef,b,classes=linear_parameters(model);W=np.zeros((len(classes),A.shape[1],3));W[:,fs]=coef.reshape(len(classes),len(fs),3)
    else:W=np.zeros((len(classes),A.shape[1],3));b=np.log(counts/counts.sum())
    return {'coef':W.tolist(),'intercept':b.tolist(),'classes':classes.tolist()}

def score(B,truth,fit):
    pred=np.array(fit['classes'])[(np.einsum('ndk,qdk->nq',memberships(B),np.array(fit['coef']))+np.array(fit['intercept'])).argmax(axis=1)]
    return float(np.mean([np.mean(pred[truth==c]==c) for c in np.unique(truth)]))

if __name__=='__main__':
    audit.FOLDER=audit.ROOT/'results/common-fuzzy-classifier'
    audit.independent_fit=independent_fit;audit.score=score;audit.main()
