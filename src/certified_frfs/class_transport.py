"""Class-aware distance transport; training objective, not a new certificate.

Combines the existing class-balanced robust soft-neighbor loss with sampled
distance reconstruction. Nonconvex supervised term; local solver only.
"""
import numpy as np
from scipy.optimize import minimize
from .core import _inputs
from .metric_transport import fit_transport_weights
from .softvote_metric import softvote_terms,softvote_objective


def class_transport_objective(v,A,target,initial,terms,*,alpha=.1,reg=.001):
    residual=A@v-target
    geometric=float(np.mean(residual**2)+reg*np.sum((v-initial)**2))
    gradient=2*A.T@residual/len(target)+2*reg*(v-initial)
    C,same,mask,labels=terms
    loss,grad=softvote_objective(v,C,same,mask,labels,temperature=.05,reg=0.)
    return geometric+alpha*loss,gradient+alpha*grad


def fit_class_transport(X,y,features,w_full,*,alpha=.1,reg=.001,eps=.01):
    X,y,fs,w=_inputs(X,y,features,w_full)
    initial,record=fit_transport_weights(X,fs,w,reg=reg)
    if record['status']=='degenerate':return initial,record
    pairs=np.array(record['pair_indices']);components=abs(X[pairs[:,0]]-X[pairs[:,1]])
    A=components[:,fs];target=components@w;terms=softvote_terms(X[:,fs],y,eps=eps)
    if not np.isfinite(alpha) or alpha<0:raise ValueError('nonnegative class loss coefficient required')
    mass=w.sum();floor=min(1e-6,mass/(2*len(fs)))
    start=np.maximum(initial,floor);start*=mass/start.sum()
    objective=lambda v:class_transport_objective(v,A,target,initial,terms,alpha=alpha,reg=reg)
    before=objective(start)[0]
    fit=minimize(objective,start,jac=True,method='SLSQP',bounds=[(floor,mass)]*len(fs),
        constraints={'type':'eq','fun':lambda v:v.sum()-mass,'jac':lambda v:np.ones(len(fs))},
        options={'maxiter':200,'ftol':1e-9})
    chosen=start.copy();status='fallback'
    if fit.success and np.isfinite(fit.x).all():
        candidate=np.maximum(fit.x,floor);candidate*=mass/candidate.sum()
        if objective(candidate)[0]<=before+1e-12:chosen=candidate;status='local_optimized'
    return chosen,{'status':status,'solver_success':bool(fit.success),'solver_message':str(fit.message),
        'objective_before':before,'objective_after':objective(chosen)[0],
        'alpha':alpha,'reg':reg,'eps':eps,'full_mass':float(mass),
        'distance_fit':record,'scope':'local_training_class_aware_transport_not_global_optimum_or_test_risk'}
