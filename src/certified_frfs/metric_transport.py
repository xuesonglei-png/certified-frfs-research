"""Training-only convex reconstruction of the original weighted L1 distances.

Nonnegative reduced weights conserve full weight mass. A fixed sample of
training pairs defines a convex quadratic program; it is not test-risk
minimization. The existing changed-metric certificate must be recomputed.
"""
import numpy as np
from scipy.optimize import minimize
from .core import _inputs
from .redistributed_loss import transfer_weights


def transport_objective(v,A,target,initial,reg):
    residual=A@v-target
    return (float(np.mean(residual**2)+reg*np.sum((v-initial)**2)),
            2*A.T@residual/len(target)+2*reg*(v-initial))


def fit_transport_weights(X,features,w_full,*,max_pairs=1024,seed=960,reg=.001):
    X=np.asarray(X,float)
    X,_,fs,w=_inputs(X,np.zeros(len(X),int),features,w_full)
    if not len(fs):raise ValueError('nonempty retained subset required')
    if not np.isfinite(reg) or reg<=0 or int(max_pairs)!=max_pairs or max_pairs<1:
        raise ValueError('positive regularization and integer pair budget required')
    initial=transfer_weights(X,fs,w);mass=float(w.sum())
    if len(X)<2 or mass==0:
        return initial,{'status':'degenerate','training_pair_count':0,'full_mass':mass}
    total=len(X)*(len(X)-1)//2
    if total<=max_pairs:
        ii,jj=np.triu_indices(len(X),1)
    else:
        rng=np.random.default_rng(seed);pairs=set()
        while len(pairs)<max_pairs:
            i,j=rng.integers(0,len(X),2)
            if i!=j:pairs.add((int(min(i,j)),int(max(i,j))))
        ii,jj=np.asarray(sorted(pairs),int).T
    components=abs(X[ii]-X[jj]);target=components@w;A=components[:,fs]
    before,_=transport_objective(initial,A,target,initial,reg)
    fit=minimize(lambda v:transport_objective(v,A,target,initial,reg),initial,jac=True,
        method='SLSQP',bounds=[(0,mass)]*len(fs),
        constraints={'type':'eq','fun':lambda v:v.sum()-mass,'jac':lambda v:np.ones(len(fs))},
        options={'maxiter':200,'ftol':1e-10})
    candidate=initial.copy();status='fallback'
    if fit.success and np.isfinite(fit.x).all():
        projected=np.maximum(fit.x,0)
        if projected.sum()>0:
            projected*=mass/projected.sum();after,_=transport_objective(projected,A,target,initial,reg)
            if after<=before+1e-12:candidate=projected;status='optimized'
    residual_before=A@initial-target;residual_after=A@candidate-target
    return candidate,{'status':status,'solver_success':bool(fit.success),'solver_message':str(fit.message),
        'training_pair_count':len(target),'full_mass':mass,'seed':seed,'reg':reg,
        'training_mse_before':float(np.mean(residual_before**2)),
        'training_mse_after':float(np.mean(residual_after**2)),
        'regularized_objective_before':before,
        'regularized_objective_after':transport_objective(candidate,A,target,initial,reg)[0],
        'pair_indices':np.column_stack([ii,jj]).tolist(),
        'scope':'training_distance_transport_not_prediction_or_dependency_certificate'}
