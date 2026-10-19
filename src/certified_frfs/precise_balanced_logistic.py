"""Stationarity-checked balanced logistic fitting for future experiments.

Frozen earlier sklearn fits are not altered. A numeric optimizer success flag
alone does not certify convergence; the final objective gradient is recorded.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp


def fit_precise_balanced_logistic(design,y,C=1.):
    X,y = np.asarray(design,float),np.asarray(y); n,p = X.shape; classes,own = np.unique(y,return_inverse=True); q = len(classes)
    if n == 0 or y.shape != (n,) or q < 2 or not np.isfinite(X).all() or not np.isfinite(C) or C <= 0:
        raise ValueError('finite design, multiple classes and positive C required')
    mass = np.array([1/(q*sum(own == c)) for c in own]); strength = 1/(C*n); r = 1 if q == 2 else q
    def value_gradient(vector):
        theta = vector[:r*p].reshape(r,p); bias = vector[r*p:]; scores = X@theta.T+bias
        if q == 2:
            eta = scores[:,0]; residual = mass*(expit(eta)-own)
            value = float(mass@(np.logaddexp(0,eta)-own*eta)); gradient = residual[None]@X; gbias = np.array([sum(residual)])
        else:
            lse = logsumexp(scores,axis=1); probability = np.exp(scores-lse[:,None])
            residual = mass[:,None]*(probability-np.eye(q)[own]); value = float(mass@(lse-scores[np.arange(n),own]))
            gradient = residual.T@X; gbias = residual.sum(axis=0)
        value += strength*float(np.sum(theta**2))/2; gradient += strength*theta
        return value,np.r_[gradient.ravel(),gbias]
    result = minimize(value_gradient,np.zeros(r*(p+1)),method='L-BFGS-B',jac=True,
        options={'maxiter':10000,'gtol':1e-10,'ftol':1e-15,'maxls':50,'maxcor':30})
    cost,gradient = value_gradient(result.x); norm = float(abs(gradient).max()); theta = result.x[:r*p].reshape(r,p); bias = result.x[r*p:]
    if q == 2: theta = np.vstack([np.zeros(p),theta[0]]); bias = np.r_[0.,bias[0]]
    else: theta = theta-theta[:1]; bias = bias-bias[:1]
    return {'coef':theta,'intercept':bias,'classes':classes,'optimizer_iterations':int(result.nit),
        'optimizer_converged':bool(norm <= 1e-8 and result.nit < 10000),
        'optimizer_success':bool(result.success),'optimizer_message':str(result.message),
        'gradient_inf':norm,'objective':float(cost),'objective_strength':strength,
        'scope':'numeric_stationarity_check_not_exact_convex_optimality_certificate'}
