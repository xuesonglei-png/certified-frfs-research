"""Exact Hessian-vector Newton correction of a frozen logistic objective.

No labels beyond the supplied training reference. Earlier experimental fits
remain untouched; this is an explicit numerical refinement, not a new loss.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
from scipy.sparse.linalg import cg,LinearOperator

def refine_balanced_logistic(design,y,coef,intercept,C,*,maxiter=100):
    X,y = np.asarray(design,float),np.asarray(y); n,p = X.shape; classes,own = np.unique(y,return_inverse=True); q = len(classes)
    coef,intercept = np.asarray(coef,float),np.asarray(intercept,float)
    if coef.shape != (q,p) or intercept.shape != (q,) or q < 2 or not np.isfinite(C) or C <= 0:
        raise ValueError('compatible class scores and positive C required')
    r = 1 if q == 2 else q; strength = 1/(C*n); mass = np.array([1/(q*sum(own == c)) for c in own])
    theta,bias = (coef[1:2]-coef[:1],intercept[1:2]-intercept[:1]) if q == 2 else (coef-coef.mean(axis=0),intercept-intercept.mean())
    start = np.r_[theta.ravel(),bias]
    def pieces(vector):
        theta,bias = vector[:r*p].reshape(r,p),vector[r*p:]; logits = X@theta.T+bias
        if q == 2:
            probability = expit(logits[:,0]); residual = mass*(probability-own)
            value = float(mass@(np.logaddexp(0,logits[:,0])-own*logits[:,0])); gt = residual[None]@X; gb = np.array([sum(residual)])
        else:
            normalizer = logsumexp(logits,axis=1); probability = np.exp(logits-normalizer[:,None]); residual = mass[:,None]*(probability-np.eye(q)[own])
            value = float(mass@(normalizer-logits[np.arange(n),own])); gt = residual.T@X; gb = residual.sum(axis=0)
        return value+strength*float((theta**2).sum())/2,np.r_[(gt+strength*theta).ravel(),gb],probability
    def objective(vector):
        value,gradient,_ = pieces(vector); return value,gradient
    def hessian_vector(vector,direction):
        _,_,probability = pieces(vector); dt,db = direction[:r*p].reshape(r,p),direction[r*p:]; delta = X@dt.T+db
        if q == 2:
            residual = mass*probability*(1-probability)*delta[:,0]; ht = residual[None]@X; hb = np.array([sum(residual)])
        else:
            adjusted = delta-np.sum(probability*delta,axis=1,keepdims=True); residual = mass[:,None]*probability*adjusted
            ht = residual.T@X; hb = residual.sum(axis=0)
        return np.r_[(ht+strength*dt).ravel(),hb]
    result = minimize(objective,start,method='trust-ncg',jac=True,hessp=hessian_vector,
        options={'maxiter':maxiter,'gtol':1e-11})
    oldcost,oldgrad,_ = pieces(start); cost,gradient,_ = pieces(result.x)
    # Solver status alone is not acceptance. Keep the old point if the new
    # one is numerically worse in either cost or stationarity.
    accepted = cost <= oldcost+1e-12 and abs(gradient).max() < abs(oldgrad).max()
    vector = result.x if accepted else start; cost,gradient,_ = pieces(vector)
    polish_steps = 0
    # Trust-region termination can be limited by objective roundoff while a
    # useful Newton step remains. Solve the linearized stationarity equation
    # directly and independently check the resulting cost and gradient.
    for _ in range(3):
        if abs(gradient).max() <= 1e-10: break
        hessian = LinearOperator((len(vector),len(vector)),matvec=lambda delta:hessian_vector(vector,delta))
        delta,status = cg(hessian,-gradient,atol=1e-14,rtol=1e-6,maxiter=3000)
        if status != 0 or not np.isfinite(delta).all(): break
        proposed = vector+delta; nc,ng,_ = pieces(proposed)
        if nc > cost+1e-12 or abs(ng).max() >= abs(gradient).max(): break
        vector,cost,gradient = proposed,nc,ng; polish_steps += 1; accepted = True
    theta,bias = vector[:r*p].reshape(r,p),vector[r*p:]
    if q == 2: theta = np.vstack([np.zeros(p),theta[0]]); bias = np.r_[0.,bias[0]]
    else: theta = theta-theta[:1]; bias = bias-bias[:1]
    return {'coef':theta,'intercept':bias,'classes':classes,'objective':float(cost),'gradient_inf':float(abs(gradient).max()),
        'optimizer_converged':bool(abs(gradient).max() <= 1e-8 and result.nit < maxiter),
        'refinement_iterations':int(result.nit),'refinement_accepted':bool(accepted),'refinement_success':bool(result.success),
        'refinement_message':str(result.message),'initial_gradient_inf':float(abs(oldgrad).max()),
        'stationarity_polish_steps':polish_steps,
        'initial_objective':float(oldcost),'objective_strength':strength,
        'scope':'same_objective_training_only_numeric_refinement_not_new_regularizer_or_prediction_gain'}
