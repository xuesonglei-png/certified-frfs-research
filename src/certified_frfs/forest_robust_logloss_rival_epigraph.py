"""Numerical diagnostic: equivalent rival-only nominal epigraph cuts.

Binary envelope is the exact box-adversarial logistic loss. Multiclass envelope
can be strictly larger than box-adversarial softmax loss. Solver diagnostics
are numerical, not rigorous global optimization or population certificates.
"""
import cvxpy as cp
import numpy as np
from scipy.special import logsumexp
from .triangular_classifier import triangular_memberships
from .orthogonal_forest_logistic import CONTRAST
from .forest_fuzzy_classifier import forest_box_margins


def orthogonal_design(X,features,edges):
    Z = triangular_memberships(X)@CONTRAST
    return np.column_stack([Z[:,features].reshape(len(X),-1)]+[
        (Z[:,u,:,None]*Z[:,v,None,:]).reshape(len(X),4) for u,v in edges])


def decode_forest(beta,bias,d,features,edges,classes):
    q = len(classes); W = np.zeros((q,d,3)); V = np.zeros((q,len(edges),3,3))
    W[:,features] = beta[:,:2*len(features)].reshape(q,len(features),2)@CONTRAST.T
    for e in range(len(edges)):
        V[:,e] = np.einsum('ki,qij,lj->qkl',CONTRAST,beta[:,2*len(features)+4*e:2*len(features)+4*(e+1)].reshape(q,2,2),CONTRAST)
    return {'coef':W.tolist(),'pair_coef':V.tolist(),'intercept':np.asarray(bias).tolist(),
        'features':list(features),'edges':[list(e) for e in edges],'classes':np.asarray(classes).tolist()}


def all_pair_box_witnesses(X,y,fit,*,eps=.01):
    X,y = np.asarray(X,float),np.asarray(y); classes = np.asarray(fit['classes']); q,d = len(classes),X.shape[1]
    own = np.searchsorted(classes,y)
    if np.any(own >= q) or not np.array_equal(classes[own],y): raise ValueError('unknown target class')
    W,b = np.asarray(fit['coef']),np.asarray(fit['intercept']); V = np.asarray(fit['pair_coef']).reshape(q,len(fit['edges']),3,3)
    margins = np.zeros((len(X),q)); raw = np.repeat(X[:,None,:],q,axis=1)
    for target in range(q):
        ids = np.flatnonzero(own == target)
        if not len(ids): continue
        for rival in range(q):
            if rival == target: continue
            pair = [target,rival]
            oracle = forest_box_margins(X[ids],np.zeros(len(ids),int),W[pair],b[pair],[0,1],fit['edges'],V[pair],eps=eps,witnesses=True)
            margins[ids,rival] = oracle['margins']; raw[ids,rival] = oracle['witness_raw']
    return {'pair_margins':margins,'pair_witness_raw':raw,'envelope_losses':logsumexp(-margins,axis=1)}


def fit_robust_forest_logloss(X,y,features,edges,*,C=1.,eps=.01,max_rounds=40,separation_tol=2e-7):
    X,y = np.asarray(X,float),np.asarray(y); features = list(features); classes = np.unique(y); q,d = len(classes),X.shape[1]; n = len(X)
    if q < 2 or not features or not np.isfinite(C) or C <= 0 or not np.isfinite(eps) or eps < 0: raise ValueError('invalid model settings')
    if not np.isfinite(separation_tol) or separation_tol <= 0 or int(max_rounds) != max_rounds or max_rounds < 1: raise ValueError('invalid separation settings')
    if len(set(features)) != len(features) or any(int(a) != a or not 0 <= a < d for a in features): raise ValueError('invalid feature support')
    raw_edges = [list(e) for e in edges]
    forest_box_margins(X[:1],y[:1],np.zeros((q,d,3)),np.zeros(q),classes,raw_edges,np.zeros((q,len(raw_edges),3,3)),eps=eps)
    edges = [e for e in raw_edges if all(a in features for a in e)]
    Z = orthogonal_design(X,features,edges); p = Z.shape[1]; own = np.searchsorted(classes,y)
    alpha = np.array([1/(q*sum(y == c)) for c in y])
    beta,bias,upper = cp.Variable((q,p)),cp.Variable(q),cp.Variable((n,q))
    score = Z@beta.T+cp.reshape(bias,(1,q),order='C')
    constraints = [beta[0] == 0,bias[0] == 0,upper[np.arange(n),own] == 0]
    initial_ids,initial_rivals = np.where(np.arange(q)[None,:] != own[:,None])
    constraints.append(upper[initial_ids,initial_rivals] >= score[initial_ids,initial_rivals]-score[initial_ids,own[initial_ids]])
    penalized = beta[1] if q == 2 else beta-cp.reshape(cp.sum(beta,axis=0)/q,(1,p),order='C')
    objective = cp.Minimize(alpha@cp.log_sum_exp(upper,axis=1)+cp.sum_squares(penalized)/(2*C*n))
    trace,cuts,seen = [],[],set(); result = None
    for step in range(max_rounds):
        problem = cp.Problem(objective,constraints)
        try:
            problem.solve(solver='CLARABEL',max_iter=250,time_limit=30.,tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9)
        except cp.error.SolverError as error:
            return {'usable':False,'status':'solver_error','message':str(error),'trace':trace,'cuts':cuts}
        if problem.status != 'optimal' or beta.value is None or bias.value is None:
            return {'usable':False,'status':str(problem.status),'trace':trace,'cuts':cuts}
        fit = decode_forest(beta.value,bias.value,d,features,edges,classes)
        oracle = all_pair_box_witnesses(X,y,fit,eps=eps); violation = -oracle['pair_margins']-upper.value
        violation[np.arange(n),own] = 0.; maximum = float(violation.max())
        values = beta.value[1] if q == 2 else beta.value-beta.value.mean(axis=0,keepdims=True)
        exact_objective = float(alpha@oracle['envelope_losses']+np.sum(values**2)/(2*C*n))
        trace.append({'round':step,'status':str(problem.status),'relaxation_objective':float(problem.value),
            'exact_envelope_objective':exact_objective,'max_separation_violation':maximum,'cuts':len(cuts)})
        result = {**fit,'usable':maximum <= separation_tol,'status':'separated' if maximum <= separation_tol else 'unseparated',
            'C':float(C),'pair_penalty':1.,'eps':float(eps),'separation_tol':float(separation_tol),
            'beta':beta.value.tolist(),'upper':upper.value.tolist(),'envelope_losses':oracle['envelope_losses'].tolist(),
            'envelope_pair_margins':oracle['pair_margins'].tolist(),'exact_envelope_objective':exact_objective,
            'relaxation_objective':float(problem.value),'trace':trace,'cuts':cuts,
            'class_balanced':True,'teacher_guards_enforced':False,'epigraph_layout':'rival_only_equivalent',
            'scope':'numerical_convex_label_robust_envelope_separation_not_teacher_retention_or_population_accuracy'}
        if result['usable']: return result
        ids,rivals = np.where(violation > separation_tol); added_ids,added_rivals,rows = [],[],[]
        for i,rival in zip(ids,rivals):
            point = oracle['pair_witness_raw'][i,rival]; identity = (int(i),int(rival),tuple(point.tolist()))
            if identity in seen: continue
            seen.add(identity); cuts.append({'object':int(i),'rival':int(rival),'raw':point.tolist()})
            added_ids.append(int(i)); added_rivals.append(int(rival)); rows.append(orthogonal_design(point[None],features,edges)[0])
        if not rows:
            result['usable'] = False; result['status'] = 'stalled_separation'; return result
        ii,rr = np.array(added_ids),np.array(added_rivals); rows = np.array(rows)
        constraints.append(upper[ii,rr] >= cp.sum(cp.multiply(beta[rr]-beta[own[ii]],rows),axis=1)+bias[rr]-bias[own[ii]])
    result['usable'] = False; result['status'] = 'separation_limit'; return result
