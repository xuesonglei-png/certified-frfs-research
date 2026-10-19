"""Independent interpolation, classifier refit, leaf elimination and LP."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.special import expit,logsumexp
from sklearn.linear_model import LogisticRegression


def membership(X,knots):
    X,knots = np.asarray(X,float),np.asarray(knots,float)
    n,d = X.shape; k = knots.shape[1]; output = np.zeros((n,d,k))
    for a in range(d):
        for j in range(k):
            values = np.zeros(k); values[j] = 1.
            output[:,a,j] = np.interp(X[:,a],knots[a],values)
    return output


def score(X,fit):
    Z = membership(X,fit['knots']); W,b = np.array(fit['coef']),np.array(fit['intercept'])
    q,k = len(b),Z.shape[-1]; V = np.array(fit['pair_coef']).reshape(q,len(fit['edges']),k,k)
    values = np.einsum('ndk,qdk->nq',Z,W)+b
    for e,(u,v) in enumerate(fit['edges']): values += np.einsum('nk,qkl,nl->nq',Z[:,u],V[:,e],Z[:,v])
    return values


def refit(A,y,fit):
    fs,edges = fit['features'],fit['edges']; q,d,k = len(fit['classes']),A.shape[1],len(fit['knots'][0])
    W = np.array(fit['coef']); V = np.array(fit['pair_coef']).reshape(q,len(edges),k,k)
    assert fit['classes'] == np.unique(y).tolist() and fit['class_balanced'] and fit['teacher_guards_enforced'] is False
    np.testing.assert_allclose(W[:,[a for a in range(d) if a not in fs]],0,atol=1e-12)
    np.testing.assert_allclose(W.sum(axis=-1),0,atol=1e-9)
    np.testing.assert_allclose(V.sum(axis=-1),0,atol=1e-9); np.testing.assert_allclose(V.sum(axis=-2),0,atol=1e-9)
    if not fs:
        np.testing.assert_allclose(score(A,fit),0,atol=1e-12)
        assert not edges and fit['optimizer_converged'] and fit['optimizer_iterations'] == 0
        return {'coefficient_error':0.,'intercept_error':0.,'score_error':0.,'gradient_inf':0.,'objective_gap':0.}
    # Construct the contrast basis by its defining formula, not scipy.helmert.
    basis = np.zeros((k,k-1))
    for j in range(k-1):
        basis[:j+1,j] = 1/np.sqrt((j+1)*(j+2)); basis[j+1,j] = -(j+1)/np.sqrt((j+1)*(j+2))
    Z = membership(A,fit['knots'])@basis; width = k-1
    blocks = [Z[:,fs].reshape(len(A),-1)]
    blocks += [(Z[:,u,:,None]*Z[:,v,None,:]).reshape(len(A),width**2)/np.sqrt(fit['pair_penalty']) for u,v in edges]
    design = np.column_stack(blocks)
    model = LogisticRegression(C=fit['C'],class_weight='balanced',max_iter=4000,tol=1e-8,random_state=0).fit(design,y)
    coef,bias = model.coef_,model.intercept_
    if q == 2: coef = np.vstack([np.zeros_like(coef[0]),coef[0]]); bias = np.r_[0.,bias[0]]
    decoded = [(W[:,fs]@basis).reshape(q,-1)]
    decoded += [(np.einsum('ki,qkl,lj->qij',basis,V[:,e],basis)*np.sqrt(fit['pair_penalty'])).reshape(q,width**2) for e in range(len(edges))]
    decoded = np.column_stack(decoded)
    # Independent interpolation/basis arithmetic can change LBFGS stopping
    # at sub-micro precision. Validate stationarity/objective as well as a
    # declared numerical refit tolerance; do not silently hide the discrepancy.
    np.testing.assert_allclose(decoded,coef-coef[:1],atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(fit['intercept'],bias-bias[:1],atol=1e-5,rtol=1e-5)
    raw = score(A,fit); expected = design@coef.T+bias
    np.testing.assert_allclose(raw-raw[:,:1],expected-expected[:,:1],atol=1e-5,rtol=1e-5)
    assert fit['optimizer_converged'] == (model.n_iter_.max() < 4000)
    own = np.argmax(y[:,None] == np.array(fit['classes'])[None],axis=1)
    balance = np.array([1/(q*sum(y == label)) for label in y]); strength = 1/(fit['C']*len(A))
    def objective_and_gradient(theta,intercept):
        logits = design@theta.T+intercept
        if q == 2:
            eta = logits[:,0]; residual = balance*(expit(eta)-own)
            loss = float(balance@(np.logaddexp(0,eta)-own*eta))
            gradient = residual[None]@design+strength*theta; gbias = np.array([sum(residual)])
        else:
            probability = np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
            residual = balance[:,None]*(probability-np.eye(q)[own])
            loss = float(balance@(logsumexp(logits,axis=1)-logits[np.arange(len(A)),own]))
            gradient = residual.T@design+strength*theta; gbias = residual.sum(axis=0)
        return loss+strength*float(np.sum(theta**2))/2,max(float(abs(gradient).max()),float(abs(gbias).max()))
    theta = decoded[1:2] if q == 2 else decoded-decoded.mean(axis=0,keepdims=True)
    intercept = np.array(fit['intercept'])[1:2] if q == 2 else np.array(fit['intercept'])-np.mean(fit['intercept'])
    reference_theta = coef[1:2] if q == 2 else coef-coef.mean(axis=0,keepdims=True)
    reference_bias = bias[1:2] if q == 2 else bias-bias.mean()
    cost,gradient = objective_and_gradient(theta,intercept)
    reference_cost,_ = objective_and_gradient(reference_theta,reference_bias)
    assert gradient <= 2e-6 and abs(cost-reference_cost) <= 1e-9
    return {'coefficient_error':float(abs(decoded-(coef-coef[:1])).max()),
        'intercept_error':float(abs(np.array(fit['intercept'])-(bias-bias[:1])).max()),
        'score_error':float(abs((raw-raw[:,:1])-(expected-expected[:,:1])).max()),
        'gradient_inf':gradient,'objective_gap':abs(cost-reference_cost)}


def box_potentials(X,y,fit,eps):
    knots = np.array(fit['knots']); q,d,k = len(fit['classes']),X.shape[1],knots.shape[1]
    own = np.argmax(y[:,None] == np.array(fit['classes'])[None],axis=1)
    lo,hi = np.maximum(0,X-eps),np.minimum(1,X+eps)
    points = np.concatenate([lo[:,:,None],hi[:,:,None],np.clip(knots[None,:,1:-1],lo[:,:,None],hi[:,:,None])],axis=-1)
    Z = np.empty((len(X),d,k,k))
    for s in range(k): Z[:,:,s] = membership(points[:,:,s],knots)
    W,V,b = np.array(fit['coef']),np.array(fit['pair_coef']).reshape(q,len(fit['edges']),k,k),np.array(fit['intercept'])
    unary = np.einsum('nqdk,ndsk->nqds',W[own,None]-W[None],Z)
    energies = [np.einsum('nqkl,nsk,ntl->nqst',V[own,e][:,None]-V[None,:,e],Z[:,u],Z[:,v]) for e,(u,v) in enumerate(fit['edges'])]
    return unary,energies,b[own,None]-b[None],own,points,Z


def leaf_margins(X,y,fit,eps):
    unary,energies,bias,own,points,Z = box_potentials(X,y,fit,eps); d = X.shape[1]
    neighbors = {a:set() for a in range(d)}; edges = {}
    for e,(u,v) in enumerate(fit['edges']): neighbors[u].add(v); neighbors[v].add(u); edges[(u,v)] = energies[e]
    cost = {a:unary[:,:,a].copy() for a in range(d)}; active = set(range(d)); value = bias.copy()
    while active:
        leaves = sorted(a for a in active if len(neighbors[a]) <= 1); assert leaves
        for a in leaves:
            if neighbors[a]:
                other = next(iter(neighbors[a])); key = tuple(sorted((a,other)))
                pair = edges[key] if key[0] == a else edges[key].swapaxes(-1,-2)
                cost[other] += (cost[a][:,:,:,None]+pair).min(axis=-2)
                neighbors[other].remove(a); neighbors[a].clear()
            else: value += cost[a].min(axis=-1)
            active.remove(a)
    value[np.arange(len(X)),own] = np.inf
    return value,points,Z


def lp_margins_first(X,y,fit,eps):
    unary,energies,bias,own,points,Z = box_potentials(X[:1],y[:1],fit,eps)
    d,k = X.shape[1],len(fit['knots'][0]); edges = fit['edges']; count = d*k+len(edges)*k*k
    rows,cols,values,rhs = [],[],[],[]
    def add(entries,value):
        row = len(rhs); rhs.append(value)
        for col,coef in entries: rows.append(row); cols.append(col); values.append(coef)
    for a in range(d): add([(a*k+j,1.) for j in range(k)],1.)
    for e,(u,v) in enumerate(edges):
        start = d*k+e*k*k
        for j in range(k):
            add([(start+j*k+s,1.) for s in range(k)]+[(u*k+j,-1.)],0.)
            add([(start+s*k+j,1.) for s in range(k)]+[(v*k+j,-1.)],0.)
    matrix = coo_matrix((values,(rows,cols)),shape=(len(rhs),count)).tocsr(); result = {}
    for rival in range(len(fit['classes'])):
        if rival == own[0]: continue
        objective = np.r_[unary[0,rival].ravel(),*[energy[0,rival].ravel() for energy in energies]]
        solved = linprog(objective,A_eq=matrix,b_eq=np.array(rhs),bounds=(0,None),method='highs',
            options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'ipm_optimality_tolerance':1e-10})
        assert solved.success
        result[rival] = float(solved.fun+bias[0,rival])
    return result
