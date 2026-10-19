"""Independent cell quadrature and softmax fit for the Sobolev classifier."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
from spline_forest_audit_oracle import membership,score

def basis(knots,alpha):
    outputs,Rs = [],[]
    for t in knots:
        t = np.array(t); k = len(t); H,K = np.zeros((k,k)),np.zeros((k,k)); mean = np.zeros(k)
        nodes,weights = np.polynomial.legendre.leggauss(2)
        for j,(lo,hi) in enumerate(zip(t[:-1],t[1:])):
            points = (lo+hi)/2+nodes*(hi-lo)/2; masses = weights*(hi-lo)/2
            values = np.column_stack([np.interp(points,t,np.eye(k)[a]) for a in range(k)])
            derivatives = np.zeros((2,k)); derivatives[:,j] = -1/(hi-lo); derivatives[:,j+1] = 1/(hi-lo)
            H += values.T@(masses[:,None]*values); K += derivatives.T@(masses[:,None]*derivatives); mean += masses@values
        raw = np.zeros((k,k-1))
        for j in range(k-1): raw[:j+1,j] = 1/np.sqrt((j+1)*(j+2)); raw[j+1,j] = -(j+1)/np.sqrt((j+1)*(j+2))
        raw -= mean@raw; R = H+alpha*K; L = np.linalg.cholesky(raw.T@R@raw); B = np.linalg.solve(L,raw.T).T
        np.testing.assert_allclose(mean@B,0,atol=1e-12); np.testing.assert_allclose(B.T@R@B,np.eye(k-1),atol=1e-12)
        outputs.append(B); Rs.append(R)
    return np.array(outputs),np.array(Rs)

def refit(A,y,fit):
    q,d,k = np.array(fit['coef']).shape; width = k-1; fs,edges = fit['features'],fit['edges']
    assert fit['classes'] == np.unique(y).tolist() and fit['class_balanced'] and not fit['teacher_guards_enforced']
    assert fit['penalty_basis'] == 'raw_centered_Sobolev_tensor_product_L2_plus_alpha_stiffness'
    B,R = basis(fit['knots'],fit['alpha']); Z = np.einsum('ndk,dki->ndi',membership(A,fit['knots']),B)
    blocks = [Z[:,fs].reshape(len(A),-1)]+[(Z[:,u,:,None]*Z[:,v,None,:]).reshape(len(A),width**2) for u,v in edges]
    design = np.column_stack(blocks); p = design.shape[1]; r = 1 if q == 2 else q
    own = np.argmax(y[:,None] == np.array(fit['classes'])[None],axis=1)
    mass = np.array([1/(q*sum(y == c)) for c in y]); strength = 1/(fit['C']*len(y))
    def objective(vector):
        theta,bias = vector[:r*p].reshape(r,p),vector[r*p:]; logits = design@theta.T+bias
        if q == 2:
            values = logits[:,0]; residual = mass*(expit(values)-own)
            cost = float(mass@(np.logaddexp(0,values)-own*values)); gcoef = residual[None]@design; gbias = np.array([np.sum(residual)])
        else:
            normalizer = logsumexp(logits,axis=1); probs = np.exp(logits-normalizer[:,None]); residual = mass[:,None]*(probs-np.eye(q)[own])
            cost = float(mass@(normalizer-logits[np.arange(len(y)),own])); gcoef = residual.T@design; gbias = residual.sum(axis=0)
        return cost+strength*float((theta**2).sum())/2,np.r_[(gcoef+strength*theta).ravel(),gbias]
    W,V = np.array(fit['coef']),np.array(fit['pair_coef']).reshape(q,len(edges),k,k); dual = np.einsum('dkl,dli->dki',R,B)
    theta = [np.einsum('qdk,dki->qdi',W[:,fs],dual[fs]).reshape(q,-1)]
    theta += [np.einsum('ki,qkl,lj->qij',dual[u],V[:,e],dual[v]).reshape(q,width**2) for e,(u,v) in enumerate(edges)]
    decoded = np.column_stack(theta); bias = np.array(fit['intercept'])
    owncoef,ownbias = (decoded[1:2],bias[1:2]) if q == 2 else (decoded-decoded.mean(axis=0),bias-bias.mean())
    cost,gradient = objective(np.r_[owncoef.ravel(),ownbias]); norm = float(abs(gradient).max())
    np.testing.assert_allclose(fit['objective'],cost,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(fit['gradient_inf'],norm,atol=1e-12,rtol=1e-5)
    assert fit['optimizer_converged'] == (norm <= 1e-8 and fit['optimizer_iterations'] < 10000)
    result = minimize(objective,np.zeros(r*(p+1)),jac=True,method='L-BFGS-B',
        options={'maxiter':10000,'gtol':1e-10,'ftol':1e-15,'maxls':50,'maxcor':30})
    reference_cost,_ = objective(result.x)
    coef,intercept = result.x[:r*p].reshape(r,p),result.x[r*p:]
    if q == 2: coef = np.vstack([np.zeros(p),coef[0]]); intercept = np.r_[0.,intercept[0]]
    else: coef -= coef[:1].copy(); intercept -= intercept[:1].copy()
    actual,expected = score(A,fit),design@coef.T+intercept
    np.testing.assert_allclose(decoded,coef,atol=1e-4,rtol=1e-4)
    np.testing.assert_allclose(bias,intercept,atol=1e-4,rtol=1e-4)
    np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-4)
    assert abs(cost-reference_cost) <= 1e-10
    return {'coef_error':float(abs(decoded-coef).max()),'bias_error':float(abs(bias-intercept).max()),
        'score_error':float(abs(actual-expected).max()),'gradient_inf':norm,'objective_gap':abs(cost-reference_cost)}
