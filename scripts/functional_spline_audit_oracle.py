"""Independent Gauss-integrated functional bases and numerical refits."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import expit,logsumexp
try:
    from spline_forest_audit_oracle import membership,score
except ModuleNotFoundError:
    from scripts.spline_forest_audit_oracle import membership,score


def bases(knots):
    knots = np.asarray(knots,float); d,k = knots.shape
    output,grams,means = [],[],[]; nodes,weights = np.polynomial.legendre.leggauss(2)
    raw = np.zeros((k,k-1))
    for j in range(k-1):
        raw[:j+1,j] = 1/np.sqrt((j+1)*(j+2)); raw[j+1,j] = -(j+1)/np.sqrt((j+1)*(j+2))
    for a in range(d):
        H = np.zeros((k,k)); mean = np.zeros(k)
        for lo,hi in zip(knots[a,:-1],knots[a,1:]):
            locations = (lo+hi)/2+nodes*(hi-lo)/2; masses = weights*(hi-lo)/2
            Z = np.column_stack([np.interp(locations,knots[a],np.eye(k)[j]) for j in range(k)])
            H += Z.T@(masses[:,None]*Z); mean += masses@Z
        centered = raw-mean@raw
        L = np.linalg.cholesky(centered.T@H@centered); B = np.linalg.solve(L,centered.T).T
        np.testing.assert_allclose(mean@B,0,atol=1e-12)
        np.testing.assert_allclose(B.T@H@B,np.eye(k-1),atol=1e-10)
        output.append(B); grams.append(H); means.append(mean)
    return np.array(output),np.array(grams),np.array(means)


def refit_functional(A,y,fit,*,strict=True):
    fs,edges = fit['features'],fit['edges']; q,d,k = len(fit['classes']),A.shape[1],len(fit['knots'][0])
    assert fit['classes'] == np.unique(y).tolist() and fit['class_balanced'] and not fit['teacher_guards_enforced']
    assert fit['penalty_basis'] == 'raw_uniform_L2_centered_unary_and_pure_pair'
    W,V = np.array(fit['coef']),np.array(fit['pair_coef']).reshape(q,len(edges),k,k)
    B,H,mu = bases(fit['knots'])
    np.testing.assert_allclose(np.einsum('qdk,dk->qd',W,mu),0,atol=1e-9)
    np.testing.assert_allclose(W[:,[a for a in range(d) if a not in fs]],0,atol=1e-12)
    for e,(u,v) in enumerate(edges):
        np.testing.assert_allclose(V[:,e]@mu[v],0,atol=1e-9)
        np.testing.assert_allclose(np.einsum('k,qkl->ql',mu[u],V[:,e]),0,atol=1e-9)
    if not fs:
        np.testing.assert_allclose(score(A,fit),0,atol=1e-12)
        assert not edges and fit['optimizer_converged'] and fit['optimizer_iterations'] == 0
        return {'coef_error':0.,'bias_error':0.,'score_error':0.,'gradient_inf':0.,'objective_gap':0.}
    Z = np.einsum('ndk,dki->ndi',membership(A,fit['knots']),B); width = k-1
    blocks = [Z[:,fs].reshape(len(A),-1)]
    blocks += [(Z[:,u,:,None]*Z[:,v,None,:]).reshape(len(A),width**2)/np.sqrt(fit['pair_penalty']) for u,v in edges]
    design = np.column_stack(blocks)
    model = LogisticRegression(C=fit['C'],class_weight='balanced',max_iter=4000,tol=1e-8,random_state=0).fit(design,y)
    coef,bias = model.coef_,model.intercept_
    if q == 2: coef = np.vstack([np.zeros_like(coef[0]),coef[0]]); bias = np.r_[0.,bias[0]]
    dual = np.einsum('dkl,dli->dki',H,B)
    decoded = [np.einsum('qdk,dki->qdi',W[:,fs],dual[fs]).reshape(q,-1)]
    decoded += [(np.einsum('ki,qkl,lj->qij',dual[u],V[:,e],dual[v])*np.sqrt(fit['pair_penalty'])).reshape(q,width**2)
        for e,(u,v) in enumerate(edges)]
    decoded = np.column_stack(decoded); raw = score(A,fit); expected = design@coef.T+bias
    # Whitened floating point coordinates and a converged numeric optimizer
    # need not give byte-identical parameters. Record maxima and verify loss
    # and stationarity, as well as declared1e-4 numerical parameter tolerance.
    comparisons = {'coef':bool(np.allclose(decoded,coef-coef[:1],atol=1e-4,rtol=1e-4)),
        'bias':bool(np.allclose(fit['intercept'],bias-bias[:1],atol=1e-4,rtol=1e-4)),
        'score':bool(np.allclose(raw-raw[:,:1],expected-expected[:,:1],atol=2e-4,rtol=2e-4)),
        'convergence_flag':bool(fit['optimizer_converged'] == (model.n_iter_.max() < 4000))}
    own = np.argmax(y[:,None] == np.array(fit['classes'])[None],axis=1)
    balance = np.array([1/(q*sum(y == label)) for label in y]); strength = 1/(fit['C']*len(A))
    def objective(theta,intercept):
        logits = design@theta.T+intercept
        if q == 2:
            eta = logits[:,0]; residual = balance*(expit(eta)-own)
            empirical = float(balance@(np.logaddexp(0,eta)-own*eta))
            gradient = residual[None]@design+strength*theta; gbias = np.array([sum(residual)])
        else:
            probability = np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
            residual = balance[:,None]*(probability-np.eye(q)[own])
            empirical = float(balance@(logsumexp(logits,axis=1)-logits[np.arange(len(A)),own]))
            gradient = residual.T@design+strength*theta; gbias = residual.sum(axis=0)
        return empirical+strength*float(np.sum(theta**2))/2,max(float(abs(gradient).max()),float(abs(gbias).max()))
    theta = decoded[1:2] if q == 2 else decoded-decoded.mean(axis=0,keepdims=True)
    intercept = np.array(fit['intercept'])[1:2] if q == 2 else np.array(fit['intercept'])-np.mean(fit['intercept'])
    rt = coef[1:2] if q == 2 else coef-coef.mean(axis=0,keepdims=True)
    rb = bias[1:2] if q == 2 else bias-bias.mean()
    cost,gradient = objective(theta,intercept); reference_cost,_ = objective(rt,rb)
    diagnostics = {'coef_error':float(abs(decoded-(coef-coef[:1])).max()),
        'bias_error':float(abs(np.array(fit['intercept'])-(bias-bias[:1])).max()),
        'score_error':float(abs((raw-raw[:,:1])-(expected-expected[:,:1])).max()),
        'gradient_inf':gradient,'objective_gap':abs(cost-reference_cost)}
    if strict:
        assert all(comparisons.values()), (comparisons,diagnostics)
        assert gradient <= 2e-6 and abs(cost-reference_cost) <= 1e-8, diagnostics
        return diagnostics
    # Diagnosis only: never use this branch to issue a passed audit.
    return {**diagnostics,'comparisons':comparisons,
        'stationarity_passed':bool(gradient <= 2e-6),
        'objective_passed':bool(abs(cost-reference_cost) <= 1e-8),
        'refit_iterations':int(model.n_iter_.max()),
        'training_predictions_changed':int(np.sum(raw.argmax(axis=1) != expected.argmax(axis=1)))}
