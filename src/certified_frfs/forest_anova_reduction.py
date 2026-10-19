"""Uniform raw-domain ANOVA energy and exact forest cardinality DP.

This is a task reconstruction surrogate for fixed raw-mean-centered unary
and pure-pair scores. It is not a globally optimal FRS-certified reduct or a
population prediction result. Dynamic programming uses ordinary float energy.
"""
import numpy as np
from .functional_spline_forest import hat_integral_gram
from .spline_forest import validate_knots,validate_edges


def canonical_raw_anova(fit):
    """Reparameterize arbitrary unary/pair hats without changing scores."""
    W = np.asarray(fit['coef'],float).copy(); b = np.asarray(fit['intercept'],float).copy()
    q,d,k = W.shape; knots = validate_knots(fit['knots'],d); edges = validate_edges(fit['edges'],d)
    V = np.asarray(fit['pair_coef'],float).reshape(q,len(edges),k,k).copy()
    mu = np.stack([hat_integral_gram(t)@np.ones(k) for t in knots])
    for e,(u,v) in enumerate(edges):
        left = V[:,e]@mu[v]; right = np.einsum('k,qkl->ql',mu[u],V[:,e])
        constant = left@mu[u]
        V[:,e] -= left[:,:,None]+right[:,None,:]-constant[:,None,None]
        W[:,u] += left; W[:,v] += right; b -= constant
    means = np.einsum('qdk,dk->qd',W,mu); W -= means[:,:,None]; b += means.sum(axis=1)
    result = {**fit,'coef':W.tolist(),'pair_coef':V.tolist(),'intercept':b.tolist(),
        'scope':'exact_fixed_model_raw_uniform_ANOVA_reparameterization_no_refit_no_prediction_change'}
    functional_score_energies(result)
    return result


def functional_score_energies(fit):
    W = np.asarray(fit['coef'],float); q,d,k = W.shape
    knots = validate_knots(fit['knots'],d); edges = validate_edges(fit['edges'],d)
    V = np.asarray(fit['pair_coef'],float).reshape(q,len(edges),k,k)
    if not np.isfinite(W).all() or not np.isfinite(V).all(): raise ValueError('finite scores required')
    H = [hat_integral_gram(t) for t in knots]; mean = np.stack([h@np.ones(k) for h in H])
    centeredW = W-W.mean(axis=0,keepdims=True); centeredV = V-V.mean(axis=0,keepdims=True)
    if not np.allclose(np.einsum('qdk,dk->qd',centeredW,mean),0,atol=1e-8,rtol=0):
        raise ValueError('class-centered unary scores must have zero raw mean')
    unary = np.array([sum(centeredW[c,a]@H[a]@centeredW[c,a] for c in range(q))/q for a in range(d)])
    pair = []
    for e,(u,v) in enumerate(edges):
        if not np.allclose(centeredV[:,e]@mean[v],0,atol=1e-8,rtol=0) or not np.allclose(np.einsum('k,qkl->ql',mean[u],centeredV[:,e]),0,atol=1e-8,rtol=0):
            raise ValueError('class-centered pairs must have zero integrated marginals')
        pair.append(sum(np.trace(centeredV[c,e].T@H[u]@centeredV[c,e]@H[v]) for c in range(q))/q)
    if np.any(unary < -1e-10) or np.any(np.array(pair) < -1e-10): raise RuntimeError('negative integrated squared norm')
    return {'unary_energy':np.maximum(unary,0).tolist(),'pair_energy':np.maximum(pair,0).tolist(),
        'edges':edges,'q':q,'total_energy':float(sum(unary)+sum(pair)),
        'scope':'class_centered_score_mean_squared_uniform_raw_cube_energy_not_true_distribution_or_test_accuracy'}


def projected_support_fit(fit,features):
    d = len(fit['coef'][0]); fs = sorted(features)
    if len(set(fs)) != len(fs) or any(int(a) != a or not 0 <= a < d for a in fs): raise ValueError('invalid raw support')
    # Validation also rejects non-ANOVA scores; no label/query argument exists.
    functional_score_energies(fit)
    W = np.asarray(fit['coef'],float).copy(); omitted = [a for a in range(d) if a not in fs]; W[:,omitted] = 0
    ids = [e for e,(u,v) in enumerate(fit['edges']) if u in fs and v in fs]
    V = np.asarray(fit['pair_coef'],float).reshape(len(fit['classes']),len(fit['edges']),W.shape[-1],W.shape[-1])[:,ids]
    return {**fit,'coef':W.tolist(),'pair_coef':V.tolist(),'edges':[fit['edges'][e] for e in ids],'features':fs,
        'scope':'fixed_teacher_uniform_conditional_expectation_support_projection_not_refitted_classifier'}


def omission_energy(energies,features):
    keep = set(features)
    return float(sum(v for a,v in enumerate(energies['unary_energy']) if a not in keep)
        +sum(v for (u,w),v in zip(energies['edges'],energies['pair_energy']) if u not in keep or w not in keep))


def forest_energy_optimal_support(unary_energy,edges,pair_energy,count):
    unary = np.asarray(unary_energy,float); pair = np.asarray(pair_energy,float); d = len(unary)
    edges = validate_edges(edges,d)
    if unary.ndim != 1 or d < 1 or pair.shape != (len(edges),) or not np.isfinite(unary).all() or not np.isfinite(pair).all() or np.any(unary < 0) or np.any(pair < 0):
        raise ValueError('nonnegative finite unary/pair energies required')
    if int(count) != count or not 0 <= count <= d: raise ValueError('valid cardinality required')
    count = int(count); neighbors = [[] for _ in range(d)]
    for e,(u,v) in enumerate(edges): neighbors[u].append((v,e)); neighbors[v].append((u,e))
    parents = [-2]*d; order = []; roots = []; edgeparent = [-1]*d
    for start in range(d):
        if parents[start] != -2: continue
        roots.append(start); parents[start] = -1; stack = [start]
        while stack:
            u = stack.pop(); order.append(u)
            for v,e in neighbors[u]:
                if v == parents[u]: continue
                parents[v] = u; edgeparent[v] = e; stack.append(v)
    def update(table,k,value,support):
        old = table.get(k)
        if old is None or value > old[0] or value == old[0] and support < old[1]: table[k] = (value,support)
    tables = {}
    for u in reversed(order):
        node = [{0:(0.,())},{1:(float(unary[u]),(u,))} if count >= 1 else {}]
        children = sorted(v for v,_ in neighbors[u] if parents[v] == u)
        for v in children:
            merged = [{},{}]
            for state in (0,1):
                for n,(value,support) in node[state].items():
                    for childstate in (0,1):
                        for m,(cv,cs) in tables[v][childstate].items():
                            if n+m > count: continue
                            gained = value+cv+(float(pair[edgeparent[v]]) if state and childstate else 0.)
                            update(merged[state],n+m,gained,tuple(sorted(support+cs)))
            node = merged
        tables[u] = node
    result = {0:(0.,())}
    for root in roots:
        merged = {}
        for n,(value,support) in result.items():
            for state in (0,1):
                for m,(rv,rs) in tables[root][state].items():
                    if n+m <= count: update(merged,n+m,value+rv,tuple(sorted(support+rs)))
        result = merged
    retained,support = result[count]; total = float(unary.sum()+pair.sum())
    return {'features':list(support),'retained_energy':retained,'omission_energy':max(0.,total-retained),
        'count':count,'scope':'exact_combinatorial_forest_DP_for_declared_float_surrogate_not_FRS_feasible_optimum_or_test_risk'}
