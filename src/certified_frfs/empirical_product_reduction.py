"""Reference-marginal product ANOVA and feasible fixed-count refinement.

The declared product law is not the observed joint distribution. This module
accepts only training references; it provides no population accuracy claim.
"""
import numpy as np
from .spline_forest import spline_memberships,validate_edges
from .forest_anova_reduction import forest_energy_optimal_support


def marginal_product_anova(fit,X,weights):
    X = np.asarray(X,float); weights = np.asarray(weights,float)
    W = np.asarray(fit['coef'],float).copy(); bias = np.asarray(fit['intercept'],float).copy()
    q,d,k = W.shape; edges = validate_edges(fit['edges'],d)
    V = np.asarray(fit['pair_coef'],float).reshape(q,len(edges),k,k).copy()
    if X.ndim != 2 or X.shape[1] != d or weights.shape != (len(X),) or len(X) == 0:
        raise ValueError('reference dimensions do not agree')
    if not np.isfinite(X).all() or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(),1):
        raise ValueError('finite unit-mass reference required')
    Z = spline_memberships(X,fit['knots']); mu = np.einsum('n,ndk->dk',weights,Z)
    H = np.einsum('n,ndk,ndl->dkl',weights,Z,Z)
    for e,(u,v) in enumerate(edges):
        left = V[:,e]@mu[v]; right = np.einsum('k,qkl->ql',mu[u],V[:,e]); constant = left@mu[u]
        V[:,e] -= left[:,:,None]+right[:,None,:]-constant[:,None,None]
        W[:,u] += left; W[:,v] += right; bias -= constant
    means = np.einsum('qdk,dk->qd',W,mu); W -= means[:,:,None]; bias += means.sum(axis=1)
    CW,CV = W-W.mean(axis=0,keepdims=True),V-V.mean(axis=0,keepdims=True)
    unary = np.array([sum(CW[c,a]@H[a]@CW[c,a] for c in range(q))/q for a in range(d)])
    pair = np.array([sum(np.trace(CV[c,e].T@H[u]@CV[c,e]@H[v]) for c in range(q))/q for e,(u,v) in enumerate(edges)])
    if np.any(unary < -1e-8) or np.any(pair < -1e-8): raise RuntimeError('negative squared energy')
    centered = {**fit,'coef':W.tolist(),'pair_coef':V.tolist(),'intercept':bias.tolist(),
        'scope':'unchanged_teacher_ANOVA_under_training_marginal_product_law'}
    energy = {'unary_energy':np.maximum(unary,0).tolist(),'pair_energy':np.maximum(pair,0).tolist(),
        'edges':edges,'q':q,'total_energy':float(sum(unary)+sum(pair)),
        'scope':'training_marginal_product_reconstruction_not_training_joint_or_population_risk'}
    return centered,energy,{'marginal_means':mu.tolist(),'marginal_grams':H.tolist(),'weights':weights.tolist()}


def feasible_count_refinement(energies,initial,candidate,*,swap_pool=8,rounds=2):
    """Same count, unchanged certificates, strictly improving declared energy.

    Global DP solves the surrogate only. Feasible local refinement is bounded
    and heuristic, with an explicit feasible initial fallback. No completeness
    or globally FRS-constrained optimum claim.
    """
    from .forest_anova_reduction import omission_energy
    fs = sorted(initial); d = len(energies['unary_energy']); count = len(fs)
    if not isinstance(swap_pool,int) or swap_pool < 1 or not isinstance(rounds,int) or rounds < 0:
        raise ValueError('positive pool and nonnegative rounds required')
    if len(set(fs)) != count or any(int(a) != a or not 0 <= a < d for a in fs):
        raise ValueError('invalid support')
    original = candidate(fs)
    if not original['certificate']['feasible']: raise ValueError('feasible initial support required')
    oldloss = omission_energy(energies,fs); attempts = []; stages = []
    dp = forest_energy_optimal_support(energies['unary_energy'],energies['edges'],energies['pair_energy'],count)
    if dp['omission_energy'] < oldloss-1e-12 and dp['features'] != fs:
        geometry = candidate(dp['features']); accepted = geometry['certificate']['feasible']
        attempts.append({'source':'global_surrogate_DP','candidate':geometry,'loss':omission_energy(energies,dp['features']),'accepted':accepted})
        if accepted: fs = dp['features']; oldloss = omission_energy(energies,fs)
    for step in range(rounds):
        choices = []
        for remove in fs:
            for add in range(d):
                if add in fs: continue
                support = sorted([a for a in fs if a != remove]+[add]); loss = omission_energy(energies,support)
                if loss < oldloss-1e-12: choices.append((loss,tuple(support),remove,add))
        choices.sort(key=lambda row:(row[0],row[1])); stages.append({'round':step,'features':fs.copy(),
            'loss':oldloss,'ranked_improvements':len(choices),'tested_limit':swap_pool})
        accepted = None
        for loss,support,remove,add in choices[:swap_pool]:
            geometry = candidate(list(support)); ok = geometry['certificate']['feasible']
            attempts.append({'source':'swap','round':step,'remove':remove,'add':add,'candidate':geometry,'loss':loss,'accepted':ok})
            if ok: accepted = list(support); oldloss = loss; break
        if accepted is None: break
        fs = accepted
    final = candidate(fs)
    assert len(fs) == count and final['certificate']['feasible']
    assert omission_energy(energies,fs) <= omission_energy(energies,initial)+1e-10
    return {'features':fs,'candidate':final,'initial_features':sorted(initial),'dp':dp,'attempts':attempts,
        'stages':stages,'initial_loss':omission_energy(energies,initial),'final_loss':omission_energy(energies,fs),
        'swap_pool':swap_pool,'rounds':rounds,'scope':'same_count_feasible_surrogate_descent_not_accuracy_guarantee'}
