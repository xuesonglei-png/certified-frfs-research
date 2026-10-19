"""Exact coordinate sensitivities of triangular pairwise class contrasts.

They define a unit-mass task-informed reference relation. This changes the
reference metric explicitly and does not certify a newly refitted classifier.
"""
import numpy as np
from .forest_fuzzy_classifier import forest_scores


def forest_contrast_lipschitz(fit):
    W = np.asarray(fit['coef'],float); q,d,_ = W.shape
    edges = fit['edges']; V = np.asarray(fit['pair_coef'],float).reshape(q,len(edges),3,3)
    forest_scores(np.zeros((1,d)),W,fit['intercept'],edges,V)
    pairs = [(c,r) for c in range(q) for r in range(c+1,q)]
    unary = np.stack([W[c]-W[r] for c,r in pairs])
    interactions = np.stack([V[c]-V[r] for c,r in pairs])
    lower = np.zeros((len(pairs),d,2)); upper = lower.copy()
    for a in range(d):
        for branch in range(2):
            delta = 2*(unary[:,a,branch+1]-unary[:,a,branch]); lo,hi = delta.copy(),delta.copy()
            for e,(u,v) in enumerate(edges):
                if a == u: contribution = 2*(interactions[:,e,branch+1,:]-interactions[:,e,branch,:])
                elif a == v: contribution = 2*(interactions[:,e,:,branch+1]-interactions[:,e,:,branch])
                else: continue
                lo += contribution.min(axis=1); hi += contribution.max(axis=1)
            lower[:,a,branch],upper[:,a,branch] = lo,hi
    contrast_constants = np.maximum(abs(lower),abs(upper)).max(axis=2)
    constants = contrast_constants.max(axis=0); total = float(constants.sum())
    weights = constants/total if total > 0 else np.ones(d)/d
    return {'coordinate_constants':constants.tolist(),'contrast_constants':contrast_constants.tolist(),
        'class_index_pairs':[list(p) for p in pairs],'slope_lower':lower.tolist(),'slope_upper':upper.tolist(),
        'total_constant':total,'unit_mass_weights':weights.tolist(),'uniform_fallback':total == 0,
        'scope':'exact_full_cube_coordinate_contrast_sensitivity_fixed_model_unit_mass_reference_not_refit_or_population_guarantee'}
