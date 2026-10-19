"""Stratified bootstrap logit averaging on ONE fixed forest.

The merged score remains a forest score. Averaging probability predictions or
mixing different forest graphs would not have this representation guarantee.
"""
import numpy as np
from .orthogonal_forest_logistic import fit_orthogonal_forest


def stratified_bootstrap_indices(y, *, members=12, seed=20261027):
    y = np.asarray(y); rng = np.random.default_rng(seed)
    if int(members) != members or members < 1: raise ValueError('positive integer members required')
    groups = [np.flatnonzero(y == c) for c in np.unique(y)]
    return [np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in groups]).tolist() for _ in range(members)]


def average_fixed_forest(fits):
    if not fits: raise ValueError('nonempty fixed-forest ensemble required')
    first = fits[0]
    for fit in fits:
        for key in ('features','edges','classes','C','pair_penalty'):
            if fit[key] != first[key]: raise ValueError('all members must share forest and settings')
    out = {k:first[k] for k in ('features','edges','classes','C','pair_penalty')}
    for key in ('coef','pair_coef','intercept'):
        out[key] = np.mean([np.array(fit[key]) for fit in fits],axis=0).tolist()
    out.update({'class_balanced':True,'teacher_guards_enforced':False,
        'optimizer_converged':all(fit['optimizer_converged'] for fit in fits),
        'scope':'mean_logits_fixed_forest_exact_representation_no_population_or_margin_retention_claim'})
    return out


def fit_bagged_fixed_forest(X,y,features,edges,*,C=1.,pair_penalty=1.,members=12,seed=20261027):
    X,y = np.asarray(X,float),np.asarray(y)
    indices = stratified_bootstrap_indices(y,members=members,seed=seed)
    fits = [fit_orthogonal_forest(X[a],y[a],features,edges,C=C,pair_penalty=pair_penalty) for a in indices]
    return {'fit':average_fixed_forest(fits),'members':fits,'bootstrap_indices':indices,'seed':seed,
        'scope':'bootstrap_only_current_training_table_fixed_graph_mean_scores_not_probability_averaging'}
