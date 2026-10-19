"""Cross-fitted recipe selection for the certified sparse metric pipeline."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .validated_pool import preprocess_fit, validation_scores
from .feasible_transport import weight_certificate
from .feasible_query_descent import fit_feasible_query
from .trajectory_certificate import certify_capped_trajectory
from .capped_sparse import sparse_trajectory_extension


def fit_crossfit_capped_sparse(X, y, *, seed=14100, folds=3):
    """Choose full/3-step/10-step by disjoint crossfit query scores.

    Each recipe is fitted only on a fold's training reference and scored on
    that fold's query. After the recipe is fixed, it is fitted and certified
    once on the complete input. No external labels enter selection.
    """
    X, y = np.asarray(X, float), np.asarray(y)
    labels, counts = np.unique(y, return_counts=True)
    repeated = np.flatnonzero(np.isin(y, labels[counts > 1]))
    if not len(repeated):
        raise ValueError('insufficient repeated class support for crossfit')
    folds = min(folds, int(counts[counts > 1].min()))
    rows = {r: [] for r in ('full', '3', '10')}
    def recipe_fit(table, labels, recipe):
        A, _ = preprocess_fit(table, table); d = table.shape[1]; full = np.ones(d)/np.sqrt(d)
        if recipe == 'full':
            return {'features': list(range(d)), 'weights': full.tolist()}
        trajectory = fit_feasible_query(table, labels, certified=False, weight_cap_ratio=2, steps=int(recipe))
        certified = certify_capped_trajectory(table, labels, trajectory)
        fit = sparse_trajectory_extension(table, labels, certified)
        return fit['compact_selected']
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for inner, (ti, qi) in enumerate(splitter.split(repeated, y[repeated])):
        train_ids, query_ids = repeated[ti], repeated[qi]
        # Singleton labels remain training-only in each fold.
        single = np.flatnonzero(np.isin(y, labels[counts == 1]))
        fold_table, fold_labels = X[np.r_[train_ids, single]], y[np.r_[train_ids, single]]
        A, B = preprocess_fit(fold_table, X[query_ids])
        for recipe in ('full', '3', '10'):
            row = recipe_fit(fold_table, fold_labels, recipe)
            fs, w = row['features'], np.asarray(row['weights'])
            rows[recipe].append({'inner': inner, 'features': fs, 'weights': row['weights'],
                'train_indices': np.r_[train_ids, single].tolist(), 'query_indices': query_ids.tolist(),
                'metrics': validation_scores(A, fold_labels, B, y[query_ids], fs, w, seed+31+inner, joint=True)})
    summary = []
    for recipe, values in rows.items():
        summary.append({'recipe': recipe, 'utility': float(np.mean([(v['metrics']['clean']+v['metrics']['noise01'])/2 for v in values])),
            'mean_size': float(np.mean([len(v['features']) for v in values])), 'folds': values})
    chosen = max(range(len(summary)), key=lambda i: (summary[i]['utility'], -summary[i]['mean_size'], -i))
    recipe = summary[chosen]['recipe']
    if recipe == 'full':
        A, _ = preprocess_fit(X, X); full = np.ones(X.shape[1])/np.sqrt(X.shape[1]); fs = list(range(X.shape[1]))
        selected = {'features': fs, 'weights': full.tolist(), 'certificate': weight_certificate(A, y, fs, full, full)}
        refit = None
    else:
        trajectory = fit_feasible_query(X, y, certified=False, weight_cap_ratio=2, steps=int(recipe))
        certified = certify_capped_trajectory(X, y, trajectory); final = sparse_trajectory_extension(X, y, certified)
        selected, refit = final['compact_selected'], {'trajectory': trajectory, 'certified_trajectory': certified, 'fit': final}
    if not selected['certificate']['feasible']: raise RuntimeError('crossfit refit failed certificate')
    return {'selected': selected, 'chosen_recipe': recipe, 'summary': summary, 'seed': seed, 'folds': folds,
        'scope': 'crossfit_recipe_selection_full_refit_fresh_certificate_not_test_risk', 'refit': refit}
