"""Reference-only holdout selection and freshly certified metric refit."""
import numpy as np
from sklearn.model_selection import train_test_split
from .feasible_query_descent import fit_feasible_query
from .trajectory_certificate import certify_capped_trajectory
from .capped_sparse import sparse_trajectory_extension
from .validated_pool import preprocess_fit, validation_scores


def truncate_trajectory(trajectory, steps):
    trace = [node for node in trajectory['trace'] if node['iteration'] <= steps]
    return {**trajectory, 'trace': trace, 'steps': steps, 'oracles': trajectory['oracles'][:steps],
        'weights': trace[-1]['weights'], 'features': np.flatnonzero(np.array(trace[-1]['weights']) > 0).tolist()}


def fit_validated_capped_sparse(X, y, *, seed=14100):
    """Select one of three frozen recipes using a disjoint training holdout.

    Singleton classes remain discovery-only. The validation winner determines
    a recipe, not transferable features or a certificate. Refit selected recipe
    from scratch on the entire reference set; full has its original certificate.
    """
    X, y = np.asarray(X, float), np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    repeated = np.flatnonzero(np.isin(y, classes[counts > 1]))
    single = np.flatnonzero(np.isin(y, classes[counts == 1]))
    if not len(repeated): raise ValueError('repeated classes required for recipe validation')
    n_classes = len(classes[counts > 1]); n_validation = max(n_classes, int(np.ceil(.3*len(repeated))))
    n_validation = min(n_validation, len(repeated)-n_classes)
    discovery, validation = train_test_split(repeated, test_size=n_validation, stratify=y[repeated], random_state=seed)
    discovery = np.r_[discovery, single]
    A, B = preprocess_fit(X[discovery], X[validation]); d = X.shape[1]
    uniform = np.ones(d)/np.sqrt(d); full_features = list(range(d))
    trace = [{'recipe': 'full', 'features': full_features, 'weights': uniform.tolist(),
        'validation': validation_scores(A, y[discovery], B, y[validation], full_features, uniform, seed+1, joint=True)}]
    trajectory = fit_feasible_query(X[discovery], y[discovery], certified=False, weight_cap_ratio=2, steps=10)
    fits = {}
    for steps in (3, 10):
        training = truncate_trajectory(trajectory, steps)
        cert = certify_capped_trajectory(X[discovery], y[discovery], training)
        fit = sparse_trajectory_extension(X[discovery], y[discovery], cert)
        fits[str(steps)] = {'trajectory': training, 'certified_trajectory': cert, 'fit': fit}
        c = fit['compact_selected']
        trace.append({'recipe': str(steps), 'features': c['features'], 'weights': c['weights'],
            'validation': validation_scores(A, y[discovery], B, y[validation], c['features'], np.array(c['weights']), seed+1, joint=True)})
    for row in trace: row['utility'] = (row['validation']['clean']+row['validation']['noise01'])/2
    chosen = max(range(len(trace)), key=lambda i: (trace[i]['utility'], -len(trace[i]['features']), -i))
    recipe = trace[chosen]['recipe']
    if recipe == 'full':
        Z, _ = preprocess_fit(X, X)
        from .feasible_transport import weight_certificate
        selected = {'features': full_features, 'weights': uniform.tolist(),
            'certificate': weight_certificate(Z, y, full_features, uniform, uniform)}
        refit = None
    else:
        final_trajectory = fit_feasible_query(X, y, certified=False, weight_cap_ratio=2, steps=int(recipe))
        certified = certify_capped_trajectory(X, y, final_trajectory)
        final_fit = sparse_trajectory_extension(X, y, certified)
        selected = final_fit['compact_selected']
        refit = {'trajectory': final_trajectory, 'certified_trajectory': certified, 'fit': final_fit}
    if not selected['certificate']['feasible']: raise RuntimeError('final refit did not certify')
    return {'selected': selected, 'chosen_recipe': recipe, 'trace': trace, 'discovery_indices': discovery.tolist(),
        'validation_indices': validation.tolist(), 'training_fits': fits, 'refit': refit, 'seed': seed,
        'scope': 'reference_holdout_recipe_selection_fresh_refit_certificate_not_test_risk'}
