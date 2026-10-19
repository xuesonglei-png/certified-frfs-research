"""Training-only finite-grid metric learning under unchanged certificates.

Feasibility is checked separately at every grid point; no convexity or
monotonicity of the majority certificate is assumed. An empty feasible set
returns None, never an uncertified fallback.
"""
import numpy as np
from .core import _inputs
from .redistributed_loss import transfer_weights, redistributed_loss_bound
from .class_transport import fit_class_transport, class_transport_objective
from .softvote_metric import softvote_terms
from .majority_metric import MajorityTerms


def weight_certificate(X, y, features, w_full, w_reduced, *, eps=.01,
                       retention=.9, dependency_eta=.1, protected=None):
    X, y, fs, w = _inputs(X, y, features, w_full)
    if not 0 < retention <= 1 or not 0 <= dependency_eta <= 1:
        raise ValueError('invalid certificate constraints')
    if protected is None:
        protected = MajorityTerms(X, y, weights=w, eps=eps).bounds(
            range(X.shape[1]), allowed_enemies=2) > 1e-10
    protected = np.asarray(protected, bool)
    if protected.shape != y.shape:
        raise ValueError('invalid original protected mask')
    groups = [(y == q) & protected for q in np.unique(y)]
    required = [int(np.ceil(retention*g.sum()-1e-12)) for g in groups]
    bounds = MajorityTerms(X[:, fs], y, weights=w_reduced, eps=eps).bounds(
        range(len(fs)), allowed_enemies=2)
    counts = [int(((bounds > 1e-10) & g).sum()) for g in groups]
    dep = redistributed_loss_bound(X, y, fs, w, w_reduced, eps=eps)
    majority = all(c >= r for c, r in zip(counts, required))
    return {'feasible': majority and dep['upper_bound'] <= dependency_eta,
            'majority_feasible': majority, 'retained_counts': counts,
            'required': required, 'dependency_certificate': dep}


def training_neighbor_utility(X, y, features, weights, *, eps=.01, seed=983):
    """Fixed clean/shared-noise LOO BA; training selection, not validation."""
    fs = list(features); labels = np.unique(y); rng = np.random.default_rng(seed)
    values = []
    for Z in (X, np.clip(X+rng.uniform(-eps, eps, X.shape), 0, 1)):
        D = abs(Z[:, None, fs]-Z[None, :, fs])@weights
        np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, axis=1, kind='stable')[:, :min(5, len(y)-1)]
        votes = np.stack([(y[nn] == q).sum(axis=1) for q in labels], axis=1)
        pred = labels[votes.argmax(axis=1)]
        values.append(np.mean([(pred[y == q] == q).mean() for q in labels]))
    return float(np.mean(values))


def crossfit_interpolation_scores(X, y, fs, grid, *, family, alpha, reg, eps):
    """Nested reference-only selection, including independently fitted scaling."""
    from sklearn.model_selection import StratifiedKFold
    from .validated_pool import preprocess_fit
    from .softvote_metric import learn_softvote_weights
    labels, counts = np.unique(y, return_counts=True)
    if counts.min() < 2: raise ValueError('nested selector requires repeated classes')
    rows = []; scores = []
    for ii, jj in StratifiedKFold(min(3, int(counts.min())), shuffle=True, random_state=984).split(X, y):
        A, B = preprocess_fit(X[ii], X[jj]); yt, yv = y[ii], y[jj]
        w = np.ones(X.shape[1])/np.sqrt(X.shape[1]) if family == 'uniform' else learn_softvote_weights(A, yt)
        corr = transfer_weights(A, fs, w)
        learned, _ = fit_class_transport(A, yt, fs, w, alpha=alpha, reg=reg, eps=eps)
        rng = np.random.default_rng(985+len(rows)); Z = np.clip(A+rng.uniform(-eps, eps, A.shape), 0, 1); Q = np.clip(B+rng.uniform(-eps, eps, B.shape), 0, 1)
        fold_scores = []
        for t in grid:
            v = (1-t)*corr+t*learned; values = []
            for ref, query in [(A, B), (Z, Q)]:
                D = abs(query[:, None, fs]-ref[None, :, fs])@v
                nn = np.argsort(D, axis=1, kind='stable')[:, :min(5, len(yt))]
                labs = np.unique(yt)
                pred = labs[np.stack([(yt[nn] == q).sum(axis=1) for q in labs], axis=1).argmax(axis=1)]
                values.append(np.mean([(pred[yv == q] == q).mean() for q in np.unique(yv)]))
            fold_scores.append(float(np.mean(values)))
        scores.append(fold_scores); rows.append({'train': ii.tolist(), 'validation': jj.tolist(), 'scores': fold_scores})
    return np.mean(scores, axis=0), rows


def fit_feasible_transport(X, y, features, w_full, *, certified=True,
                           grid=(0., .25, .5, .75, 1.), alpha=.1,
                           reg=.001, eps=.01, retention=.9, dependency_eta=.1,
                           selection='smooth', family='uniform'):
    X, y, fs, w = _inputs(X, y, features, w_full)
    grid = np.asarray(grid, float)
    if selection not in ('smooth', 'loo', 'crossfit'): raise ValueError('unknown training selector')
    if family not in ('uniform', 'softvote'): raise ValueError('unknown metric family')
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or np.any((grid < 0) | (grid > 1)):
        raise ValueError('nonempty interpolation grid in [0,1] required')
    correlation = transfer_weights(X, fs, w)
    learned, fit = fit_class_transport(X, y, fs, w, alpha=alpha, reg=reg, eps=eps)
    if 'distance_fit' not in fit:
        raise ValueError('transport requires at least two objects and positive mass')
    pairs = np.asarray(fit['distance_fit']['pair_indices'], int)
    components = abs(X[pairs[:, 0]]-X[pairs[:, 1]])
    # The same objective as the class-aware optimizer, including its anchor.
    from .metric_transport import fit_transport_weights
    anchor, _ = fit_transport_weights(X, fs, w, reg=reg)
    terms = softvote_terms(X[:, fs], y, eps=eps)
    protected = (MajorityTerms(X, y, weights=w, eps=eps).bounds(
        range(X.shape[1]), allowed_enemies=2) > 1e-10) if certified else None
    scores, splits = crossfit_interpolation_scores(X, y, fs, grid, family=family,
        alpha=alpha, reg=reg, eps=eps) if selection == 'crossfit' else (None, [])
    rows = []
    for index, t in enumerate(grid):
        v = (1-t)*correlation+t*learned
        objective = class_transport_objective(v, components[:, fs], components@w,
            anchor, terms, alpha=alpha, reg=reg)[0]
        cert = weight_certificate(X, y, fs, w, v, eps=eps, retention=retention,
            dependency_eta=dependency_eta, protected=protected) if certified else {'feasible': True}
        rows.append({'interpolation': float(t), 'weights': v.tolist(),
                     'selection_utility': None if scores is None else float(scores[index]),
                     'training_utility': training_neighbor_utility(X, y, fs, v, eps=eps),
                     'training_objective': objective, **cert})
    feasible = [r for r in rows if r['feasible']]
    key = (lambda r: (-r['selection_utility'], r['interpolation'])) if selection == 'crossfit' else (lambda r: (r['training_objective'], r['interpolation'])) if selection == 'smooth' else (
        lambda r: (-r['training_utility'], r['training_objective'], r['interpolation']))
    selected = min(feasible, key=key) if feasible else None
    return (None if selected is None else np.asarray(selected['weights'])), {
        'status': 'no_feasible_grid_point' if selected is None else 'selected',
        'selected': selected, 'candidates': rows, 'endpoint_fit': fit,
        'certified': certified, 'selection': selection, 'selection_splits': splits, 'eps': eps, 'retention': retention,
        'dependency_eta': dependency_eta,
        'scope': 'finite_training_grid_selection_not_global_optimum_or_test_risk'}


def fit_repaired_transport(X, y, features, w_full, **kwargs):
    """Restore omitted coordinates in descending original weight order.

    Restoration is fixed using reference training weights, never held-out
    accuracy. Full-metric endpoint is explicit if all learned grids fail.
    """
    X, y, fs, w = _inputs(X, y, features, w_full)
    current = fs.tolist(); trace = []
    order = sorted(set(range(X.shape[1]))-set(current), key=lambda a: (-w[a], a))
    while True:
        v, record = fit_feasible_transport(X, y, current, w, **kwargs)
        trace.append({'features': current.copy(), 'fit': record})
        if v is not None:
            return current, v, {'trace': trace, 'restored_count': len(current)-len(fs), 'full_fallback': False}
        if order:
            current = sorted(current+[order.pop(0)])
        else:
            cert = weight_certificate(X, y, current, w, w, eps=kwargs.get('eps', .01),
                retention=kwargs.get('retention', .9), dependency_eta=kwargs.get('dependency_eta', .1))
            if not cert['feasible']: raise RuntimeError('original full metric failed its own certificate')
            return current, w.copy(), {'trace': trace, 'restored_count': len(current)-len(fs),
                                      'full_fallback': True, 'fallback_certificate': cert}
