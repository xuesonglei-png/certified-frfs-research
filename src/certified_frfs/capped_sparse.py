"""Cardinality and concentration controlled training metric candidates."""
import numpy as np
from .query_probability import QueryProbabilityObjective
from .validated_pool import preprocess_fit
from .majority_metric import MajorityTerms
from .feasible_transport import weight_certificate


def capped_sparse_projection(z, k, cap):
    """Euclidean projection onto nonnegative unit mass, cap, and support <=k.

    By coordinate exchange, a projection can use the k largest coordinates.
    The remaining capped-simplex projection has a single scalar multiplier.
    """
    z = np.asarray(z, float)
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
        raise ValueError('finite nonempty vector required')
    if int(k) != k or not 1 <= k <= len(z) or not np.isfinite(cap) or cap <= 0 or k*cap < 1-1e-14:
        raise ValueError('infeasible support or cap')
    ids = np.argsort(-z, kind='stable')[:k]; values = z[ids]
    low, high = float(values.min()-cap), float(values.max())
    for _ in range(80):
        threshold = (low+high)/2
        if np.clip(values-threshold, 0, cap).sum() > 1:
            low = threshold
        else:
            high = threshold
    out = np.zeros(len(z)); out[ids] = np.clip(values-(low+high)/2, 0, cap)
    if abs(out.sum()-1) > 1e-10:
        raise RuntimeError('projection lost unit mass')
    return out


def optimize_capped_sparse(objective, initial, k, *, cap, maxiter=30):
    w = capped_sparse_projection(initial, k, cap); value, gradient = objective(w)
    trace = [{'iteration': 0, 'objective': value, 'weights': w.tolist()}]
    step = .05; status = 'maxiter'
    for iteration in range(1, maxiter+1):
        accepted = False
        for _ in range(20):
            candidate = capped_sparse_projection(w-step*gradient, k, cap)
            new_value, new_gradient = objective(candidate)
            if new_value <= value-1e-4*np.sum((candidate-w)**2)/step+1e-12:
                accepted = True
                break
            step *= .5
        if not accepted:
            status = 'line_search_stopped'
            break
        delta = np.linalg.norm(candidate-w); w, value, gradient = candidate, new_value, new_gradient
        trace.append({'iteration': iteration, 'objective': value, 'weights': w.tolist()})
        if delta < 1e-7:
            status = 'stationary_projection'
            break
        step = min(step*1.2, .2)
    return w, {'trace': trace, 'status': status, 'requested_cardinality': int(k), 'cap': cap}


def sparse_trajectory_extension(X, y, previous, *, fractions=(.5, .65, .8), tolerance=.01, geometry='minmax'):
    """Fixed sparse extensions with independent certification, no outer labels.

    Objective tolerance is a training selection rule, not a risk bound.
    Both exact minimum-loss and compression-within-tolerance are returned.
    """
    X, y = np.asarray(X, float), np.asarray(y)
    if tolerance < 0 or not np.isfinite(tolerance) or any(not 0 < f <= 1 for f in fractions):
        raise ValueError('invalid training selection rule')
    A, _ = preprocess_fit(X, X,geometry=geometry); d = X.shape[1]; full = np.ones(d)/np.sqrt(d); mass = full.sum()
    objective = QueryProbabilityObjective(X, y,geometry=geometry)
    if previous['training_splits'] != objective.splits:
        raise ValueError('training split mismatch')
    protected = MajorityTerms(A, y, weights=full).bounds(range(d), allowed_enemies=2) > 1e-10
    candidates = []
    for old in previous['candidates']:
        v = np.asarray(old['weights']); fs = np.flatnonzero(v > 0).tolist()
        cert = weight_certificate(A, y, fs, full, v[fs], protected=protected)
        candidates.append({'origin': 'old_trajectory', 'features': fs, 'weights': v.tolist(),
            'objective': objective(v/mass)[0], 'certificate': cert})
    initial = np.asarray(previous['unconstrained_selected']['weights'])/mass
    for k in sorted({max(int(np.ceil(d/2)), int(np.ceil(f*d))) for f in fractions}):
        w, fit = optimize_capped_sparse(objective, initial, k, cap=2/d)
        v = w*mass; fs = np.flatnonzero(v > 0).tolist()
        cert = weight_certificate(A, y, fs, full, v[fs], protected=protected)
        candidates.append({'origin': 'capped_sparse', 'features': fs, 'weights': v.tolist(),
            'objective': objective(w)[0], 'certificate': cert, 'fit': fit})
    feasible = [c for c in candidates if c['certificate']['feasible']]
    loss_key = lambda c: (c['objective'], len(c['features']))
    exact = min(feasible, key=loss_key)
    compact = min((c for c in feasible if c['objective'] <= exact['objective']+tolerance),
        key=lambda c: (len(c['features']), c['objective']))
    return {'candidates': candidates, 'selected': exact, 'compact_selected': compact,
        'unconstrained_selected': min(candidates, key=loss_key), 'tolerance': tolerance,
        'training_splits': objective.splits, 'scope': 'training_sparse_cap_candidates_independent_empirical_certificates'}
