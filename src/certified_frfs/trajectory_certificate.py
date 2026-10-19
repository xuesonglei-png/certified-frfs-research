"""Training-objective selection of independently certified trajectory mixtures."""
import numpy as np
from .validated_pool import preprocess_fit
from .query_probability import QueryProbabilityObjective
from .majority_metric import MajorityTerms
from .feasible_transport import weight_certificate


def certify_capped_trajectory(X, y, trajectory, *, geometry='minmax'):
    """Fixed four mixtures per noninitial node; no validation data enter."""
    X, y = np.asarray(X, float), np.asarray(y)
    A, _ = preprocess_fit(X, X,geometry=geometry); d = A.shape[1]; full = np.ones(d)/np.sqrt(d); mass = full.sum()
    objective = QueryProbabilityObjective(X, y,geometry=geometry)
    if objective.splits != trajectory['training_splits']:
        raise ValueError('trajectory query splits differ from fixed objective')
    protected = MajorityTerms(A, y, weights=full).bounds(range(d), allowed_enemies=2) > 1e-10
    candidates = []; seen = set()
    proposals = [(0, 0., full)]
    for node in trajectory['trace'][1:]:
        endpoint = np.asarray(node['weights'], float)
        if endpoint.shape != (d,) or np.any(endpoint < 0) or not np.isfinite(endpoint).all():
            raise ValueError('invalid trajectory endpoint')
        if abs(endpoint.sum()-mass) > 1e-10 or np.any(endpoint > 2*mass/d+1e-10):
            raise ValueError('trajectory endpoint violates fixed mass or cap')
        proposals.extend((node['iteration'], alpha, (1-alpha)*full+alpha*endpoint)
                         for alpha in (.25, .5, .75, 1.))
    for iteration, alpha, v in proposals:
        key = tuple(v)
        if key in seen:
            continue
        seen.add(key); fs = np.flatnonzero(v > 0).tolist()
        cert = weight_certificate(A, y, fs, full, v[fs], protected=protected)
        score = objective(v/mass)[0]
        candidates.append({'iteration': iteration, 'alpha': alpha, 'features': fs,
            'weights': v.tolist(), 'objective': score, 'certificate': cert})
    feasible = [c for c in candidates if c['certificate']['feasible']]
    if not candidates[0]['certificate']['feasible']:
        raise RuntimeError('original full metric fails its certificate')
    key = lambda c: (c['objective'], len(c['features']), c['iteration'], c['alpha'])
    selected = min(feasible, key=key); unconstrained = min(candidates, key=key)
    return {'selected': selected, 'unconstrained_selected': unconstrained,
        'candidates': candidates, 'training_splits': objective.splits,
        'scope': 'fixed_training_trajectory_grid_dynamic_witness_certificate_not_test_risk'}
