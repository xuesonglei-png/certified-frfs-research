"""Budgeted support search that can traverse uncertified intermediate sets.

The two outputs share one crossfit-scored pool. Certification filters a fixed
shortlist of endpoints; it never licenses a claim about test-set accuracy.
"""
import numpy as np
from .cached_transfer import CachedTransfer
from .cached_transfer_utility import CachedCrossfitTransferUtility
from .feasible_transport import weight_certificate
from .majority_metric import MajorityTerms
from .validated_pool import preprocess_fit


def support_pool(d, evaluate, *, budget=64, seed=101):
    """Deterministic restarts and add/delete/swap/block moves; unique scoring."""
    if not isinstance(d, int) or d < 1 or budget < 1:
        raise ValueError('positive dimension and evaluation budget required')
    rng = np.random.default_rng(seed)
    rows = {}

    def add(fs, origin):
        key = tuple(sorted(set(fs)))
        if not key or key in rows or len(rows) >= budget:
            return False
        if key[0] < 0 or key[-1] >= d:
            raise ValueError('invalid support')
        scores = evaluate(key)
        if not np.isfinite(scores['utility']):
            raise ValueError('nonfinite search utility')
        rows[key] = {'features': list(key), 'origin': origin, **scores}
        return True

    add(range(d), 'original_full')
    # Near-full restarts make multi-coordinate deletions accessible even when
    # every one-coordinate intermediate fails a certificate.
    for fraction in (.9, .8, .65, .5):
        for _ in range(3):
            size = max(1, int(np.ceil(fraction*d)))
            add(rng.choice(d, size, replace=False).tolist(), 'restart')
    idle = 0
    while len(rows) < budget and idle < 8*budget:
        ranked = sorted(rows.values(), key=lambda r: (-r['utility'], len(r['features']), r['features']))
        parent = ranked[int(rng.integers(min(6, len(ranked))))]['features']
        present = np.array(parent, int)
        absent = np.setdiff1d(np.arange(d), present)
        operation = int(rng.integers(4))
        fs = set(parent)
        if operation == 0 and len(present) > 1:
            fs.remove(int(rng.choice(present)))
        elif operation == 1 and len(absent):
            fs.add(int(rng.choice(absent)))
        elif operation == 2 and len(absent):
            fs.remove(int(rng.choice(present)))
            fs.add(int(rng.choice(absent)))
        elif operation == 3 and len(present) > 2:
            count = min(len(present)-1, max(2, int(np.ceil(.15*d))))
            fs.difference_update(rng.choice(present, count, replace=False).tolist())
        else:
            size = int(rng.integers(1, d+1))
            fs = set(rng.choice(d, size, replace=False).tolist())
        idle = 0 if add(fs, ('delete', 'add', 'swap', 'block')[operation]) else idle+1
    return sorted(rows.values(), key=lambda r: (-r['utility'], len(r['features']), r['features']))


def fit_joint_support(X, y, *, budget=64, certificate_budget=12, seed=101):
    """Raw training data only; unchanged .01/.1/.9 certificate protocol."""
    if certificate_budget < 1:
        raise ValueError('positive certificate budget required')
    X = np.asarray(X, float)
    y = np.asarray(y)
    d = X.shape[1]
    workspace = CachedCrossfitTransferUtility(X, y, family='uniform', seed=seed)
    pool = support_pool(d, workspace.evaluate, budget=budget, seed=seed)
    A, _ = preprocess_fit(X, X)
    original = np.ones(d)/np.sqrt(d)
    transfer = CachedTransfer(A, original)
    protected = MajorityTerms(A, y, weights=original, eps=.01).bounds(
        range(d), allowed_enemies=2) > 1e-10
    full = tuple(range(d))
    shortlist = [tuple(r['features']) for r in pool if tuple(r['features']) != full][:certificate_budget]
    shortlist.append(full)
    certificates = {}
    for fs in shortlist:
        certificates[fs] = weight_certificate(A, y, fs, original, transfer.weights(fs),
            eps=.01, retention=.9, dependency_eta=.1, protected=protected)
    if not certificates[full]['feasible']:
        raise RuntimeError('original metric fails its own certificate')
    eligible = [r for r in pool if certificates.get(tuple(r['features']), {}).get('feasible', False)]
    certified = eligible[0]
    unconstrained = pool[0]

    def fitted(row, cert):
        fs = row['features']
        wide = np.zeros(d)
        wide[fs] = transfer.weights(fs)
        return {'features': fs, 'weights': wide.tolist(), 'search_utility': row['utility'],
                'certificate': cert, 'original_full': tuple(fs) == full}

    return {'certified': fitted(certified, certificates[tuple(certified['features'])]),
            'unconstrained': fitted(unconstrained, certificates.get(tuple(unconstrained['features']))),
            'pool': pool, 'certificates': [{'features': list(fs), **cert} for fs, cert in certificates.items()],
            'search_splits': workspace.splits, 'budget': budget, 'certificate_budget': certificate_budget,
            'evaluations': len(pool), 'seed': seed,
            'scope': 'adaptive_training_crossfit_endpoint_certification_not_test_risk',
            'unchecked_candidates': len(pool)-len(certificates)}
