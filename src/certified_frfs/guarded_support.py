"""Independent discovery/guard selection and fresh whole-reference certificates."""
import numpy as np
from .cached_transfer import CachedTransfer
from .cached_transfer_utility import CachedCrossfitTransferUtility
from .joint_support_search import support_pool
from .validated_pool import preprocess_fit
from .feasible_transport import weight_certificate
from .majority_metric import MajorityTerms


def discovery_guard_split(y, *, seed=101):
    """Classwise holdout; classes with fewer than four stay discovery-only."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed+73)
    discovery, guard = [], []
    for label in np.unique(y):
        ids = rng.permutation(np.flatnonzero(y == label))
        count = max(1, int(np.floor(.25*len(ids)))) if len(ids) >= 4 else 0
        guard.extend(ids[:count].tolist())
        discovery.extend(ids[count:].tolist())
    return np.array(sorted(discovery), int), np.array(sorted(guard), int)


def paired_guard_gain(correct, baseline, labels):
    """Class-balanced paired mean minus one stratified standard-error penalty.

    Heuristic selection penalty only: multiple candidates share the guard.
    Inputs average clean and repeated noise correctness per guard observation.
    """
    correct, baseline, labels = np.asarray(correct), np.asarray(baseline), np.asarray(labels)
    if correct.shape != labels.shape or baseline.shape != labels.shape or len(labels) == 0:
        raise ValueError('nonempty matching correctness and labels required')
    gains = correct-baseline
    classes = np.unique(labels)
    means, variances = [], []
    for q in classes:
        v = gains[labels == q]
        means.append(float(v.mean()))
        # A singleton class has no variance estimate: conservative range-based
        # contribution, rather than treating its estimated variance as zero.
        variances.append(float(np.var(v, ddof=1)/len(v)) if len(v) > 1 else 1.)
    mean = float(np.mean(means))
    penalty = float(np.sqrt(np.sum(variances))/len(classes))
    return {'mean_gain': mean, 'variability_penalty': penalty, 'penalized_gain': mean-penalty}


def guard_correctness(A, yt, B, yv, features, weights, *, seed=101):
    fs = tuple(features)
    rng = np.random.default_rng(seed+79)
    tables = [(A, B)]
    for _ in range(3):
        tables.append((np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1),
                       np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1)))
    labels = np.unique(yt); correct = []
    for reference, query in tables:
        D = np.abs(query[:, None, fs]-reference[None, :, fs])@weights
        ids = np.argsort(D, axis=1, kind='stable')[:, :min(5, len(yt))]
        votes = np.stack([(yt[ids] == q).sum(axis=1) for q in labels], axis=1)
        correct.append((labels[votes.argmax(axis=1)] == yv).astype(float))
    return np.mean(correct, axis=0)


def fit_guarded_support(X, y, *, seed=101, budget=64, certificate_budget=12):
    X, y = np.asarray(X, float), np.asarray(y)
    d = X.shape[1]; full = tuple(range(d)); original = np.ones(d)/np.sqrt(d)
    if certificate_budget < 1:
        raise ValueError('positive certificate budget required')
    discovery, guard = discovery_guard_split(y, seed=seed)
    A, _ = preprocess_fit(X, X)
    transfer = CachedTransfer(A, original)
    if len(guard) == 0:
        raise ValueError('no guard observations; independent selection unavailable')
    workspace = CachedCrossfitTransferUtility(X[discovery], y[discovery], seed=seed)
    pool = support_pool(d, workspace.evaluate, budget=budget, seed=seed)
    shortlist = [c for c in pool if tuple(c['features']) != full][:certificate_budget]
    shortlist.append(next(c for c in pool if tuple(c['features']) == full))
    D, G = preprocess_fit(X[discovery], X[guard])
    discovery_transfer = CachedTransfer(D, original)
    baseline = guard_correctness(D, y[discovery], G, y[guard], full, original, seed=seed)
    guarded = []
    for c in shortlist:
        fs = c['features']
        correctness = guard_correctness(D, y[discovery], G, y[guard], fs, discovery_transfer.weights(fs), seed=seed)
        gain = paired_guard_gain(correctness, baseline, y[guard])
        # The full-vs-itself difference is exactly zero even for singleton
        # guard classes; it supplies the reference endpoint for this heuristic.
        if tuple(fs) == full:
            gain = {'mean_gain': 0., 'variability_penalty': 0., 'penalized_gain': 0.}
        guarded.append({**c, **gain, 'guard_correctness': correctness.tolist()})
    guarded.sort(key=lambda c: (-c['penalized_gain'], -c['mean_gain'],
        tuple(c['features']) != full, len(c['features']), c['features']))
    protected = MajorityTerms(A, y, weights=original).bounds(range(d), allowed_enemies=2) > 1e-10
    certificates = {}
    for c in guarded:
        fs = c['features']
        certificates[tuple(fs)] = weight_certificate(A, y, fs, original, transfer.weights(fs), protected=protected)
    assert certificates[full]['feasible']
    feasible = [c for c in guarded if certificates[tuple(c['features'])]['feasible']]
    fits = {}
    for name, c in [('certified', feasible[0]), ('unconstrained', guarded[0])]:
        fs = c['features']; wide = np.zeros(d); wide[fs] = transfer.weights(fs)
        fits[name] = {'features': fs, 'weights': wide.tolist(), 'guard_gain': c['mean_gain'],
            'penalized_guard_gain': c['penalized_gain'], 'certificate': certificates[tuple(fs)]}
    return {**fits, 'pool': pool, 'guarded_candidates': guarded,
        'guard_baseline_correctness': baseline.tolist(), 'discovery_indices': discovery.tolist(),
        'guard_indices': guard.tolist(), 'discovery_crossfit_splits': workspace.splits,
        'certificates': [{'features': list(fs), **c} for fs, c in certificates.items()],
        'scope': 'independent_candidate_discovery_shared_guard_selection_fresh_finite_certificate_not_test_risk',
        'budget': budget, 'certificate_budget': certificate_budget, 'seed': seed}
