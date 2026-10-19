"""Independent friend/interval replay and original certificate verification."""
import json
import sys
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import gamma
from run_flexible_majority_repair_admission import FOLDER, BASE, digest
from audit_forest_compact_transfer import equivalent


def friend_enemy_bounds(A, y, weights, anchor, *, friend_weights=None):
    """Own cityblock friend selection and scalar-box endpoint calculation."""
    fw = weights if friend_weights is None else friend_weights
    friends = np.flatnonzero((y == y[anchor]) & (np.arange(len(y)) != anchor))
    if len(friends) < 3:
        return None
    distances = np.sum(abs(A[friends]-A[anchor])*fw, axis=1)
    friends = friends[np.lexsort((friends, distances))][:3]
    enemies = np.flatnonzero(y != y[anchor])
    lower, upper = np.maximum(0, A-.01), np.minimum(1, A+.01)
    values = []
    for t in (lower[anchor], upper[anchor]):
        near = np.maximum(0, np.maximum(lower[enemies]-t, t-upper[enemies]))
        far = np.maximum(abs(t-lower[friends]), abs(t-upper[friends]))
        values.append(near[:, None, :]-far[None, :, :])
    terms = np.minimum(*values)
    return np.einsum('efd,d->ef', terms, weights).min(axis=1)


def independent_protected(A, y, weights):
    margins = np.full(len(A), -np.inf)
    for i in range(len(A)):
        value = friend_enemy_bounds(A, y, weights, i)
        if value is not None:
            raw = np.partition(value, 2)[2]
            margins[i] = raw-1e-10*(1+abs(raw))
    return margins > 1e-10


def check_repair(A, y, info, counts):
    fs, d = info['features'], A.shape[1]
    weights = np.array(info['weights'])
    full = np.ones(d)/np.sqrt(d)
    assert np.min(weights) >= 0
    np.testing.assert_allclose(weights.sum(), np.sqrt(d), atol=1e-10)
    assert weights.max()/np.sqrt(d) <= 2/d+1e-10
    protected = independent_protected(A, y, full)
    cert = weight_certificate(A, y, fs, full, weights, protected=protected)
    equivalent(cert, info['certificate'])
    targets = np.array(info['selected_targets'], dtype=int)
    assert len(set(targets)) == len(targets) and np.all(protected[targets])
    for c, required in zip(np.unique(y), cert['required']):
        assert np.sum(y[targets] == c) >= required
    for i in targets:
        bounds = friend_enemy_bounds(A[:, fs], y, weights, i, friend_weights=np.array(info['initial_weights']))
        # Verify the actual sufficient friend/exception property rather than
        # treating loose solver feasibility tolerance as a certificate.
        assert np.sum(bounds <= 1e-10) <= 2
        counts['independent_fixed_friend_targets'] += 1
    if info['status'] == 'original_certificate_passed':
        assert cert['feasible']
    counts['fresh_original_certificate_replays'] += 1
    rng = np.random.default_rng(20261045)
    wide = np.zeros(d)
    wide[fs] = weights
    for _ in range(12):
        Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
        loss = gamma(Z, y, weights=full)-gamma(Z, y, fs, weights=wide)
        assert loss <= cert['dependency_certificate']['upper_bound']+1e-10
        distances = cdist(Z[:, fs]*weights, Z[:, fs]*weights, 'cityblock')
        np.fill_diagonal(distances, np.inf)
        nn = np.argsort(distances, axis=1, kind='stable')[:, :5]
        correct = (y[nn] == y[:, None]).sum(axis=1) >= 3
        for c, required in zip(np.unique(y), cert['required']):
            assert np.sum(correct & protected & (y == c)) >= required
        counts['sampled_shared_tables'] += 1
    return cert


def audit(name):
    threadpool_limits(1)
    path = FOLDER/(name+'.json')
    saved = json.loads(path.read_text())
    assert saved['complete'] and not saved['query_labels_or_accuracy_used']
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    prior = json.loads((BASE/(name+'.json')).read_text())
    assert saved['reference_indices'] == prior['reference_indices']
    stage = prior['outer']['stage']
    counts = Counter(datasets=1)
    if saved['attempts']:
        X, y = load_dataset(name)
        ri = np.array(saved['reference_indices'])
        A, _ = preprocess_fit(X[ri], X[ri])
        choices = stage['path']['rankings'][0]['choices']
        for entry, attempt in zip(choices, saved['attempts']):
            assert entry['removed'] == attempt['removed'] and entry['loss'] == attempt['fixed_OOF_loss']
            info = attempt['fit']
            assert info['features'] == [a for a in range(saved['d']) if a != entry['removed']]
            assert info['time_limit'] == 10.
            if 'weights' in info:
                check_repair(A, y[ri], info, counts)
            else:
                # Time-limited no-incumbent reports are search records, never
                # independently certified mathematical infeasibility results.
                assert info['status'] in ('no_incumbent', 'incumbent_constraints_failed', 'normalized_cap_failed')
            counts['repair_attempts'] += 1
        if saved['status'] == 'original_certified_repair_selected':
            assert saved['attempts'][-1]['fit']['status'] == 'original_certificate_passed'
            assert all(a['fit']['status'] != 'original_certificate_passed' for a in saved['attempts'][:-1])
            assert saved['actual_count'] == saved['target_count']
            equivalent(saved['weights'], saved['attempts'][-1]['fit']['weights'])
            counts['repaired_references'] += 1
        else:
            assert len(saved['attempts']) == len(choices)
    else:
        assert saved['actual_count'] == saved['original_count']
    return counts


if __name__ == '__main__':
    path = FOLDER/'complete.json'
    saved = json.loads(path.read_text())
    counts = Counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(audit, r['dataset']) for r in saved['rows']]):
            counts.update(future.result())
    assert counts['repaired_references'] == saved['repaired_blocked_references']
    assert saved['admission_gate_passed'] == (counts['repaired_references'] >= 1)
    report = {'passed': True, 'counts': dict(counts), 'summary_sha256': digest(path),
              'auditor_sha256': digest(Path(__file__)), 'goal_achieved': False,
              'scope': 'independent_scalar_friend_MILP_witness_original_FRS_replay_and_samples_not_global_solver_optimum'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report, flush=True)
