"""Independent cdist LOO votes and vector Newton membership replay of profiles."""
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from diagnose_soft5_calibration import ROOT, FOLDER, PARENT, digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from soft_topk_metric_objective import soft_topk_membership


def check(A, y, weights, mass, saved):
    n, p = A.shape; weights = np.asarray(weights); classes = np.unique(y)
    lo, hi = np.maximum(0, A-.01), np.minimum(1, A+.01)
    same = y[:, None] == y[None, :]; np.fill_diagonal(same, False)
    D = np.zeros((n, n))
    for a in range(p):
        near = np.maximum(0, np.maximum(lo[:, None, a]-hi[None, :, a], lo[None, :, a]-hi[:, None, a]))
        far = np.maximum(abs(lo[:, None, a]-hi[None, :, a]), abs(hi[:, None, a]-lo[None, :, a]))
        D += (weights[a]/mass)*np.where(same, far, near)
    mask = ~np.eye(n, dtype=bool)
    Q = soft_topk_membership(D, mask, temperature=.05)
    ranked = D.copy(); np.fill_diagonal(ranked, np.inf)
    order = np.argsort(ranked, axis=1, kind='stable'); hard = np.zeros_like(mask)
    hard[np.arange(n)[:, None], order[:, :5]] = True
    leak = np.sum(Q*(mask & ~hard), axis=1)/5
    soft_own = np.sum(Q*same, axis=1)/5
    hard_own = np.sum(hard*same, axis=1)/5
    gaps = D[np.arange(n), order[:, 5]]-D[np.arange(n), order[:, 4]]
    bounds = []
    for g in gaps:
        h, m = np.exp(-g/.05), n-1
        root = brentq(lambda t: t*t-h*(5-t)*(m-5-t), 0., min(5, m-5), xtol=1e-13)
        bounds.append(root/5)
    error = abs(soft_own-hard_own)
    assert np.all(error <= leak+1e-9) and np.all(leak <= np.array(bounds)+1e-9)
    clean = cdist(A*weights, A*weights, 'cityblock'); np.fill_diagonal(clean, np.inf)
    neighbors = np.argsort(clean, axis=1, kind='stable')[:, :5]
    votes = np.stack([(y[neighbors] == c).sum(axis=1) for c in classes], axis=1)
    correct = classes[votes.argmax(axis=1)] == y
    balanced = lambda v: float(np.mean([np.mean(v[y == c]) for c in classes]))
    fresh = {'clean_hard5_LOO_BA': balanced(correct),
             'balanced_robust_soft_own_mass': balanced(soft_own),
             'balanced_robust_hard5_own_fraction': balanced(hard_own),
             'mean_robust_soft_to_hard_leak_fraction': float(leak.mean()),
             'mean_conditional_leak_bound_fraction': float(np.mean(bounds)),
             'median_robust_boundary_gap': float(np.median(gaps)),
             'fraction_positive_robust_boundary_gaps': float(np.mean(gaps > 0)),
             'mean_robust_class_fraction_error': float(error.mean()),
             'max_robust_class_fraction_error': float(error.max())}
    assert saved['objects'] == n and saved['temperature'] == .05 and saved['max_mass_residual'] <= 1e-10
    for key, value in fresh.items(): np.testing.assert_allclose(saved[key], value, atol=1e-9, rtol=0, err_msg=key)
    assert np.max(abs(Q.sum(axis=1)-5)) <= 1e-10


def audit(name):
    threadpool_limits(1); verify(PARENT)
    saved = json.loads((FOLDER/(name+'.json')).read_text())
    assert saved['complete'] and not saved['query_labels_or_accuracy_used']
    for rel, sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
    parent = json.loads((PARENT/(name+'.json')).read_text()); assert len(parent['records']) == len(saved['records'])
    X, y = load_dataset(name); ri = np.array(parent['reference_indices']); A, _ = preprocess_fit(X[ri], X[ri]); labels = y[ri]
    counts = Counter(datasets=1)
    for source, record in zip(parent['records'], saved['records']):
        fs = source['features']; assert record['features'] == fs; mass = source['fit']['mass']
        for family in ('anchor', 'target'):
            check(A[:, fs], labels, source[family], mass, record[family]); counts['profiles'] += 1
        gain = 100*(record['target']['clean_hard5_LOO_BA']-record['anchor']['clean_hard5_LOO_BA'])
        np.testing.assert_allclose(gain, record['clean_hard5_LOO_gain_pp'], atol=1e-10)
        counts['training_hard5_improved' if gain > 1e-8 else 'training_hard5_worsened' if gain < -1e-8 else 'training_hard5_tied'] += 1
        counts['supports'] += 1
    print(name, 'independent vector membership/cdist LOO calibration audit passed', flush=True)
    return {'counts': dict(counts), 'artifact_sha256': digest(FOLDER/(name+'.json')), 'dataset': name}


def main():
    summary = json.loads((FOLDER/'complete.json').read_text()); counts, reports = Counter(), []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(audit, row['dataset']) for row in summary['rows']]):
            result = future.result(); reports.append(result); counts.update(result['counts'])
    assert counts['datasets'] == 15 and counts['supports'] == 86 and counts['profiles'] == 172
    assert {k: v for k, v in counts.items() if k not in ('datasets', 'profiles')} == summary['counts']
    hashes = {r['dataset']: r['artifact_sha256'] for r in reports}; assert hashes == summary['artifact_hashes']
    result = {'passed': True, 'counts': dict(counts), 'artifact_hashes': hashes,
              'summary_sha256': digest(FOLDER/'complete.json'), 'auditor_sha256': digest(Path(__file__)),
              'goal_achieved': False, 'scope': 'training_only_scalarBrent_vs_vectorNewton_and_cdist_LOO_not_query_accuracy'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
