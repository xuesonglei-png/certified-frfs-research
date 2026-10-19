"""Independent robust soft-neighbor arithmetic, metric caps and FRS replay."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from run_predictive_target_metric_admission import ROOT, FOLDER, PARENT, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from audit_forest_compact_transfer import equivalent


def objective(A, y, weights):
    n, p = A.shape
    lo, hi = np.maximum(0, A-.01), np.minimum(1, A+.01)
    same = y[:, None] == y[None, :]; np.fill_diagonal(same, False)
    valid = same.any(axis=1); classes = np.unique(y[valid])
    total = np.zeros((n, n))
    for a, w in enumerate(weights):
        near = np.maximum(0, np.maximum(lo[:, a, None]-hi[None, :, a], lo[None, :, a]-hi[:, a, None]))
        far = np.maximum(abs(lo[:, a, None]-hi[None, :, a]), abs(hi[:, a, None]-lo[None, :, a]))
        total += w*np.where(same, far, near)
    scores = -total/.05; np.fill_diagonal(scores, -np.inf)
    values = []
    for i in np.flatnonzero(valid):
        values.append((logsumexp(scores[i])-logsumexp(scores[i, same[i]]))/(len(classes)*sum(y[valid] == y[i])))
    return float(sum(values)+.01*np.sum(weights*np.log(p*weights)))


def audit(name):
    threadpool_limits(1)
    artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    assert saved['complete'] and not saved['query_labels_or_accuracy_used']
    for rel, sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha, rel
    parent_path = PARENT/(name+'.json'); parent = json.loads(parent_path.read_text()); parent_audit = json.loads((PARENT/'complete-audit.json').read_text())
    assert parent_audit['passed'] and parent_audit['artifact_hashes'][name] == digest(parent_path)
    assert saved['reference_indices'] == parent['reference_indices']
    X, y = load_dataset(name); ri = np.array(saved['reference_indices']); labels = y[ri]
    A, _ = preprocess_fit(X[ri], X[ri]); d = A.shape[1]; mass = np.sqrt(d); full = np.ones(d)/mass
    protected = independent_protected(A, labels, full); counts = Counter(datasets=1)
    expected = [r for r in parent['pool'] if r['accepted']]; assert len(expected) == len(saved['records'])
    nontrivial = 0
    for record, old in zip(saved['records'], expected):
        fs = record['features']; p = len(fs)
        assert fs == old['features'] and record['origins'] == old['origins']
        anchor = np.array(old['selected_metric']['weights'])[fs]
        equivalent(anchor.tolist(), record['anchor']); target = np.array(record['target']); fit = record['fit']
        equivalent(target.tolist(), fit['weights']); np.testing.assert_allclose(target.sum(), mass, atol=1e-10)
        assert target.min() >= 0 and target.max()/mass <= 2/d+1e-10
        assert fit['features'] == fs and fit['temperature'] == .05 and fit['reg'] == .01 and fit['max_iter'] == 200
        classes, numbers = np.unique(labels, return_counts=True)
        assert fit['singleton_training_classes'] == classes[numbers == 1].tolist()
        if fit['status'] == 'unique_capped_simplex':
            assert abs(p*2/d-1) <= 1e-12 and fit['optimizer_iterations'] == 0
            np.testing.assert_allclose(target, np.full(p, mass/p), atol=1e-10)
        elif fit['status'] == 'no_LOO_same_class_supervision':
            assert np.all(numbers == 1); equivalent(target.tolist(), anchor.tolist())
        else:
            initial = (1-1e-5)*anchor/mass+1e-5*np.ones(p)/p
            initial_value = objective(A[:, fs], labels, initial)
            equivalent(initial_value, fit['training_objective_initial'])
            if fit['status'] == 'local_predictive_target':
                assert fit['optimizer_success'] and fit['optimizer_status'] == 0
                value = objective(A[:, fs], labels, target/mass)
                equivalent(value, fit['training_objective_final']); assert value <= initial_value+1e-8
            else:
                assert 'fallback' in fit['status']; equivalent(target.tolist(), anchor.tolist())
            counts['independent_objective_replays'] += 1
        assert [r['interpolation'] for r in record['mixtures']] == [.25, .5, .75, 1.]
        for r in record['mixtures']:
            w = (1-r['interpolation'])*anchor+r['interpolation']*target
            equivalent(w.tolist(), r['weights']); np.testing.assert_allclose(w.sum(), mass, atol=1e-10)
            assert w.max()/mass <= 2/d+1e-10 and w.min() >= 0
            cert = weight_certificate(A, labels, fs, full, w, protected=protected)
            equivalent(cert, r['certificate']); classes = independent_class_certificate(labels, cert)
            equivalent(classes, r['class_certificate']); accepted = cert['feasible'] and classes['feasible_all_classes']
            assert accepted == r['accepted']; shift = float(np.sum(abs(w-anchor))); equivalent(shift, r['weight_shift_l1'])
            counts['mixtures'] += 1; counts['admitted_mixtures'] += accepted
            if accepted and shift > 1e-8:
                counts['admitted_nontrivial_mixtures'] += 1; nontrivial += 1
        counts['supports'] += 1; counts['fit_'+fit['status']] += 1; counts['optimizer_failures'] += not fit['optimizer_success']
    row = {'dataset': name, 'supports': len(expected), 'admitted_nontrivial_mixtures': nontrivial}
    print(name, 'independent target/objective/FRS/class audit passed', flush=True)
    return {'row': row, 'counts': dict(counts), 'artifact_sha256': digest(artifact)}


if __name__ == '__main__':
    summary = json.loads((FOLDER/'complete.json').read_text()); reports = []; counts = Counter()
    with ProcessPoolExecutor(max_workers=2) as ex:
        for future in as_completed([ex.submit(audit, r['dataset']) for r in summary['rows']]):
            result = future.result(); reports.append(result); counts.update(result['counts'])
    equivalent(sorted([r['row'] for r in reports], key=lambda r: r['dataset']), summary['rows'])
    assert counts['datasets'] == len(summary['rows']) == 15
    expected_counts = {k: v for k, v in counts.items() if k not in ('independent_objective_replays', 'datasets')}
    equivalent(expected_counts, summary['counts'])
    output = {'passed': True, 'counts': dict(counts), 'artifact_hashes': {r['row']['dataset']: r['artifact_sha256'] for r in reports},
              'summary_sha256': digest(FOLDER/'complete.json'), 'auditor_sha256': digest(Path(__file__)), 'goal_achieved': False,
              'scope': 'independent_predictive_target_objective_mass_cap_mixture_and_original_classwise_FRS_admission_not_test_accuracy'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8'); print(output, flush=True)
