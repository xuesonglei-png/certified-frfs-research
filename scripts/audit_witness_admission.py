"""Shared-box majority samples and unchanged-selected-endpoint audit."""
import hashlib
import json
import zipfile
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from audit_sparse_pilots import preprocessing
from certified_frfs.datasets import load_dataset


def main():
    threadpool_limits(1)
    source = ROOT/'results/witness-admission-diagnostic/complete.json'
    p = json.loads(source.read_text()); samples = 0; objects = 0; selected = 0
    for relative, expected in p['source_hashes'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected
    for row in p['rows']:
        X, y = load_dataset(row['dataset'])
        prior_path = ROOT/'results/joint-support-diagnostic'/f"{row['dataset']}.json"
        assert row['prior_sha256'] == hashlib.sha256(prior_path.read_bytes()).hexdigest()
        for s in row['splits']:
            ri = np.array(s['reference_indices']); A, _ = preprocessing(X[ri], X[ri]); labels = y[ri]
            for c in s['certificates']:
                fs = c['features']; wide = np.zeros(row['d'])
                from certified_frfs.redistributed_loss import transfer_weights
                full = np.ones(row['d'])/np.sqrt(row['d']); wide[fs] = transfer_weights(A, fs, full)
                protected = np.array(c['witness_bounds']) > 1e-10
                rng = np.random.default_rng(1207)
                for _ in range(3):
                    Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
                    D = cdist(Z*wide, Z*wide, 'cityblock'); np.fill_diagonal(D, np.inf)
                    nn = np.argsort(D, axis=1, kind='stable')[:, :min(5, len(ri)-1)]
                    counts = (labels[nn] == labels[:, None]).sum(axis=1)
                    assert np.all(counts[protected] >= 3)
                    samples += 1; objects += int(protected.sum())
            assert s['features'] == s['original_features']
            assert all(abs(s['metrics'][k]-s['reference_metrics']['certified'][k]) < 1e-12 for k in s['metrics'])
            selected += 1
        print(row['dataset'], 'witness sampled majority audited', flush=True)
    audit = {'table_perturbations': samples, 'protected_object_checks': objects,
        'unchanged_selected_endpoints': selected, 'observed_violations': 0,
        'result_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope': 'sampled_robust_LOO_majority_and_selection_not_box_exhaustion_or_test_risk'}
    (source.parent/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    target = ROOT/'output/joint-support-followup-evidence.zip'
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        folders = [ROOT/'results/paired-stability-diagnostic', ROOT/'results/witness-admission-diagnostic']
        paths = [q for f in folders for q in f.glob('*.json')]
        paths += [ROOT/q for q in p['source_hashes']]
        paths += [ROOT/'scripts'/q for q in ['diagnose_paired_stability.py', 'audit_paired_stability.py',
            'audit_witness_admission.py', 'analyze_joint_support_failures.py']]
        paths += [ROOT/'research'/q for q in ['paired-stability-protocol.md', 'joint-support-development-review.md']]
        paths += [ROOT/'results/joint-support-diagnostic/failure-analysis.json']
        for q in paths:
            z.write(q, str(q.relative_to(ROOT)))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
