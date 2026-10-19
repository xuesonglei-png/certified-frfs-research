"""Recompute partition bounds and sampled majority for the admission diagnostic."""
import hashlib
import json
import zipfile
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_partition_admission import FOLDER
from diagnose_guarded_support import NAMES
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.partition_majority import partition_majority_bounds
from certified_frfs.redistributed_loss import transfer_weights


def main():
    threadpool_limits(1); source = FOLDER/'partition-admission.json'; p = json.loads(source.read_text())
    count = {'bounds': 0, 'sampled_cells': 0, 'objects': 0, 'violations': 0}
    for row in p['rows']:
        path = FOLDER/f"{row['dataset']}.json"; saved = json.loads(path.read_text())
        assert row['prior_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        s = saved['splits'][row['inner']]; X, y = load_dataset(row['dataset']); ri = np.array(s['reference_indices'])
        A, _ = preprocess_fit(X[ri], X[ri]); labels = y[ri]; d = A.shape[1]; w = np.ones(d)/np.sqrt(d)
        fs = row['features']; wr = transfer_weights(A, fs, w); protected = MajorityTerms(A, labels, weights=w).bounds(range(d), allowed_enemies=2) > 1e-10
        old = MajorityTerms(A[:, fs], labels, weights=wr).bounds(range(len(fs)), allowed_enemies=2)
        bounds, leaves = partition_majority_bounds(A[:, fs], labels, wr, protected=protected, max_leaves=16)
        combined = np.maximum(old, bounds); retained = [int(((combined > 1e-10) & ((labels == q) & protected)).sum()) for q in np.unique(labels)]
        assert retained == row['new_retained']; assert leaves.tolist() == row['leaf_counts']
        count['bounds'] += len(bounds)
        rng = np.random.default_rng(1491)
        for _ in range(20):
            Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1); D = cdist(Z[:, fs]*wr, Z[:, fs]*wr, 'cityblock'); np.fill_diagonal(D, np.inf)
            nn = np.argsort(D, axis=1, kind='stable')[:, :5]; correct = (labels[nn] == labels[:, None]).sum(axis=1) >= 3
            if np.any(~correct[combined > 1e-10]): count['violations'] += 1
            count['sampled_cells'] += 1; count['objects'] += int((combined > 1e-10).sum())
        print(row['dataset'], row['inner'], 'partition audited', flush=True)
    result = {**count, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope': 'partition_bound_reconstruction_and_sampled_majority_not_box_exhaustion_or_test_risk'}
    (FOLDER/'partition-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    archive = ROOT/'output/partition-majority-evidence.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for file in [source, FOLDER/'partition-audit.json', ROOT/'scripts/diagnose_partition_admission.py', ROOT/'scripts/audit_partition_admission.py', ROOT/'src/certified_frfs/partition_majority.py', ROOT/'research/partition-majority-proof.md']:
            z.write(file, str(file.relative_to(ROOT)))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
