"""Fixed16cell anchor diagnostic on the twelve rejected guard choices."""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_guarded_support import FOLDER, NAMES
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.partition_majority import partition_majority_bounds


def main():
    threadpool_limits(1); out = FOLDER/'partition-admission.json'
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for name in NAMES:
        path = FOLDER/f'{name}.json'; p = json.loads(path.read_text()); X, y = load_dataset(name)
        for index, s in enumerate(p['splits']):
            c = s['fit']['unconstrained']
            if len(c['features']) == p['d'] or c['certificate']['feasible']:
                continue
            ri = np.array(s['reference_indices']); A, _ = preprocess_fit(X[ri], X[ri]); labels = y[ri]
            w = np.ones(p['d'])/np.sqrt(p['d']); fs = c['features']; wr = np.array(c['weights'])[fs]
            protected = MajorityTerms(A, labels, weights=w).bounds(range(p['d']), allowed_enemies=2) > 1e-10
            old = MajorityTerms(A[:, fs], labels, weights=wr).bounds(range(len(fs)), allowed_enemies=2)
            partition, leaves = partition_majority_bounds(A[:, fs], labels, wr, protected=protected, max_leaves=16)
            combined = np.maximum(old, partition)
            groups = [(labels == q) & protected for q in np.unique(labels)]
            required = [int(np.ceil(.9*g.sum()-1e-12)) for g in groups]
            retained = [int(((combined > 1e-10) & g).sum()) for g in groups]
            assert all(a >= b for a, b in zip(retained, c['certificate']['retained_counts']))
            majority = all(a >= b for a, b in zip(retained, required))
            feasible = bool(majority and c['certificate']['dependency_certificate']['upper_bound'] <= .1)
            rows.append({'dataset': name, 'inner': index, 'features': fs,
                'old_retained': c['certificate']['retained_counts'], 'new_retained': retained,
                'required': required, 'newly_majority_certified_objects': int(((combined > 1e-10) & (old <= 1e-10) & protected).sum()),
                'majority_feasible': majority, 'feasible': feasible, 'leaf_counts': leaves.tolist(),
                'partition_bounds': [float(v) if np.isfinite(v) else None for v in partition],
                'prior_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            print(name, index, 'new objects', rows[-1]['newly_majority_certified_objects'], 'feasible', feasible, flush=True)
    sources = [ROOT/'scripts/diagnose_partition_admission.py', ROOT/'src/certified_frfs/partition_majority.py', ROOT/'research/partition-majority-proof.md']
    result = {'rows': rows, 'checked_supports': len(rows), 'new_joint_admissions': sum(r['feasible'] for r in rows),
        'new_majority_certified_objects': sum(r['newly_majority_certified_objects'] for r in rows),
        'max_leaves': 16, 'scope': 'training_admission_only_not_new_classification_or_test_superiority',
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
