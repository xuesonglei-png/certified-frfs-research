"""Look for counterexamples to rejected supports' actual finite-table targets.

Sampling cannot certify absence of a violation. A found violation supplies
evidence that changing only a sufficient bound cannot admit that metric.
"""
import hashlib
import json
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_guarded_support import FOLDER, NAMES
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.core import gamma


def main():
    threadpool_limits(1)
    target = FOLDER/'sampled-actual-violations.json'
    if target.exists():
        raise FileExistsError(target)
    rows = []
    for name in NAMES:
        path = FOLDER/f'{name}.json'; p = json.loads(path.read_text()); X, y = load_dataset(name)
        for index, s in enumerate(p['splits']):
            c = s['fit']['unconstrained']
            if len(c['features']) == p['d']:
                continue
            ri = np.array(s['reference_indices']); labels = y[ri]; A, _ = preprocess_fit(X[ri], X[ri])
            original = np.ones(p['d'])/np.sqrt(p['d']); w = np.array(c['weights']); fs = c['features']
            protected = MajorityTerms(A, labels, weights=original).bounds(range(p['d']), allowed_enemies=2) > 1e-10
            groups = [(labels == q) & protected for q in np.unique(labels)]
            required = [int(np.ceil(.9*g.sum()-1e-12)) for g in groups]
            assert required == c['certificate']['required']
            rng = np.random.default_rng(1323); draws = []
            tables = [(0, A)]
            for j in range(1, 65):
                noise = rng.uniform(-.01, .01, A.shape)
                tables.append((j, np.clip(A+noise, 0, 1)))
            for draw, Z in tables:
                D = cdist(Z*w, Z*w, 'cityblock'); np.fill_diagonal(D, np.inf)
                nn = np.argsort(D, axis=1, kind='stable')[:, :min(5, len(ri)-1)]
                correct_majority = (labels[nn] == labels[:, None]).sum(axis=1) >= 3
                retained = [int((correct_majority & g).sum()) for g in groups]
                actual_loss = gamma(Z, labels, weights=original)-gamma(Z, labels, fs, weights=w)
                draws.append({'draw': draw, 'retained_counts': retained,
                    'majority_violation': any(a < b for a, b in zip(retained, required)),
                    'dependency_loss': actual_loss, 'dependency_violation': actual_loss > .1+1e-10})
            rows.append({'dataset': name, 'inner': index, 'features': fs, 'certificate': c['certificate'],
                'required': required, 'protected_counts': [int(g.sum()) for g in groups], 'draws': draws,
                'majority_counterexample': any(d['majority_violation'] for d in draws),
                'dependency_counterexample': any(d['dependency_violation'] for d in draws),
                'prior_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            print(name, index, 'counterexamples', rows[-1]['majority_counterexample'], rows[-1]['dependency_counterexample'], flush=True)
    rejected = [r for r in rows if not r['certificate']['feasible']]
    result = {'rows': rows, 'supports': len(rows), 'rejected_supports': len(rejected),
        'rejected_majority_counterexamples': sum(r['majority_counterexample'] for r in rejected),
        'rejected_dependency_counterexamples': sum(r['dependency_counterexample'] for r in rejected),
        'rejected_with_no_sampled_violation': sum(not r['majority_counterexample'] and not r['dependency_counterexample'] for r in rejected),
        'tables_checked': sum(len(r['draws']) for r in rows), 'seed': 1323,
        'script_sha256': hashlib.sha256((ROOT/'scripts/diagnose_guarded_actual_violations.py').read_bytes()).hexdigest(),
        'scope': 'actual_finite_training_counterexample_search_absence_is_not_a_certificate_or_test_risk_bound'}
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
