"""Unchanged original-metric certificates on every query-refitted inner metric."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import gamma


def main():
    threadpool_limits(1); folder = ROOT/'results/query-majority-diagnostic'
    source = json.loads((folder/'all-inner-fixed-support.json').read_text()); rows = []; sampled = 0
    for r in source['rows']:
        p = json.loads((ROOT/'results/predictive-iteration41'/f"{r['dataset']}-s101-f0.json").read_text())
        X, y = load_dataset(r['dataset']); tr = np.array(p['train_indices']); candidates = []
        for s in r['splits']:
            ids = tr[np.array(s['train'])]; A, _ = preprocess_fit(X[ids], X[ids]); full = np.ones(A.shape[1])/np.sqrt(A.shape[1])
            for c in s['candidates']:
                w = np.array(c['weights'])*full.sum(); fs = c['features']; certificate = weight_certificate(A, y[ids], fs, full, w[fs])
                rng = np.random.default_rng(1017); losses = []
                for _ in range(5):
                    Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
                    losses.append(gamma(Z, y[ids], weights=full)-gamma(Z, y[ids], fs, weights=w))
                assert max(losses) <= certificate['dependency_certificate']['upper_bound']+1e-10; sampled += 5
                candidates.append({'inner': s['inner'], 'config': c['config'], 'features': fs,
                    'certificate': certificate, 'sampled_losses': losses, 'utility_delta': c['utility_delta']})
        rows.append({'dataset': r['dataset'], 'candidates': candidates}); print(r['dataset'], sum(c['certificate']['feasible'] for c in candidates), '/', len(candidates), flush=True)
    out = folder/'all-inner-certificates.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps({'rows': rows, 'sampled_loss_checks': sampled, 'observed_violations': 0,
        'scope': 'finite_reference_table_query_metric_certification_not_test_risk'}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
