"""Run all44 unconstrained fits against immutable ten-second constrained fits."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_guarded_support import NAMES
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit, validation_scores
from certified_frfs.feasible_query_descent import fit_feasible_query


def main():
    threadpool_limits(1); folder = ROOT/'results/feasible-query-ablation'; folder.mkdir(exist_ok=True)
    out = folder/'complete.json'
    if out.exists(): raise FileExistsError(out)
    paths = sorted((ROOT/'src/certified_frfs').glob('*.py'))+[Path(__file__), ROOT/'research/feasible-query-ablation-protocol.md']
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows = []; utility = lambda m: (m['clean']+m['noise01'])/2
    for name in NAMES:
        prior = ROOT/'results/feasible-query-long-diagnostic'/f'{name}.json'; p = json.loads(prior.read_text())
        X, y = load_dataset(name); splits = []
        for index, s in enumerate(p['splits']):
            ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices']); A, B = preprocess_fit(X[ri], X[vi])
            started = time.perf_counter(); fit = fit_feasible_query(X[ri], y[ri], certified=False, oracle_seconds=10.)
            elapsed = time.perf_counter()-started
            assert fit['training_splits'] == s['fit']['training_splits']
            scores = validation_scores(A, y[ri], B, y[vi], fit['features'], np.array(fit['weights']), 144+index, joint=True)
            splits.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(), 'fit': fit,
                'metrics': scores, 'certified_metrics': s['metrics'], 'reference_metrics': s['reference_metrics'], 'seconds': elapsed})
        means = {'unconstrained': float(np.mean([utility(s['metrics']) for s in splits])),
            'certified': float(np.mean([utility(s['certified_metrics']) for s in splits]))}
        means.update({m: float(np.mean([utility(s['reference_metrics'][m]) for s in splits])) for m in ('full', 'fixed_majority41')})
        rows.append({'dataset': name, 'd': p['d'], 'splits': splits, 'means': means,
            'prior_sha256': hashlib.sha256(prior.read_bytes()).hexdigest(),
            'data_sha256': hashlib.sha256((ROOT/'data/processed'/f'{name}.npz').read_bytes()).hexdigest()})
        print(name, means, flush=True)
    means = {m: float(np.mean([r['means'][m] for r in rows])) for m in rows[0]['means']}
    result = {'rows': rows, 'means': means, 'source_hashes': source_hashes,
        'certified_positive_vs_unconstrained': sum(r['means']['certified'] > r['means']['unconstrained']+1e-12 for r in rows),
        'unconstrained_positive_vs_full': sum(r['means']['unconstrained'] > r['means']['full']+1e-12 for r in rows),
        'scope': 'same_objective_callbudget_all_inner_development_ablation_not_confirmation'}
    assert source_hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'source_hashes')}, indent=2), flush=True)


if __name__ == '__main__': main()
