"""All44 exposed development screen for a fixed certified sparse extension."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit, validation_scores
from certified_frfs.capped_sparse import sparse_trajectory_extension

FOLDER = ROOT/'results/capped-sparse-extension'


def fingerprints(name):
    paths = [Path(__file__), ROOT/'src/certified_frfs/capped_sparse.py',
        ROOT/'research/capped-sparse-extension-protocol.md',
        ROOT/'results/trajectory-certificate-diagnostic'/f'{name}.json',
        ROOT/'data/processed'/f'{name}.npz']
    paths += [ROOT/'src/certified_frfs'/f'{m}.py' for m in
        ('query_probability', 'feasible_transport', 'majority_metric', 'validated_pool', 'core', 'redistributed_loss')
        if (ROOT/'src/certified_frfs'/f'{m}.py').exists()]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(name):
    threadpool_limits(1); out = FOLDER/f'{name}.json'; hashes = fingerprints(name)
    if out.exists():
        p = json.loads(out.read_text()); assert p['fingerprints'] == hashes
        return
    prior = json.loads((ROOT/'results/trajectory-certificate-diagnostic'/f'{name}.json').read_text())
    X, y = load_dataset(name); rows = []
    for index, s in enumerate(prior['splits']):
        ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
        start = time.perf_counter(); fit = sparse_trajectory_extension(X[ri], y[ri], s['fit'])
        elapsed = time.perf_counter()-start; A, B = preprocess_fit(X[ri], X[vi]); metrics = {}
        for branch, key in [('exact', 'selected'), ('compact', 'compact_selected'), ('unconstrained', 'unconstrained_selected')]:
            c = fit[key]; metrics[branch] = validation_scores(A, y[ri], B, y[vi], c['features'], np.array(c['weights']), 144+index, joint=True)
        rows.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(), 'fit': fit,
            'selection_seconds': elapsed, 'metrics': metrics, 'old_metrics': s['metrics']['certified']})
        print(name, index, metrics, 'sizes', len(fit['selected']['features']), len(fit['compact_selected']['features']), 'seconds', round(elapsed, 2), flush=True)
    assert fingerprints(name) == hashes
    out.write_text(json.dumps({'dataset': name, 'd': X.shape[1], 'splits': rows, 'fingerprints': hashes}, indent=2), encoding='utf-8')


def summarize(names):
    utility = lambda m: (m['clean']+m['noise01'])/2
    recent = json.loads((ROOT/'results/trajectory-recent-comparison/complete.json').read_text())
    rows = []
    for name in names:
        p = json.loads((FOLDER/f'{name}.json').read_text()); assert p['fingerprints'] == fingerprints(name)
        base = next(r for r in recent['rows'] if r['dataset'] == name)
        means = {m: float(np.mean([utility(s['metrics'][m]) for s in p['splits']])) for m in ('exact', 'compact', 'unconstrained')}
        means.update(base['utility'])
        removed = {m: float(np.mean([1-len(s['fit'][key]['features'])/p['d'] for s in p['splits']]))
            for m,key in [('exact', 'selected'), ('compact', 'compact_selected')]}
        rows.append({'dataset': name, 'utility': means, 'removed': removed,
            'sparse_candidate_feasible': sum(c['certificate']['feasible'] for s in p['splits'] for c in s['fit']['candidates'] if c['origin']=='capped_sparse')})
    means = {m: float(np.mean([r['utility'][m] for r in rows])) for m in rows[0]['utility']}
    comparisons = {m: {b: {'gain': means[m]-means[b], 'positive_datasets': sum(r['utility'][m]>r['utility'][b]+1e-12 for r in rows)}
        for b in ('proposed', 'fgs2023', 'sfrip2025')} for m in ('exact', 'compact')}
    result = {'rows': rows, 'means': means, 'comparisons': comparisons,
        'removed': {m: float(np.mean([r['removed'][m] for r in rows])) for m in ('exact', 'compact')},
        'scope': 'all15_all44_exposed_development_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2), flush=True)


if __name__ == '__main__':
    names = [r['dataset'] for r in json.loads((ROOT/'results/trajectory-certificate-diagnostic/complete-analysis.json').read_text())['rows']]
    FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, n) for n in names]): future.result()
    summarize(names)
