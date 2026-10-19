"""Fixed local certified deletion development screen."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import hashlib
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit, validation_scores
from certified_frfs.certified_cap_backward import certified_cap_backward

FOLDER = ROOT/'results/certified-cap-backward'


def fingerprints(name):
    paths = [Path(__file__), ROOT/'research/certified-cap-backward-protocol.md',
        ROOT/'results/capped-sparse-extension'/f'{name}.json', ROOT/'data/processed'/f'{name}.npz']
    paths.extend(sorted((ROOT/'src/certified_frfs').glob('*.py')))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(name):
    threadpool_limits(1); hashes = fingerprints(name); out = FOLDER/f'{name}.json'
    if out.exists():
        assert json.loads(out.read_text())['fingerprints'] == hashes
        return
    prior = json.loads((ROOT/'results/capped-sparse-extension'/f'{name}.json').read_text())
    X, y = load_dataset(name); rows = []
    for index, s in enumerate(prior['splits']):
        ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
        start = time.perf_counter(); fit = certified_cap_backward(X[ri], y[ri], s['fit']); elapsed = time.perf_counter()-start
        A, B = preprocess_fit(X[ri], X[vi]); c = fit['selected']
        metrics = validation_scores(A, y[ri], B, y[vi], c['features'], np.array(c['weights']), 144+index, joint=True)
        rows.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
            'fit': fit, 'metrics': metrics, 'incremental_seconds': elapsed})
        print(name, index, metrics, 'removed', len(s['fit']['compact_selected']['features'])-len(c['features']), flush=True)
    assert fingerprints(name) == hashes
    out.write_text(json.dumps({'dataset': name, 'd': X.shape[1], 'splits': rows, 'fingerprints': hashes}, indent=2), encoding='utf-8')


def summarize(names):
    prior = json.loads((ROOT/'results/capped-sparse-extension/complete.json').read_text()); rows = []
    for name in names:
        p = json.loads((FOLDER/f'{name}.json').read_text()); assert p['fingerprints'] == fingerprints(name)
        base = next(r for r in prior['rows'] if r['dataset']==name)
        means = {k:float(np.mean([s['metrics'][k] for s in p['splits']])) for k in ('clean', 'noise01')}
        rows.append({'dataset':name, 'utility':(means['clean']+means['noise01'])/2, 'metrics':means,
            'removed':float(np.mean([1-len(s['fit']['selected']['features'])/p['d'] for s in p['splits']])),
            'control':base['utility'], 'incremental_seconds':float(np.mean([s['incremental_seconds'] for s in p['splits']]))})
    mean = float(np.mean([r['utility'] for r in rows]))
    result = {'rows': rows, 'mean_utility':mean, 'mean_removed':float(np.mean([r['removed'] for r in rows])),
        'comparisons': {m:{'gain':float(np.mean([r['utility']-r['control'][m] for r in rows])),
            'wins':sum(r['utility']>r['control'][m]+1e-12 for r in rows)} for m in ('compact', 'fgs2023', 'sfrip2025')},
        'scope':'all15_all44_exposed_development_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2), flush=True)


if __name__ == '__main__':
    names = [r['dataset'] for r in json.loads((ROOT/'results/capped-sparse-extension/complete.json').read_text())['rows']]
    FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]): f.result()
    summarize(names)
