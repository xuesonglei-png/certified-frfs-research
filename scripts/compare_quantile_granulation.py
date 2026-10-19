"""Same-geometry nested baseline comparison on all exposed development tasks."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from compare_trajectory_recent import GRIDS, SELECTORS, score, utility
from certified_frfs.datasets import load_dataset
from certified_frfs.feasible_query_descent import fit_feasible_query
from certified_frfs.trajectory_certificate import certify_capped_trajectory
from certified_frfs.capped_sparse import sparse_trajectory_extension
from certified_frfs.validated_pool import preprocess_fit, validation_scores

FOLDER = ROOT/'results/quantile-granulation'


def fingerprints(name):
    paths = [Path(__file__), ROOT/'scripts/compare_trajectory_recent.py',
             ROOT/'research/quantile-granulation-protocol.md',
             ROOT/'results/long-capped-sparse'/f'{name}.json', ROOT/'data/processed'/f'{name}.npz']
    paths.extend(sorted((ROOT/'src/certified_frfs').glob('*.py')))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(name):
    threadpool_limits(1); hashes = fingerprints(name); out = FOLDER/f'{name}.json'
    if out.exists():
        result = json.loads(out.read_text()); assert result['fingerprints'] == hashes
        return
    prior = json.loads((ROOT/'results/long-capped-sparse'/f'{name}.json').read_text())
    X, y = load_dataset(name); rows = []
    for index, split in enumerate(prior['splits']):
        ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
        start = time.perf_counter(); raw, labels = X[ri], y[ri]
        trajectory = fit_feasible_query(raw, labels, steps=10, certified=False, weight_cap_ratio=2, geometry='quantile')
        certified = certify_capped_trajectory(raw, labels, trajectory, geometry='quantile')
        fit = sparse_trajectory_extension(raw, labels, certified, geometry='quantile')
        seconds = time.perf_counter()-start
        A, B = preprocess_fit(raw, X[vi], geometry='quantile'); d = X.shape[1]
        candidates = {'proposed': fit['compact_selected'], 'dense': certified['selected'],
                      'unconstrained': fit['unconstrained_selected'],
                      'full': {'features': list(range(d)), 'weights': (np.ones(d)/np.sqrt(d)).tolist()}}
        metrics = {m: validation_scores(A, labels, B, y[vi], c['features'], np.array(c['weights']), 144+index, joint=True)
                   for m, c in candidates.items()}
        baselines = {}
        for method, grid in GRIDS.items():
            begin = time.perf_counter(); trace = [{'config': c, 'folds': []} for c in grid]
            for j, inner in enumerate(trajectory['training_splits']):
                ti, qi = np.array(inner['reference']), np.array(inner['query'])
                assert set(ti).isdisjoint(qi) and len(set(ti).union(qi)) == len(ri)
                Ai, Bi = preprocess_fit(raw[ti], raw[qi], geometry='quantile')
                for candidate in trace:
                    selected = SELECTORS[method](Ai, labels[ti], **candidate['config'])
                    values = score(Ai, labels[ti], Bi, labels[qi], selected['features'], 5000+j)
                    candidate['folds'].append({'features': selected['features'], 'metrics': values})
            for candidate in trace:
                candidate['utility'] = float(np.mean([utility(f['metrics']) for f in candidate['folds']]))
                candidate['mean_size'] = float(np.mean([len(f['features']) for f in candidate['folds']]))
            chosen = max(range(len(trace)), key=lambda j: (trace[j]['utility'], -trace[j]['mean_size'], -j))
            selected = SELECTORS[method](A, labels, **trace[chosen]['config'])
            metrics[method] = score(A, labels, B, y[vi], selected['features'], 144+index)
            baselines[method] = {'trace': trace, 'chosen_index': chosen, 'selection': selected,
                                 'seconds': time.perf_counter()-begin}
        rows.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
                     'trajectory': trajectory, 'certified_trajectory': certified, 'fit': fit,
                     'metrics': metrics, 'baselines': baselines, 'seconds': seconds,
                     'old_minmax_metrics': split['metrics']})
        print(name, index, {m: round(utility(v), 5) for m, v in metrics.items()},
              'size', len(fit['compact_selected']['features']), flush=True)
    assert fingerprints(name) == hashes
    out.write_text(json.dumps({'dataset': name, 'd': X.shape[1], 'fingerprints': hashes,
                              'splits': rows, 'geometry': 'quantile'}, indent=2), encoding='utf-8')


def summarize(names):
    records = []; methods = ('proposed', 'dense', 'unconstrained', 'full', *GRIDS)
    for name in names:
        p = json.loads((FOLDER/f'{name}.json').read_text()); assert p['fingerprints'] == fingerprints(name)
        s = p['splits']
        records.append({'dataset': name,
            'means': {m: {k: float(np.mean([r['metrics'][m][k] for r in s])) for k in ('clean', 'noise01')} for m in methods},
            'utility': {m: float(np.mean([utility(r['metrics'][m]) for r in s])) for m in methods},
            'removed': float(np.mean([1-len(r['fit']['compact_selected']['features'])/p['d'] for r in s])),
            'clean_gain_vs_old_minmax': float(np.mean([r['metrics']['proposed']['clean']-r['old_minmax_metrics']['clean'] for r in s]))})
    comparisons = {}
    for m in methods[1:]:
        delta = np.array([r['utility']['proposed']-r['utility'][m] for r in records])
        comparisons[m] = {'gain': float(delta.mean()), 'wins': int(sum(delta > 1e-12)),
                          'descriptive_wilcoxon_raw': float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.}
    running = 0.
    for rank, m in enumerate(sorted(comparisons, key=lambda m: comparisons[m]['descriptive_wilcoxon_raw'])):
        running = max(running, min(1., (len(comparisons)-rank)*comparisons[m]['descriptive_wilcoxon_raw']))
        comparisons[m]['descriptive_wilcoxon_holm'] = running
    removed = float(np.mean([r['removed'] for r in records]))
    clean_gain = float(np.mean([r['clean_gain_vs_old_minmax'] for r in records]))
    result = {'rows': records, 'comparisons': comparisons,
              'means': {m: float(np.mean([r['utility'][m] for r in records])) for m in methods},
              'mean_removed': removed, 'mean_clean_gain_vs_old_minmax': clean_gain,
              'gate_passed': all(comparisons[m]['gain'] >= .005 and comparisons[m]['wins'] >= 10 for m in GRIDS)
                  and removed >= .1 and clean_gain >= .005,
              'scope': 'all15_all44_same_quantile_geometry_exposed_development_not_confirmation'}
    assert all(r['fit']['compact_selected']['certificate']['feasible'] for name in names
               for r in json.loads((FOLDER/f'{name}.json').read_text())['splits'])
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    names = [r['dataset'] for r in json.loads((ROOT/'results/long-capped-sparse/complete.json').read_text())['rows']]
    FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, n) for n in names]):
            future.result()
    summarize(names)
