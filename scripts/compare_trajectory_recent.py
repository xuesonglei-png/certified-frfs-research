"""Training-only nested recent baselines on all exposed trajectory splits."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import wilcoxon
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.graph_recent import select_fgs
from certified_frfs.recent import select_sfrip
from certified_frfs.validated_pool import preprocess_fit, validation_scores

FOLDER = ROOT/'results/trajectory-recent-comparison'
GRIDS = {
    'fgs2023': [{'beta': b, 'xi': x} for b in (.6, .8, .9) for x in (.5, 1., 2.)],
    'sfrip2025': [{'beta': b, 'confidence': c} for b in (2., 3., 5.) for c in (.6, .8)]}
SELECTORS = {'fgs2023': select_fgs, 'sfrip2025': select_sfrip}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(name):
    paths = [Path(__file__), ROOT/'research/trajectory-recent-comparison-protocol.md',
        ROOT/'src/certified_frfs/graph_recent.py', ROOT/'src/certified_frfs/recent.py',
        ROOT/'src/certified_frfs/validated_pool.py', ROOT/'src/certified_frfs/datasets.py',
        ROOT/'results/trajectory-certificate-diagnostic'/f'{name}.json',
        ROOT/'data/processed'/f'{name}.npz']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def utility(metrics):
    return (metrics['clean']+metrics['noise01'])/2


def score(A, y, B, truth, fs, seed):
    if fs:
        return validation_scores(A, y, B, truth, fs, np.ones(A.shape[1])/np.sqrt(A.shape[1]), seed, joint=True)
    model = DummyClassifier(strategy='prior').fit(np.zeros((len(y), 1)), y)
    value = float(balanced_accuracy_score(truth, model.predict(np.zeros((len(truth), 1)))))
    return {'clean': value, 'noise01': value}


def run(name):
    threadpool_limits(1)
    out = FOLDER/f'{name}.json'; hashes = fingerprints(name)
    if out.exists():
        cached = json.loads(out.read_text())
        if cached['fingerprints'] != hashes:
            raise ValueError(f'stale comparison cache: {name}')
        return cached
    prior = json.loads((ROOT/'results/trajectory-certificate-diagnostic'/f'{name}.json').read_text())
    X, y = load_dataset(name); rows = []
    for index, split in enumerate(prior['splits']):
        ri = np.array(split['reference_indices']); vi = np.array(split['validation_indices'])
        raw, labels = X[ri], y[ri]
        folds = split['fit']['training_splits']; baselines = {}
        for method, grid in GRIDS.items():
            start = time.perf_counter(); trace = [{'config': c, 'folds': []} for c in grid]
            for j, inner in enumerate(folds):
                ti, qi = np.array(inner['reference']), np.array(inner['query'])
                assert not set(ti).intersection(qi)
                assert len(set(ti).union(qi)) == len(ri)
                A, B = preprocess_fit(raw[ti], raw[qi])
                for candidate in trace:
                    selected = SELECTORS[method](A, labels[ti], **candidate['config'])
                    fs = selected['features']; metrics = score(A, labels[ti], B, labels[qi], fs, 5000+j)
                    candidate['folds'].append({'features': fs, 'metrics': metrics})
            for candidate in trace:
                candidate['utility'] = float(np.mean([utility(f['metrics']) for f in candidate['folds']]))
                candidate['mean_size'] = float(np.mean([len(f['features']) for f in candidate['folds']]))
            chosen = max(range(len(trace)), key=lambda j: (trace[j]['utility'], -trace[j]['mean_size'], -j))
            A, B = preprocess_fit(raw, X[vi])
            selected = SELECTORS[method](A, labels, **trace[chosen]['config'])
            elapsed = time.perf_counter()-start
            metrics = score(A, labels, B, y[vi], selected['features'], 144+index)
            baselines[method] = {'trace': trace, 'chosen_index': chosen, 'selection': selected,
                'metrics': metrics, 'nested_selection_seconds': elapsed}
            print(name, index, method, metrics, 'size', len(selected['features']), 'seconds', round(elapsed, 2), flush=True)
        rows.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(), 'baselines': baselines,
            'proposed_metrics': split['metrics']['certified'],
            'proposed_features': split['fit']['selected']['features']})
    assert fingerprints(name) == hashes
    result = {'dataset': name, 'd': X.shape[1], 'splits': rows, 'fingerprints': hashes,
        'scope': 'training_only_tuned_independent_recent_implementations_exposed_development'}
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def summarize(names):
    rows = []
    for name in names:
        p = json.loads((FOLDER/f'{name}.json').read_text()); assert p['fingerprints'] == fingerprints(name)
        metrics = {'proposed': [s['proposed_metrics'] for s in p['splits']]}
        metrics.update({m: [s['baselines'][m]['metrics'] for s in p['splits']] for m in GRIDS})
        sizes = {'proposed': [len(s['proposed_features']) for s in p['splits']]}
        sizes.update({m: [len(s['baselines'][m]['selection']['features']) for s in p['splits']] for m in GRIDS})
        rows.append({'dataset': name, 'd': p['d'], 'means': {m: {k: float(np.mean([v[k] for v in values]))
            for k in ('clean', 'noise01')} for m, values in metrics.items()},
            'utility': {m: float(np.mean([utility(v) for v in values])) for m, values in metrics.items()},
            'removed': {m: 1-float(np.mean(v))/p['d'] for m, v in sizes.items()},
            'seconds': {m: float(np.mean([s['baselines'][m]['nested_selection_seconds'] for s in p['splits']])) for m in GRIDS}})
    comparisons = {}
    for m in GRIDS:
        delta = np.array([r['utility']['proposed']-r['utility'][m] for r in rows])
        comparisons[m] = {'gain': float(delta.mean()), 'wins': int(sum(delta > 1e-12)),
            'ties': int(sum(abs(delta) <= 1e-12)), 'losses': int(sum(delta < -1e-12)),
            'descriptive_wilcoxon_raw': float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.,
            'leave_one_dataset_out_gain_range': [float(min((delta.sum()-delta)/(len(delta)-1))),
                float(max((delta.sum()-delta)/(len(delta)-1)))] if len(delta)>1 else None}
    running = 0.
    for rank, m in enumerate(sorted(comparisons, key=lambda m: comparisons[m]['descriptive_wilcoxon_raw'])):
        running = max(running, min(1., (len(comparisons)-rank)*comparisons[m]['descriptive_wilcoxon_raw']))
        comparisons[m]['descriptive_wilcoxon_holm'] = running
    result = {'rows': rows, 'comparisons': comparisons, 'means': {m: float(np.mean([r['utility'][m] for r in rows]))
        for m in ('proposed', *GRIDS)}, 'removed': {m: float(np.mean([r['removed'][m] for r in rows]))
        for m in ('proposed', *GRIDS)}, 'scope': 'exposed_development_descriptive_not_confirmatory'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--datasets', nargs='+'); args = parser.parse_args()
    names = args.datasets or [r['dataset'] for r in json.loads((ROOT/'results/trajectory-certificate-diagnostic/complete-analysis.json').read_text())['rows']]
    FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]): future.result()
    summarize(names)
