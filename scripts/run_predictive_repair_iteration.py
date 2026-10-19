"""Iteration 2 nested development experiment with paired repair ablation.

Six configurations each: MI/Relief rankings x 0.2/0.5/0.8 initial budget.
Unrepaired and repaired procedures independently choose by the same inner CV.
The classifier, preprocessing, outer splits and objective match iteration 1.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from run_predictive_certified_iteration import scaled_split, prediction_scores


def run(job):
    from sklearn.model_selection import StratifiedKFold
    from threadpoolctl import threadpool_limits
    from certified_frfs import baselines
    from certified_frfs.datasets import load_dataset
    from certified_frfs.core import certificate
    from certified_frfs.predictive_repair import repair_to_certificate
    threadpool_limits(1)
    name, seed, fold = job
    paths = [Path(__file__), ROOT / 'scripts/run_predictive_certified_iteration.py',
             *sorted((ROOT / 'src/certified_frfs').glob('*.py'))]
    fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()
    out = ROOT / 'results/predictive-iteration2' / f'{name}-s{seed}-f{fold}.json'
    if out.exists():
        if json.loads(out.read_text())['code_sha256'] != fingerprint:
            raise ValueError('stale cache')
        return {'dataset': name, 'status': 'cached'}
    X, y = load_dataset(name)
    nfold = min(3, np.unique(y, return_counts=True)[1].min())
    train, test = list(StratifiedKFold(n_splits=nfold, shuffle=True,
                                      random_state=seed).split(X, y))[fold]
    ni = min(3, np.unique(y[train], return_counts=True)[1].min())
    if ni < 2:
        return {'dataset': name, 'status': 'insufficient_inner_class_counts'}
    splits = list(StratifiedKFold(n_splits=ni, shuffle=True, random_state=seed+1)
                  .split(X[train], y[train]))
    config_grid = [{'ranking': r, 'fraction': f} for r in ['mi', 'relief']
                   for f in [.2, .5, .8]]
    cache = {}

    def fitted(A, labels, configuration, cache_key, repair):
        key = (cache_key, configuration['ranking'])
        if key not in cache:
            score = (baselines.mi_scores(A, labels, random_state=seed)
                     if configuration['ranking'] == 'mi'
                     else baselines.relieff_scores(A, labels, random_state=seed))
            cache[key] = np.argsort(-score, kind='stable').tolist()
        ranking = cache[key]
        k = max(1, int(np.ceil(configuration['fraction'] * A.shape[1])))
        if repair:
            result = repair_to_certificate(A, labels, ranking, k)
            return result['features'], result
        return ranking[:k], None

    rows = []
    for method, repair in [('ranking_unrepaired', False), ('ranking_certified_repair', True)]:
        start = time.perf_counter()
        candidates = []
        for config in config_grid:
            scores, sizes = [], []
            for i, (it, iv) in enumerate(splits):
                A, B = scaled_split(X[train][it], X[train][iv])
                fs, _ = fitted(A, y[train][it], config, i, repair)
                metrics = prediction_scores(A, y[train][it], B, y[train][iv], fs, seed+2)
                scores.append((metrics['0.0']['balanced_accuracy'] +
                               metrics['0.01']['balanced_accuracy']) / 2)
                sizes.append(len(fs))
            candidates.append({'config': config, 'score': float(np.mean(scores)),
                               'inner_scores': scores, 'mean_k': float(np.mean(sizes))})
        chosen = max(candidates, key=lambda c: (c['score'], -c['mean_k']))
        A, B = scaled_split(X[train], X[test])
        fs, trace = fitted(A, y[train], chosen['config'], 'outer', repair)
        cert = certificate(A, y[train], fs, eps=.01)
        if repair and cert['upper_bound'] > .1 + 1e-10:
            raise AssertionError('repair certificate violated')
        metrics = prediction_scores(A, y[train], B, y[test], fs, seed+3)
        rows.append({'method': method, 'features': fs, 'config': chosen['config'],
                     'inner_candidates': candidates, 'metrics': metrics,
                     'certificate_upper': cert['upper_bound'],
                     'repair_history': trace['history'] if trace else None,
                     'seconds': time.perf_counter()-start})
        print(json.dumps({'dataset': name, 'method': method, 'k': len(fs),
                          'clean': metrics['0.0']['balanced_accuracy'],
                          'upper': cert['upper_bound']}), flush=True)
    result = {'dataset': name, 'seed': seed, 'fold': fold, 'd': X.shape[1],
              'rows': rows, 'train_indices': train.tolist(), 'test_indices': test.tolist(),
              'status': 'development_not_confirmatory', 'code_sha256': fingerprint,
              'inner_indices': [[a.tolist(), b.tolist()] for a, b in splits]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return {'dataset': name, 'status': 'complete'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['iris', 'wine', 'sonar', 'wdbc', 'ionosphere'])
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run, (name, args.seed, args.fold)) for name in args.datasets]
        for f in as_completed(jobs):
            print(json.dumps(f.result()), flush=True)
