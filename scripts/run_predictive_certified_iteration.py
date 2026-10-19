"""Development-only nested evaluation of prediction-aware certification.

New outer splits remain development evidence: datasets have already been seen.
Every inner split refits preprocessing AND feature selection. Each comparator
receives six configurations with a fixed common classifier. No test tuning.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def configurations(method):
    if method == 'predictive_certified':
        return [{'eta': eta, 'order': order} for eta in [.03, .06, .10]
                for order in ['mi', 'relief']]
    if method == 'sfrip':
        return [{'beta': beta, 'confidence': confidence}
                for beta in [2., 3., 5.] for confidence in [.6, .8]]
    if method in ['mi', 'mrmr', 'relief', 'maf_instance']:
        return [{'fraction': f} for f in [.2, .35, .5, .65, .8, 1.]]
    return [{}]


def subset(X, y, method, config, seed):
    import numpy as np
    from certified_frfs import baselines
    from certified_frfs.selection import deletion_order, select_certified
    from certified_frfs.recent import select_sfrip
    if method == 'all':
        return list(range(X.shape[1]))
    if method in ['original_certified', 'predictive_certified']:
        order = (np.argsort(baselines.relieff_scores(X, y, random_state=seed),
                           kind='stable').tolist() if config.get('order') == 'relief'
                 else deletion_order(X, y, seed))
        return select_certified(X, y, eps=.01, eta=config.get('eta', .1),
                                order=order)['features']
    if method == 'sfrip':
        return select_sfrip(X, y, **config)['features']
    k = max(1, int(np.ceil(config['fraction'] * X.shape[1])))
    funcs = {'mi': baselines.select_mi, 'mrmr': baselines.select_mrmr,
             'relief': baselines.select_relieff,
             'maf_instance': baselines.select_mafrfs_additive}
    return funcs[method](X, y, k=k)


def scaled_split(Xtr, Xva):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = MinMaxScaler(clip=True)
    return (scaler.fit_transform(imputer.fit_transform(Xtr)),
            scaler.transform(imputer.transform(Xva)))


def prediction_scores(Xtr, ytr, Xva, yva, features, seed):
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score
    if features:
        model = KNeighborsClassifier(n_neighbors=min(5, len(ytr)))
        model.fit(Xtr[:, features], ytr)
    else:
        model = DummyClassifier(strategy='prior').fit(np.zeros((len(ytr), 1)), ytr)
    rng = np.random.default_rng(seed)
    results = {}
    for eps in [0., .01, .05]:
        versions = [Xva] if eps == 0 else [np.clip(Xva + rng.uniform(-eps, eps, Xva.shape), 0, 1)
                                          for _ in range(3)]
        ba, f1 = [], []
        for z in versions:
            pred = model.predict(z[:, features] if features else np.zeros((len(z), 1)))
            ba.append(balanced_accuracy_score(yva, pred))
            f1.append(f1_score(yva, pred, average='macro', zero_division=0))
        results[str(eps)] = {'balanced_accuracy': float(np.mean(ba)),
                             'macro_f1': float(np.mean(f1))}
    return results


def run(job):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from sklearn.model_selection import StratifiedKFold
    from certified_frfs.datasets import load_dataset
    from certified_frfs.core import certificate
    threadpool_limits(1)
    name, seed, fold = job
    out = ROOT / 'results/predictive-iteration1' / f'{name}-s{seed}-f{fold}.json'
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() + b''.join(
        p.read_bytes() for p in sorted((ROOT / 'src/certified_frfs').glob('*.py')))).hexdigest()
    if out.exists():
        cached = json.loads(out.read_text())
        if cached['code_sha256'] != fingerprint:
            raise ValueError('stale development cache')
        return {'dataset': name, 'status': 'cached'}
    X, y = load_dataset(name)
    outer_count = min(3, np.unique(y, return_counts=True)[1].min())
    train, test = list(StratifiedKFold(n_splits=outer_count, shuffle=True,
                                      random_state=seed).split(X, y))[fold]
    inner_count = min(3, np.unique(y[train], return_counts=True)[1].min())
    if inner_count < 2:
        return {'dataset': name, 'status': 'skipped_insufficient_inner_class_count'}
    inner = list(StratifiedKFold(n_splits=inner_count, shuffle=True,
                                 random_state=seed + 1).split(X[train], y[train]))
    rows = []
    for method in ['all', 'original_certified', 'predictive_certified',
                   'sfrip', 'mi', 'mrmr', 'relief', 'maf_instance']:
        start = time.perf_counter()
        candidates = []
        for config in configurations(method):
            scores, sizes = [], []
            for it, iv in inner:
                A, B = scaled_split(X[train][it], X[train][iv])
                features = subset(A, y[train][it], method, config, seed)
                metrics = prediction_scores(A, y[train][it], B, y[train][iv], features, seed + 2)
                scores.append((metrics['0.0']['balanced_accuracy'] +
                               metrics['0.01']['balanced_accuracy']) / 2)
                sizes.append(len(features))
            candidates.append({'config': config, 'inner_scores': scores,
                               'score': float(np.mean(scores)),
                               'mean_k': float(np.mean(sizes))})
        # Objective fixed before outer evaluation; compression breaks ties only.
        chosen = max(candidates, key=lambda c: (c['score'], -c['mean_k']))
        A, B = scaled_split(X[train], X[test])
        features = subset(A, y[train], method, chosen['config'], seed)
        cert = certificate(A, y[train], features, eps=.01)
        metrics = prediction_scores(A, y[train], B, y[test], features, seed + 3)
        row = {'method': method, 'features': features, 'config': chosen['config'],
               'inner_candidates': candidates, 'metrics': metrics,
               'certificate_upper': cert['upper_bound'],
               'seconds': time.perf_counter() - start}
        rows.append(row)
        print(json.dumps({'dataset': name, 'method': method, 'k': len(features),
                          'clean': metrics['0.0']['balanced_accuracy']}), flush=True)
    result = {'dataset': name, 'seed': seed, 'fold': fold, 'd': X.shape[1],
              'train_indices': train.tolist(), 'test_indices': test.tolist(),
              'inner_indices': [[a.tolist(), b.tolist()] for a, b in inner],
              'rows': rows, 'code_sha256': fingerprint,
              'status': 'development_not_confirmatory',
              'selection_objective': 'mean_inner_clean_and_noise_0.01_balanced_accuracy'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return {'dataset': name, 'status': 'complete'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['iris', 'wine', 'sonar', 'wdbc'])
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run, (name, args.seed, args.fold)) for name in args.datasets]
        for f in as_completed(jobs):
            print(json.dumps(f.result()), flush=True)
