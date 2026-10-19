"""Iteration 3: train-only relation weights with nested, matched opportunity.

All selectors share the same train-learned metric family. Fixed nonnegative
weights are never renormalized after deletion. Observed data are scaled first;
the certificate is computed on unweighted coordinates with learned weights.
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


def learned_weights(X, y, power, seed):
    from certified_frfs.baselines import mi_scores
    if power == 0:
        return np.ones(X.shape[1]) / np.sqrt(X.shape[1])
    relevance = mi_scores(X, y, random_state=seed)
    if relevance.max() <= 0:
        return np.ones(X.shape[1]) / np.sqrt(X.shape[1])
    raw = (.05 + .95 * relevance / relevance.max()) ** power
    return raw / np.linalg.norm(raw)


def configurations(method):
    if method == 'certified_metric':
        return [{'power': p, 'eta': e} for p in [0., .5, 1.] for e in [.05, .1]]
    if method == 'sfrip_metric':
        return [{'power': p, 'beta': b} for p in [0., .5, 1.] for b in [2., 5.]]
    if method == 'all_metric':
        return [{'power': p} for p in [0., .5, 1.]]
    return [{'power': p, 'fraction': f} for p in [0., .5, 1.] for f in [.3, .7]]


def choose_subset(X, y, method, config, seed, weights):
    from certified_frfs import baselines
    from certified_frfs.recent import select_sfrip
    from certified_frfs.selection import select_certified
    if method == 'all_metric':
        return list(range(X.shape[1]))
    if method == 'certified_metric':
        order = np.argsort(baselines.relieff_scores(X * weights, y,
                            random_state=seed), kind='stable').tolist()
        return select_certified(X, y, eps=.01, eta=config['eta'],
                                weights=weights, order=order)['features']
    if method == 'sfrip_metric':
        return select_sfrip(X * weights, y, beta=config['beta'])['features']
    k = max(1, int(np.ceil(config['fraction'] * X.shape[1])))
    return {'mi_metric': baselines.select_mi, 'mrmr_metric': baselines.select_mrmr,
            'relief_metric': baselines.select_relieff,
            'maf_metric_instance': baselines.select_mafrfs_additive}[method](X * weights, y, k=k)


def scores(A, labels, B, validation_labels, fs, weights, seed):
    # Generate common perturbations in original [0,1] coordinates FIRST.
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score
    model = (KNeighborsClassifier(n_neighbors=min(5, len(labels))) if fs
             else DummyClassifier(strategy='prior'))
    model.fit(A[:, fs] * weights[fs] if fs else np.zeros((len(A), 1)), labels)
    rng = np.random.default_rng(seed)
    results = {}
    for epsilon in [0., .01, .05]:
        variants = [B] if epsilon == 0 else [np.clip(B + rng.uniform(-epsilon, epsilon, B.shape), 0, 1)
                                            for _ in range(3)]
        values = []
        for variant in variants:
            pred = model.predict(variant[:, fs] * weights[fs] if fs else np.zeros((len(B), 1)))
            values.append([balanced_accuracy_score(validation_labels, pred),
                           f1_score(validation_labels, pred, average='macro', zero_division=0)])
        mean = np.mean(values, axis=0)
        results[str(epsilon)] = {'balanced_accuracy': float(mean[0]), 'macro_f1': float(mean[1])}
    return results


def run(job):
    from sklearn.model_selection import StratifiedKFold
    from threadpoolctl import threadpool_limits
    from certified_frfs.datasets import load_dataset
    from certified_frfs.core import certificate
    threadpool_limits(1)
    name, seed, fold = job
    out = ROOT / 'results/predictive-iteration3' / f'{name}-s{seed}-f{fold}.json'
    paths = [Path(__file__), ROOT / 'scripts/run_predictive_certified_iteration.py',
             *sorted((ROOT / 'src/certified_frfs').glob('*.py'))]
    fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()
    if out.exists():
        if json.loads(out.read_text())['code_sha256'] != fingerprint:
            raise ValueError('stale cache')
        return {'dataset': name, 'status': 'cached'}
    X, y = load_dataset(name)
    nf = min(3, np.unique(y, return_counts=True)[1].min())
    train, test = list(StratifiedKFold(n_splits=nf, shuffle=True, random_state=seed).split(X, y))[fold]
    ni = min(3, np.unique(y[train], return_counts=True)[1].min())
    if ni < 2:
        return {'dataset': name, 'status': 'insufficient_inner_counts'}
    splits = list(StratifiedKFold(n_splits=ni, shuffle=True, random_state=seed+1)
                  .split(X[train], y[train]))
    rows = []
    for method in ['all_metric', 'certified_metric', 'sfrip_metric',
                   'mi_metric', 'mrmr_metric', 'relief_metric', 'maf_metric_instance']:
        start = time.perf_counter()
        candidates = []
        for config in configurations(method):
            vals, sizes = [], []
            for it, iv in splits:
                A, B = scaled_split(X[train][it], X[train][iv])
                w = learned_weights(A, y[train][it], config['power'], seed)
                fs = choose_subset(A, y[train][it], method, config, seed, w)
                metrics = scores(A, y[train][it], B, y[train][iv], fs, w, seed+2)
                vals.append((metrics['0.0']['balanced_accuracy'] + metrics['0.01']['balanced_accuracy']) / 2)
                sizes.append(len(fs))
            candidates.append({'config': config, 'score': float(np.mean(vals)),
                               'inner_scores': vals, 'mean_k': float(np.mean(sizes))})
        chosen = max(candidates, key=lambda c: (c['score'], -c['mean_k']))
        A, B = scaled_split(X[train], X[test])
        w = learned_weights(A, y[train], chosen['config']['power'], seed)
        fs = choose_subset(A, y[train], method, chosen['config'], seed, w)
        cert = certificate(A, y[train], fs, weights=w, eps=.01)
        if method == 'certified_metric' and cert['upper_bound'] > chosen['config']['eta'] + 1e-10:
            raise AssertionError('certificate failed')
        metrics = scores(A, y[train], B, y[test], fs, w, seed+3)
        rows.append({'method': method, 'features': fs, 'weights': w.tolist(),
                     'config': chosen['config'], 'inner_candidates': candidates,
                     'metrics': metrics, 'certificate_upper': cert['upper_bound'],
                     'seconds': time.perf_counter()-start})
        print(json.dumps({'dataset': name, 'method': method, 'k': len(fs),
                          'clean': metrics['0.0']['balanced_accuracy'],
                          'power': chosen['config']['power']}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'dataset': name, 'seed': seed, 'fold': fold, 'd': X.shape[1],
        'train_indices': train.tolist(), 'test_indices': test.tolist(), 'rows': rows,
        'code_sha256': fingerprint, 'status': 'development_not_confirmatory',
        'certificate_scope': 'fixed_training_learned_relation_only_not_original_uniform_relation'}, indent=2), encoding='utf-8')
    return {'dataset': name, 'status': 'complete'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['iris', 'wine', 'sonar', 'wdbc', 'ionosphere'])
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run, (n, args.seed, args.fold)) for n in args.datasets]
        for f in as_completed(jobs):
            print(json.dumps(f.result()), flush=True)
