"""Development iteration 4: decision coverage with aligned L1 1NN.

All search is fit on training folds; outer tests are diagnostic, not confirmation.
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
sys.path.insert(0, str(ROOT/'src'))
from run_predictive_certified_iteration import scaled_split
from certified_frfs import baselines
from certified_frfs.decision_coverage import certified_decision_coverage, select_decision_coverage
from certified_frfs.recent import select_sfrip


def configs(method):
    if method.endswith('_knn_majority'):
        return [{'rank':r,'penalty':p} for r in ['mi','relief'] for p in [0.,.02,.05]]
    if method.endswith('_screen'):
        return [{'rank':r,'tolerance':t} for r in ['mi','relief'] for t in [0.,.01,.02]]
    if method.endswith('_constrained'):
        return [{'rank':r, 'constraint_tolerance':t, 'penalty':.02}
                for r in ['mi','relief'] for t in [0.,.01,.02]]
    if method.endswith('_crossfit') or method.endswith('_constrained'):
        return [{'rank': r, 'penalty': p} for r in ['mi', 'relief'] for p in [0., .02, .05]]
    if method.endswith('_balanced_query'):
        return [{'rank': r, 'fraction': f, 'certificate_weight': .25}
                for r in ['mi', 'relief'] for f in [.2, .35, .5, .65, .8, 1.]]
    if method in ['decision_query_aware', 'decision_nominal_query_aware']:
        return [{'rank': r, 'fraction': f, 'certificate_weight': w}
                for r in ['mi', 'relief'] for f in [.2, .35, .5, .65, .8, 1.]
                for w in [.15, .25, .4]]
    if method.endswith('_local'):
        return [{'rank': r, 'penalty': .02, 'margin_scale': s}
                for r in ['mi', 'relief'] for s in [.02, .05, .1]]
    if method in ['decision_margin', 'decision_nominal_margin']:
        return [{'rank': r, 'penalty': .02, 'margin_scale': s}
                for r in ['mi', 'relief'] for s in [.02, .05, .1]]
    if method in ['decision_robust', 'decision_nominal']:
        return [{'rank': r, 'penalty': p} for r in ['mi', 'relief'] for p in [0., .02, .05]]
    if method == 'sfrip':
        return [{'beta': b, 'confidence': c} for b in [2., 3., 5.] for c in [.6, .8]]
    if method == 'all':
        return [{}]
    return [{'fraction': f} for f in [.2, .35, .5, .65, .8, 1.]]


def subset(A, y, method, config, seed, cache=None, raw=None):
    if method.endswith('_knn_majority'):
        from certified_frfs.knn_decision import select_knn_majority
        relevance=(baselines.mi_scores(A,y,random_state=seed) if config['rank']=='mi'
                   else baselines.relieff_scores(A,y,random_state=seed))
        ranking=np.argsort(-relevance,kind='stable').tolist()
        return select_knn_majority(A,y,ranking,eps=.01 if method=='decision_robust_knn_majority' else 0.,
                                   penalty=config['penalty'])['features']
    if method.endswith('_screen'):
        from certified_frfs.independent_screen import select_independent_screen
        if raw is None:
            raise ValueError('screen requires raw training table')
        mode={'decision_robust_screen':'robust','decision_nominal_screen':'nominal',
              'decision_compression_screen':'accuracy'}[method]
        result=select_independent_screen(raw,y,mode=mode,seed=seed,**config)
        if cache is not None:
            key=(method,seed,hashlib.sha256(raw.tobytes()+y.tobytes()).hexdigest()+json.dumps(config,sort_keys=True))
            cache[key]=result
        return result['features']
    if method.endswith('_crossfit') or method.endswith('_constrained'):
        from certified_frfs.crossfit_search import select_crossfit_queries
        if raw is None:
            raise ValueError('crossfit search requires raw fold training table')
        mode = {'decision_robust_crossfit':'robust', 'decision_nominal_crossfit':'nominal',
                'decision_accuracy_crossfit':'accuracy', 'decision_robust_constrained':'robust',
                'decision_nominal_constrained':'nominal', 'decision_compression_constrained':'accuracy'}[method]
        result = select_crossfit_queries(raw, y, seed=seed, mode=mode, **config)
        if cache is not None:
            key = (method, seed, hashlib.sha256(raw.tobytes()+y.tobytes()).hexdigest()+json.dumps(config,sort_keys=True))
            cache[key] = result
        return result['features']
    if method in ['decision_query_aware', 'decision_nominal_query_aware'] or method.endswith('_balanced_query'):
        values = (baselines.mi_scores(A, y, random_state=seed) if config['rank'] == 'mi'
                  else baselines.relieff_scores(A, y, random_state=seed))
        ranking = np.argsort(-values, kind='stable').tolist()
        k = max(1, int(np.ceil(config['fraction']*A.shape[1])))
        return ranking[:k]
    if method.endswith('_stable'):
        from certified_frfs.decision_stability import stability_ranking
        key = (method, seed, hashlib.sha256(A.tobytes()+y.tobytes()).hexdigest())
        cache = {} if cache is None else cache
        if key not in cache:
            mode = {'decision_robust_stable': 'robust', 'decision_nominal_stable': 'nominal',
                    'relief_stable': 'relief'}[method]
            cache[key] = stability_ranking(A, y, seed=seed, mode=mode)
        k = max(1, int(np.ceil(config['fraction']*A.shape[1])))
        return cache[key]['ranking'][:k]
    if method == 'all':
        return list(range(A.shape[1]))
    if method == 'sfrip':
        return select_sfrip(A, y, **config)['features']
    if method.startswith('decision_'):
        values = (baselines.mi_scores(A, y, random_state=seed) if config['rank'] == 'mi'
                  else baselines.relieff_scores(A, y, random_state=seed))
        ranking = np.argsort(-values, kind='stable').tolist()
        if method.endswith('_local'):
            from certified_frfs.decision_search import select_local_decisions
            return select_local_decisions(A, y, ranking,
                eps=.01 if method == 'decision_robust_local' else 0.,
                margin_scale=config['margin_scale'], penalty=config['penalty'],
                budget=24, accuracy_only=method == 'decision_accuracy_local')['features']
        return select_decision_coverage(A, y, ranking,
            eps=.01 if method in ['decision_robust', 'decision_margin'] else 0.,
            penalty=config['penalty'], margin_scale=config.get('margin_scale'))['features']
    functions = {'mi': baselines.select_mi, 'relief': baselines.select_relieff,
                 'mrmr': baselines.select_mrmr, 'maf_instance': baselines.select_mafrfs_additive}
    return functions[method](A, y, k=max(1, int(np.ceil(config['fraction']*A.shape[1]))))


def scores(A, y, B, labels, fs, seed, k=1):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score
    model = (KNeighborsClassifier(n_neighbors=k, metric='manhattan', algorithm='brute')
             if fs else DummyClassifier(strategy='prior'))
    model.fit(A[:, fs] if fs else np.zeros((len(A), 1)), y)
    rng = np.random.default_rng(seed)
    result = {}
    for eps in [0., .01, .05]:
        versions = [B] if eps == 0 else [np.clip(B+rng.uniform(-eps, eps, B.shape), 0, 1) for _ in range(3)]
        metrics = []
        for Z in versions:
            pred = model.predict(Z[:, fs] if fs else np.zeros((len(Z), 1)))
            metrics.append([balanced_accuracy_score(labels, pred),
                            f1_score(labels, pred, average='macro', zero_division=0)])
        ba, f1 = np.mean(metrics, axis=0)
        result[str(eps)] = {'balanced_accuracy': float(ba), 'macro_f1': float(f1)}
    return result


def query_aware_score(A, y, B, labels, fs, seed, certificate_weight=.25, eps=.01,
                      balanced=False):
    """Validation score coupling 1NN BA with held-out certified correctness."""
    from sklearn.neighbors import KNeighborsClassifier
    from certified_frfs.query_certificate import certify_queries
    model = KNeighborsClassifier(n_neighbors=1, metric='manhattan', algorithm='brute').fit(A[:, fs], y)
    pred = model.predict(B[:, fs])
    cert = certify_queries(A, y, B, pred, fs, eps=eps)
    correct = pred == labels
    certified_correct = float(np.mean(np.asarray(cert['certified']) & correct))
    if balanced:
        protected_correct = np.asarray(cert['certified']) & correct
        certified_correct = float(np.mean([protected_correct[labels == c].mean() for c in np.unique(labels)]))
    from sklearn.metrics import balanced_accuracy_score
    ba = balanced_accuracy_score(labels, pred)
    return float((1-certificate_weight)*ba + certificate_weight*certified_correct), certified_correct


def run(job):
    from sklearn.model_selection import StratifiedKFold
    from threadpoolctl import threadpool_limits
    from certified_frfs.datasets import load_dataset
    threadpool_limits(1)
    name, seed, fold, iteration = job
    out = ROOT/f'results/predictive-iteration{iteration}'/f'{name}-s{seed}-f{fold}.json'
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()+b''.join(
        p.read_bytes() for p in sorted((ROOT/'src/certified_frfs').glob('*.py')))).hexdigest()
    if out.exists():
        if json.loads(out.read_text())['code_sha256'] != fingerprint:
            raise ValueError('stale results; preserve them and choose a new directory')
        return {'dataset': name, 'status': 'cached'}
    X, y = load_dataset(name)
    nf = min(3, np.unique(y, return_counts=True)[1].min())
    tr, te = list(StratifiedKFold(n_splits=nf, shuffle=True, random_state=seed).split(X, y))[fold]
    ni = min(3, np.unique(y[tr], return_counts=True)[1].min())
    if ni < 2:
        return {'dataset': name, 'status': 'insufficient_inner_counts'}
    splits = list(StratifiedKFold(n_splits=ni, shuffle=True, random_state=seed+1).split(X[tr], y[tr]))
    rows = []
    stability_cache = {}
    decision_methods = {4: ['decision_robust', 'decision_nominal'],
                        5: ['decision_margin', 'decision_nominal_margin'],
                        6: ['decision_robust_local', 'decision_nominal_local', 'decision_accuracy_local'],
                        7: ['decision_robust_stable', 'decision_nominal_stable', 'relief_stable'],
                        8: ['decision_query_aware', 'decision_nominal_query_aware'],
                        9: ['decision_robust_balanced_query', 'decision_nominal_balanced_query',
                            'decision_accuracy_balanced_query'],
                        10: ['decision_robust_crossfit', 'decision_nominal_crossfit',
                             'decision_accuracy_crossfit'],
                        11: ['decision_robust_constrained', 'decision_nominal_constrained',
                             'decision_compression_constrained'],
                        12: ['decision_robust_constrained', 'decision_nominal_constrained',
                             'decision_compression_constrained'],
                        13: ['decision_robust_screen','decision_nominal_screen',
                             'decision_compression_screen'],
                        14: ['decision_robust_knn_majority','decision_nominal_knn_majority']}[iteration]
    classifier_k=5 if iteration==14 else 1
    for method in ['all', *decision_methods, 'sfrip', 'mi', 'relief', 'mrmr', 'maf_instance']:
        start = time.perf_counter()
        candidates = []
        for config in configs(method):
            vals, sizes, inner_protected = [], [], []
            for it, iv in splits:
                A, B = scaled_split(X[tr][it], X[tr][iv])
                fs = subset(A, y[tr][it], method, config, seed, stability_cache, raw=X[tr][it])
                if method.endswith('_query_aware') or (method.endswith('_balanced_query') and
                                                     method != 'decision_accuracy_balanced_query'):
                    value, certified_correct = query_aware_score(
                        A, y[tr][it], B, y[tr][iv], fs, seed+2,
                        certificate_weight=config['certificate_weight'],
                        eps=0.01 if method in ['decision_query_aware', 'decision_robust_balanced_query'] else 0.,
                        balanced=method.endswith('_balanced_query'))
                    m = scores(A, y[tr][it], B, y[tr][iv], fs, seed+2,k=classifier_k)
                    vals.append(value)
                else:
                    m = scores(A, y[tr][it], B, y[tr][iv], fs, seed+2,k=classifier_k)
                    certified_correct = None
                    vals.append((m['0.0']['balanced_accuracy']+m['0.01']['balanced_accuracy'])/2)
                sizes.append(len(fs))
                inner_protected.append(certified_correct)
            candidates.append({'config': config, 'score': float(np.mean(vals)), 'inner_scores': vals,
                               'inner_certified_correct': inner_protected,
                               'mean_k': float(np.mean(sizes))})
        chosen = max(candidates, key=lambda c: (c['score'], -c['mean_k']))
        A, B = scaled_split(X[tr], X[te])
        fs = subset(A, y[tr], method, chosen['config'], seed, stability_cache, raw=X[tr])
        m = scores(A, y[tr], B, y[te], fs, seed+3,k=classifier_k)
        if iteration==14:
            from certified_frfs.knn_decision import certified_knn_coverage
            coverage=certified_knn_coverage(A,y[tr],fs,k=classifier_k) if fs else None
        else:
            coverage = certified_decision_coverage(A, y[tr], fs) if fs else None
        rows.append({'method': method, 'features': fs, 'config': chosen['config'],
                     'inner_candidates': candidates, 'metrics': m, 'coverage': coverage,
                     'seconds': time.perf_counter()-start})
        print(json.dumps({'dataset': name, 'fold': fold, 'method': method, 'k': len(fs),
                          'clean': m['0.0']['balanced_accuracy']}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'dataset': name, 'seed': seed, 'fold': fold, 'd': X.shape[1],
        'rows': rows, 'train_indices': tr.tolist(), 'test_indices': te.tolist(),
        'code_sha256': fingerprint, 'status': 'development_not_confirmatory',
        'stability_records': [{'method': key[0], 'training_sha256': key[2], **value}
                              for key, value in stability_cache.items()],
        'classifier': f'L1_{classifier_k}NN', 'certificate_scope': 'finite_training_table_LOO_not_test_accuracy'}, indent=2), encoding='utf-8')
    return {'dataset': name, 'status': 'complete'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['iris', 'wine', 'sonar', 'wdbc', 'ionosphere'])
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--iteration', type=int, choices=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], default=4)
    ap.add_argument('--all-folds', action='store_true')
    args = ap.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run, (n, args.seed, fold, args.iteration)) for n in args.datasets
                for fold in (range(3) if args.all_folds else [args.fold])]
        for future in as_completed(jobs):
            print(json.dumps(future.result()), flush=True)
