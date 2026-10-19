"""Frozen complete45-task development expansion; same-objective ablation."""
import argparse, json, hashlib, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.sparse_majority import sparse_majority_candidates
from certified_frfs.repaired_majority import repair_majority_metric
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.validated_pool import preprocess_fit, validation_scores


DATASETS = ['banknote', 'ecoli', 'glass', 'haberman', 'heart_statlog', 'ionosphere',
            'iris', 'seeds', 'sonar', 'transfusion', 'vehicle', 'vertebral', 'wdbc', 'wine', 'yeast']
METHODS = ['certified_repaired', 'unconstrained_majority', 'unlimited_majority']


def candidates(A, y):
    d = A.shape[1]; full = np.ones(d)/np.sqrt(d); mass = full.sum(); raw = []; repaired = []
    for ridge in (.001, .01, .1):
        fitted = sparse_majority_candidates(A, y, eps=.01, ridge=ridge)
        dense = np.asarray(fitted[-1]['weights'])
        for c in fitted:
            w = np.asarray(c['weights'])*mass; fs = c['features']
            cert = weight_certificate(A, y, fs, full, w[fs])
            r = {**c, 'config': {**c['config'], 'ridge': ridge}, 'weights': w.tolist(),
                 'full_weights': full.tolist(), 'certificate': cert}
            raw.append(r); repaired.append(repair_majority_metric(A, y, r, dense, ridge=ridge))
    original = {'config': {'eps': .01, 'fraction': 1., 'ridge': 'original'},
        'features': list(range(d)), 'weights': full.tolist(), 'full_weights': full.tolist(),
        'certificate': weight_certificate(A, y, range(d), full, full), 'fit': {'scope': 'original_full_metric'}}
    raw.append(original); repaired.append(original)
    return {'certified_repaired': repaired, 'unconstrained_majority': raw,
            'unlimited_majority': [r for r in raw if r['config']['fraction'] == 1.]}


def run(job):
    name, fold = job; threadpool_limits(1)
    folder = ROOT/'results/predictive-iteration40'; folder.mkdir(exist_ok=True)
    out = folder/f'{name}-s101-f{fold}.json'
    if out.exists(): raise FileExistsError(out)
    started = time.perf_counter(); source = ROOT/'results/predictive-iteration36'/out.name
    old = json.loads(source.read_text()); X, y = load_dataset(name)
    tr, te = np.array(old['train_indices']), np.array(old['test_indices']); inner = []
    traces = {m: {} for m in METHODS}
    for index, split in enumerate(old['inner_splits']):
        ii, jj = np.array(split['train']), np.array(split['validation'])
        A, B = preprocess_fit(X[tr][ii], X[tr][jj]); pools = candidates(A, y[tr][ii]); cache = {}
        for method, rows in pools.items():
            for r in rows:
                key = (tuple(r['features']), tuple(r['weights']))
                if key not in cache:
                    cache[key] = validation_scores(A, y[tr][ii], B, y[tr][jj], r['features'], np.asarray(r['weights']), 144+index, joint=True)
                scores = cache[key]; r = {**r, 'validation': scores, 'score': (scores['clean']+scores['noise01'])/2}
                config = json.dumps(r['config'], sort_keys=True)
                traces[method].setdefault(config, []).append(r)
            pools[method] = [{**r, 'validation': cache[(tuple(r['features']), tuple(r['weights']))]} for r in rows]
        inner.append({'train': ii.tolist(), 'validation': jj.tolist(), 'fits': pools})
        print(name, fold, 'inner', index, 'complete', flush=True)
    configs = {}; choices = {}
    for method, trace in traces.items():
        configs[method] = [{'config': json.loads(key), 'score': float(np.mean([r['score'] for r in records])),
            'mean_size': float(np.mean([len(r['features']) for r in records]))} for key, records in trace.items()]
        choices[method] = min(configs[method], key=lambda c: (-c['score'], c['mean_size'], json.dumps(c['config'], sort_keys=True)))
    F, Q = preprocess_fit(X[tr], X[te]); pools = candidates(F, y[tr]); results = []
    for method in METHODS:
        chosen = choices[method]; fitted = next(r for r in pools[method] if r['config'] == chosen['config'])
        if method == 'certified_repaired': assert fitted['certificate']['feasible']
        scores = validation_scores(F, y[tr], Q, y[te], fitted['features'], np.asarray(fitted['weights']), 104, joint=True)
        results.append({'method': method, 'chosen': chosen, 'fit': fitted, 'metrics': scores})
        print(name, fold, method, scores['clean'], 'size', len(fitted['features']), flush=True)
    selected = results[0]['fit']; fs = selected['features']
    uniform_control = validation_scores(F, y[tr], Q, y[te], fs, np.ones(F.shape[1])/np.sqrt(F.shape[1]), 104, joint=True)
    sources = [ROOT/'src/certified_frfs'/f for f in ['sparse_neighbor.py', 'sparse_majority.py', 'repaired_majority.py',
        'feasible_transport.py', 'redistributed_loss.py', 'majority_metric.py', 'margin.py', 'core.py', 'validated_pool.py']]
    sources += [ROOT/'scripts/run_joint_majority_iteration40.py', ROOT/'research/joint-majority-iteration40-protocol.md']
    payload = {'dataset': name, 'fold': fold, 'd': X.shape[1], 'train_indices': tr.tolist(), 'test_indices': te.tolist(),
        'inner_splits': inner, 'configurations': configs, 'rows': results, 'final_candidate_fits': pools,
        'uniform_on_selected_support': {'features': fs, 'metrics': uniform_control},
        'baseline_rows': [r for r in old['rows'] if r['method'] in ['all', 'fgs', 'mi', 'relief', 'mrmr', 'sfrip']],
        'baseline_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'seconds': time.perf_counter()-started,
        'scope': 'repeated_development_nested_joint_majority_not_untouched_confirmation_or_test_risk',
        'budgets': '10 each primary method;4 unlimited;legacy budgets12/18 differ',
        'certificate_protocol': 'original uniform L2 full metric;.1 loss/.9 retention/.01 radius;preserve full mass'}
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return {'dataset': name, 'fold': fold, 'seconds': payload['seconds']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--datasets', nargs='+', default=DATASETS)
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2]); p.add_argument('--workers', type=int, default=2); args = p.parse_args()
    pilot = json.loads((ROOT/'results/repaired-majority-pilot/verified-analysis.json').read_text())
    if not pilot['gate_passed']: raise RuntimeError('pilot expansion gate failed')
    jobs = [(name, fold) for name in args.datasets for fold in args.folds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for f in as_completed(futures): print(json.dumps(f.result()), flush=True)
