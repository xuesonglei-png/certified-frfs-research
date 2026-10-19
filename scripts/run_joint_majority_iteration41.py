"""Robust majority-margin joint metrics against a fixed original uniform full metric.

Frozen nine learned candidates: ridge .001/.01/.1, fraction .35/.65/1,
robust surrogate radius .01. Explicit tenth candidate is original full metric.
Internal validation chooses strength; outer outcomes never choose configs.
"""
import json, hashlib, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.sparse_majority import sparse_majority_candidates
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.validated_pool import preprocess_fit, validation_scores


def fits(A, y):
    d = A.shape[1]; full = np.ones(d)/np.sqrt(d); mass = full.sum(); rows = []
    for ridge in (.001, .01, .1):
        for c in sparse_majority_candidates(A, y, eps=.01, ridge=ridge):
            w = np.asarray(c['weights'])*mass; fs = c['features']
            cert = weight_certificate(A, y, fs, full, w[fs])
            rows.append({**c, 'config': {**c['config'], 'ridge': ridge},
                'weights': w.tolist(), 'full_weights': full.tolist(), 'certificate': cert})
    fs = list(range(d)); cert = weight_certificate(A, y, fs, full, full)
    assert cert['feasible']
    rows.append({'config': {'eps': .01, 'fraction': 1., 'ridge': 'original'},
        'features': fs, 'weights': full.tolist(), 'full_weights': full.tolist(),
        'certificate': cert, 'fit': {'scope': 'original_full_metric_fallback'}})
    return rows


def run(job):
    name, fold = job
    threadpool_limits(1); directory = ROOT/'results/predictive-iteration41'; directory.mkdir(exist_ok=True)
    out = directory/f'{name}-s101-f{fold}.json'
    if out.exists(): raise FileExistsError(out)
    started = time.perf_counter(); old_path = ROOT/'results/predictive-iteration36'/out.name
    old = json.loads(old_path.read_text()); X, y = load_dataset(name)
    tr = np.array(old['train_indices']); te = np.array(old['test_indices']); inner = []; traces = {}
    for index, split in enumerate(old['inner_splits']):
        ii = np.array(split['train']); jj = np.array(split['validation'])
        A, B = preprocess_fit(X[tr][ii], X[tr][jj]); rows = fits(A, y[tr][ii])
        for r in rows:
            r['validation'] = validation_scores(A, y[tr][ii], B, y[tr][jj], r['features'], np.asarray(r['weights']), 144+index, joint=True)
            r['score'] = (r['validation']['clean']+r['validation']['noise01'])/2
            traces.setdefault(json.dumps(r['config'], sort_keys=True), []).append(r)
        inner.append({'train': ii.tolist(), 'validation': jj.tolist(), 'fits': rows})
        print(name, 'inner', index, 'complete', flush=True)
    configs = [{'config': json.loads(key), 'score': float(np.mean([r['score'] for r in rows])),
        'mean_size': float(np.mean([len(r['features']) for r in rows])),
        'all_inner_feasible': all(r['certificate']['feasible'] for r in rows)} for key, rows in traces.items()]
    ranking = lambda c: (-c['score'], c['mean_size'], json.dumps(c['config'], sort_keys=True))
    choices = {'sparse': min(configs, key=ranking),
        'certified_sparse': min([c for c in configs if c['all_inner_feasible']], key=ranking),
        'unlimited_joint_metric': min([c for c in configs if c['config']['fraction'] == 1.], key=ranking)}
    F, Q = preprocess_fit(X[tr], X[te]); final = fits(F, y[tr]); results = []
    for method, selected in choices.items():
        fitted = next(r for r in final if r['config'] == selected['config']); fallback = False
        if method == 'certified_sparse' and not fitted['certificate']['feasible']:
            fitted = final[-1]; fallback = True
        metrics = validation_scores(F, y[tr], Q, y[te], fitted['features'], np.asarray(fitted['weights']), 104, joint=True)
        results.append({'method': method, 'chosen': selected, 'fit': fitted, 'full_fallback': fallback, 'metrics': metrics})
        print(name, method, metrics['clean'], 'size', len(fitted['features']), 'cert', fitted['certificate']['feasible'], flush=True)
    sources = sorted((ROOT/'src/certified_frfs').glob('*.py')) + [
        ROOT/'scripts/run_joint_majority_iteration41.py', ROOT/'scripts/run_decision_iteration.py',
        ROOT/'research/joint-majority-iteration41-protocol.md']
    payload = {'dataset': name, 'train_indices': tr.tolist(), 'test_indices': te.tolist(), 'inner_splits': inner,
        'configurations': configs, 'rows': results, 'final_candidate_fits': final,
        'baseline_rows': [r for r in old['rows'] if r['method'] in ['all', 'fgs', 'mi', 'relief', 'mrmr', 'sfrip']],
        'fold': fold, 'seconds': time.perf_counter()-started, 'baseline_source_sha256': hashlib.sha256(old_path.read_bytes()).hexdigest(),
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'data_sha256': hashlib.sha256((ROOT/'data/processed'/f'{name}.npz').read_bytes()).hexdigest(),
        'scope': 'complete_development_nested_majority_joint_sparse_metric_not_confirmation_or_novelty_proof',
        'budgets': '10 configs sparse/certified;4 unlimited;cached baseline budgets differ',
        'certificate_protocol': 'original uniform L2 full reference;reduced mass preserved;.1 loss/.9 retention/.01 radius'}
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['banknote','ecoli','glass','haberman','heart_statlog','ionosphere','iris','seeds','sonar','transfusion','vehicle','vertebral','wdbc','wine','yeast'])
    p.add_argument('--folds', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    pilot = json.loads((ROOT/'results/majority-sparse-pilot/verified-analysis.json').read_text())
    means = pilot['means']; wins = sum(r['metrics']['sparse']['clean'] > r['metrics']['fgs']['clean']+1e-12 for r in pilot['rows'])
    assert means['sparse']['clean'] >= means['fgs']['clean']+.005 and means['sparse']['clean'] >= means['all']['clean'] and wins >=3
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run,(name,fold)) for name in args.datasets for fold in args.folds]
        for f in as_completed(futures): f.result()
