"""Nested development pilot for joint sparse metric learning and certification.

Six frozen validation configurations, identical-objective constrained control.
All original full reference metrics are L2-normalized for certification,
and reduced metric mass is conserved. No new test-risk theorem is claimed.
"""
import json, hashlib, time, argparse
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.sparse_neighbor import sparse_metric_candidates
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.validated_pool import preprocess_fit, validation_scores


def fits(A, y):
    rows = []
    for eps in (0., .01):
        candidates = sparse_metric_candidates(A, y, eps=eps)
        dense = np.asarray(candidates[-1]['weights']); scale = 1/np.linalg.norm(dense)
        full = dense*scale
        for c in candidates:
            w = np.asarray(c['weights'])*scale; fs = c['features']
            cert = weight_certificate(A, y, fs, full, w[fs])
            rows.append({**c, 'weights': w.tolist(), 'full_weights': full.tolist(), 'certificate': cert})
    return rows


def run(name):
    threadpool_limits(1); directory = ROOT/'results/sparse-neighbor-pilot'; directory.mkdir(exist_ok=True)
    out = directory/f'{name}-s101-f0.json'
    if out.exists(): raise FileExistsError(out)
    started = time.perf_counter(); old_path = ROOT/'results/predictive-iteration36'/out.name
    old = json.loads(old_path.read_text()); X, y = load_dataset(name)
    tr = np.array(old['train_indices']); te = np.array(old['test_indices']); inner = []; traces = {}
    for index, split in enumerate(old['inner_splits']):
        ii = np.array(split['train']); jj = np.array(split['validation'])
        A, B = preprocess_fit(X[tr][ii], X[tr][jj]); rows = fits(A, y[tr][ii])
        for r in rows:
            w = np.asarray(r['weights']); fs = r['features']
            r['validation'] = validation_scores(A, y[tr][ii], B, y[tr][jj], fs, w, 144+index, joint=True)
            r['score'] = (r['validation']['clean']+r['validation']['noise01'])/2
            key = json.dumps(r['config'], sort_keys=True); traces.setdefault(key, []).append(r)
        inner.append({'train': ii.tolist(), 'validation': jj.tolist(), 'fits': rows})
        print(name, 'inner', index, 'completed', flush=True)
    # Configuration eligibility is fixed by certification in all reference
    # inner folds. A selected final candidate is recertified and can fall back.
    configs = []
    for key, rows in traces.items():
        configs.append({'config': json.loads(key), 'score': float(np.mean([r['score'] for r in rows])),
            'mean_size': float(np.mean([len(r['features']) for r in rows])),
            'all_inner_feasible': all(r['certificate']['feasible'] for r in rows)})
    ranking = lambda c: (-c['score'], c['mean_size'], c['config']['eps'], c['config']['fraction'])
    unconstrained = min(configs, key=ranking)
    certified = min([c for c in configs if c['all_inner_feasible']], key=ranking)
    dense_choice = min([c for c in configs if c['config']['fraction'] == 1.], key=ranking)
    # Final metric fitting receives training data only. Query preprocessing
    # uses frozen reference parameters and cannot influence fitting.
    F, Q = preprocess_fit(X[tr], X[te]); final = fits(F, y[tr]); results = []
    for method, selected in [('sparse', unconstrained), ('certified_sparse', certified), ('dense_joint_metric', dense_choice)]:
        fit = next(r for r in final if r['config'] == selected['config']); fallback = False
        if method == 'certified_sparse' and not fit['certificate']['feasible']:
            fit = next(r for r in final if r['config'] == {'eps': selected['config']['eps'], 'fraction': 1.})
            fallback = True
            assert fit['certificate']['feasible']
        metrics = validation_scores(F, y[tr], Q, y[te], fit['features'], np.array(fit['weights']), 104, joint=True)
        results.append({'method': method, 'chosen': selected, 'fit': fit, 'full_fallback': fallback, 'metrics': metrics})
        print(name, method, metrics['clean'], 'size', len(fit['features']), 'certified', fit['certificate']['feasible'], flush=True)
    # Cached baselines enter only after selection and final predictions.
    baseline_rows = [r for r in old['rows'] if r['method'] in ['all', 'fgs', 'mi', 'relief', 'mrmr', 'sfrip']]
    sources = [ROOT/'src/certified_frfs/sparse_neighbor.py', ROOT/'src/certified_frfs/feasible_transport.py',
        ROOT/'src/certified_frfs/redistributed_loss.py', ROOT/'src/certified_frfs/softvote_metric.py',
        ROOT/'src/certified_frfs/validated_pool.py', ROOT/'scripts/run_sparse_neighbor_pilot.py']
    payload = {'dataset': name, 'train_indices': tr.tolist(), 'test_indices': te.tolist(), 'inner_splits': inner,
        'configurations': configs, 'rows': results, 'baseline_rows': baseline_rows, 'seconds': time.perf_counter()-started,
        'final_candidate_fits': final, 'baseline_source_sha256': hashlib.sha256(old_path.read_bytes()).hexdigest(),
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'scope': 'repeated_development_nested_joint_sparse_metric_not_confirmation_or_novelty_proof',
        'budgets': '6 configurations sparse/certified_sparse;2 dense;cached baseline budgets differ',
        'certificate_protocol': 'new learned full reference L2 metric;reduced full-mass conserved;.1 loss/.9 retention/.01 radius'}
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--datasets', nargs='+', default=['iris', 'wine', 'sonar', 'wdbc', 'ionosphere']); args = p.parse_args()
    for name in args.datasets: run(name)
