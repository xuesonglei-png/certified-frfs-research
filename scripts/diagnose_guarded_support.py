"""Independent discovery/guard experiment on all existing inner splits."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_joint_support import NAMES, FOLDER as PRIOR
from certified_frfs.datasets import load_dataset
from certified_frfs.guarded_support import fit_guarded_support
from certified_frfs.validated_pool import preprocess_fit, validation_scores


FOLDER = ROOT/'results/guarded-support-diagnostic'


def hashes():
    paths = sorted((ROOT/'src/certified_frfs').glob('*.py'))
    paths += [ROOT/'scripts/diagnose_guarded_support.py', ROOT/'research/guarded-support-protocol.md']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(name):
    threadpool_limits(1)
    out = FOLDER/f'{name}.json'
    if out.exists():
        raise FileExistsError(out)
    prior = PRIOR/f'{name}.json'; p = json.loads(prior.read_text())
    source_hashes = hashes(); X, y = load_dataset(name); splits = []; start = time.perf_counter()
    for index, s in enumerate(p['splits']):
        ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
        A, B = preprocess_fit(X[ri], X[vi]); begin = time.perf_counter()
        fit = fit_guarded_support(X[ri], y[ri]); elapsed = time.perf_counter()-begin
        metrics = {}
        for branch in ('certified', 'unconstrained'):
            c = fit[branch]
            metrics[branch] = validation_scores(A, y[ri], B, y[vi], c['features'],
                np.array(c['weights']), 144+index, joint=True)
        splits.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
            'fit': fit, 'metrics': metrics, 'reference_metrics': s['metrics'], 'fit_seconds': elapsed})
        print(name, index, metrics, flush=True)
    assert source_hashes == hashes()
    result = {'dataset': name, 'd': p['d'], 'splits': splits, 'source_hashes': source_hashes,
        'prior_sha256': hashlib.sha256(prior.read_bytes()).hexdigest(),
        'data_sha256': hashlib.sha256((ROOT/'data/processed'/f'{name}.npz').read_bytes()).hexdigest(),
        'seconds': time.perf_counter()-start, 'scope': 'independent_discovery_guard_all_inner_development_no_outer_scores'}
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')


def analyze():
    rows = []; utility = lambda m: (m['clean']+m['noise01'])/2
    for name in NAMES:
        p = json.loads((FOLDER/f'{name}.json').read_text()); assert p['source_hashes'] == hashes()
        means = {m: float(np.mean([utility(s['metrics'][m]) for s in p['splits']])) for m in ('certified', 'unconstrained')}
        means.update({m: float(np.mean([utility(s['reference_metrics'][m]) for s in p['splits']]))
            for m in ('full', 'fixed_majority41')})
        rows.append({'dataset': name, 'means': means,
            'full_selected': sum(len(s['fit']['certified']['features']) == p['d'] for s in p['splits']),
            'removed': float(np.mean([1-len(s['fit']['certified']['features'])/p['d'] for s in p['splits']]))})
    means = {m: float(np.mean([r['means'][m] for r in rows])) for m in rows[0]['means']}
    positive = sum(r['means']['certified']-r['means']['full'] > 1e-12 for r in rows)
    gates = {'gain_full_half_point': means['certified']-means['full'] >= .005,
        'gain_fixed41_half_point': means['certified']-means['fixed_majority41'] >= .005,
        'ten_datasets_improve_full': positive >= 10}
    result = {'rows': rows, 'means': means, 'positive_datasets_full': positive, 'gates': gates,
        'gate_passed': all(gates.values()), 'full_selected': sum(r['full_selected'] for r in rows),
        'mean_removed': float(np.mean([r['removed'] for r in rows])),
        'scope': 'complete_reused_inner_development_not_significance_or_confirmation'}
    (FOLDER/'complete-analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(); FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, name) for name in NAMES]):
            future.result()
    analyze()
