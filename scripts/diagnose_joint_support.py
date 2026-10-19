"""All15, all saved inner splits, no outer-test scoring or tuning."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.joint_support_search import fit_joint_support
from certified_frfs.validated_pool import preprocess_fit, validation_scores


NAMES = ['banknote', 'ecoli', 'glass', 'haberman', 'heart_statlog',
         'ionosphere', 'iris', 'seeds', 'sonar', 'transfusion', 'vehicle',
         'vertebral', 'wdbc', 'wine', 'yeast']
FOLDER = ROOT/'results/joint-support-diagnostic'


def hashes():
    paths = sorted((ROOT/'src/certified_frfs').glob('*.py'))
    paths += [ROOT/'scripts/diagnose_joint_support.py', ROOT/'research/joint-support-diagnostic-protocol.md']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(name):
    threadpool_limits(1)
    out = FOLDER/f'{name}.json'
    if out.exists():
        raise FileExistsError(out)
    source_hashes = hashes()
    prior = ROOT/'results/predictive-iteration41'/f'{name}-s101-f0.json'
    p = json.loads(prior.read_text())
    X, y = load_dataset(name)
    tr = np.array(p['train_indices'])
    splits = []
    started = time.perf_counter()
    for index, s in enumerate(p['inner_splits']):
        ii, jj = np.array(s['train']), np.array(s['validation'])
        reference, truth = X[tr][ii], y[tr][ii]
        A, B = preprocess_fit(reference, X[tr][jj])
        begin = time.perf_counter()
        fit = fit_joint_support(reference, truth)
        elapsed = time.perf_counter()-begin
        metrics = {}
        for branch in ('certified', 'unconstrained'):
            c = fit[branch]
            metrics[branch] = validation_scores(A, truth, B, y[tr][jj], c['features'],
                np.array(c['weights']), 144+index, joint=True)
        metrics['full'] = validation_scores(A, truth, B, y[tr][jj], list(range(A.shape[1])),
            np.ones(A.shape[1])/np.sqrt(A.shape[1]), 144+index, joint=True)
        fixed = next(c for c in s['fits'] if c['config'].get('ridge') == .01 and c['config']['fraction'] == .65)
        metrics['fixed_majority41'] = validation_scores(A, truth, B, y[tr][jj], fixed['features'],
            np.array(fixed['weights']), 144+index, joint=True)
        # Recompute the fixed reference to detect stale scoring conventions.
        assert all(abs(metrics['fixed_majority41'][k]-fixed['validation'][k]) < 1e-12 for k in ('clean', 'noise01'))
        splits.append({'inner': index, 'reference_indices': tr[ii].tolist(),
            'validation_indices': tr[jj].tolist(), 'fit': fit, 'metrics': metrics,
            'fit_seconds': elapsed})
        print(name, index, metrics, 'seconds', round(elapsed, 2), flush=True)
    assert source_hashes == hashes(), 'generation source changed during task'
    result = {'dataset': name, 'd': X.shape[1], 'splits': splits, 'seconds': time.perf_counter()-started,
        'source_hashes': source_hashes, 'prior_sha256': hashlib.sha256(prior.read_bytes()).hexdigest(),
        'data_sha256': hashlib.sha256((ROOT/'data/processed'/f'{name}.npz').read_bytes()).hexdigest(),
        'scope': 'all_inner_development_no_outer_predictions', 'complete': True}
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')


def analyze():
    rows = []
    for name in NAMES:
        path = FOLDER/f'{name}.json'
        if not path.exists():
            raise RuntimeError(f'incomplete collection: {name}')
        p = json.loads(path.read_text())
        if p['source_hashes'] != hashes():
            raise RuntimeError(f'source provenance mismatch: {name}')
        scores = {m: {k: float(np.mean([s['metrics'][m][k] for s in p['splits']]))
            for k in ('clean', 'noise01')} for m in p['splits'][0]['metrics']}
        utility = {m: (s['clean']+s['noise01'])/2 for m, s in scores.items()}
        rows.append({'dataset': name, 'scores': scores, 'utilities': utility,
            'delta_full': utility['certified']-utility['full'],
            'delta_fixed41': utility['certified']-utility['fixed_majority41'],
            'certified_removed': float(np.mean([1-len(s['fit']['certified']['features'])/p['d'] for s in p['splits']])),
            'full_fallbacks': sum(s['fit']['certified']['original_full'] for s in p['splits'])})
    means = {m: float(np.mean([r['utilities'][m] for r in rows])) for m in rows[0]['utilities']}
    gates = {'gain_full_half_point': means['certified']-means['full'] >= .005,
             'gain_fixed41_half_point': means['certified']-means['fixed_majority41'] >= .005,
             'ten_dataset_improvements_full': sum(r['delta_full'] > 1e-12 for r in rows) >= 10}
    result = {'complete': True, 'rows': rows, 'mean_utility': means, 'gates': gates,
        'gate_passed': all(gates.values()), 'total_full_fallbacks': sum(r['full_fallbacks'] for r in rows),
        'scope': 'repeatedly_exposed_inner_development_not_significance_or_confirmation'}
    (FOLDER/'complete-analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    FOLDER.mkdir(exist_ok=True)
    if not args.analyze_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run, n) for n in NAMES]):
                future.result()
    analyze()
