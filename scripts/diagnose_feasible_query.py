"""All15/all44 held-out inner scores for feasible probability descent."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_guarded_support import FOLDER as PRIOR, NAMES
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit, validation_scores
from certified_frfs.feasible_query_descent import fit_feasible_query


FOLDER = ROOT/'results/feasible-query-diagnostic'
EXTRA_SOURCE_PATHS = []


def hashes():
    paths = sorted((ROOT/'src/certified_frfs').glob('*.py'))
    paths += [Path(__file__), ROOT/'research/feasible-query-protocol.md']
    paths += EXTRA_SOURCE_PATHS
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


ORACLE_SECONDS = 2.
WEIGHT_CAP_RATIO = None


def run(name):
    threadpool_limits(1); out = FOLDER/f'{name}.json'
    if out.exists(): raise FileExistsError(out)
    prior = PRIOR/f'{name}.json'; p = json.loads(prior.read_text()); source = hashes()
    X, y = load_dataset(name); splits = []
    for index, s in enumerate(p['splits']):
        ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
        A, B = preprocess_fit(X[ri], X[vi]); start = time.perf_counter()
        fit = fit_feasible_query(X[ri], y[ri], oracle_seconds=ORACLE_SECONDS, weight_cap_ratio=WEIGHT_CAP_RATIO); seconds = time.perf_counter()-start
        metrics = validation_scores(A, y[ri], B, y[vi], fit['features'], np.array(fit['weights']), 144+index, joint=True)
        splits.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
            'fit': fit, 'metrics': metrics, 'reference_metrics': s['reference_metrics'], 'seconds': seconds})
        print(name, index, metrics, 'accepted steps', len(fit['trace'])-1, 'seconds', round(seconds, 2), flush=True)
    assert hashes() == source
    result = {'dataset': name, 'd': p['d'], 'splits': splits, 'source_hashes': source,
        'prior_sha256': hashlib.sha256(prior.read_bytes()).hexdigest(),
        'data_sha256': hashlib.sha256((ROOT/'data/processed'/f'{name}.npz').read_bytes()).hexdigest()}
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')


def analyze():
    utility = lambda m: (m['clean']+m['noise01'])/2; rows = []
    for name in NAMES:
        p = json.loads((FOLDER/f'{name}.json').read_text()); assert p['source_hashes'] == hashes()
        means = {'feasible_query': float(np.mean([utility(s['metrics']) for s in p['splits']]))}
        means.update({m: float(np.mean([utility(s['reference_metrics'][m]) for s in p['splits']])) for m in ('full', 'fixed_majority41')})
        rows.append({'dataset': name, 'means': means,
            'accepted_steps': sum(len(s['fit']['trace'])-1 for s in p['splits']),
            'mean_removed': float(np.mean([1-len(s['fit']['features'])/p['d'] for s in p['splits']]))})
    means = {m: float(np.mean([r['means'][m] for r in rows])) for m in rows[0]['means']}
    positive = sum(r['means']['feasible_query'] > r['means']['full']+1e-12 for r in rows)
    gates = {'mean_gain_over_full_half_point': means['feasible_query']-means['full'] >= .005,
        'mean_gain_over_fixed41_half_point': means['feasible_query']-means['fixed_majority41'] >= .005,
        'ten_datasets_improve_full': positive >= 10}
    result = {'rows': rows, 'means': means, 'positive_datasets_full': positive, 'gates': gates,
        'gate_passed': all(gates.values()), 'scope': 'all_inner_exposed_development_not_test_confirmation'}
    (FOLDER/'complete-analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run, name) for name in NAMES]): f.result()
    analyze()
