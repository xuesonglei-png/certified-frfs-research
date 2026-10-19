"""Independently certify fixed capped mixtures and score all44 inner splits."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit, validation_scores
from certified_frfs.trajectory_certificate import certify_capped_trajectory


FOLDER = ROOT/'results/trajectory-certificate-diagnostic'


def hashes():
    paths = sorted((ROOT/'src/certified_frfs').glob('*.py'))+[Path(__file__), ROOT/'research/trajectory-certificate-protocol.md']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(old):
    threadpool_limits(1); name = old['dataset']; out = FOLDER/f'{name}.json'
    if out.exists(): raise FileExistsError(out)
    source = hashes(); X, y = load_dataset(name); splits = []
    prior_path = ROOT/'results/certified-capped-query-diagnostic'/f'{name}.json'; prior = json.loads(prior_path.read_text())
    for index, s in enumerate(old['splits']):
        ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices']); A, B = preprocess_fit(X[ri], X[vi])
        assert ri.tolist() == prior['splits'][index]['reference_indices']
        assert vi.tolist() == prior['splits'][index]['validation_indices']
        fit = certify_capped_trajectory(X[ri], y[ri], s['fit']); metrics = {}
        for branch, key in [('certified', 'selected'), ('unconstrained', 'unconstrained_selected')]:
            c = fit[key]
            metrics[branch] = validation_scores(A, y[ri], B, y[vi], c['features'], np.array(c['weights']), 144+index, joint=True)
        splits.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(), 'fit': fit,
            'metrics': metrics, 'polytope_certified_metrics': prior['splits'][index]['metrics']})
        print(name, index, metrics, 'feasible', sum(c['certificate']['feasible'] for c in fit['candidates']), '/', len(fit['candidates']), flush=True)
    assert hashes() == source
    result = {'dataset': name, 'd': old['d'], 'splits': splits, 'source_hashes': source,
        'polytope_prior_sha256': hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        'data_sha256': hashlib.sha256((ROOT/'data/processed'/f'{name}.npz').read_bytes()).hexdigest()}
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')


def analyze(prior, prior_sha256):
    utility = lambda m: (m['clean']+m['noise01'])/2; rows = []
    for old in prior['rows']:
        p = json.loads((FOLDER/f"{old['dataset']}.json").read_text()); assert p['source_hashes'] == hashes()
        means = {m: float(np.mean([utility(s['metrics'][m]) for s in p['splits']])) for m in ('certified', 'unconstrained')}
        means.update({m: old['means'][m] for m in ('full', 'fixed_majority41')})
        means['polytope_certified'] = float(np.mean([utility(s['polytope_certified_metrics']) for s in p['splits']]))
        rows.append({'dataset': p['dataset'], 'means': means,
            'mean_removed': float(np.mean([1-len(s['fit']['selected']['features'])/p['d'] for s in p['splits']])),
            'full_original_selected': sum(s['fit']['selected']['iteration'] == 0 for s in p['splits'])})
    means = {m: float(np.mean([r['means'][m] for r in rows])) for m in rows[0]['means']}
    positive = sum(r['means']['certified'] > r['means']['full']+1e-12 for r in rows)
    gates = {'mean_gain_full_half_point': means['certified']-means['full'] >= .005,
        'mean_gain_fixed41_half_point': means['certified']-means['fixed_majority41'] >= .005,
        'ten_positive_full': positive >= 10}
    result = {'rows': rows, 'means': means, 'positive_full': positive, 'gates': gates,
        'gate_passed': all(gates.values()), 'prior_sha256': prior_sha256,
        'scope': 'all_inner_exposed_training_grid_independent_candidate_certification_not_confirmation'}
    (FOLDER/'complete-analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True); source = ROOT/'results/capped-query-diagnostic/complete.json'
    prior = json.loads(source.read_text()); prior_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, r) for r in prior['rows']]): future.result()
    analyze(prior, prior_sha256)
