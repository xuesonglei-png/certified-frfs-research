"""Rebuild subfolds; verify retained-grid selection and sampled loss bounds."""
import json, hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.redistributed_search import CrossfitRedistributedUtilityWorkspace
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import gamma


def main():
    threadpool_limits(1); checks = []; comparisons = []
    for name in ['feasible-transport-diagnostic', 'repaired-loo-transport-diagnostic', 'repaired-crossfit-transport-diagnostic']:
        source = ROOT/'results'/name/'training-subfold-gate.json'; p = json.loads(source.read_text())
        deltas = []; matched = []
        for r in p['rows']:
            o = json.loads((ROOT/'results/predictive-iteration36'/f"{r['dataset']}-s101-f0.json").read_text())
            X, y = load_dataset(r['dataset']); tr = np.array(o['train_indices'])
            workspace = CrossfitRedistributedUtilityWorkspace(X[tr], y[tr], family=r['family'])
            for index, ((A, B, yt, yv, w), f) in enumerate(zip(workspace.folds, r['folds'])):
                if f['certified'] is None: continue
                fit = f['fit']; chosen = fit['selected']; fs = f.get('retained_features', r['features']); v = np.asarray(chosen['weights'])
                assert not (f.get('repair') or {}).get('full_fallback', False), 'audit fallback separately'
                feasible = [q for q in fit['candidates'] if q['feasible']]
                if fit.get('selection', 'smooth') == 'smooth': key = lambda q: (q['training_objective'], q['interpolation'])
                elif fit['selection'] == 'loo': key = lambda q: (-q['training_utility'], q['training_objective'], q['interpolation'])
                else: key = lambda q: (-q['selection_utility'], q['interpolation'])
                assert chosen == min(feasible, key=key)
                cert = weight_certificate(A, yt, fs, w, v); assert cert['feasible']
                assert cert['retained_counts'] == chosen['retained_counts']
                assert abs(cert['dependency_certificate']['upper_bound']-chosen['dependency_certificate']['upper_bound']) < 1e-12
                assert abs(v.sum()-w.sum()) < 1e-12
                rng = np.random.default_rng(986+index); wide = np.zeros(A.shape[1]); wide[fs] = v; losses = []
                for _ in range(10):
                    Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
                    losses.append(gamma(Z, yt, weights=w)-gamma(Z, yt, fs, weights=wide))
                assert max(losses) <= cert['dependency_certificate']['upper_bound']+1e-10
                checks.append({'experiment': name, 'dataset': r['dataset'], 'subfold': index,
                    'features': fs, 'bound': cert['dependency_certificate']['upper_bound'], 'sampled_losses': losses})
            if r['utility_delta'] is not None: deltas.append(r['utility_delta'])
            if r['complete_feasibility']:
                matched.append(float(np.mean([f['certified']['utility']-f.get('matched_correlation', f['correlation'])['utility'] for f in r['folds']])))
        comparisons.append({'experiment': name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'complete_datasets': len(deltas), 'all_dataset_mean_delta': p['mean_utility_delta'],
            'matched_subset_mean_delta_complete_only': float(np.mean(matched)), 'gate_passed': p['gate_passed']})
    sources = {str(q.relative_to(ROOT)): hashlib.sha256(q.read_bytes()).hexdigest() for q in [
        ROOT/'src/certified_frfs/feasible_transport.py', ROOT/'scripts/diagnose_feasible_transport.py',
        ROOT/'scripts/audit_feasible_transport.py', ROOT/'tests/test_feasible_transport.py']}
    payload = {'scope': 'development_training_diagnostics_not_test_superiority', 'comparisons': comparisons,
        'checks': checks, 'sampled_loss_checks': 10*len(checks), 'observed_violations': 0, 'source_hashes': sources}
    out = ROOT/'results/feasible-transport-diagnostic/combined-audit.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8'); print(json.dumps(comparisons), flush=True)


if __name__ == '__main__': main()
