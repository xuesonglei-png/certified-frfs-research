"""Fixed five-friend witness pool, same old protected masks and endpoints."""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_joint_support import FOLDER, NAMES
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit, validation_scores
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.witness_majority import witness_majority_bounds
from certified_frfs.redistributed_loss import transfer_weights


def main():
    threadpool_limits(1)
    folder = ROOT/'results/witness-admission-diagnostic'; folder.mkdir(exist_ok=True)
    out = folder/'complete.json'
    if out.exists():
        raise FileExistsError(out)
    sources = [ROOT/'scripts/diagnose_witness_admission.py', ROOT/'src/certified_frfs/witness_majority.py',
               ROOT/'research/witness-majority-extension.md']
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    rows = []; admitted = 0; lost = 0; endpoints = 0
    for name in NAMES:
        path = FOLDER/f'{name}.json'; p = json.loads(path.read_text()); X, y = load_dataset(name); splits = []
        for index, s in enumerate(p['splits']):
            ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
            A, B = preprocess_fit(X[ri], X[vi]); labels = y[ri]; full = np.ones(p['d'])/np.sqrt(p['d'])
            protected = MajorityTerms(A, labels, weights=full).bounds(range(p['d']), allowed_enemies=2) > 1e-10
            groups = [(labels == q) & protected for q in np.unique(labels)]
            required = [int(np.ceil(.9*g.sum()-1e-12)) for g in groups]
            certificates = []
            for c in s['fit']['certificates']:
                fs = c['features']; weights = transfer_weights(A, fs, full)
                bounds = witness_majority_bounds(A[:, fs], labels, weights, pool_size=5)
                counts = [int(((bounds > 1e-10) & g).sum()) for g in groups]
                assert required == c['required']
                assert all(a >= b for a, b in zip(counts, c['retained_counts']))
                majority = all(a >= b for a, b in zip(counts, required))
                feasible = bool(majority and c['dependency_certificate']['upper_bound'] <= .1)
                admitted += int(feasible and not c['feasible']); lost += int(c['feasible'] and not feasible)
                certificates.append({**c, 'original_feasible': c['feasible'], 'original_retained_counts': c['retained_counts'],
                    'feasible': feasible, 'majority_feasible': majority, 'retained_counts': counts,
                    'witness_bounds': bounds.tolist()})
                endpoints += 1
            by_key = {tuple(c['features']): c for c in certificates}
            chosen = next(c for c in s['fit']['pool'] if by_key.get(tuple(c['features']), {}).get('feasible', False))
            fs = chosen['features']; wide = np.zeros(p['d']); wide[fs] = transfer_weights(A, fs, full)
            metrics = validation_scores(A, labels, B, y[vi], fs, wide, 144+index, joint=True)
            splits.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
                'certificates': certificates, 'features': fs, 'weights': wide.tolist(), 'metrics': metrics,
                'original_features': s['fit']['certified']['features'], 'reference_metrics': s['metrics']})
        rows.append({'dataset': name, 'd': p['d'], 'splits': splits,
            'prior_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(name, 'cumulative newly admitted', admitted, flush=True)
    utility = lambda s: (s['clean']+s['noise01'])/2
    means = {'witness_certified': float(np.mean([np.mean([utility(s['metrics']) for s in r['splits']]) for r in rows]))}
    means.update({m: float(np.mean([np.mean([utility(s['reference_metrics'][m]) for s in r['splits']]) for r in rows]))
        for m in ('certified', 'full', 'fixed_majority41', 'unconstrained')})
    positive = int(sum(np.mean([utility(s['metrics'])-utility(s['reference_metrics']['full']) for s in r['splits']]) > 1e-12 for r in rows))
    gates = {'gain_full_half_point': bool(means['witness_certified']-means['full'] >= .005),
        'gain_fixed41_half_point': bool(means['witness_certified']-means['fixed_majority41'] >= .005),
        'ten_dataset_improvements_full': positive >= 10}
    result = {'rows': rows, 'means': means, 'newly_admitted': admitted, 'lost_admissions': lost,
        'endpoints': endpoints, 'positive_datasets_full': positive, 'gates': gates,
        'gate_passed': all(gates.values()), 'source_hashes': source_hashes,
        'scope': 'stronger_sufficient_majority_fixed_old_mask_reused_pool_all_inner_not_test_risk'}
    assert lost == 0
    assert source_hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
