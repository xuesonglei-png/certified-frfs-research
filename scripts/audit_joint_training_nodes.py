"""Incremental full-study training candidate/admitted-node certificate audit."""
import json, hashlib, argparse
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from audit_sparse_pilots import preprocessing
from certified_frfs.datasets import load_dataset
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import gamma


def audit(path):
    p = json.loads(path.read_text()); X, y = load_dataset(p['dataset']); tr = np.array(p['train_indices']); records = []
    seen = set(); sampled = 0; nodes = []
    for index, split in enumerate(p['inner_splits']):
        ids = tr[np.array(split['train'])]; A, _ = preprocessing(X[ids], X[ids])
        nodes.append((f'inner{index}', A, y[ids], split['fits']))
    A, _ = preprocessing(X[tr], X[tr]); nodes.append(('final', A, y[tr], p['final_candidate_fits']))
    for scope, A, labels, candidates in nodes:
        for c in candidates:
            metrics = [(c['features'], c['weights'], c['certificate'])]
            metrics += [(q['features'], q['weights'], q['certificate']) for q in c.get('repair', {}).get('trace', []) if q['certificate']['feasible']]
            for fs, weights, saved in metrics:
                key = (scope, tuple(fs), tuple(weights))
                if key in seen: continue
                seen.add(key); w = np.asarray(weights); full = np.ones(A.shape[1])/np.sqrt(A.shape[1])
                assert abs(w.sum()-full.sum()) < 1e-10 and np.all(w >= 0)
                rebuilt = weight_certificate(A, labels, fs, full, w[fs])
                assert rebuilt['feasible'] == saved['feasible'] and rebuilt['retained_counts'] == saved['retained_counts']
                assert abs(rebuilt['dependency_certificate']['upper_bound']-saved['dependency_certificate']['upper_bound']) < 1e-12
                rng = np.random.default_rng(1009); losses = []
                for _ in range(3):
                    Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
                    losses.append(gamma(Z, labels, weights=full)-gamma(Z, labels, fs, weights=w))
                assert max(losses) <= rebuilt['dependency_certificate']['upper_bound']+1e-10; sampled += 3
                records.append({'scope': scope, 'features': fs, 'weights': weights, 'feasible': rebuilt['feasible'],
                    'bound': rebuilt['dependency_certificate']['upper_bound'], 'sampled_losses': losses})
    return {'dataset': p['dataset'], 'fold': p['fold'], 'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'nodes_checked': len(records), 'sampled_loss_checks': sampled, 'observed_violations': 0, 'rows': records,
        'scope': 'all_saved_unique_inner_final_candidates_and_admitted_repair_nodes;not_optimizer_rebuild_or_mathematical_proof'}


def main():
    p = argparse.ArgumentParser(); p.add_argument('iteration', choices=['41', '42']); p.add_argument('--datasets', nargs='+'); args = p.parse_args()
    threadpool_limits(1); folder = ROOT/f'results/predictive-iteration{args.iteration}'; target = folder/'training-audits'; target.mkdir(exist_ok=True)
    for path in sorted(folder.glob('*-s101-f*.json')):
        if args.datasets and not any(path.name.startswith(n+'-') for n in args.datasets): continue
        out = target/path.name
        if out.exists():
            assert json.loads(out.read_text())['result_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest(); continue
        result = audit(path); temporary = out.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2), encoding='utf-8'); temporary.replace(out)
        print(path.name, result['nodes_checked'], 'training nodes audited', flush=True)


if __name__ == '__main__': main()
