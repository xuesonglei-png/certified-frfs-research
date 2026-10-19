"""Audit every saved inner/final admitted repaired metric, not only selection."""
import json, argparse
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from audit_sparse_pilots import preprocessing
from certified_frfs.datasets import load_dataset
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import gamma


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('directory'); args = parser.parse_args()
    threadpool_limits(1); folder = ROOT/'results'/args.directory; records = []; seen = set(); sampled = 0
    for name in ['iris', 'wine', 'sonar', 'wdbc', 'ionosphere']:
        p = json.loads((folder/f'{name}-s101-f0.json').read_text()); X, y = load_dataset(name); tr = np.array(p['train_indices'])
        tables = []
        for index, split in enumerate(p['inner_splits']):
            ids = tr[np.array(split['train'])]; A, _ = preprocessing(X[ids], X[ids])
            tables.append((f'inner{index}', A, y[ids], split['fits']))
        A, _ = preprocessing(X[tr], X[tr]); tables.append(('final', A, y[tr], p['final_candidate_fits']))
        for scope, A, labels, candidates in tables:
            for c in candidates:
                nodes = [(c['features'], c['weights'], c['certificate'])]
                nodes += [(r['features'], r['weights'], r['certificate']) for r in c.get('repair', {}).get('trace', []) if r['certificate']['feasible']]
                for fs, weights, old in nodes:
                    key = (name, scope, tuple(fs), tuple(weights))
                    if key in seen: continue
                    seen.add(key); w = np.asarray(weights); full = np.asarray(c['full_weights'])
                    rebuilt = weight_certificate(A, labels, fs, full, w[fs])
                    assert rebuilt['feasible'] == old['feasible']
                    assert rebuilt['retained_counts'] == old['retained_counts']
                    assert abs(rebuilt['dependency_certificate']['upper_bound']-old['dependency_certificate']['upper_bound']) < 1e-12
                    rng = np.random.default_rng(996); losses = []
                    for _ in range(3):
                        Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
                        losses.append(gamma(Z, labels, weights=full)-gamma(Z, labels, fs, weights=w))
                    assert max(losses) <= rebuilt['dependency_certificate']['upper_bound']+1e-10; sampled += 3
                    records.append({'dataset': name, 'scope': scope, 'features': fs, 'feasible': rebuilt['feasible'],
                        'bound': rebuilt['dependency_certificate']['upper_bound'], 'sampled_losses': losses})
        print(name, 'all training metric nodes audited', flush=True)
    out = folder/'training-metric-audit.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps({'scope': 'all_unique_inner_final_candidates_and_admitted_repair_nodes',
        'nodes_checked': len(records), 'sampled_loss_checks': sampled, 'observed_violations': 0, 'rows': records}, indent=2), encoding='utf-8')
    print('nodes', len(records), 'sampled', sampled, flush=True)


if __name__ == '__main__': main()
