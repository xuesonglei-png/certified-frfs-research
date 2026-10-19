"""Fixed five reference-training diagnostic; no outer classifications."""
import json, time
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.coupled_redistributed import coupled_redistributed_bound
from certified_frfs.core import gamma


def main():
    threadpool_limits(1); rows = []
    for name in ['iris', 'wine', 'sonar', 'wdbc', 'ionosphere']:
        p = json.loads((ROOT/'results/anchored-majority-pilot'/f'{name}-s101-f0.json').read_text())
        X, y = load_dataset(name); tr = np.array(p['train_indices']); A, _ = preprocess_fit(X[tr], X[tr])
        # Evaluate every frozen final learned config, not only favorable ones.
        records = []
        for c in p['final_candidate_fits']:
            fs = c['features']; w = np.array(c['weights']); full = np.array(c['full_weights']); start = time.perf_counter()
            r = coupled_redistributed_bound(A, y[tr], fs, full, w[fs])
            rng = np.random.default_rng(1012); losses = []
            for _ in range(10):
                Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
                losses.append(gamma(Z, y[tr], weights=full)-gamma(Z, y[tr], fs, weights=w))
            assert max(losses) <= r['upper_bound']+1e-10
            records.append({'config': c['config'], 'features': fs, 'certificate': r,
                'sampled_losses': losses, 'seconds': time.perf_counter()-start})
        rows.append({'dataset': name, 'candidates': records})
        print(name, 'tightened', sum(r['certificate']['upper_bound'] < r['certificate'].get('objectwise_bound',r['certificate']['upper_bound'])-1e-12 for r in records), flush=True)
    folder = ROOT/'results/signed-coupling-diagnostic'; folder.mkdir(exist_ok=True); out = folder/'training-candidates.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps({'rows': rows, 'scope': 'all_frozen_final_training_candidates_no_classification_or_novelty_claim',
        'sampled_loss_checks': 500, 'observed_violations': 0}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
