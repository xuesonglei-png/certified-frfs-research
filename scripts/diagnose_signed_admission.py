"""Does shared coupling admit formerly rejected raw metrics? Training only."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.coupled_redistributed import coupled_redistributed_bound


def main():
    threadpool_limits(1); rows = []
    for name in ['iris', 'wine', 'sonar', 'wdbc', 'ionosphere']:
        p = json.loads((ROOT/'results/predictive-iteration41'/f'{name}-s101-f0.json').read_text())
        X, y = load_dataset(name); tr = np.array(p['train_indices']); candidates = []
        for index, s in enumerate(p['inner_splits']):
            ids = tr[np.array(s['train'])]; A, _ = preprocess_fit(X[ids], X[ids])
            for c in s['fits']:
                fs = c['features']; w = np.array(c['weights']); full = np.array(c['full_weights'])
                original = c['certificate']; result = coupled_redistributed_bound(A, y[ids], fs, full, w[fs])
                new_feasible = original['majority_feasible'] and result['upper_bound'] <= .1
                candidates.append({'inner': index, 'config': c['config'], 'features': fs,
                    'old_bound': original['dependency_certificate']['upper_bound'], 'new_certificate': result,
                    'majority_feasible': original['majority_feasible'], 'old_feasible': original['feasible'],
                    'new_feasible': new_feasible, 'new_admission': new_feasible and not original['feasible']})
        rows.append({'dataset': name, 'candidates': candidates}); print(name, 'new admissions', sum(c['new_admission'] for c in candidates), flush=True)
    out = ROOT/'results/signed-coupling-diagnostic/raw-inner-admission.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps({'rows': rows, 'scope': 'all_frozen_raw_training_inner_candidates_not_accuracy_or_novelty_claim',
        'new_admissions': sum(c['new_admission'] for r in rows for c in r['candidates'])}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
