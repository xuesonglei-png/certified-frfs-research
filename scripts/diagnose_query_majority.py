"""Eight fixed development reference/query experiments; no outer test scoring."""
import json, time
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit, validation_scores
from certified_frfs.query_majority import query_majority_candidates


NAMES = ['iris', 'wine', 'sonar', 'wdbc', 'ionosphere', 'glass', 'haberman', 'heart_statlog']


def main():
    threadpool_limits(1); directory = ROOT/'results/query-majority-diagnostic'; directory.mkdir(exist_ok=True); out = directory/'fixed-inner0.json'
    if out.exists(): raise FileExistsError(out)
    rows = []
    for name in NAMES:
        p = json.loads((ROOT/'results/predictive-iteration41'/f'{name}-s101-f0.json').read_text())
        X, y = load_dataset(name); tr = np.array(p['train_indices']); s = p['inner_splits'][0]
        ii, jj = np.array(s['train']), np.array(s['validation']); A, B = preprocess_fit(X[tr][ii], X[tr][jj]); started = time.perf_counter()
        fitted = query_majority_candidates(X[tr][ii], y[tr][ii], ridge=.01)
        candidates = []
        for c in fitted:
            w = np.asarray(c['weights']); scores = validation_scores(A, y[tr][ii], B, y[tr][jj], c['features'], w, 144, joint=True)
            old = next(r for r in s['fits'] if r['config'].get('ridge') == .01 and r['config']['fraction'] == c['config']['fraction'])
            prior = old['validation']; delta = (scores['clean']+scores['noise01']-prior['clean']-prior['noise01'])/2
            candidates.append({**c, 'metrics': scores, 'prior_majority_same_config': prior, 'utility_delta': delta})
        rows.append({'dataset': name, 'reference_indices': tr[ii].tolist(), 'validation_indices': tr[jj].tolist(),
                     'candidates': candidates, 'seconds': time.perf_counter()-started})
        print(name, 'mean delta across3frozen configs', np.mean([c['utility_delta'] for c in candidates]), flush=True)
        (directory/'partial.json').write_text(json.dumps({'complete': False, 'rows': rows}, indent=2), encoding='utf-8')
    # Descriptive fixed-config comparison, no selecting the best validation
    # configuration per dataset and relabeling it as a learned method.
    means = {str(f): float(np.mean([next(c['utility_delta'] for c in r['candidates'] if c['config']['fraction'] == f) for r in rows])) for f in [.35, .65, 1.]}
    out.write_text(json.dumps({'complete': True, 'rows': rows, 'fixed_fraction_mean_deltas': means,
        'scope': 'eight_fixed_inner_training_experiments_no_outer_score_or_significance_claim'}, indent=2), encoding='utf-8')
    print(json.dumps(means), flush=True)


if __name__ == '__main__': main()
