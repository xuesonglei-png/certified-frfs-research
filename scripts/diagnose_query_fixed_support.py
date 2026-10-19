"""Same-support query-objective refitting; one fixed initialization/config."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.query_majority import QueryMajorityObjective
from certified_frfs.sparse_neighbor import optimize_cardinality_objective
from certified_frfs.validated_pool import preprocess_fit, validation_scores
from diagnose_query_majority import NAMES


def main():
    threadpool_limits(1); rows = []
    for name in NAMES:
        p = json.loads((ROOT/'results/predictive-iteration41'/f'{name}-s101-f0.json').read_text())
        X, y = load_dataset(name); tr = np.array(p['train_indices']); s = p['inner_splits'][0]
        ii, jj = np.array(s['train']), np.array(s['validation']); A, B = preprocess_fit(X[tr][ii], X[tr][jj])
        objective = QueryMajorityObjective(X[tr][ii], y[tr][ii], ridge=.01); candidates = []
        for fraction in [.35, .65, 1.]:
            old = next(r for r in s['fits'] if r['config']['ridge'] == .01 and r['config']['fraction'] == fraction)
            fs = np.array(old['features']); initial = np.asarray(old['weights'])[fs]; initial /= initial.sum()
            def reduced(v):
                wide = np.zeros(A.shape[1]); wide[fs] = v
                value, gradient = objective(wide)
                return value, gradient[fs]
            v, fit = optimize_cardinality_objective(reduced, initial, len(fs), ridge=.01)
            wide = np.zeros(A.shape[1]); wide[fs] = v
            scores = validation_scores(A, y[tr][ii], B, y[tr][jj], fs.tolist(), wide, 144, joint=True)
            prior = old['validation']; delta = (scores['clean']+scores['noise01']-prior['clean']-prior['noise01'])/2
            candidates.append({'fraction': fraction, 'allowed_support': fs.tolist(), 'actual_features': np.flatnonzero(wide > 0).tolist(),
                'weights': wide.tolist(), 'fit': fit, 'old_query_objective': objective(np.asarray(old['weights'])/sum(old['weights']))[0],
                'new_query_objective': objective(wide)[0], 'metrics': scores, 'prior': prior, 'utility_delta': delta})
        rows.append({'dataset': name, 'candidates': candidates, 'training_splits': objective.splits})
        print(name, [round(c['utility_delta'], 5) for c in candidates], flush=True)
    means = {str(f): float(np.mean([next(c['utility_delta'] for c in r['candidates'] if c['fraction'] == f) for r in rows])) for f in [.35, .65, 1.]}
    out = ROOT/'results/query-majority-diagnostic/fixed-support-inner0.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps({'rows': rows, 'fixed_fraction_mean_deltas': means,
        'scope': 'eight_fixed_inner_queries_same_allowed_support_diagnostic_not_test_superiority'}, indent=2), encoding='utf-8')
    print(json.dumps(means), flush=True)


if __name__ == '__main__': main()
