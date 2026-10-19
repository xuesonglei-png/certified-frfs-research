"""Frozen eight datasets/all inner folds, training-only joint refit diagnostic."""
import json, hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_query_majority import NAMES
from certified_frfs.datasets import load_dataset
from certified_frfs.query_majority import QueryMajorityObjective, refit_query_support
from certified_frfs.validated_pool import preprocess_fit, validation_scores


def main():
    threadpool_limits(1); rows = []; folder = ROOT/'results/query-majority-diagnostic'; out = folder/'all-inner-fixed-support.json'
    if out.exists(): raise FileExistsError(out)
    for name in NAMES:
        path = ROOT/'results/predictive-iteration41'/f'{name}-s101-f0.json'; p = json.loads(path.read_text())
        X, y = load_dataset(name); tr = np.array(p['train_indices']); splits = []
        for index, s in enumerate(p['inner_splits']):
            ii, jj = np.array(s['train']), np.array(s['validation']); A, B = preprocess_fit(X[tr][ii], X[tr][jj])
            objective = QueryMajorityObjective(X[tr][ii], y[tr][ii], ridge=.01); fitted = []
            for fraction in [.35, .65, 1.]:
                old = next(r for r in s['fits'] if r['config']['ridge'] == .01 and r['config']['fraction'] == fraction)
                # Strip earlier validation scores from fitting arguments.
                candidate = {k: old[k] for k in ['config', 'features', 'weights', 'fit']}
                new = refit_query_support(X[tr][ii], y[tr][ii], candidate, objective=objective)
                scores = validation_scores(A, y[tr][ii], B, y[tr][jj], new['features'], np.array(new['weights']), 144+index, joint=True)
                prior = old['validation']; delta = (scores['clean']+scores['noise01']-prior['clean']-prior['noise01'])/2
                fitted.append({**new, 'metrics': scores, 'prior': prior, 'utility_delta': delta})
            splits.append({'inner': index, 'train': ii.tolist(), 'validation': jj.tolist(), 'candidates': fitted})
        deltas = {str(f): float(np.mean([next(c['utility_delta'] for c in s['candidates'] if c['config']['fraction'] == f) for s in splits])) for f in [.35, .65, 1.]}
        rows.append({'dataset': name, 'splits': splits, 'deltas': deltas, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        (folder/'all-inner-partial.json').write_text(json.dumps({'complete': False, 'rows': rows}, indent=2), encoding='utf-8')
        print(name, deltas, flush=True)
    means = {str(f): float(np.mean([r['deltas'][str(f)] for r in rows])) for f in [.35, .65, 1.]}
    # Require broad gain at all frozen budgets, without choosing a winning
    # budget retrospectively. This is a screening rule, not significance.
    gates = {'each_fraction_mean_gain_at_least_half_point': all(v >= .005 for v in means.values()),
        'at_least_five_positive_datasets_each_fraction': all(sum(r['deltas'][str(f)] > 1e-12 for r in rows) >= 5 for f in [.35, .65, 1.])}
    out.write_text(json.dumps({'complete': True, 'rows': rows, 'mean_deltas': means, 'gates': gates, 'gate_passed': all(gates.values()),
        'scope': 'eight_reused_development_outerfold0_all_training_innerfolds_not_outer_test_advantage'}, indent=2), encoding='utf-8')
    print(json.dumps(means), json.dumps(gates), flush=True)


if __name__ == '__main__': main()
