"""Descriptive guard-to-refit transfer and sufficient-certificate blockers."""
import json
import numpy as np
from run_decision_iteration import ROOT
from diagnose_guarded_support import FOLDER, NAMES


def main():
    rows = []
    utility = lambda m: (m['clean']+m['noise01'])/2
    for name in NAMES:
        p = json.loads((FOLDER/f'{name}.json').read_text()); observations = []
        blockers = {'feasible': 0, 'majority_only': 0, 'dependency_only': 0, 'both': 0}
        for s in p['splits']:
            fit = s['fit']; full_utility = utility(s['reference_metrics']['full'])
            for branch in ('certified', 'unconstrained'):
                c = fit[branch]; fs = c['features']; cert = c['certificate']
                observations.append({'branch': branch, 'guard_gain': c['guard_gain'],
                    'penalized_guard_gain': c['penalized_guard_gain'],
                    'external_refit_gain': utility(s['metrics'][branch])-full_utility,
                    'features': len(fs), 'full': len(fs) == p['d'],
                    'feasible': cert['feasible'], 'majority_feasible': cert['majority_feasible'],
                    'dependency_feasible': cert['dependency_certificate']['upper_bound'] <= .1})
            for c in fit['certificates']:
                majority = c['majority_feasible']; dep = c['dependency_certificate']['upper_bound'] <= .1
                key = 'feasible' if majority and dep else 'both' if not majority and not dep else 'majority_only' if not majority else 'dependency_only'
                blockers[key] += 1
        rows.append({'dataset': name, 'observations': observations, 'blockers': blockers})
    selected = [o for r in rows for o in r['observations'] if o['branch'] == 'unconstrained' and not o['full']]
    result = {'rows': rows, 'blocker_counts': {k: sum(r['blockers'][k] for r in rows) for k in rows[0]['blockers']},
        'selected_unconstrained_nonfull': len(selected),
        'selected_unconstrained_nonfull_rejected': sum(not o['feasible'] for o in selected),
        'selected_unconstrained_nonfull_external_positive': sum(o['external_refit_gain'] > 1e-12 for o in selected),
        'mean_selected_nonfull_guard_gain': float(np.mean([o['guard_gain'] for o in selected])),
        'mean_selected_nonfull_external_gain': float(np.mean([o['external_refit_gain'] for o in selected])),
        'scope': 'dependent_selected_support_diagnostics_not_inference_guard_gain_is_selection_optimistic'}
    (FOLDER/'failure-analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
