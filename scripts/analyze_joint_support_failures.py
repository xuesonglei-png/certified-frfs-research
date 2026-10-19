"""Descriptive search optimism and constraint blockers, no new selection."""
import json
import numpy as np
from run_decision_iteration import ROOT
from diagnose_joint_support import FOLDER, NAMES


def main():
    rows = []
    for name in NAMES:
        p = json.loads((FOLDER/f'{name}.json').read_text())
        observations = []; blockers = {'majority_only': 0, 'dependency_only': 0, 'both': 0, 'feasible': 0}
        for s in p['splits']:
            fit = s['fit']; full = next(c for c in fit['pool'] if len(c['features']) == p['d'])
            utility = lambda m: (m['clean']+m['noise01'])/2
            external_full = utility(s['metrics']['full'])
            for branch in ('certified', 'unconstrained'):
                training_gain = fit[branch]['search_utility']-full['utility']
                external_gain = utility(s['metrics'][branch])-external_full
                observations.append({'branch': branch, 'training_search_gain': training_gain,
                    'external_inner_gain': external_gain, 'optimism_gap': training_gain-external_gain})
            for c in fit['certificates']:
                majority = c['majority_feasible']; dependency = c['dependency_certificate']['upper_bound'] <= .1
                key = 'feasible' if majority and dependency else 'both' if not majority and not dependency else 'majority_only' if not majority else 'dependency_only'
                blockers[key] += 1
        rows.append({'dataset': name, 'blockers': blockers, 'observations': observations,
            'mean_optimism': {b: float(np.mean([o['optimism_gap'] for o in observations if o['branch'] == b]))
                for b in ('certified', 'unconstrained')}})
    summary = {'rows': rows, 'mean_dataset_optimism': {b: float(np.mean([r['mean_optimism'][b] for r in rows]))
        for b in ('certified', 'unconstrained')},
        'blocker_totals': {k: sum(r['blockers'][k] for r in rows) for k in rows[0]['blockers']},
        'scope': 'adaptive_search_vs_external_inner_descriptive_dependent_records_not_inference'}
    (FOLDER/'failure-analysis.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
