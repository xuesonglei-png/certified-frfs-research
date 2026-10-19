"""Leave-one-dataset influence and surrogate/vote mismatch, descriptive only."""
import json
import numpy as np
from run_decision_iteration import ROOT


def main():
    folder = ROOT/'results/feasible-query-ablation'; p = json.loads((folder/'complete.json').read_text())
    rows = []; utility = lambda m: (m['clean']+m['noise01'])/2
    for r in p['rows']:
        differences = {'certified_vs_unconstrained': r['means']['certified']-r['means']['unconstrained'],
            'certified_vs_full': r['means']['certified']-r['means']['full'],
            'unconstrained_vs_full': r['means']['unconstrained']-r['means']['full']}
        losses = [s['fit']['trace'][0]['objective']-s['fit']['trace'][-1]['objective'] for s in r['splits']]
        rows.append({'dataset': r['dataset'], 'deltas': differences,
            'mean_training_surrogate_drop_unconstrained': float(np.mean(losses)),
            'removed_unconstrained': float(np.mean([1-len(s['fit']['features'])/r['d'] for s in r['splits']])),
            'mean_seconds_unconstrained': float(np.mean([s['seconds'] for s in r['splits']]))})
    leave_one_out = []
    for r in rows:
        other = [q for q in rows if q['dataset'] != r['dataset']]
        leave_one_out.append({'omitted_for_influence_only': r['dataset'],
            'mean_deltas': {k: float(np.mean([q['deltas'][k] for q in other])) for k in r['deltas']}})
    result = {'rows': rows, 'leave_one_dataset_out': leave_one_out,
        'scope': 'descriptive_influence_every_dataset_retained_in_primary_analysis_not_selected_study_or_inference'}
    (folder/'influence-analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'mean_deltas': {k: float(np.mean([r['deltas'][k] for r in rows])) for k in rows[0]['deltas']},
        'leave_one_out_ranges': {k: [min(r['mean_deltas'][k] for r in leave_one_out), max(r['mean_deltas'][k] for r in leave_one_out)] for k in rows[0]['deltas']}}, indent=2))


if __name__ == '__main__': main()
