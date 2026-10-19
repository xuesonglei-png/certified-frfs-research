"""Describe failure modes without selecting a model by external scores."""
import json
import numpy as np
from diagnose_vote_aligned_refinement import runner


def main():
    records = []
    for path in sorted(runner.FOLDER.glob('*.json')):
        if path.name.startswith('complete'):
            continue
        p = json.loads(path.read_text()); splits = p['splits']
        rejection = {'no_descent': 0, 'hard_vote_drop': 0, 'certificate_failed': 0, 'accepted': 0}
        for s in splits:
            trace = s['fit']['trace']
            for a in s['fit']['attempts']:
                previous = trace[a['iteration']-1]
                if a['objective'] >= previous['objective']-1e-9:
                    rejection['no_descent'] += 1
                elif a['training_utility'] < previous['training_utility']-1e-12:
                    rejection['hard_vote_drop'] += 1
                elif not a['certificate']['feasible']:
                    rejection['certificate_failed'] += 1
                else:
                    rejection['accepted'] += 1
        metrics = {key: float(np.mean([s['metrics'][key] for s in splits])) for key in ('clean', 'noise01')}
        gains = {key: float(np.mean([s['metrics'][key]-s['initial_metrics'][key] for s in splits])) for key in metrics}
        records.append({'dataset': p['dataset'], 'metrics': metrics, 'gains': gains,
                        'training_gain': float(np.mean([s['fit']['trace'][-1]['training_utility']-s['fit']['trace'][0]['training_utility'] for s in splits])),
                        'rejection': rejection})
    summary = {'rows': records,
               'rejection': {k: sum(r['rejection'][k] for r in records) for k in records[0]['rejection']},
               'mean_training_gain': float(np.mean([r['training_gain'] for r in records])),
               'mean_gains': {k: float(np.mean([r['gains'][k] for r in records])) for k in ('clean', 'noise01')},
               'scope': 'adaptive_training_and_exposed_development_failure_diagnostic_not_confirmation'}
    (runner.FOLDER/'complete-review.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
