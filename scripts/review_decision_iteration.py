"""Report complete folds and explicit failures without dataset selection."""
import json
import argparse
from pathlib import Path
import numpy as np
from summarize_predictive_iterations import summarize

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--iteration', type=int, choices=[4, 5], default=4)
args = ap.parse_args()
directory = ROOT/f'results/predictive-iteration{args.iteration}'
target = 'decision_robust' if args.iteration == 4 else 'decision_margin'
summarize(directory)
summary = json.loads((directory/'summary.json').read_text())
failures = []
for path in sorted(directory.glob('*-s*-f*.json')):
    obj = json.loads(path.read_text())
    rows = {r['method']: r for r in obj['rows']}
    proposed = rows[target]
    reference = rows['all']
    delta = proposed['metrics']['0.0']['balanced_accuracy']-reference['metrics']['0.0']['balanced_accuracy']
    if delta < -.02 or proposed['coverage']['min_class_coverage'] == 0:
        failures.append({'dataset': obj['dataset'], 'fold': obj['fold'],
                         'delta_vs_all': delta, 'selected': len(proposed['features']),
                         'class_coverage': proposed['coverage']['class_coverage']})
lines = ['# Decision coverage iteration review', '',
         'Development evidence only; dataset means average outer folds before comparisons.', '',
         '| Method | Clean balanced accuracy | Radius .05 BA | Removed | Minimum class coverage |',
         '| --- | ---: | ---: | ---: | ---: |']
for method, row in summary['methods'].items():
    m = row['means']
    lines.append(f"| {method} | {m['accuracy']:.4f} | {m['noise05']:.4f} | {m['removed']:.4f} | {m['min_class_coverage']:.4f} |")
lines += ['', '## Paired dataset comparisons', '']
for row in summary['pairwise']:
    lines.append(f"- vs {row['comparator']}: delta {100*row['mean_delta']:.3f} percentage points; "
                 f"wins/ties/losses {row['wins_ties_losses']}; Holm p={row['holm_p']:.4f}.")
lines += ['', '## Explicit failure records', '', '```json', json.dumps(failures, indent=2), '```', '',
          'Coverage certificates concern robust LOO training decisions. They do not certify test risk.',
          'SFRIP is an independent implementation/configuration. MAF instance is not FD+ or FCE+.',
          'Positive average deltas alone do not establish significant superiority.',
          'Further direction: replace the discontinuous positive-margin count with a clipped',
          'continuous class-balanced margin utility; test it as an ablation with unchanged folds.',
          'Do not use outer scores to choose a configuration or omit failing datasets.']
(ROOT/f'research/predictive-iteration{args.iteration}-review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps({'splits': summary['completed_splits'], 'failure_count': len(failures)}, indent=2))
