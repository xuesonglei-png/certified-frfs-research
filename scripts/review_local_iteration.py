"""Pilot review including paired nominal/robust search and runtime."""
import json
from pathlib import Path
import numpy as np
from summarize_predictive_iterations import summarize

ROOT = Path(__file__).resolve().parents[1]
directory = ROOT/'results/predictive-iteration6'
summarize(directory)
lines = ['# Local-search pilot review', '',
         'All pilot data are development evidence. No dataset is omitted by score.', '',
         '| Dataset | Robust | Nominal local | Accuracy local | All | Robust removed | Robust seconds |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
records = []
for path in sorted(directory.glob('*-s*-f*.json')):
    obj = json.loads(path.read_text())
    rows = {r['method']: r for r in obj['rows']}
    metrics = {m: r['metrics']['0.0']['balanced_accuracy'] for m, r in rows.items()}
    robust = rows['decision_robust_local']
    records.append({'dataset': obj['dataset'], 'metrics': metrics,
                    'robust_nominal_same_subset': robust['features']==rows['decision_nominal_local']['features'],
                    'removed': 1-len(robust['features'])/obj['d'],
                    'seconds': {m:r['seconds'] for m,r in rows.items()},
                    'robust_coverage': robust['coverage']})
    lines.append(f"| {obj['dataset']} | {metrics['decision_robust_local']:.4f} | "
                 f"{metrics['decision_nominal_local']:.4f} | {metrics['decision_accuracy_local']:.4f} | "
                 f"{metrics['all']:.4f} | {records[-1]['removed']:.4f} | {robust['seconds']:.2f} |")
lines += ['', 'Runtime includes all inner configuration fits, outer fit and final coverage.',
          'Ordinary prefix baselines use fewer subset evaluations; differences versus them',
          'do not isolate the robust objective. Matched local ablations are required.',
          'Accuracy-only configurations contain duplicate margin scales, an explicitly',
          'recorded inefficiency rather than distinct tuning opportunities.', '',
          'A local search can improve its training objective and degrade outer predictions.',
          'Do not expand a costly pilot merely to seek a significant p-value. Further',
          'iterations should assess cross-fitted/stability-based search rather than make',
          'the same training-table objective greedier. Independent confirmation remains',
          'necessary after freezing the method.']
(directory/'pilot-review.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
(ROOT/'research/predictive-iteration6-review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'completed':len(records), 'same_robust_nominal_subset':sum(r['robust_nominal_same_subset'] for r in records)},indent=2))
