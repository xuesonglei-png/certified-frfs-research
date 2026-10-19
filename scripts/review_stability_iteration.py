"""Audit training-only aggregation traces and report all pilot outcomes."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from summarize_predictive_iterations import summarize

ROOT = Path(__file__).resolve().parents[1]
directory = ROOT/'results/predictive-iteration7'
summarize(directory)
lines = ['# Stability-selection pilot review', '',
         'Development evidence only. Subsample certificates do not transfer to aggregate subsets.', '',
         '| Dataset | Fold | Robust stable | Nominal stable | Relief stable | All | Removed |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
records = []
for path in sorted(directory.glob('*-s*-f*.json')):
    obj = json.loads(path.read_text())
    rows = {r['method']: r for r in obj['rows']}
    robust = rows['decision_robust_stable']
    diagnostics = []
    for record in obj['stability_records']:
        subsets = [set(r['features']) for r in record['subsamples']]
        count = np.zeros(obj['d'])
        for sample in record['subsamples']:
            assert len(sample['indices']) == len(set(sample['indices']))
            count[sample['features']] += 1
        assert np.allclose(count/len(subsets), record['frequency'])
        assert sorted(record['ranking']) == list(range(obj['d']))
        jaccard = [len(a&b)/len(a|b) for a,b in combinations(subsets, 2)]
        diagnostics.append({'method': record['method'], 'training_sha256':record['training_sha256'],
                            'mean_jaccard': float(np.mean(jaccard)),
                            'mean_selected': float(np.mean([len(s) for s in subsets]))})
    matched = {}
    for r in obj['stability_records']:
        memberships = [sample['indices'] for sample in r['subsamples']]
        if r['training_sha256'] in matched:
            assert memberships == matched[r['training_sha256']]
        matched[r['training_sha256']] = memberships
    metrics = {m:r['metrics']['0.0']['balanced_accuracy'] for m,r in rows.items()}
    record = {'dataset':obj['dataset'], 'fold':obj['fold'], 'metrics':metrics,
              'robust_nominal_same_subset':robust['features']==rows['decision_nominal_stable']['features'],
              'removed':1-len(robust['features'])/obj['d'], 'diagnostics':diagnostics,
              'seconds':{m:r['seconds'] for m,r in rows.items()},
              'coverage':robust['coverage']}
    records.append(record)
    lines.append(f"| {obj['dataset']} | {obj['fold']} | {metrics['decision_robust_stable']:.4f} | "
                 f"{metrics['decision_nominal_stable']:.4f} | {metrics['relief_stable']:.4f} | "
                 f"{metrics['all']:.4f} | {record['removed']:.4f} |")
lines += ['', 'Audit: frequencies reconstructed from memberships; matched subsamples across',
          'objectives; source rankings are permutations. Report mean Jaccard in pilot-review.json.',
          'Runtime includes cached aggregation, inner prediction evaluations and final coverage.',
          'Cache key uses training-table and labels digest; validation labels never enter ranking.',
          'This pilot cannot establish significant predictive superiority or publication readiness.']
(directory/'pilot-review.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
(ROOT/'research/predictive-iteration7-review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'completed':len(records), 'same_robust_nominal':sum(r['robust_nominal_same_subset'] for r in records)},indent=2))
