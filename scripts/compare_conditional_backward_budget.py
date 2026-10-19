"""Exact-count balanced support controls; no claim of native reproduction."""
import hashlib
import json
from pathlib import Path
import compare_nested_support_budget as original
from balanced_fuzzy_controls import train, final_metrics

original.train = train
original.final_metrics = final_metrics
SOURCE = 'nested-conditional-backward'

if __name__ == '__main__':
    original.run(SOURCE)
    folder = original.ROOT/'results'/(SOURCE+'-common-budget')
    summary = json.loads((folder/'complete.json').read_text())
    for row in summary['rows']:
        path = folder/(row['dataset']+'.json')
        saved = json.loads(path.read_text())
        for f in (Path(__file__), original.ROOT/'scripts/balanced_fuzzy_controls.py'):
            saved['fingerprints'][str(f.relative_to(original.ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
        saved['class_balanced'] = True
        path.write_text(json.dumps(saved, indent=2), encoding='utf-8')
    summary['class_balanced'] = True
    summary['limitations'] = ('FGS/SFRIP are explicit budget adaptations. Proposed conditional policy frozen after inner selection. '
        'All use class-balanced unconstrained logistic; support ablation, not certified deployment or confirmation.')
    (folder/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
