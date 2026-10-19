"""Balanced support ablation; no assertion of native baseline reproduction."""
import hashlib
import json
from pathlib import Path
from compare_conditional_backward_budget import original

SOURCE = 'nested-crossfit-classification'

if __name__ == '__main__':
    original.run(SOURCE)
    folder = original.ROOT/'results'/(SOURCE+'-common-budget')
    summary = json.loads((folder/'complete.json').read_text())
    for row in summary['rows']:
        path = folder/(row['dataset']+'.json')
        saved = json.loads(path.read_text())
        for f in (Path(__file__), original.ROOT/'scripts/compare_conditional_backward_budget.py',
                  original.ROOT/'scripts/balanced_fuzzy_controls.py'):
            saved['fingerprints'][str(f.relative_to(original.ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
        saved['class_balanced'] = True
        path.write_text(json.dumps(saved, indent=2), encoding='utf-8')
    summary['class_balanced'] = True
    summary['limitations'] = ('FGS/SFRIP are explicit budget adaptations, not native algorithms. '
        'Common balanced unconstrained classifier; frozen crossfit-selected certified supports, development ablation.')
    (folder/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
