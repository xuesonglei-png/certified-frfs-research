"""Training-only class coverage of the unchanged global dependency contract."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from certified_frfs.datasets import load_dataset
BASE = ROOT/'results/repaired-crossfit-recent-pilot'


if __name__ == '__main__':
    report = json.loads((BASE/'complete.json').read_text())
    rows = []
    for row in report['rows']:
        name = row['dataset']
        saved = json.loads((BASE/(name+'.json')).read_text())
        X, y = load_dataset(name)
        ri = np.array(saved['reference_indices'])
        for index, block in enumerate(saved['inner']+[saved['outer']]):
            labels = y[ri[np.array(block['reference_indices'])]] if index < len(saved['inner']) else y[ri]
            cert = block['stage']['candidate']['certificate']
            dep = cert['dependency_certificate']
            if 'redistributed_object_bounds' in dep:
                values = np.array(dep['redistributed_object_bounds'])
            else:
                assert block['stage']['actual_count'] == saved['d'] and dep['upper_bound'] == 0.
                values = np.zeros(len(labels))
            classes = np.unique(labels)
            bounds = {}
            for c, required in zip(classes, cert['required']):
                raw = float(np.mean(values[labels == c]))
                bounds[str(c)] = {'count': int(np.sum(labels == c)), 'majority_required': required,
                                  'classwise_dependency_upper': min(1., raw+1e-10*(1+abs(raw)))}
            rows.append({'dataset': name, 'stage_index': index,
                         'outer_stage': index == len(saved['inner']),
                         'actual_count': block['stage']['actual_count'],
                         'accepted_repairs': sum(a['fit']['status'] == 'original_certificate_passed' for r in block['stage']['path']['repair_blocks'] for a in r['attempts']),
                         'global_dependency_upper': dep['upper_bound'], 'classes': bounds,
                         'all_class_dependency_leq_point1': all(v['classwise_dependency_upper'] <= .1 for v in bounds.values())})
    result = {'rows': rows, 'query_labels_or_accuracy_used': False,
              'scope': 'classwise_reaveraging_of_sound_objectwise_reference_bounds_not_classifier_risk_or_novelty'}
    (ROOT/'results/classwise-repair-dependency-diagnosis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print([r for r in rows if r['outer_stage'] and r['accepted_repairs']], flush=True)
