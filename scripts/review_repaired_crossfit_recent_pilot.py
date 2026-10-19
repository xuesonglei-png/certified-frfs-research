"""Retain all class-level adverse outcomes and distinguish repairs from risk."""
import json
from pathlib import Path
import numpy as np
from run_flexible_majority_repair_admission import digest
from certified_frfs.datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'results/crossfit-label-loss-recent-pilot'
NEW = ROOT/'results/repaired-crossfit-recent-pilot'


if __name__ == '__main__':
    old, new = [json.loads((folder/'complete.json').read_text()) for folder in (OLD, NEW)]
    lookup = {r['dataset']: r for r in old['rows']}
    rows = []
    for row in new['rows']:
        name = row['dataset']
        a, b = [json.loads((folder/(name+'.json')).read_text()) for folder in (OLD, NEW)]
        X, y = load_dataset(name)
        ri, vi = np.array(a['reference_indices']), np.array(a['validation_indices'])
        assert a['reference_indices'] == b['reference_indices'] and a['validation_indices'] == b['validation_indices']
        repair_records = [r for block in b['inner']+[b['outer']] for repair in block['stage']['path']['repair_blocks'] for r in repair['attempts']]
        choices, recalls = {}, {}
        for method in ('proposed', 'full', 'fgs_matched', 'sfrip_matched'):
            selected = b['outer']['results'][method]
            model = b['outer']['models'][selected['model_id']]
            choices[method] = {'C': model['C'], 'knot_count': len(model['knots']), 'model_index': selected['model_index']}
            predictions = np.array(selected['evaluation']['predictions'])
            recalls[method] = {str(c): float(np.mean(predictions[y[vi] == c] == c)) if np.any(y[vi] == c) else None for c in np.unique(y)}
        rows.append({'dataset': name, 'd': row['d'], 'old_count': lookup[name]['actual_counts']['proposed'],
                     'new_count': row['actual_counts']['proposed'],
                     'new_minus_old_proposed_BA_pp': 100*(row['metrics']['proposed']['clean_BA']-lookup[name]['metrics']['proposed']['clean_BA']),
                     'new_minus_full_BA_pp': 100*(row['metrics']['proposed']['clean_BA']-row['metrics']['full']['clean_BA']),
                     'training_class_counts': {str(c): int(np.sum(y[ri] == c)) for c in np.unique(y)},
                     'query_class_counts': {str(c): int(np.sum(y[vi] == c)) for c in np.unique(y)},
                     'majority_required': b['outer']['stage']['candidate']['certificate']['required'],
                     'selected_classifier_configs': choices, 'query_class_recalls': recalls,
                     'repair_attempts': len(repair_records),
                     'accepted_repairs': sum(r['fit']['status'] == 'original_certificate_passed' for r in repair_records),
                     'same_support': a['outer']['stage']['candidate']['features'] == b['outer']['stage']['candidate']['features']})
    result = {'rows': rows, 'source_sha256': {'old': digest(OLD/'complete.json'), 'new': digest(NEW/'complete.json')},
              'new_complete_audit_present': (NEW/'complete-audit-v2.json').exists(),
              'preserved_v1_audit_failed_bitwise_cost_comparison': True,
              'validated_v2_audit': 'results/repaired-crossfit-recent-pilot/complete-audit-v2.json',
              'proposed_BA_pct': 100*new['means']['proposed']['clean_BA'],
              'contrasts': new['contrasts_clean_BA'], 'mean_removed': new['mean_removed'],
              'fixed_gate_passed': new['pilot_gate_passed'], 'goal_achieved': False,
              'scope': 'all15_exposed_development_adverse_results_retained_not_confirmatory_or_test_selected'}
    (ROOT/'research/repaired-crossfit-recent-pilot-review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print([(r['dataset'], r['old_count'], r['new_count'], r['new_minus_old_proposed_BA_pp']) for r in rows], flush=True)
