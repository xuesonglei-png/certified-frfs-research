"""All15 paired recovery/compression review; preserve negative predecessors."""
import json
from pathlib import Path
import numpy as np
from run_safeguarded_crossfit_recent_pilot import FOLDER
from run_flexible_majority_repair_admission import digest
from audit_forest_compact_transfer import equivalent
from review_box_iterations import paired

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'results/repaired-crossfit-recent-pilot'
OOF = ROOT/'results/crossfit-label-loss-recent-pilot'


if __name__ == '__main__':
    summaries = {arm: json.loads((folder/'complete.json').read_text()) for arm, folder in [('repaired', OLD), ('OOF', OOF), ('guarded', FOLDER)]}
    lookups = {arm: {r['dataset']: r for r in summary['rows']} for arm, summary in summaries.items()}
    rows = []
    for name in lookups['guarded']:
        current = lookups['guarded'][name]
        old, initial = lookups['repaired'][name], lookups['OOF'][name]
        for comparator in (old, initial):
            for method in ('full', 'fgs_native', 'sfrip_native'):
                equivalent(current['metrics'][method], comparator['metrics'][method])
        artifact = json.loads((FOLDER/(name+'.json')).read_text())
        repairs = [a for b in artifact['inner']+[artifact['outer']] for r in b['stage']['path']['repair_blocks'] for a in r['attempts']]
        rows.append({'dataset': name, 'proposed_BA': current['metrics']['proposed']['clean_BA'],
                     'guarded_minus_repaired_BA': current['metrics']['proposed']['clean_BA']-old['metrics']['proposed']['clean_BA'],
                     'guarded_minus_OOF_BA': current['metrics']['proposed']['clean_BA']-initial['metrics']['proposed']['clean_BA'],
                     'count_repaired': old['actual_counts']['proposed'], 'count_guarded': current['actual_counts']['proposed'],
                     'guard_rejections': sum(a['fit']['status'] == 'classwise_guard_failed' for a in repairs),
                     'all_newly_accepted_repairs_classwise_safe': all(a['fit']['classwise_dependency_certificate']['feasible_all_classes'] for a in repairs if a['fit']['status'] == 'original_certificate_passed')})
    result = {'rows': rows, 'source_sha256': {arm: digest(folder/'complete.json') for arm, folder in [('repaired', OLD), ('OOF', OOF), ('guarded', FOLDER)]},
              'guarded_independent_audit_complete': (FOLDER/'complete-audit.json').exists(),
              'guarded_minus_repaired_descriptive': paired([r['guarded_minus_repaired_BA'] for r in rows]),
              'guarded_minus_OOF_descriptive': paired([r['guarded_minus_OOF_BA'] for r in rows]),
              'mean_removed': summaries['guarded']['mean_removed'], 'means': summaries['guarded']['means'],
              'contrasts': summaries['guarded']['contrasts_clean_BA'], 'gate_passed': summaries['guarded']['pilot_gate_passed'],
              'goal_achieved': False,
              'scope': 'all15_exposed_development_targeted_recovery_not_confirmation_or_whole_path_classwise_guarantee'}
    (ROOT/'research/safeguarded-crossfit-recent-pilot-review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k not in ('rows', 'means', 'contrasts')}, flush=True)
