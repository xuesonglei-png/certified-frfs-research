"""Complete-only local-search review; all endpoints and negative cases retained."""
import json
from collections import Counter
import numpy as np
from run_guarded_soft5_descent_admission import ROOT, FOLDER, PARENT, digest
from continue_soft_topk_knn_pilot import verify


def main():
    audit = verify(FOLDER); verify(PARENT)
    assert audit['counts']['datasets'] == 15 and audit['counts']['supports'] == 86
    rows, counts, improvements = [], Counter(), []
    for name in sorted(audit['artifact_hashes']):
        saved = json.loads((FOLDER/(name+'.json')).read_text()); targets = []
        for r in saved['records']:
            fit = r['fit']; gain = fit['training_objective_initial']-fit['training_objective_final']
            assert fit['endpoint_guard']['accepted']
            counts['supports'] += 1; counts['status_'+fit['status']] += 1
            counts['guard_checks'] += len(fit['checks'])
            counts['rejected_guard_checks'] += sum(not q['accepted'] for q in fit['checks'])
            counts['accepted_steps'] += len(fit['accepted_step_ids'])
            counts['improved_actual_target_objectives'] += gain > 1e-8
            counts['certified_endpoints'] += r['mixtures'][-1]['accepted']
            improvements.append(gain)
            targets.append({'features': r['features'], 'status': fit['status'],
                            'guard_checks': len(fit['checks']), 'accepted_steps': len(fit['accepted_step_ids']),
                            'actual_training_objective_decrease': gain,
                            'admitted_mixtures': sum(m['accepted'] for m in r['mixtures']),
                            'parent_admitted_mixtures': r['previous_admitted_mixtures'],
                            'weight_shift_l1': float(abs(np.array(r['target'])-r['anchor']).sum())})
        rows.append({'dataset': name, 'targets': targets})
    assert counts['supports'] == counts['certified_endpoints'] == 86
    result = {'rows': rows, 'counts': {k: int(v) for k, v in counts.items()},
              'admission_counts': audit['counts'], 'mean_training_objective_decrease': float(np.mean(improvements)),
              'query_labels_or_accuracy_used': False, 'goal_achieved': False,
              'scope': 'audited_local_certified_training_descent_not_classification_or_optimality',
              'audit_sha256': digest(FOLDER/'complete-audit.json')}
    (ROOT/'research/guarded-soft5-descent-admission-review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__': main()
