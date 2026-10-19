"""All86 audited training pairs: optimization headroom versus rejected targets."""
import json
from collections import Counter
import numpy as np
from run_guarded_soft5_descent_admission import ROOT, FOLDER, PARENT, digest
from continue_soft_topk_knn_pilot import verify


def main():
    ga = verify(FOLDER)
    simplex = ROOT/'results/simplex-soft5-target-admission'; sa = verify(simplex)
    review = json.loads((ROOT/'research/simplex-soft5-target-admission-review.json').read_text())
    assert review['audit_sha256'] == digest(simplex/'complete-audit.json')
    assert review['objective_source'] == 'independent_scalar_Brent_actual_returned_targets_including_fallbacks'
    scores = {row['dataset']: row['targets'] for row in review['rows']}
    rows, counts = [], Counter()
    for name in sorted(ga['artifact_hashes']):
        guarded = json.loads((FOLDER/(name+'.json')).read_text())
        unconstrained_domain = json.loads((simplex/(name+'.json')).read_text())
        assert guarded['reference_indices'] == unconstrained_domain['reference_indices']
        assert len(guarded['records']) == len(unconstrained_domain['records']) == len(scores[name])
        targets = []
        for g, s, score in zip(guarded['records'], unconstrained_domain['records'], scores[name]):
            assert g['features'] == s['features'] == score['features']
            np.testing.assert_array_equal(g['anchor'], s['anchor'])
            joint_loss = g['fit']['training_objective_final']
            gap = joint_loss-score['actual_returned_target_objective']
            admitted = s['mixtures'][-1]['accepted']
            counts['supports'] += 1
            counts['joint_budget_exhaustions'] += 'budget_exhausted' in g['fit']['status']
            if gap > 1e-8:
                counts['certified_lower_loss_simplex_targets' if admitted else 'rejected_lower_loss_simplex_targets'] += 1
            elif gap < -1e-8: counts['lower_loss_joint_targets'] += 1
            else: counts['tied_targets'] += 1
            targets.append({'features': g['features'], 'joint_status': g['fit']['status'],
                            'joint_loss_minus_simplex_actual_loss': gap,
                            'simplex_target_original_admitted': admitted})
        rows.append({'dataset': name, 'targets': targets})
    assert counts['supports'] == 86 and ga['counts']['supports'] == sa['counts']['supports'] == 86
    result = {'counts': dict(counts), 'rows': rows, 'goal_achieved': False,
              'query_labels_or_accuracy_used': False,
              'scope': 'audited_training_optimization_headroom_not_accuracy_or_optimality',
              'joint_audit_sha256': digest(FOLDER/'complete-audit.json'),
              'simplex_audit_sha256': digest(simplex/'complete-audit.json')}
    (ROOT/'research/guarded-soft5-budget-headroom.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__': main()
