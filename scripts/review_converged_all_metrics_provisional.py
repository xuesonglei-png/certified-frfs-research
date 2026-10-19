"""All70reported metric contrasts, no selective certified-accuracy significance."""
import json
import numpy as np
from run_converged_geometric_benchmark import ROOT, FOLDER, digest
from review_box_iterations import paired
from run_nested_joint_path import holm


def main():
    path = FOLDER/'complete.json'; summary = json.loads(path.read_text())
    assert summary['counts']['datasets'] == 15 and not summary['goal_achieved']
    direction = {'clean_BA': 1, 'balanced_logloss': -1, 'macro_F1': 1, 'query_certified_BA': 1, 'mean_perturbed_BA': 1}
    contrasts = {}
    for method in summary['means']:
        if method == 'proposed':
            continue
        for metric, sign in direction.items():
            differences = [sign*(row['metrics']['proposed'][metric]-row['metrics'][method][metric]) for row in summary['rows']]
            contrasts[method+'/'+metric] = {**paired(differences), 'oriented_raw_mean_difference': float(np.mean(differences)),
                'unit': '100_times_natural_log_loss_difference_not_percentage_points' if metric == 'balanced_logloss'
                        else 'percentage_point_change_of_0_to_1_metric',
                'positive_direction_favors_proposed': True}
    assert len(contrasts) == 70
    corrected = holm({key: value['sign_flip_two_sided_p'] for key, value in contrasts.items()})
    result = {'contrasts': contrasts, 'holm_all70': corrected,
        'nominal_significant_after_all70_holm': [key for key, value in corrected.items() if value < .05],
        'mean_removed': {method: float(np.mean([1-r['actual_counts'][method]/r['d'] for r in summary['rows']]))
                         for method in summary['means']},
        'summary_sha256': digest(path), 'reviewer_sha256': digest(ROOT/'scripts/review_converged_all_metrics_provisional.py'),
        'input_independent_audit_pending': True, 'goal_achieved': False, 'classification_superiority_established': False,
        'scope': 'PROVISIONAL_exposed_development_all_recorded_metric_contrasts_not_confirmation',
        'limits': ['Original full pipeline independent audit remains required.',
                   'All70 recorded contrasts are included to avoid selecting a favorable unadjusted certified-BA result.',
                   'This is an exploratory development analysis; correction within70 cannot undo prior sequential method development.',
                   'Native and count-matched/equally enhanced controls have different compression and learning budgets as recorded.',
                   'No algorithm/threshold/selection is changed from these outer outcomes.']}
    (ROOT/'research/converged-all70-metrics-PROVISIONAL-review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'tests': len(contrasts), 'significant': result['nominal_significant_after_all70_holm']}, indent=2))


if __name__ == '__main__':
    main()
