"""Review all audited training rows, including every failure and loss."""
import json
from collections import Counter
from run_latent_frs_training import ROOT, FOLDER, digest
from continue_soft_topk_knn_pilot import verify


def main():
    verify(FOLDER); summary = json.loads((FOLDER/'complete.json').read_text()); counts = Counter()
    for row in summary['rows']:
        for fold in row['folds']:
            gain = fold['selected_training_bound_decrease_vs_fixed_three_pp']
            counts['better_training_bound_folds'] += gain > 1e-8
            counts['worse_training_bound_folds'] += gain < -1e-8
            counts['equal_training_bound_folds'] += abs(gain) <= 1e-8
            counts['same_weight_strict_tightening_folds'] += fold['same_weight_tightening_at_fixed_three_endpoint_pp'] > 1e-8
            counts['additional_latent_optimization_improved_folds'] += fold['latent_objective_improvement_beyond_fixed_three_endpoint'] > 1e-9
            counts['additional_latent_optimization_worse_folds'] += fold['latent_objective_improvement_beyond_fixed_three_endpoint'] < -1e-9
            for point in fold['at_identical_weights'].values():
                assert point['latent_five_bound'] <= point['fixed_three_bound']+1e-10
    review = {'audit_sha256': digest(FOLDER/'complete-audit.json'), 'all_dataset_rows': summary['rows'],
        'dataset_balanced_metrics': summary['dataset_balanced_metrics'], 'optimizer_failures': summary['optimizer_failures'],
        'counts': dict(counts), 'datasets': summary['datasets'], 'inner_training_folds': summary['inner_training_folds'],
        'classification_scored': False, 'classification_superiority_established': False, 'goal_achieved': False,
        'decision': 'Review witness tightening, optimization and all failures separately before any equally enhanced classifier family.',
        'limits': ['Training bounds are not validation/test classification.', 'All15 are exposed development.',
            'Latent majorization and minimum-of-bounds are generic prior art.', 'No favorable-data subset or significance claim.',
            'Ordinary floating point; no directed interval or global optimum guarantee.']}
    (ROOT/'research/latent-FRS-training-review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in review.items() if k != 'all_dataset_rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
