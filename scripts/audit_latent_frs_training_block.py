"""Independent matched-start and scalar-bound replay for latent ablation."""
import numpy as np
from audit_capped_frs_warm_fit import check_capped_warm_fit, ScalarRankValidator
from audit_latent_frs_fit_cached import check_latent_fit
from independent_latent_friend_rank_cached import ScalarLatentValidator
from audit_forest_compact_transfer import equivalent

ARMS = ('fixed_three', 'latent_five')
STARTS = ('anchor', 'capped_endpoint')


def independent_evaluation(A, y, block):
    # Validation uses stored witnesses already independently reconstructed by
    # check_block, scalar contrasts and direct order statistics. No production
    # loss oracle or production summary helper is called.
    fits = {a: block['arms'][a]['fits'][block['arms'][a]['selected_start']] for a in ARMS}
    f_info = fits['fixed_three']['fit']['training_witnesses']
    l_info = fits['latent_five']['fit']['training_witnesses']
    mass = A.shape[1]**.5; fs = block['features']; anchor = np.asarray(block['anchor'])/mass
    fixed = ScalarRankValidator(A[:, fs], f_info); latent = ScalarLatentValidator(A[:, fs], l_info)
    alpha = np.asarray([1/(len(np.unique(y))*sum(y == c)) for c in y])
    points = {'anchor': block['anchor'], 'capped_endpoint': block['capped_endpoint'],
        'fixed_three_selected': fits['fixed_three']['target'], 'latent_five_selected': fits['latent_five']['target']}
    same_weights = {}
    for name, coordinates in points.items():
        point = np.asarray(coordinates); w = point/mass
        f = np.minimum(1., fixed.losses(w)); l = latent.losses(w)
        assert np.all(l <= f+1e-10)
        penalty = .01*float(sum(abs(w-anchor)))
        same_weights[name] = {'fixed_three_bound': float(alpha@f), 'latent_five_bound': float(alpha@l),
            'fixed_three_regularized_objective': float(alpha@f+penalty),
            'latent_five_regularized_objective': float(alpha@l+penalty),
            'same_weight_witness_tightening_pp': 100*float(alpha@(f-l)),
            'queries_strictly_tightened': int(sum(l < f-1e-10)),
            'queries_constant_one': int(sum(r['constant_loss'] == 1 for r in f_info['witnesses'])),
            'queries_constant_zero': int(sum(r['constant_loss'] == 0 for r in f_info['witnesses'])),
            'exact_zero_weights': int(sum(point == 0))}
    control, final = same_weights['fixed_three_selected'], same_weights['latent_five_selected']
    return {'at_identical_weights': same_weights,
        'selected_training_bound_decrease_vs_fixed_three_pp': 100*(control['fixed_three_bound']-final['latent_five_bound']),
        'same_weight_tightening_at_fixed_three_endpoint_pp': control['same_weight_witness_tightening_pp'],
        'latent_objective_improvement_beyond_fixed_three_endpoint':
            control['latent_five_regularized_objective']-final['latent_five_regularized_objective'],
        'regularized_training_objective_decrease_vs_fixed_three':
            control['fixed_three_regularized_objective']-final['latent_five_regularized_objective'],
        'optimizer_failures': {a: int(sum(r['fit']['subproblem_failures'] for r in block['arms'][a]['fits'].values())) for a in ARMS},
        'scope': 'training_bound_tightening_and_learning_not_prediction_or_significance'}


def check_block(A, y, block, counts):
    assert set(block['arms']) == set(ARMS)
    for arm, checker in (('fixed_three', check_capped_warm_fit), ('latent_five', check_latent_fit)):
        a = block['arms'][arm]; assert set(a['fits']) == set(STARTS)
        for start in STARTS:
            r = a['fits'][start]; fit = r['fit']
            assert r['features'] == block['features']; equivalent(r['anchor'], block['anchor'])
            expected = block['anchor'] if start == 'anchor' else block['capped_endpoint']
            equivalent(fit['requested_initial_weights'], expected)
            assert fit['initialization_scope'] == ('anchor' if start == 'anchor' else 'audited_capped_selected_training_endpoint')
            assert fit['reference_scope'] == 'original_full' and fit['constrained']
            assert fit['solver_primal_feasibility_tolerance'] == fit['solver_dual_feasibility_tolerance'] == 1e-9
            checker(A, y, r, counts)
        expected_start = min(STARTS, key=lambda s: (a['fits'][s]['fit']['training_objective_final'], STARTS.index(s)))
        assert a['selected_start'] == expected_start
    # Feasible region and anchors are identical for BOTH models and starts.
    reference_fit = block['arms']['fixed_three']['fits']['anchor']['fit']
    for arm in ARMS:
        for r in block['arms'][arm]['fits'].values():
            equivalent(r['fit']['class_cell'], reference_fit['class_cell'])
            equivalent(r['fit']['polytope'], reference_fit['polytope'])
    independent = independent_evaluation(A, y, block)
    equivalent(independent, block['evaluation'])
    counts['matched_training_blocks'] += 1
    return independent
