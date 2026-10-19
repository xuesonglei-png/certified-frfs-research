"""Matched two-start fixed-three/latent-five training experiment primitives.

Inputs are training arrays and already audited capped endpoints. There is
no validation/test-label API. Original anchors, cells and regularization
remain fixed. No fit failure is removed or converted into a success.
"""
import time
import numpy as np
from fit_strict_capped_frs_hinge import fit_strict_capped_frs_target
from fit_latent_frs_hinge import fit_latent_frs_target
from capped_third_enemy_hinge import CappedThirdEnemyHinge
from latent_friend_rank_hinge import LatentFriendRankHinge

ARMS = ('fixed_three', 'latent_five')
STARTS = ('anchor', 'capped_endpoint')
SCOPES = {'anchor': 'anchor', 'capped_endpoint': 'audited_capped_selected_training_endpoint'}


def selected_start(fits):
    return min(STARTS, key=lambda s: (fits[s]['fit']['training_objective_final'], STARTS.index(s)))


def fit_block(A, y, features, anchor, full_reference, capped_endpoint):
    anchor = np.asarray(anchor, float)
    endpoints = {'anchor': None, 'capped_endpoint': np.asarray(capped_endpoint, float)}
    block = {'features': list(features), 'anchor': anchor.tolist(),
             'capped_endpoint': np.asarray(capped_endpoint, float).tolist(), 'arms': {}}
    for arm, fit_function in (('fixed_three', fit_strict_capped_frs_target),
                              ('latent_five', fit_latent_frs_target)):
        fits = {}
        for start in STARTS:
            timer = time.perf_counter()
            target, fit = fit_function(A, y, features, anchor, full_reference=full_reference,
                reference_scope='original_full', initialization=endpoints[start],
                initialization_scope=SCOPES[start])
            fits[start] = {'features': list(features), 'anchor': anchor.tolist(),
                'target': target.tolist(), 'fit': fit, 'fit_seconds': time.perf_counter()-timer}
        block['arms'][arm] = {'fits': fits, 'selected_start': selected_start(fits)}
    block['evaluation'] = evaluate_block(A, y, block)
    return block


def evaluate_block(A, y, block):
    """Separate witness tightening at identical weights from metric changes."""
    mass = float(np.sqrt(A.shape[1])); anchor = np.asarray(block['anchor'])/mass
    fs = block['features']; fixed = CappedThirdEnemyHinge(A[:, fs], y, anchor)
    latent = LatentFriendRankHinge(A[:, fs], y, anchor)
    points = {'anchor': np.asarray(block['anchor']),
              'capped_endpoint': np.asarray(block['capped_endpoint'])}
    for arm in ARMS:
        a = block['arms'][arm]
        points[arm+'_selected'] = np.asarray(a['fits'][a['selected_start']]['target'])
    same_weights = {}
    for name, point in points.items():
        w = point/mass; f, l = fixed.losses(w), latent.losses(w)
        assert np.all(l <= f+1e-10)
        penalty = .01*float(abs(w-anchor).sum())
        same_weights[name] = {'fixed_three_bound': float(fixed.alpha@f),
            'latent_five_bound': float(latent.alpha@l),
            'fixed_three_regularized_objective': float(fixed.alpha@f+penalty),
            'latent_five_regularized_objective': float(latent.alpha@l+penalty),
            'same_weight_witness_tightening_pp': 100*float(fixed.alpha@(f-l)),
            'queries_strictly_tightened': int(sum(l < f-1e-10)),
            'queries_constant_one': int(sum(r['constant_loss'] == 1 for r in fixed.rows)),
            'queries_constant_zero': int(sum(r['constant_loss'] == 0 for r in fixed.rows)),
            'exact_zero_weights': int(sum(point == 0))}
    control = same_weights['fixed_three_selected']; final = same_weights['latent_five_selected']
    return {'at_identical_weights': same_weights,
        'selected_training_bound_decrease_vs_fixed_three_pp':
            100*(control['fixed_three_bound']-final['latent_five_bound']),
        'same_weight_tightening_at_fixed_three_endpoint_pp': control['same_weight_witness_tightening_pp'],
        'latent_objective_improvement_beyond_fixed_three_endpoint':
            control['latent_five_regularized_objective']-final['latent_five_regularized_objective'],
        'regularized_training_objective_decrease_vs_fixed_three':
            control['fixed_three_regularized_objective']-final['latent_five_regularized_objective'],
        'optimizer_failures': {a: int(sum(r['fit']['subproblem_failures'] for r in block['arms'][a]['fits'].values()))
                               for a in ARMS},
        'scope': 'training_bound_tightening_and_learning_not_prediction_or_significance'}
