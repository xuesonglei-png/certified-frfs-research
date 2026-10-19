"""Matched two-start prototype; no held-out label API or real-data driver yet."""
import time
import numpy as np
from fit_pair_capped_frs_hinge import fit_pair_capped_frs_target
from fit_saturation_capped_frs_hinge import fit_saturation_capped_frs_target
from frs_pair_transition_cell import evaluate_class_cell as old_value
from frs_saturation_transition_cell import evaluate_class_cell as new_value

ARMS = ('nonnegative_pair', 'signed_saturation_pair')
STARTS = ('anchor', 'capped_endpoint')


def fit_block(A, y, features, anchor, full_reference, capped_endpoint):
    initial = {'anchor':None, 'capped_endpoint':np.asarray(capped_endpoint)}
    block = {'features':list(features), 'anchor':np.asarray(anchor).tolist(),
        'capped_endpoint':np.asarray(capped_endpoint).tolist(), 'arms':{}, 'classification_scored':False}
    for arm, function in (('nonnegative_pair', fit_pair_capped_frs_target),
                          ('signed_saturation_pair', fit_saturation_capped_frs_target)):
        fits = {}
        for start in STARTS:
            timer = time.perf_counter()
            target, fit = function(A, y, features, anchor, full_reference=full_reference,
                reference_scope='original_full', initialization=initial[start],
                initialization_scope='anchor' if start == 'anchor' else 'audited_capped_selected_training_endpoint')
            fits[start] = {'features':list(features), 'anchor':np.asarray(anchor).tolist(),
                'target':target.tolist(), 'fit':fit, 'fit_seconds':time.perf_counter()-timer}
        selected = min(STARTS, key=lambda s:(fits[s]['fit']['training_objective_final'], STARTS.index(s)))
        block['arms'][arm] = {'fits':fits, 'selected_start':selected}
    old = block['arms'][ARMS[0]]; new = block['arms'][ARMS[1]]
    a, z = old['fits'][old['selected_start']], new['fits'][new['selected_start']]
    cell = new['fits']['anchor']['fit']['class_cell']; same = {}
    for name, point in (('anchor', anchor), ('capped_endpoint', capped_endpoint),
                        ('nonnegative_selected', a['target']), ('signed_selected', z['target'])):
        before, after = old_value(cell, point), new_value(cell, point)
        assert all(b <= v+1e-12 for b, v in zip(after['raw_upper_bounds_with_inflation'], before['raw_upper_bounds_with_inflation'], strict=True))
        same[name] = {'nonnegative_cell':before, 'signed_saturation_cell':after,
            'negative_class_bounds':sum(v < 0 for v in after['raw_upper_bounds_with_inflation'])}
    block['evaluation'] = {'training_objective_decrease_vs_nonnegative':a['fit']['training_objective_final']-z['fit']['training_objective_final'],
        'clipped_training_bound_decrease_pp':100*(a['fit']['training_error_upper_bound_final']-z['fit']['training_error_upper_bound_final']),
        'same_weight_cell_values':same, 'signed_endpoint_outside_old_cell':not same['signed_selected']['nonnegative_cell']['feasible'],
        'exact_zero_weights':{arm:sum(v == 0 for v in block['arms'][arm]['fits'][block['arms'][arm]['selected_start']]['target']) for arm in ARMS},
        'optimizer_failures':{arm:sum(r['fit']['subproblem_failures'] for r in block['arms'][arm]['fits'].values()) for arm in ARMS}}
    return block
