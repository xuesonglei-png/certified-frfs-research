"""Matched three-start training-only attribution:cell geometry versus sparsity."""
import time
import numpy as np
from fit_saturation_capped_frs_hinge import fit_saturation_capped_frs_target
from fit_global_transition_capped_frs_hinge import fit_global_transition_capped_target
from fit_sparse_global_transition_frs import fit_sparse_global_transition_target
from frs_saturation_transition_cell import build_class_cell as fixed_cell, evaluate_class_cell as fixed_value
from frs_global_saturation_transition import build_class_cell as global_cell, evaluate_class_cell as global_value
from soft_topk_predictive_metric import anchor_witness_polytope
from capped_third_enemy_hinge import CappedThirdEnemyHinge

ARMS = ('fixed_signed_risk', 'global_signed_risk', 'global_signed_sparse')
STARTS = ('anchor', 'common_capped', 'common_zero')


def direct_feasible(info, C, rhs, point, value):
    v = np.asarray(point, float)
    return bool(v.shape == (len(info['features']),) and np.isfinite(v).all() and v.min() >= 0
        and abs(v.sum()-np.sqrt(len(info['full_weights']))) < 1e-8
        and np.all(C@v >= rhs) and value(info, v)['feasible'])


def common_design(A, y, features, anchor, capped, diagnostic):
    mass = np.sqrt(A.shape[1]); full = np.ones(A.shape[1])/mass
    C, rhs, polytope = anchor_witness_polytope(A, y, features, anchor)
    FC, FR, fixed = fixed_cell(A, y, features, anchor, full)
    GC, GR, glob = global_cell(A, y, features, anchor, full)
    oldC, oldR = np.vstack((C, FC)), np.r_[rhs, FR]
    newC, newR = np.vstack((C, GC)), np.r_[rhs, GR]
    accepted = lambda p:direct_feasible(fixed, oldC, oldR, p, fixed_value) and direct_feasible(glob, newC, newR, p, global_value)
    warm_ok = accepted(capped)
    candidates = []
    for row in diagnostic['regimes']['pair_transition']['coordinates']:
        if row['success'] and row['accepted_individual_zero_witness']:
            candidates.append({'feature':row['feature'], 'coordinate':row['coordinate'],
                'weights':row['candidate_weights'], 'common_region_accepted':accepted(row['candidate_weights'])})
    chosen = next((r for r in sorted(candidates, key=lambda r:r['feature']) if r['common_region_accepted']), None)
    starts = {'anchor':np.asarray(anchor).tolist(),
        'common_capped':np.asarray(capped if warm_ok else anchor).tolist(),
        'common_zero':np.asarray(anchor).tolist() if chosen is None else chosen['weights']}
    return {'features':list(features), 'anchor':np.asarray(anchor).tolist(), 'capped':np.asarray(capped).tolist(),
        'fixed_probe':fixed, 'global_probe':glob, 'polytope':polytope, 'starts':starts,
        'common_capped_available':warm_ok, 'zero_candidate_screen':candidates,
        'common_zero_feature':None if chosen is None else chosen['feature']}


def fit_block(A, y, features, anchor, capped, diagnostic):
    block = common_design(A, y, features, anchor, capped, diagnostic)
    full = np.ones(A.shape[1])/np.sqrt(A.shape[1]); block['arms'] = {}
    for arm, function in ((ARMS[0], fit_saturation_capped_frs_target),
                          (ARMS[1], fit_global_transition_capped_target),
                          (ARMS[2], fit_sparse_global_transition_target)):
        probe = block['fixed_probe'] if arm == ARMS[0] else block['global_probe']
        if not probe['anchor_strictly_feasible']:
            block['arms'][arm] = {'available':False, 'fits':{}, 'selected_start':None,
                'target':np.asarray(anchor).tolist(), 'optimizer_success':None,
                'scope':'original_full_region_unavailable_anchor_retained_no_search'}
            continue
        fits = {}
        for start in STARTS:
            timer = time.perf_counter()
            kwargs = {'sparsity_lambda':.05, 'sparsity_delta':.01} if arm == ARMS[2] else {}
            target, fit = function(A, y, features, anchor, full_reference=full,
                reference_scope='original_full', initialization=np.asarray(block['starts'][start]),
                initialization_scope=start, **kwargs)
            fits[start] = {'features':list(features), 'anchor':np.asarray(anchor).tolist(),
                'target':target.tolist(), 'fit':fit, 'fit_seconds':time.perf_counter()-timer}
        selected = min(STARTS, key=lambda s:(fits[s]['fit']['training_objective_final'], STARTS.index(s)))
        block['arms'][arm] = {'available':True, 'fits':fits, 'selected_start':selected,
            'target':fits[selected]['target'], 'optimizer_success':all(r['fit']['optimizer_success'] for r in fits.values())}
    block['evaluation'] = evaluation(A, y, block)
    return block


def evaluation(A, y, block):
    mass = np.sqrt(A.shape[1]); anchor = np.asarray(block['anchor'])/mass
    loss = CappedThirdEnemyHinge(A[:, block['features']], y, anchor)
    risks, zeros, failures = {}, {}, {}
    for arm in ARMS:
        a = block['arms'][arm]; point = np.asarray(a['target'])
        risks[arm] = float(loss.alpha@loss.losses(point/mass)); zeros[arm] = int(sum(point == 0))
        failures[arm] = sum(r['fit']['subproblem_failures'] for r in a['fits'].values())
    return {'clipped_training_bounds':risks, 'exact_zero_weights':zeros, 'optimizer_failures':failures,
        'global_region_bound_decrease_vs_fixed_pp':100*(risks[ARMS[0]]-risks[ARMS[1]]),
        'sparse_bound_decrease_vs_global_pp':100*(risks[ARMS[1]]-risks[ARMS[2]]),
        'sparse_additional_exact_zeros_vs_global':zeros[ARMS[2]]-zeros[ARMS[1]],
        'available':{a:block['arms'][a]['available'] for a in ARMS},
        'classification_scored':False}
