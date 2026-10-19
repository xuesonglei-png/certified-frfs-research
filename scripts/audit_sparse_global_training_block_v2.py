"""Independent common-start choice, geometry, fits/duals and sparsity summary."""
import numpy as np
from audit_saturation_capped_frs_fit import check_saturation_capped_fit
from audit_global_transition_capped_fit import check_global_transition_capped_fit
from audit_sparse_global_transition_frs_v2 import check_sparse_global_transition_fit
from independent_frs_saturation_transition import reconstruct_cell as fixed_cell, scalar_cell_evaluation as fixed_value
from independent_global_frs_transition import reconstruct_cell as global_cell, scalar_cell_evaluation as global_value
from audit_frs_coordinate_minimum import reconstruct_geometry
from audit_pair_capped_frs_fit import ScalarRankValidator
from audit_forest_compact_transfer import equivalent

ARMS = ('fixed_signed_risk', 'global_signed_risk', 'global_signed_sparse')
STARTS = ('anchor', 'common_capped', 'common_zero')


def audit_block(A, y, block, diagnostic, counts):
    fs, anchor = block['features'], np.asarray(block['anchor']); mass = np.sqrt(A.shape[1])
    C, rhs, _, _ = reconstruct_geometry(A, y, diagnostic['geometry'], counts)
    equivalent(block['polytope'], diagnostic['geometry']['polytope'])
    equivalent(fs, diagnostic['geometry']['features']); equivalent(anchor.tolist(), diagnostic['geometry']['anchor'])
    FC, FR = fixed_cell(A, y, block['fixed_probe'], counts)
    GC, GR = global_cell(A, y, block['global_probe'], counts)
    for probe in (block['fixed_probe'], block['global_probe']):
        equivalent(probe['full_weights'], (np.ones(A.shape[1])/mass).tolist())
    def feasible(point):
        p = np.asarray(point, float)
        return bool(p.shape == anchor.shape and np.isfinite(p).all() and p.min() >= 0
            and abs(p.sum()-mass) < 1e-8 and np.all(C@p >= rhs) and np.all(FC@p >= FR)
            and np.all(GC@p >= GR) and fixed_value(block['fixed_probe'], p)['feasible']
            and global_value(block['global_probe'], p)['feasible'])
    warm_ok = feasible(block['capped']); assert block['common_capped_available'] == warm_ok
    screen = [{'feature':r['feature'], 'coordinate':r['coordinate'], 'weights':r['candidate_weights'],
               'common_region_accepted':feasible(r['candidate_weights'])}
              for r in diagnostic['regimes']['pair_transition']['coordinates']
              if r['success'] and r['accepted_individual_zero_witness']]
    equivalent(screen, block['zero_candidate_screen'])
    chosen = next((r for r in sorted(screen, key=lambda r:r['feature']) if r['common_region_accepted']), None)
    assert block['common_zero_feature'] == (None if chosen is None else chosen['feature'])
    starts = {'anchor':anchor.tolist(), 'common_capped':block['capped'] if warm_ok else anchor.tolist(),
              'common_zero':anchor.tolist() if chosen is None else chosen['weights']}
    equivalent(starts, block['starts']); assert tuple(block['arms']) == ARMS
    validator = None; risks, zeros, failures = {}, {}, {}
    for arm, checker in zip(ARMS, (check_saturation_capped_fit, check_global_transition_capped_fit, check_sparse_global_transition_fit)):
        a = block['arms'][arm]; probe = block['fixed_probe'] if arm == ARMS[0] else block['global_probe']
        assert a['available'] == probe['anchor_strictly_feasible']
        if not a['available']:
            assert a['fits'] == {} and a['selected_start'] is None and a['optimizer_success'] is None
            equivalent(a['target'], anchor.tolist()); counts[arm+'_unavailable'] += 1
        else:
            assert tuple(a['fits']) == STARTS
            for start in STARTS:
                r = a['fits'][start]; fit = r['fit']
                assert r['features'] == fs; equivalent(r['anchor'], anchor.tolist())
                assert fit['reference_scope'] == 'original_full' and fit['initialization_scope'] == start
                equivalent(fit['requested_initial_weights'], starts[start]); equivalent(fit['class_cell'], probe)
                if arm == ARMS[2]:
                    assert fit['sparsity_lambda'] == .05 and fit['sparsity_delta'] == .01
                checker(A, y, r, counts)
                if validator is None:
                    validator = ScalarRankValidator(A[:, fs], fit['training_witnesses'])
                    alpha = np.asarray(fit['training_witnesses']['alpha'])
                else:
                    equivalent(fit['training_witnesses'], validator.info)
            selected = min(STARTS, key=lambda s:(a['fits'][s]['fit']['training_objective_final'], STARTS.index(s)))
            assert a['selected_start'] == selected
            equivalent(a['target'], a['fits'][selected]['target'])
            assert a['optimizer_success'] == all(r['fit']['optimizer_success'] for r in a['fits'].values())
        zeros[arm] = int(sum(w == 0 for w in a['target']))
        failures[arm] = sum(r['fit']['subproblem_failures'] for r in a['fits'].values())
    if validator is None:
        raise AssertionError('Original fixed signed anchor must remain available for diagnostic source')
    for arm in ARMS:
        point = np.asarray(block['arms'][arm]['target'])/mass
        risks[arm] = float(alpha@np.minimum(1., validator.losses(point)))
    report = {'clipped_training_bounds':risks, 'exact_zero_weights':zeros, 'optimizer_failures':failures,
        'global_region_bound_decrease_vs_fixed_pp':100*(risks[ARMS[0]]-risks[ARMS[1]]),
        'sparse_bound_decrease_vs_global_pp':100*(risks[ARMS[1]]-risks[ARMS[2]]),
        'sparse_additional_exact_zeros_vs_global':zeros[ARMS[2]]-zeros[ARMS[1]],
        'available':{arm:block['arms'][arm]['available'] for arm in ARMS}, 'classification_scored':False}
    equivalent(report, block['evaluation']); counts['matched_sparse_training_folds'] += 1
