"""Independent replay of new fits/guards/predictions; audited prefixes immutable."""
import copy
import numpy as np
from audit_pair_transition_frs_fit import check_pair_transition_fit
from independent_frs_pair_transition_cell import reconstruct_cell, scalar_cell_evaluation
from audit_predictive_target_knn_pilot import check_stage
from independent_geometric_evaluation import independent_evaluate
from audit_forest_compact_transfer import equivalent

METHODS = ('proposed', 'unguarded_refined', 'raw_repair_refined',
           'full_refined', 'fgs_refined', 'sfrip_refined')
TIMES = (0., .05, .1, .25, .5, .75, 1.)


def check_append_block(A, y, B, truth, block, previous, counts):
    for key in ('reference_indices', 'validation_indices', 'bases'):
        equivalent(block[key], previous[key])
    stage, old = block['stage'], previous['stage']
    fits, records, certs = len(old['learners']), len(previous['records']), len(old['certificate_checks'])
    assert block['previous_audited_records'] == records and block['previous_audited_learners'] == fits
    equivalent(stage['learners'][:fits], old['learners'])
    equivalent(block['records'][:records], previous['records'])
    equivalent(stage['certificate_checks'][:certs], old['certificate_checks'])
    for key in old:
        if key not in ('learners', 'certificate_checks'):
            equivalent(stage[key], old[key])
    counts['previous_audited_fits_reused'] += fits
    counts['previous_audited_prediction_records_reused'] += records
    extra = copy.deepcopy(stage['certificate_checks'][certs:])
    for i, check in enumerate(extra):
        assert check['id'] == certs+i; check['id'] = i
    check_stage(A, y, {'learners':[], 'certificate_checks':extra}, counts)
    keys = set()
    for learner in stage['learners'][fits:]:
        key = (tuple(learner['features']), tuple(learner['anchor']), learner['certified_arm'])
        assert key not in keys; keys.add(key)
        assert learner['family'] == 'pair_transition_constrained'
        fit, probe = learner['fit'], learner['original_cell_probe']
        full = (np.ones(A.shape[1])/np.sqrt(A.shape[1])).tolist()
        equivalent(probe['full_weights'], full)
        scope = 'original_full' if probe['anchor_strictly_feasible'] or learner['certified_arm'] else 'own_anchor'
        assert fit['reference_scope'] == scope
        if scope == 'original_full' and not fit['cell_unavailable']:
            equivalent(probe, fit['class_cell'])
        else:
            reconstruct_cell(A, y, probe, counts)
        if fit['cell_unavailable']:
            assert learner['certified_arm'] and not probe['anchor_strictly_feasible']
            assert fit['status'] == 'original_full_cell_unavailable_anchor_retained'
            assert fit['optimizer_success'] is None and fit['subproblem_failures'] == 0
            equivalent(learner['anchor'], learner['target']); equivalent(probe, fit['class_cell'])
            counts['unavailable_proposed_cells'] += 1
        else:
            check_pair_transition_fit(A, y, learner, counts)
            assert fit['solver_primal_feasibility_tolerance'] == fit['solver_dual_feasibility_tolerance'] == 1e-9
        counts['new_pair_learner_records'] += 1
    for gi, guard in enumerate(stage['pair_certificate_checks']):
        assert guard['id'] == gi
        learner = stage['learners'][guard['learner_id']]; fit = learner['fit']
        assert guard['learner_id'] >= fits and learner['certified_arm']
        assert fit['reference_scope'] == 'original_full'
        old_guard = stage['certificate_checks'][guard['old_guard_id']]
        assert guard['features'] == old_guard['features'] == learner['features']
        equivalent(guard['weights'], old_guard['weights'])
        value = scalar_cell_evaluation(fit['class_cell'], guard['weights'])
        equivalent(value, guard['cell_evaluation'])
        accepted = bool(not fit['cell_unavailable'] and value['feasible'] and old_guard['certificate']['majority_feasible'])
        assert guard['cell_accepted'] == accepted and guard['old_accepted'] == old_guard['accepted']
        assert guard['accepted'] == bool(accepted or old_guard['accepted'])
        counts['independent_pair_guard_replays'] += 1
        counts['new_pair_guard_admissions'] += bool(accepted and not old_guard['accepted'])
    used = set()
    for m in METHODS:
        base = block['bases'][m]; options = block['options'][m]
        assert len(options) == 35 and len(previous['options'][m]) == 28
        equivalent(options[:28], previous['options'][m])
        for ti, op in enumerate(options):
            assert op['target_index'] == ti; metric = op['metric']
            assert metric['features'] == base['features']
            rid = op['record_id']; assert 0 <= rid < len(block['records']); used.add(rid)
            record = block['records'][rid]
            equivalent({k:metric[k] for k in ('features', 'weights')},
                       {k:record['metric'][k] for k in ('features', 'weights')})
            if ti < 28:
                continue
            t = TIMES[ti-28]; lid = metric['learner_id']; assert fits <= lid < len(stage['learners'])
            learner = stage['learners'][lid]
            assert learner['certified_arm'] == (m == 'proposed')
            assert metric['target_family'] == learner['family'] == 'pair_transition_constrained'
            assert metric['target_index'] == ti and metric['interpolation'] == t
            assert metric['base_origin'] == base['origin'] and learner['features'] == base['features']
            equivalent(learner['anchor'], base['weights'])
            request = (1-t)*np.asarray(base['weights'])+t*np.asarray(learner['target'])
            equivalent(request.tolist(), metric['requested_weights'])
            if m == 'proposed':
                guard = stage['pair_certificate_checks'][metric['pair_guard_id']]
                deployed = stage['pair_certificate_checks'][metric['pair_deployed_guard_id']]
                assert guard['learner_id'] == deployed['learner_id'] == lid
                assert metric['guard_id'] == guard['old_guard_id']
                assert metric['deployed_guard_id'] == deployed['old_guard_id']
                equivalent(guard['weights'], request.tolist())
                assert metric['guard_fallback'] == bool(not guard['accepted']) and deployed['accepted']
                actual = base['weights'] if metric['guard_fallback'] else request.tolist()
                equivalent(actual, metric['weights']); equivalent(actual, deployed['weights'])
            else:
                assert all(metric[k] is None for k in
                           ('guard_id', 'deployed_guard_id', 'pair_guard_id', 'pair_deployed_guard_id'))
                assert not metric['guard_fallback']; equivalent(request.tolist(), metric['weights'])
    assert used == set(range(len(block['records'])))
    for record in block['records'][records:]:
        equivalent(independent_evaluate(A, y, B, truth, record['metric']), record['evaluation'])
        counts['independent_new_inner_predictions'] += 1
    counts['inner_folds'] += 1
