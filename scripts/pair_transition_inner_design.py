"""Append-only development design; retain audited full-budget native controls.

No outer-label or outer-score API. Old 28 candidates are copied exactly,
including failures. Every weighted refined arm receives the same seven new
choices. Native controls retain their separate, much larger search spaces.
"""
import copy
import numpy as np
from pair_transition_frs_hinge_stage import GRID, PredictiveMetricStage
from monotone_frs_hinge_stage import GRID as OLD_GRID
from geometric_vote_evaluation import evaluate

METHODS = ('proposed', 'unguarded_refined', 'raw_repair_refined',
           'full_refined', 'fgs_refined', 'sfrip_refined')
STRONG = ('fgs_native120', 'fgs_weighted_native120', 'sfrip_native6',
          'sfrip_weighted_native6', 'fgs_weighted_count120',
          'fgs_uniform_count120', 'sfrip_weighted_count6',
          'sfrip_uniform_count6', 'full_uniform')


def alignment(atomic, canonical, previous=None):
    assert atomic['complete'] and canonical['complete']
    assert atomic['reference_indices'] == canonical['reference_indices']
    assert atomic['validation_indices'] == canonical['validation_indices']
    if previous is not None:
        assert previous['reference_indices'] == atomic['reference_indices']
        assert len(previous['inner']) == len(atomic['inner'])
    for fi, (a, c) in enumerate(zip(atomic['inner'], canonical['inner'], strict=True)):
        for key in ('reference_indices', 'validation_indices'):
            assert a[key] == c[key]
            if previous is not None:
                assert a[key] == previous['inner'][fi][key]
        assert a['exact_proposed_count'] == c['count']
        for m in METHODS:
            anchor = a['options'][m][atomic['selection']['selected'][m]]['metric']
            expected = {k:anchor[k] for k in ('features', 'weights')}
            assert len(c['bases'][m]) == 1
            assert expected == {k:c['bases'][m][0][k] for k in expected}
            if previous is not None:
                assert expected == {k:previous['inner'][fi]['bases'][m][k] for k in expected}
        for m in STRONG:
            assert c['options'][m]
    return len(atomic['inner'])


def append_block(A, y, B, truth, prior):
    stage = PredictiveMetricStage(A, y, prior['stage'])
    block = {k:copy.deepcopy(prior[k]) for k in
             ('reference_indices', 'validation_indices', 'bases', 'options', 'records')}
    lookup = {(tuple(r['metric']['features']), tuple(r['metric']['weights'])):i
              for i, r in enumerate(block['records'])}
    assert len(lookup) == len(block['records'])
    for m in METHODS:
        assert len(block['options'][m]) == len(OLD_GRID)
        for ti in range(len(OLD_GRID), len(GRID)):
            metric = stage.metric(block['bases'][m], ti, certified=m == 'proposed')
            key = (tuple(metric['features']), tuple(metric['weights']))
            if key not in lookup:
                lookup[key] = len(block['records'])
                block['records'].append({'metric':metric, 'evaluation':evaluate(A, y, B, truth, metric)})
            block['options'][m].append({'target_index':ti, 'metric':metric, 'record_id':lookup[key]})
    block['stage'] = stage.export()
    block['previous_audited_records'] = len(prior['records'])
    block['previous_audited_learners'] = len(prior['stage']['learners'])
    return block


def select_refined(blocks):
    selected = {}
    for m in METHODS:
        rows = []
        for ti in range(len(GRID)):
            values = [b['records'][b['options'][m][ti]['record_id']]['evaluation']['metrics'] for b in blocks]
            rows.append({'target_index':ti,
                'mean_BA':float(np.mean([r['clean_BA'] for r in values])),
                'mean_logloss':float(np.mean([r['balanced_logloss'] for r in values]))})
        index = min(range(len(rows)), key=lambda i:(-round(rows[i]['mean_BA'], 12),
                    round(rows[i]['mean_logloss'], 12), i))
        selected[m] = {'options':rows, 'selected':index,
            'selected_inner_gain_pp':100*(rows[index]['mean_BA']-rows[0]['mean_BA'])}
    return selected


def strong_selection(canonical):
    """Copy selection made with every native candidate, never a weak subset."""
    selection = {}
    for m in STRONG:
        source = canonical['selection'][m]; rows = []
        for i, op in enumerate(canonical['inner'][0]['options'][m]):
            records = [b['records'][b['options'][m][i]['record_id']] for b in canonical['inner']]
            rows.append({'config_index':op['config_index'], 'target_index':op['target_index'],
                'mean_BA':float(np.mean([r['evaluation']['metrics']['clean_BA'] for r in records])),
                'mean_logloss':float(np.mean([r['evaluation']['metrics']['balanced_logloss'] for r in records])),
                'mean_count':float(np.mean([len(r['metric']['features']) for r in records]))})
        index = min(range(len(rows)), key=lambda i:(-round(rows[i]['mean_BA'], 12),
                    round(rows[i]['mean_logloss'], 12), rows[i]['mean_count'], i))
        assert rows == source['options'] and index == source['selected_index']
        selection[m] = {'selected_index':index, 'candidate_count':len(rows),
            'selected_recipe':rows[index], 'provenance':'entire_audited_canonical_inner_candidate_family'}
    return selection


def cohort_summary(rows, failures):
    selected = lambda r, m:r['selected_inner_BA'][m]
    delta = {m:float(np.mean([100*(selected(r, 'proposed')-selected(r, m)) for r in rows]))
             for m in METHODS[1:]+STRONG}
    gain = {m:float(np.mean([r['vs_previous_DC_pp'][m] for r in rows])) for m in METHODS}
    anchor = {m:float(np.mean([r['anchor_gain_pp'][m] for r in rows])) for m in METHODS}
    wins = sum(r['vs_previous_DC_pp']['proposed'] > 1e-10 for r in rows)
    challengers = ('full_refined', 'fgs_refined', 'sfrip_refined')+STRONG
    return {'rows':rows, 'datasets':len(rows), 'mean_anchor_gain_pp':anchor,
        'mean_vs_previous_DC_pp':gain, 'improved_vs_previous_DC_datasets':wins,
        'mean_proposed_minus_controls_pp':delta, 'target_optimizer_failures':int(failures),
        'pilot_gate_passed':bool(len(rows) == 15 and gain['proposed'] >= .5 and wins >= 6
            and failures == 0 and all(delta[m] > 0 for m in challengers)),
        'goal_achieved':False, 'new_outer_evaluation_performed':False,
        'scope':'exposed_sequential_inner_development_screen_not_predictive_superiority',
        'limits':['Native controls use their original entire candidate families; only six refined arms have 35 choices.',
                  'No p-values or confirmed risk claims from repeatedly used inner data.',
                  'A positive screen would justify fresh joint nested development, not submission.']}
