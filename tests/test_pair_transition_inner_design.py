"""Scientific append-only design, independent replay and tamper rejection."""
import copy
from collections import Counter
import numpy as np
import pytest
from monotone_frs_hinge_stage import PredictiveMetricStage as OldStage, GRID as OLD_GRID
from geometric_vote_evaluation import evaluate
from pair_transition_inner_design import (METHODS, STRONG, alignment, append_block,
    select_refined, strong_selection, cohort_summary)
from audit_pair_transition_inner_block import check_append_block


def previous_block(A, y, B, truth):
    base = {'features':list(range(A.shape[1])),
        'weights':(np.ones(A.shape[1])/np.sqrt(A.shape[1])).tolist(), 'origin':'synthetic'}
    stage = OldStage(A, y); options = {}; records = []; lookup = {}
    for m in METHODS:
        options[m] = []
        for ti in range(len(OLD_GRID)):
            metric = stage.metric(base, ti, certified=m == 'proposed')
            key = (tuple(metric['features']), tuple(metric['weights']))
            if key not in lookup:
                lookup[key] = len(records)
                records.append({'metric':metric, 'evaluation':evaluate(A, y, B, truth, metric)})
            options[m].append({'target_index':ti, 'metric':metric, 'record_id':lookup[key]})
    return {'reference_indices':list(range(len(A))),
        'validation_indices':list(range(len(A), len(A)+len(B))),
        'bases':{m:copy.deepcopy(base) for m in METHODS},
        'options':options, 'records':records, 'stage':stage.export()}


def test_append_only_classifier_and_independent_tamper_checks():
    A = np.random.default_rng(451).uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    B = np.array([[.2, .3, .4], [.7, .4, .6]]); truth = np.array([0, 1])
    prior = previous_block(A, y, B, truth); original = copy.deepcopy(prior)
    fresh = append_block(A, y, B, truth, prior)
    assert prior == original
    counts = Counter(); check_append_block(A, y, B, truth, fresh, prior, counts)
    assert counts['inner_folds'] == 1 and counts['target_optimizer_failures'] == 0
    assert counts['new_pair_learner_records'] == 2
    assert all(len(fresh['options'][m]) == 35 for m in METHODS)
    selected = select_refined([fresh])
    assert all(selected[m]['selected_inner_gain_pp'] >= -1e-10 for m in METHODS)
    broken = copy.deepcopy(fresh); broken['options']['proposed'][0]['target_index'] = 999
    with pytest.raises(AssertionError):
        check_append_block(A, y, B, truth, broken, prior, Counter())
    broken = copy.deepcopy(fresh); broken['stage']['pair_certificate_checks'][0]['accepted'] = False
    with pytest.raises(AssertionError):
        check_append_block(A, y, B, truth, broken, prior, Counter())
    broken = copy.deepcopy(fresh); broken['options']['full_refined'][34]['metric']['weights'][0] += .1
    with pytest.raises(AssertionError):
        check_append_block(A, y, B, truth, broken, prior, Counter())


def test_all_native_options_retained_and_index_alignment_rejects_mismatch():
    atomic = {'complete':True, 'reference_indices':[0, 1, 2], 'validation_indices':[3],
        'selection':{'selected':{m:0 for m in METHODS}},
        'inner':[{'reference_indices':[0, 1], 'validation_indices':[2], 'exact_proposed_count':1,
            'options':{m:[{'metric':{'features':[0], 'weights':[1.]}}] for m in METHODS}}]}
    control = {'complete':True, 'reference_indices':[0, 1, 2], 'validation_indices':[3],
        'inner':[{'reference_indices':[0, 1], 'validation_indices':[2], 'count':1,
            'bases':{m:[{'features':[0], 'weights':[1.]}] for m in METHODS},
            'options':{m:[{'config_index':0, 'target_index':0, 'record_id':0},
                         {'config_index':1, 'target_index':0, 'record_id':1}] for m in STRONG},
            'records':[{'metric':{'features':[0], 'weights':[1.]},
                        'evaluation':{'metrics':{'clean_BA':.5, 'balanced_logloss':.7}}},
                       {'metric':{'features':[0, 1], 'weights':[.5, .5]},
                        'evaluation':{'metrics':{'clean_BA':.8, 'balanced_logloss':.4}}}]}],
        'selection':{m:{'options':[
            {'config_index':0, 'target_index':0, 'mean_BA':.5, 'mean_logloss':.7, 'mean_count':1.},
            {'config_index':1, 'target_index':0, 'mean_BA':.8, 'mean_logloss':.4, 'mean_count':2.}],
            'selected_index':1} for m in STRONG}}
    assert alignment(atomic, control) == 1
    selected = strong_selection(control)
    assert all(r['candidate_count'] == 2 and r['selected_index'] == 1 for r in selected.values())
    broken = copy.deepcopy(control); broken['inner'][0]['validation_indices'] = [1]
    with pytest.raises(AssertionError):
        alignment(atomic, broken)
    broken = copy.deepcopy(control); broken['selection']['fgs_weighted_native120']['selected_index'] = 0
    with pytest.raises(AssertionError):
        strong_selection(broken)


def test_screen_fails_if_strong_native_control_is_better_despite_weaker_controls():
    rows = [{'dataset':str(i), 'anchor_gain_pp':{m:1. for m in METHODS},
        'vs_previous_DC_pp':{m:1. for m in METHODS},
        'selected_inner_BA':{m:.8 if m == 'proposed' else .79 for m in METHODS+STRONG}}
        for i in range(15)]
    assert cohort_summary(rows, 0)['pilot_gate_passed']
    for row in rows:
        row['selected_inner_BA']['fgs_weighted_native120'] = .82
    summary = cohort_summary(rows, 0)
    assert not summary['pilot_gate_passed'] and not summary['goal_achieved']
    assert summary['mean_proposed_minus_controls_pp']['fgs_weighted_native120'] < 0
