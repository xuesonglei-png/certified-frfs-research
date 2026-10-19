"""All21 choices and adaptive controls survive independent transition replay."""
from pathlib import Path
import sys
from collections import Counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from hardaware_soft5_knn_stage import GRID, PredictiveMetricStage
from audit_hardaware_soft5_knn_pilot import check_stage
from hardaware_soft5_knn_stage import PredictionRegistry
from audit_certified_metric_knn_pool import independent_evaluate
from audit_forest_compact_transfer import equivalent


def test_three_families_equal_choices_and_distinct_control_certificate_provenance():
    rng = np.random.default_rng(135); y = np.repeat([0, 1], 12); d = 12
    A = rng.uniform(0, 1, (24, d)); A[:, 0] = .1+.8*y
    base = {'features': list(range(d)), 'weights': (np.ones(d)/np.sqrt(d)).tolist(), 'origin': 'test_full'}
    stage = PredictiveMetricStage(A, y)
    assert len(GRID) == 21
    for index in range(21):
        proposed = stage.metric(base, index, certified=True)
        control = stage.metric(base, index, certified=False)
        assert proposed['learner_id'] == control['learner_id']
        assert stage.guard(proposed['features'], proposed['weights'])['accepted']
    own = {'features': [1, 2], 'weights': [np.sqrt(d)/2]*2, 'origin': 'test_own'}
    metric = stage.metric(own, 20, certified=False)
    learner = stage.learners[metric['learner_id']]
    assert learner['fit']['guard_reference'] == 'own_anchor'
    assert len(learner['fit']['seed_checks']) == 12
    assert len(learner['fit']['source_learner_ids']) == 2
    assert not learner['fit']['original_initial_guard']['accepted']
    counts = Counter(); check_stage(A, y, stage.export(), counts)
    assert counts['independent_target_objective_replays'] == 6
    assert counts['hardaware_learners'] == 2
    assert counts['hardaware_guard_reference_original_full'] == counts['hardaware_guard_reference_own_anchor'] == 1


def test_all_choices_predictions_replay_and_query_labels_do_not_change_learners():
    rng = np.random.default_rng(313); y = np.repeat([0, 1], 12); d = 8
    A = rng.uniform(0, 1, (24, d)); B = rng.uniform(0, 1, (6, d)); truth = np.repeat([0, 1], 3)
    base = {'features': list(range(d)), 'weights': (np.ones(d)/np.sqrt(d)).tolist(), 'origin': 'full'}
    stage = PredictiveMetricStage(A, y)
    registry = PredictionRegistry(stage, B, truth)
    reversed_registry = PredictionRegistry(stage, B, 1-truth)
    for ti in range(len(GRID)):
        metric = stage.metric(base, ti, certified=True)
        mid = registry.score(metric)
        fresh = independent_evaluate(A, y, B, truth, metric['features'], metric['weights'])
        equivalent(fresh, registry.predictions[mid])
        other_mid = reversed_registry.score(metric)
        assert registry.predictions[mid]['predictions'] == reversed_registry.predictions[other_mid]['predictions']
    assert len(stage.learners) == 3
    counts = Counter(); check_stage(A, y, stage.export(), counts)
    assert counts['hardaware_seed_checks'] == 12
