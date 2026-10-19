"""All21 choices and adaptive controls survive independent transition replay."""
from pathlib import Path
import sys
from collections import Counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from joint_guarded_soft5_knn_stage import GRID, PredictiveMetricStage
from audit_joint_guarded_soft5_knn_pilot import check_stage


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
    assert not learner['fit']['original_initial_guard']['accepted']
    counts = Counter(); check_stage(A, y, stage.export(), counts)
    assert counts['independent_target_objective_replays'] == 4
    assert counts['joint_guarded_learners'] == 2
    assert counts['joint_guard_reference_original_full'] == counts['joint_guard_reference_own_anchor'] == 1
