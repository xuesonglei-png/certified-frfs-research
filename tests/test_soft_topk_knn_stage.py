"""The deployed hard5NN uses guarded metrics with equal soft5 fitting choices."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from soft_topk_knn_stage import GRID, PredictiveMetricStage
from independent_soft_topk_objective import independent_objective


def test_equal_families_and_objective_replay_for_certified_and_baseline_metrics():
    rng = np.random.default_rng(126); y = np.repeat([0, 1], 10)
    A = np.clip(.2+.6*y[:, None]+rng.normal(0, .1, (20, 6)), 0, 1)
    A[:, 0] = .1+.8*y
    stage = PredictiveMetricStage(A, y)
    base = {'features': list(range(6)), 'weights': (np.ones(6)/np.sqrt(6)).tolist(), 'origin': 'test_full'}
    assert len(GRID) == 14
    for index in range(14):
        proposed, control = stage.metric(base, index, certified=True), stage.metric(base, index, certified=False)
        assert proposed['learner_id'] == control['learner_id']
        assert stage.guard(proposed['features'], proposed['weights'])['accepted']
    assert len(stage.learners) == 2
    for learner in stage.learners:
        fit = learner['fit']; assert fit['neighborhood_k'] == 5 and fit['optimizer_success']
        value = independent_objective(A, y, np.array(learner['target'])/np.sqrt(6))
        np.testing.assert_allclose(value, fit['training_objective_final'], atol=1e-9)
        assert value <= fit['training_objective_initial']+1e-8
