import numpy as np
from third_enemy_hinge_stage import PredictiveMetricStage as OldStage, GRID as OLD_GRID
from frs_cell_hinge_stage import PredictiveMetricStage, GRID


def test_all_old_choices_unchanged_and_same_new_family_for_both_arms():
    rng = np.random.default_rng(451); A = rng.uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    base = {'features': [0, 1, 2], 'weights': (np.ones(3)/np.sqrt(3)).tolist(), 'origin': 'smoke'}
    previous = OldStage(A, y)
    old_metrics = [previous.metric(base, i, certified=True) for i in range(len(OLD_GRID))]
    current = PredictiveMetricStage(A, y, previous.export())
    assert len(GRID) == 21
    for i, old in enumerate(old_metrics):
        actual = current.metric(base, i, certified=True)
        assert {k:v for k,v in actual.items() if not k.startswith('cell_')} == old
    for certified in (True, False):
        for i in range(14, 21):
            metric = current.metric(base, i, certified=certified)
            learner = current.learners[metric['learner_id']]
            assert learner['family'] == 'frs_cell_constrained' and learner['fit']['reference_scope'] == 'original_full'
            assert learner['fit']['class_cell_final']['feasible'] and not metric['guard_fallback']
            if certified:
                assert current.cell_checks[metric['cell_guard_id']]['accepted']
            else:
                assert metric['cell_guard_id'] is None
