from collections import Counter
import copy
import numpy as np
import pytest
from monotone_frs_hinge_stage import PredictiveMetricStage as PreviousStage, GRID as PREVIOUS_GRID
from pair_transition_frs_hinge_stage import PredictiveMetricStage, GRID
from audit_pair_transition_frs_fit import check_pair_transition_fit


def test_old_choices_are_not_refitted_and_new_pair_family_is_equal_for_arms():
    rng = np.random.default_rng(451); A = rng.uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    base = {'features':[0, 1, 2], 'weights':(np.ones(3)/np.sqrt(3)).tolist(), 'origin':'synthetic'}
    previous = PreviousStage(A, y)
    for certified in (True, False):
        for i in range(len(PREVIOUS_GRID)):
            previous.metric(base, i, certified=certified)
    old = previous.export(); original = copy.deepcopy(old); stage = PredictiveMetricStage(A, y, old)
    assert len(GRID) == 35 and GRID[:28] == PREVIOUS_GRID
    for i in range(28):
        with pytest.raises(ValueError, match='reuse all old metrics'):
            stage.metric(base, i, certified=True)
    assert stage.export() == {**old, 'pair_certificate_checks':[]}
    for certified in (True, False):
        ids = set()
        for i in range(28, 35):
            metric = stage.metric(base, i, certified=certified); ids.add(metric['learner_id'])
            learner = stage.learners[metric['learner_id']]
            assert learner['family'] == 'pair_transition_constrained' and learner['certified_arm'] == certified
            assert learner['fit']['solver_primal_feasibility_tolerance'] == 1e-9
            assert learner['fit']['solver_dual_feasibility_tolerance'] == 1e-9
            assert not metric['guard_fallback']
            if certified:
                assert stage.pair_checks[metric['pair_deployed_guard_id']]['accepted']
            else:
                assert metric['pair_guard_id'] is metric['pair_deployed_guard_id'] is None
        assert len(ids) == 1
        check_pair_transition_fit(A, y, learner, Counter())
    exported = stage.export()
    assert exported['learners'][:len(old['learners'])] == old['learners']
    for key in ('cell_certificate_checks', 'monotone_certificate_checks'):
        assert exported[key] == old[key]
    assert exported['certificate_checks'][:len(old['certificate_checks'])] == old['certificate_checks']
    assert old == original
