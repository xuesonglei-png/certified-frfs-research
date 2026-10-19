import numpy as np
from collections import Counter
from frs_cell_hinge_stage import PredictiveMetricStage as PreviousStage, GRID as PREVIOUS_GRID
from monotone_frs_hinge_stage import PredictiveMetricStage, GRID
from audit_frs_one_sided_fit_stored import check_one_sided_fit


def test_preserves_all_twenty_one_metrics_and_adds_same_monotone_family():
    rng = np.random.default_rng(451); A = rng.uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    base = {'features': [0, 1, 2], 'weights': (np.ones(3)/np.sqrt(3)).tolist(), 'origin': 'smoke'}
    previous = PreviousStage(A, y); metrics = {}
    for certified in (True, False):
        metrics[certified] = [previous.metric(base, i, certified=certified) for i in range(len(PREVIOUS_GRID))]
    inherited = previous.export(); current = PredictiveMetricStage(A, y, inherited)
    assert len(GRID) == 28
    for certified in (True, False):
        for i, old in enumerate(metrics[certified]):
            actual = current.metric(base, i, certified=certified)
            assert {k:v for k,v in actual.items() if not k.startswith('monotone_')} == old
    assert len(current.learners) == len(inherited['learners'])
    assert current.cell_checks == inherited['cell_certificate_checks']
    for certified in (True, False):
        for i in range(21, 28):
            actual = current.metric(base, i, certified=certified)
            learner = current.learners[actual['learner_id']]; fit = learner['fit']
            assert learner['family'] == 'one_sided_cell_constrained' and learner['certified_arm'] == certified
            assert fit['solver_primal_feasibility_tolerance'] == fit['solver_dual_feasibility_tolerance'] == 1e-9
            assert not actual['guard_fallback'] and fit['class_cell_final']['feasible']
            if certified:
                assert current.monotone_checks[actual['monotone_guard_id']]['accepted']
            else:
                assert actual['monotone_guard_id'] is None
        check_one_sided_fit(A, y, learner, Counter())
