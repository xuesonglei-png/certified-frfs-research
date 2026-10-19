from collections import Counter
import copy
import numpy as np
import pytest
from frs_coordinate_minimum import solve_diagnostic
from audit_frs_coordinate_minimum import audit_diagnostic


def test_fixed_cell_coordinate_obstruction_and_compensation_are_distinguished():
    A = np.array([[x, c] for c in (0., 1.) for x in (0., 0., .1, .1, .2, .2)])
    y = np.repeat([0, 1], 6); anchor = np.ones(2)/np.sqrt(2)
    saved = solve_diagnostic(A, y, [0, 1], anchor)
    counts = Counter(); audit_diagnostic(A, y, saved, counts)
    assert counts['independent_coordinate_LP_primal_dual_replays'] == 8
    assert counts['LP_failures'] == 0
    old = saved['regimes']['one_sided']['coordinates'][0]
    new = saved['regimes']['pair_transition']['coordinates'][0]
    assert old['actual_minimum_weight'] > 0 and new['actual_minimum_weight'] == 0
    assert new['accepted_individual_zero_witness'] and not old['accepted_individual_zero_witness']
    broken = copy.deepcopy(saved)
    broken['regimes']['pair_transition']['coordinates'][0]['objective'] += .1
    with pytest.raises(AssertionError):
        audit_diagnostic(A, y, broken, Counter())


def test_coordinate_problem_with_nonempty_stability_rows_and_free_star_epigraphs():
    A = np.random.default_rng(451).uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    saved = solve_diagnostic(A, y, [0, 1, 2], np.ones(3)/np.sqrt(3))
    assert saved['geometry']['cell']['selected_objects']
    counts = Counter(); audit_diagnostic(A, y, saved, counts)
    assert counts['independent_coordinate_LP_primal_dual_replays'] == 12
    assert counts['LP_failures'] == 0
    broken = copy.deepcopy(saved)
    broken['regimes']['one_sided']['coordinates'][0]['accepted_individual_zero_witness'] = True
    with pytest.raises(AssertionError):
        audit_diagnostic(A, y, broken, Counter())
