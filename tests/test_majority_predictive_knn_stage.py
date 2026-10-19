"""Common target budgets, baseline feasibility and original guard separation."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from majority_predictive_knn_stage import GRID, PredictiveMetricStage
from adaptive_majority_predictive_metric import fit_majority_constrained_target


def table():
    rng = np.random.default_rng(62); y = np.repeat([0, 1], 8)
    A = rng.uniform(0, 1, (16, 6)); A[:, 0] = .1+.8*y
    return A, y


def test_weighted_arms_share_two_families_and_cached_learner_even_when_guarded():
    A, y = table(); stage = PredictiveMetricStage(A, y)
    base = {'features': list(range(6)), 'weights': (np.ones(6)/np.sqrt(6)).tolist(), 'origin': 'test_full'}
    assert len(GRID) == 14 and set(f for f, _ in GRID) == {'unconstrained', 'majority_constrained'}
    for index in range(14):
        proposed = stage.metric(base, index, certified=True)
        control = stage.metric(base, index, certified=False)
        assert proposed['learner_id'] == control['learner_id']
        assert proposed['target_family'] == control['target_family']
        assert stage.guard(proposed['features'], proposed['weights'])['accepted']
    assert len(stage.learners) == 2


def test_uncertified_baseline_anchor_gets_feasible_own_reference_constraints():
    # Full reference separates classes in four redundant coordinates; anchor
    # spends all permitted mass on unrelated coordinates and loses protection.
    rng = np.random.default_rng(65); y = np.repeat([0, 1], 10)
    A = rng.uniform(.4, .6, (20, 8)); A[:, :4] = .1+.8*y[:, None]
    anchor = np.r_[np.zeros(4), np.full(4, np.sqrt(8)/4)]
    target, fit = fit_majority_constrained_target(A, y, list(range(8)), anchor)
    assert fit['optimizer_success']
    assert fit['polytope']['protection_reference'] == 'own_anchor'
    assert np.isfinite(target).all()
    assert 'certificate' not in fit


def test_degenerate_target_reports_requested_budget_without_claiming_iterations():
    A, y = table(); anchor = np.ones(3)*np.sqrt(6)/3
    target, fit = fit_majority_constrained_target(A, y, [0, 1, 2], anchor)
    assert fit['max_iter'] == 200 and fit['optimizer_iterations'] == 0
    np.testing.assert_array_equal(target, anchor)
