"""Constrained top5 training targets checked by an independent scalar solver."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from soft_topk_predictive_metric import fit_majority_constrained_target
from independent_soft_topk_objective import independent_objective


def test_training_fit_improves_independent_soft5_loss_inside_witness_polytope():
    rng = np.random.default_rng(123); y = np.repeat([0, 1], 12)
    A = np.clip(.25+.5*y[:, None]+rng.normal(0, .12, (24, 6)), 0, 1)
    A[:, 0] = .1+.8*y+rng.uniform(-.01, .01, len(y))
    anchor = np.ones(6)/np.sqrt(6)
    target, fit = fit_majority_constrained_target(A, y, list(range(6)), anchor)
    assert fit['optimizer_success'] and fit['status'] == 'local_majority_constrained_target'
    assert fit['linear_constraints_verified'] and fit['polytope']['protection_reference'] == 'original_full'
    initial = independent_objective(A, y, anchor/np.sqrt(6))
    final = independent_objective(A, y, target/np.sqrt(6))
    np.testing.assert_allclose(initial, fit['training_objective_initial'], atol=1e-9)
    np.testing.assert_allclose(final, fit['training_objective_final'], atol=1e-9)
    assert final < initial
    assert target.min() >= 0 and target.max()/np.sqrt(6) <= 2/6+1e-10
    np.testing.assert_allclose(target.sum(), np.sqrt(6), atol=1e-10)


def test_singleton_query_classes_are_excluded_consistently_by_both_objectives():
    from certified_frfs.softvote_metric import softvote_terms
    from soft_topk_metric_objective import soft_topk_objective
    rng = np.random.default_rng(124); A = rng.uniform(0, 1, (9, 4)); y = np.array([0]*4+[1]*4+[2])
    w = np.array([0., .2, .3, .5])
    np.testing.assert_allclose(soft_topk_objective(w, softvote_terms(A, y))[0], independent_objective(A, y, w), atol=1e-9)
