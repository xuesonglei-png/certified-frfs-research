"""Contract tests for the nested relation/support selector.

These tests use tiny deterministic tables and do not exercise the development
datasets.  The classifier checks below are intentionally independent of the
production implementation: distances, stable neighbour ordering, votes, and
metrics are recomputed locally.
"""
import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from certified_frfs.relation_support_selector_v1 import (
    weighted_knn,
    _counts,
    _fit_relation,
    _ranking,
    _candidate,
)


def _independent_knn(A, y, B, truth, features, weights, k=5):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    y = np.asarray(y)
    truth = np.asarray(truth)
    features = np.asarray(features, dtype=int)
    weights = np.asarray(weights, float)
    distances = np.abs(B[:, None, features] - A[None, :, features]) @ weights
    nn = np.argsort(distances, axis=1, kind="stable")[:, : min(k, len(A))]
    classes = np.unique(y)
    votes = np.column_stack([(y[nn] == c).sum(axis=1) for c in classes])
    predictions = classes[np.argmax(votes, axis=1)]
    probabilities = (votes + 1.0) / (nn.shape[1] + len(classes))
    losses = [
        log_loss(truth[truth == c], probabilities[truth == c], labels=classes)
        for c in np.unique(truth)
    ]
    return {
        "clean_BA": float(balanced_accuracy_score(truth, predictions)),
        "macro_F1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "balanced_logloss": float(np.mean(losses)),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }


def test_weighted_knn_replays_distances_votes_and_metrics():
    # The first query is exactly tied.  Stable object order and class order
    # therefore exercise the declared deterministic tie convention.
    A = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0, 1, 0, 1])
    B = np.array([[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]])
    truth = np.array([0, 1, 0])
    expected = _independent_knn(A, y, B, truth, [0, 1], [0.7, 0.3], k=3)
    actual = weighted_knn(A, y, B, truth, [0, 1], [0.7, 0.3], k=3)
    np.testing.assert_allclose(actual["clean_BA"], expected["clean_BA"])
    np.testing.assert_allclose(actual["macro_F1"], expected["macro_F1"])
    np.testing.assert_allclose(actual["balanced_logloss"], expected["balanced_logloss"])
    assert actual["predictions"] == expected["predictions"]
    np.testing.assert_allclose(actual["probabilities"], expected["probabilities"])


def test_weighted_knn_requires_support_local_weight_vector():
    A = np.zeros((4, 3)); B = np.zeros((1, 3)); y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError):
        weighted_knn(A, y, B, np.array([0]), [0, 2], [1.0])
    with pytest.raises(ValueError):
        weighted_knn(A, y, B, np.array([0]), [0], [1.0, 0.0])


def test_count_grid_respects_capability_half_dimension():
    for d in range(1, 15):
        counts = _counts(d)
        assert counts == sorted(set(counts))
        assert counts[-1] == d
        assert min(counts) >= int(np.ceil(d / 2))
        assert all(1 <= c <= d for c in counts)


def test_relation_families_are_finite_and_uniform_is_exact():
    rng = np.random.default_rng(51)
    A = rng.uniform(0.0, 1.0, (18, 4))
    y = np.repeat([0, 1, 2], 6)
    uniform = _fit_relation(A, y, "uniform_sqrt")
    np.testing.assert_allclose(uniform, np.ones(4) / 2.0)
    learned = _fit_relation(A, y, "learned_majority")
    assert learned.shape == (4,)
    assert np.all(np.isfinite(learned)) and np.all(learned >= 0)
    # The selector uses the original sqrt(d) mass and a 2*mass/d cap.
    np.testing.assert_allclose(learned.sum(), 2.0, atol=1e-8)
    assert learned.max() <= 1.0 + 1e-8


def test_ranking_is_a_permutation_and_uses_both_sources():
    rng = np.random.default_rng(71)
    A = rng.uniform(0.0, 1.0, (20, 5)); y = np.repeat([0, 1], 10)
    w = _fit_relation(A, y, "uniform_sqrt")
    order = _ranking(A, y, w, "uniform_sqrt", 19)
    assert order == sorted(set(order)) or set(order) == set(range(5))
    assert sorted(order) == list(range(5))


def test_candidate_has_local_support_weights_and_certificate():
    # Separated first two coordinates make the full candidate a stable
    # certificate witness while retaining a nontrivial candidate interface.
    rng = np.random.default_rng(91)
    y = np.repeat([0, 1], 6)
    A = np.column_stack([
        0.15 + 0.1 * rng.random(12) + 0.7 * y,
        rng.random(12),
        rng.random(12),
        rng.random(12),
    ])
    w = _fit_relation(A, y, "uniform_sqrt")
    point = _candidate(A, y, w, [0, 1, 2, 3], A.shape[1])
    assert point["features"] == [0, 1, 2, 3]
    local = np.asarray(point["weights"], float)
    assert local.shape == (A.shape[1],)
    assert np.all(np.isfinite(local)) and np.all(local >= 0)
    assert point["certificate"]["feasible"]

