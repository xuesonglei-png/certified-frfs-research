"""Meaningful checks of dependency feasibility and comparator definitions."""
import hashlib

import numpy as np
import pytest

from certified_frfs.baselines import (
    mafrfs_margin_ratio,
    relieff_scores,
    select_all,
    select_mi,
    select_mrmr,
    select_mafrfs_additive,
    select_nominal_frfs,
    select_relieff,
)
from certified_frfs.core import gamma
from certified_frfs.datasets import DATA_DIR, available_datasets, load_dataset, metadata


def test_nominal_reduct_preserves_matched_dependency_and_is_inclusion_minimal():
    rng = np.random.default_rng(41)
    X = rng.uniform(size=(18, 6))
    y = np.repeat([0, 1, 2], 6)
    weights = np.asarray([0.3, 0.7, 0.1, 0.4, 0.6, 0.2])
    eta = 0.1
    selected = select_nominal_frfs(X, y, eta=eta, weights=weights)
    target = gamma(X, y, weights=weights) - eta
    assert gamma(X, y, selected, weights) >= target - 1e-12
    for feature in selected:
        subset = [a for a in selected if a != feature]
        assert gamma(X, y, subset, weights) < target - 1e-12
    assert selected == select_nominal_frfs(X, y, eta=eta, weights=weights)


def test_nominal_zero_tolerance_and_fixed_weights_are_respected():
    X = np.array([[0.0, 0.0], [0.1, 0.2], [0.8, 0.7], [1.0, 0.9]])
    y = np.array([0, 0, 1, 1])
    weights = np.array([0.3, 0.2])
    selected = select_nominal_frfs(X, y, eta=0, weights=weights)
    assert selected == [0, 1]
    assert gamma(X, y, selected, weights) == pytest.approx(gamma(X, y, weights=weights))
    assert select_nominal_frfs(X, y, eta=1, weights=weights) == []


def test_mi_and_mrmr_find_informative_feature_without_test_data():
    rng = np.random.default_rng(7)
    y = np.repeat([0, 1], 60)
    X = np.column_stack([y + rng.normal(0, 0.01, len(y)), rng.normal(size=len(y)), np.ones(len(y))])
    assert select_mi(X, y, k=1, random_state=7) == [0]
    assert select_mrmr(X, y, k=1) == [0]
    assert select_mi(X, y, k=0) == []
    assert select_mrmr(X, y, k=0) == []


def test_mrmr_penalizes_exact_redundancy_with_a_second_informative_feature():
    # Two independent class bits: each is relevant, but an exact duplicate
    # contributes zero MID after its original is selected.
    bit1 = np.tile([0, 0, 1, 1], 30)
    bit2 = np.tile([0, 1, 0, 1], 30)
    y = 2 * bit1 + bit2
    X = np.column_stack([bit1, bit1, bit2]).astype(float)
    selected = select_mrmr(X, y, k=2, n_bins=2)
    assert selected == [0, 2]


def test_relieff_known_two_class_solution_and_constant_attribute():
    X = np.array([[0, 1], [0, 1], [1, 1], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    assert relieff_scores(X, y, n_neighbors=1, max_samples=None) == pytest.approx([1, 0])
    assert select_relieff(X, y, k=1, n_neighbors=1) == [0]


def test_relieff_multiclass_prior_weighting():
    # All same-class points coincide, so hit corrections are zero. Class
    # frequencies are unequal, making unweighted multiclass misses incorrect.
    X = np.array([[0], [0], [0.5], [0.5], [0.5], [1], [1], [1], [1]], dtype=float)
    y = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    expected = (2 * ((3 / 7) * 0.5 + (4 / 7) * 1)
                + 3 * 0.5 + 4 * ((2 / 5) * 1 + (3 / 5) * 0.5)) / 9
    assert relieff_scores(X, y, n_neighbors=1, max_samples=None)[0] == pytest.approx(expected)


def test_baselines_reject_unimputed_missing_values_and_invalid_counts():
    X = np.array([[0.0, 1.0], [1.0, np.nan]])
    y = [0, 1]
    with pytest.raises(ValueError):
        select_nominal_frfs(X, y)
    with pytest.raises(ValueError):
        select_mi(np.nan_to_num(X), y, k=3)
    with pytest.raises(ValueError):
        select_relieff(np.nan_to_num(X), y, k=1, n_neighbors=0)
    assert select_all(X) == [0, 1]


def test_all_fifteen_real_datasets_match_provenance_and_content_hashes():
    names = available_datasets()
    assert len(names) == 15
    for name in names:
        X, y = load_dataset(name)
        info = metadata(name)
        assert X.dtype == np.float64
        assert y.dtype == np.int64
        assert len(np.unique(y)) == info["n_classes"]
        assert np.bincount(y).tolist() == info["class_counts"]
        assert int(np.isnan(X).sum()) == info["missing_values"]
        assert hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest() == info["array_sha256"]
        archive = DATA_DIR / "raw" / f"{name}.zip"
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == info["archive_sha256"]
        assert info["source_url"].startswith("https://archive.ics.uci.edu/dataset/")


def test_mafrfs_euclidean_margin_ratio_with_unequal_class_counts():
    # Means 0, 2 and 5; normalized within-class scatters 1, 0 and 1.
    # Local denominator |0-2|+|0-5|+|2-5|=10. Global mean is 2.4,
    # and its denominator is |0-2.4|+|2-2.4|+|5-2.4|=5.4.
    X = np.array([[-1], [1], [2], [4], [6]], dtype=float)
    y = np.array([0, 0, 1, 2, 2])
    assert mafrfs_margin_ratio(X, y, [0], "local") == pytest.approx(0.2)
    assert mafrfs_margin_ratio(X, y, [0], "global") == pytest.approx(2 / 5.4)
    assert np.isinf(mafrfs_margin_ratio(np.zeros_like(X), y, [0]))


def test_mafrfs_pool_is_recomputed_conditionally_not_a_static_top_list(monkeypatch):
    # Singletons favor 0,1,2. Once 0 enters the pool, 2 is complementary
    # and must replace 1 as the second pool member; its margin is best.
    import certified_frfs.baselines as baselines
    fd = {(): 0, (0,): 0.3, (1,): 0.2, (2,): 0.1,
          (0, 1): 0.35, (0, 2): 0.8}
    monkeypatch.setattr(baselines, "gamma", lambda X, y, features=None, weights=None:
                        fd[tuple(sorted(features))])
    X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    assert select_mafrfs_additive(X, y, k=1, sop=2) == [2]


def test_mafrfs_single_candidate_matches_forward_dependency_ranking():
    X = np.array([[0.0, 0.0, 0.2], [0.1, 0.2, 0.9],
                  [0.8, 0.7, 0.2], [1.0, 0.9, 0.8]])
    y = np.array([0, 0, 1, 1])
    weights = np.array([0.3, 0.2, 0.1])
    expected_first = max(range(3), key=lambda a: gamma(X, y, [a], weights))
    ranked = select_mafrfs_additive(X, y, k=3, sop=1, weights=weights)
    assert ranked[0] == expected_first
    assert sorted(ranked) == [0, 1, 2]
    assert select_mafrfs_additive(X, y, k=0, weights=weights) == []
