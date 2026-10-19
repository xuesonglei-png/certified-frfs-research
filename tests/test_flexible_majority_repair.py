"""Exact simplex bounds and original-contract acceptance of the new solver."""
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from flexible_majority_repair import capped_simplex_minimum, fit_flexible_majority_repair


def test_linear_bound_matches_all_capped_simplex_vertices():
    C = np.array([[1., 4., -2.], [-3., 1., .4], [0., 0., 0.]])
    vertices = np.array([[.5, .5, 0.], [.5, 0., .5], [0., .5, .5]])
    np.testing.assert_allclose(capped_simplex_minimum(C, .5), (C@vertices.T).min(axis=1), atol=1e-12)
    with pytest.raises(ValueError):
        capped_simplex_minimum(C, .2)


def test_redundant_feature_repair_preserves_original_certificate_and_mass_cap():
    x = np.r_[np.linspace(.05, .2, 8), np.linspace(.8, .95, 8)]
    X = np.column_stack([x]*5)
    y = np.repeat([0, 1], 8)
    weights, info = fit_flexible_majority_repair(X, y, [0, 1, 2, 3], time_limit=3.)
    assert weights is not None and info['certificate']['feasible']
    assert weights.sum() == pytest.approx(np.sqrt(5), abs=1e-12)
    assert weights.max()/np.sqrt(5) <= 2/5+1e-10
    assert info['certificate']['dependency_certificate']['upper_bound'] <= .1
