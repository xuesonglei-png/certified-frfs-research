import numpy as np
import pytest
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.query_probability import QueryProbabilityObjective


def test_quantile_geometry_uses_only_reference_values():
    reference = np.array([[0., np.nan], [1., 1.], [2., 2.], [100., 3.]])
    query = np.array([[-100., 0.], [1.5, 1.5], [1000., 10.]])
    A, B = preprocess_fit(reference, query, geometry='quantile')
    A2, B2 = preprocess_fit(reference, np.r_[query, [[1e12, -1e12]]], geometry='quantile')
    np.testing.assert_array_equal(A, A2)
    np.testing.assert_array_equal(B, B2[:len(B)])
    assert np.isfinite(A).all() and np.all((B >= 0) & (B <= 1))
    assert B[0, 0] == 0 and B[-1, 0] == 1
    with pytest.raises(ValueError):
        preprocess_fit(reference, query, geometry='unknown')


def test_inner_quantile_tables_refit_without_query_values():
    X = np.arange(48, dtype=float).reshape(12, 4); y = np.arange(12) % 2
    obj = QueryProbabilityObjective(X, y, geometry='quantile', eps=0.)
    for (C, _, _), split in zip(obj.terms, obj.splits):
        ti, qi = np.array(split['reference']), np.array(split['query'])
        A, B = preprocess_fit(X[ti], X[qi], geometry='quantile')
        np.testing.assert_allclose(C, np.abs(B[:, None]-A[None]))
