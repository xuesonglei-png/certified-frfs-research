import itertools
import numpy as np
from certified_frfs.multiknot_projection import MultiknotProjectionLoss
from certified_frfs.multiknot_forest import scores


def test_projection_gram_matches_raw_imputation_every_subset_nonuniform_knots():
    rng = np.random.default_rng(361)
    X, y = rng.uniform(size=(26, 4)), np.arange(26)%3
    fit = {'knots': [0., .15, .4, .65, 1.], 'classes': [0, 1, 2], 'edges': [[0, 2], [1, 2]],
           'coef': rng.normal(size=(3, 4, 5)), 'pair_coef': rng.normal(size=(3, 2, 5, 5)),
           'intercept': rng.normal(size=3)}
    projection = MultiknotProjectionLoss(X, y, fit)
    alpha = np.array([1/(3*sum(y == c)) for c in y])
    old = scores(X, fit)
    for mask in itertools.product((False, True), repeat=4):
        features = list(np.flatnonzero(mask))
        omitted = np.flatnonzero(~np.array(mask))
        raw = X.copy()
        raw[:, omitted] = projection.center[omitted]
        difference = old-scores(raw, fit)
        difference -= difference.mean(axis=1, keepdims=True)
        expected = float(alpha@np.mean(difference**2, axis=1))
        np.testing.assert_allclose(projection(features), expected, atol=2e-13)
    assert projection(range(4)) == 0.
