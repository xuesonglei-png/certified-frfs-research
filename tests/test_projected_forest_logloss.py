import itertools
import numpy as np
from scipy.special import logsumexp
from certified_frfs.multiknot_forest import scores
from certified_frfs.projected_forest_logloss import MultiknotProjectedLogloss


def test_label_loss_and_scores_equal_direct_raw_imputation_every_support():
    rng = np.random.default_rng(371)
    X, y = rng.uniform(size=(23, 4)), np.arange(23)%3
    fit = {'knots': [0., .1, .45, .7, 1.], 'classes': [0, 1, 2], 'edges': [[0, 2], [1, 2]],
           'coef': rng.normal(size=(3, 4, 5)), 'pair_coef': rng.normal(size=(3, 2, 5, 5)),
           'intercept': rng.normal(size=3)}
    objective = MultiknotProjectedLogloss(X, y, fit)
    for mask in itertools.product((False, True), repeat=4):
        fs = list(np.flatnonzero(mask))
        raw = X.copy()
        omitted = np.flatnonzero(~np.array(mask))
        raw[:, omitted] = objective.center[omitted]
        value = scores(raw, fit)
        np.testing.assert_allclose(objective.projected_scores(fs), value, atol=1e-12)
        losses = logsumexp(value, axis=1)-value[np.arange(len(X)), y]
        expected = float(np.mean([losses[y == c].mean() for c in range(3)]))
        np.testing.assert_allclose(objective(fs), expected, atol=1e-12)


def test_projected_label_loss_is_invariant_to_common_class_score_shift():
    rng = np.random.default_rng(122)
    X, y = rng.uniform(size=(18, 2)), np.arange(18)%2
    W, V, b = rng.normal(size=(2, 2, 3)), rng.normal(size=(2, 1, 3, 3)), rng.normal(size=2)
    fit = {'knots': [0., .5, 1.], 'classes': [0, 1], 'edges': [[0, 1]], 'coef': W,
           'pair_coef': V, 'intercept': b}
    common = {'coef': rng.normal(size=(1, 2, 3)), 'pair_coef': rng.normal(size=(1, 1, 3, 3)), 'intercept': 7.}
    shifted = {**fit, 'coef': W+common['coef'], 'pair_coef': V+common['pair_coef'], 'intercept': b+7.}
    original, changed = MultiknotProjectedLogloss(X, y, fit), MultiknotProjectedLogloss(X, y, shifted)
    for fs in ([], [0], [1], [0, 1]):
        np.testing.assert_allclose(original(fs), changed(fs), atol=1e-12)
