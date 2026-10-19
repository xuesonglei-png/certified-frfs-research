import itertools
import numpy as np
import pytest
from scipy.special import logsumexp
from certified_frfs.crossfit_forest_risk import ProjectedForestScores, CrossfitForestRisk
from certified_frfs.orthogonal_forest_logistic import fit_orthogonal_forest, orthogonal_scores
from certified_frfs.validated_pool import preprocess_fit


@pytest.mark.parametrize('q', [2, 3])
def test_projected_logloss_scores_match_direct_raw_substitution_all_subsets(q):
    rng = np.random.default_rng(543)
    A = rng.uniform(size=(48, 5)); y = np.arange(48) % q
    fit = fit_orthogonal_forest(A, y, list(range(5)), [[0, 1], [1, 2], [2, 4]], C=2., pair_penalty=3.)
    center = rng.uniform(size=5); B = rng.uniform(size=(17, 5))
    projected = ProjectedForestScores(B, fit, center)
    for count in range(6):
        for fs in itertools.combinations(range(5), count):
            substituted = B.copy(); omitted = [a for a in range(5) if a not in fs]
            substituted[:, omitted] = center[omitted]
            np.testing.assert_allclose(projected(fs), orthogonal_scores(substituted, fit), atol=1e-12)


def test_crossfit_preprocessing_centers_labels_and_singleton_isolation():
    rng = np.random.default_rng(21)
    raw = rng.normal(size=(19, 4)); raw[-1] *= 1000
    y = np.array([0]*9+[1]*9+[2])
    risk = CrossfitForestRisk(raw, y)
    assert risk.singleton_indices == [18]
    independent = []
    for record in risk.records:
        a, b = np.array(record['reference_indices']), np.array(record['validation_indices'])
        assert 18 in a and 18 not in b and set(a).isdisjoint(b)
        A, B = preprocess_fit(raw[a], raw[b]); labels = y[a]
        weights = np.array([1/(3*sum(labels == c)) for c in labels])
        center = weights @ A; np.testing.assert_allclose(record['center'], center, atol=1e-12)
        B[:, [1, 3]] = center[[1, 3]]
        scores = orthogonal_scores(B, record['fit']); classes = np.array(record['fit']['classes'])
        own = np.argmax(y[b, None] == classes[None], axis=1)
        balance = np.array([1/(len(np.unique(y[b]))*sum(y[b] == c)) for c in y[b]])
        independent.append(balance @ (logsumexp(scores, axis=1)-scores[np.arange(len(b)), own]))
    np.testing.assert_allclose(risk([0, 2]), np.mean(independent), atol=1e-12)
