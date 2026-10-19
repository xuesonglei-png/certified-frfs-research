"""Fixed prediction models used only for empirical evaluation."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist


class FuzzyRoughScoreClassifier(ClassifierMixin, BaseEstimator):
    """Mean of lower/upper crisp-class approximation scores.

    This is a declared standard-style fuzzy-rough scoring implementation,
    not a claim of reproducing any particular published software package.
    """
    def __init__(self, weights=None):
        self.weights = weights

    def fit(self, X, y):
        self.X_ = np.asarray(X, float)
        self.y_ = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.priors_ = counts / counts.sum()
        self.weights_ = (np.ones(X.shape[1]) if self.weights is None
                         else np.asarray(self.weights))
        return self

    def predict(self, X):
        distances = cdist(np.asarray(X) * self.weights_,
                          self.X_ * self.weights_, metric="cityblock")
        scores = []
        for c in self.classes_:
            same = self.y_ == c
            lower = np.minimum(1.0, distances[:, ~same].min(axis=1))
            upper = np.maximum(0.0, 1.0 - distances[:, same].min(axis=1))
            scores.append((lower + upper) / 2)
        scores = np.column_stack(scores)
        # Break exact numerical ties by training prior, then sorted class order.
        winners = np.isclose(scores, scores.max(axis=1, keepdims=True), atol=1e-14, rtol=0)
        return self.classes_[np.argmax(np.where(winners, self.priors_, -1), axis=1)]
