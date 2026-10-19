"""Implicit sigmoid top-k inclusion and class-balanced query vote loss."""
import numpy as np
from scipy.special import expit
from .query_probability import QueryProbabilityObjective


def soft_neighbor_inclusion(distance, *, k=5, temperature=.05):
    distance = np.asarray(distance, float)
    if distance.ndim != 2 or not np.isfinite(distance).all() or temperature <= 0 or k < 1:
        raise ValueError('finite query/reference distances and positive parameters required')
    if k >= distance.shape[1]: return np.ones_like(distance)
    left = distance.min(axis=1)-64*temperature; right = distance.max(axis=1)+64*temperature
    for _ in range(64):
        midpoint = (left+right)/2
        total = expit((midpoint[:, None]-distance)/temperature).sum(axis=1)
        left = np.where(total < k, midpoint, left); right = np.where(total >= k, midpoint, right)
    return expit((((left+right)/2)[:, None]-distance)/temperature)


class VoteMassObjective(QueryProbabilityObjective):
    def __init__(self, X, y, *, temperature=.05, top_k=5, ridge=.01, eps=.01, seed=1013):
        if int(top_k) != top_k or top_k < 1: raise ValueError('positive integer neighbor mass required')
        super().__init__(X, y, temperature=temperature, ridge=ridge, eps=eps, seed=seed)
        self.top_k = int(top_k)

    def __call__(self, w):
        w = np.asarray(w, float); value = self.ridge*np.sum(w*w); gradient = 2*self.ridge*w
        for C, same, labels in self.terms:
            distance = np.einsum('ijd,d->ij', C, w); k = min(self.top_k, distance.shape[1])
            inclusion = soft_neighbor_inclusion(distance, k=k, temperature=self.temperature)
            own = (inclusion*same).sum(axis=1); alpha = np.array([self.balance[q] for q in labels])
            # Smoothing keeps all-enemy neighborhoods finite, without claiming
            # a certificate for actual majority voting.
            denominator = own+.001*k; value -= alpha@np.log(denominator/(k*1.001))
            if k == distance.shape[1]: continue
            g = inclusion*(1-inclusion); total = g.sum(axis=1)
            center = np.einsum('ij,ijd->id', g, C)/np.maximum(total[:, None], 1e-300)
            own_derivative = np.einsum('ij,id->id', g*same, center)-np.einsum('ij,ijd->id', g*same, C)
            gradient -= np.einsum('i,id->d', alpha/denominator, own_derivative)/self.temperature
        return float(value), gradient
