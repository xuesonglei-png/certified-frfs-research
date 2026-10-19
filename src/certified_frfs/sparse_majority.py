"""Cardinality-constrained metric learning from robust 5NN majority margins.

The third-smallest enemy contrast allows at most two preceding enemies.
Three fixed same-class witnesses are selected using the original full metric.
Nonconvex/nonsmooth local optimization; no test-risk or novelty claim.
"""
import numpy as np
from scipy.special import expit
from .majority_metric import MajorityTerms
from .sparse_neighbor import optimize_cardinality_objective


class MajorityMarginObjective:
    def __init__(self, X, y, *, eps=.01, ridge=1., margin=.02, temperature=.03):
        X = np.asarray(X, float); self.y = np.asarray(y); self.ridge = ridge
        self.temperature = temperature; self.margin = margin
        if ridge < 0 or temperature <= 0 or margin < 0: raise ValueError('invalid majority objective parameters')
        self.workspace = MajorityTerms(X, y, eps=eps, weights=np.ones(X.shape[1])/X.shape[1])
        cross = self.workspace.cross
        self.ids = np.flatnonzero(self.workspace.applicable & (cross.sum(axis=1) >= 3))
        labels = self.y[self.ids]; classes = np.unique(labels)
        self.balance = np.array([1/(len(classes)*sum(labels == q)) for q in labels])

    def __call__(self, w):
        ws = self.workspace; T = np.einsum('d,dijf->ijf', w, ws.terms, optimize=True)
        loss = 0.; gradient = np.zeros_like(w)
        for alpha, i in zip(self.balance, self.ids):
            enemies = np.flatnonzero(ws.cross[i]); friend = T[i, enemies].argmin(axis=1)
            values = T[i, enemies, friend]
            # Stable ordering supplies a deterministic subgradient at ties.
            index = np.argsort(values, kind='stable')[2]; enemy = enemies[index]; witness = friend[index]
            coefficients = ws.terms[:, i, enemy, witness]; contrast = values[index]
            z = (self.margin-contrast)/self.temperature
            loss += alpha*self.temperature*np.logaddexp(0, z)
            gradient -= alpha*expit(z)*coefficients
        loss += self.ridge*np.sum(w*w); gradient += 2*self.ridge*w
        return float(loss), gradient


def sparse_majority_candidates(X, y, *, eps=.01, ridge=1., fractions=(.35, .65, 1.)):
    d = np.asarray(X).shape[1]; objective = MajorityMarginObjective(X, y, eps=eps, ridge=ridge)
    dense, full_fit = optimize_cardinality_objective(objective, np.ones(d)/d, d, ridge=ridge)
    rows = []
    for fraction in fractions:
        if not 0 < fraction <= 1: raise ValueError('invalid fraction')
        k = max(1, int(np.ceil(fraction*d)))
        w, fit = (dense.copy(), full_fit) if k == d else optimize_cardinality_objective(objective, dense, k, ridge=ridge)
        rows.append({'config': {'eps': eps, 'fraction': fraction}, 'weights': w.tolist(),
                     'features': np.flatnonzero(w > 0).tolist(), 'fit': fit})
    return rows
