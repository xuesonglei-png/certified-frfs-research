"""Scalar Brent threshold replay; no production threshold/objective imports."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


def independent_objective(A, y, w):
    A, y, w = np.asarray(A, float), np.asarray(y), np.asarray(w, float)
    lo, hi = np.maximum(0, A-.01), np.minimum(1, A+.01)
    classes, sizes = np.unique(y, return_counts=True)
    contributing = classes[sizes >= 2]; total = 0.
    for i in range(len(A)):
        if y[i] not in contributing: continue
        ids = np.flatnonzero(np.arange(len(A)) != i); same = y[ids] == y[i]
        D = np.zeros(len(ids))
        for a, weight in enumerate(w):
            lower = np.maximum(0, np.maximum(lo[ids, a]-hi[i, a], lo[i, a]-hi[ids, a]))
            upper = np.maximum(abs(lo[i, a]-hi[ids, a]), abs(hi[i, a]-lo[ids, a]))
            D += weight*np.where(same, upper, lower)
        if len(ids) == 5:
            q = np.ones(5)
        else:
            assert len(ids) > 5
            def residual(radius): return float(np.sum(expit((radius-D)/.05))-5)
            radius = brentq(residual, float(D.min()-2), float(D.max()+2), xtol=1e-13, rtol=1e-14)
            q = expit((radius-D)/.05)
            assert abs(q.sum()-5) <= 1e-10
        probability = float(q[same].sum()/5)
        total -= np.log(probability+1e-12)/(len(contributing)*sum(y == y[i]))
    return float(total+.01*np.sum(w*np.log(len(w)*(w+1e-12))))
