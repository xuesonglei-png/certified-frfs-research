"""Conservative backward selection: bounds certify feasibility, not minimality."""
from __future__ import annotations

from time import perf_counter
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from .core import certificate, gamma


def deletion_order(X, y, seed=17):
    """Ascending training-only MI; deterministic tie breaking by column index."""
    if X.shape[1] == 1:
        return [0]
    relevance = mutual_info_classif(X, y, random_state=seed)
    return np.lexsort((np.arange(X.shape[1]), relevance)).tolist()


def select_certified(X, y, *, eps=0.01, eta=0.10, weights=None,
                     method="hybrid", order=None, seed=17):
    start = perf_counter()
    if not 0 <= eta < 1:
        raise ValueError("eta must be in [0,1)")
    d = X.shape[1]
    weights = np.ones(d) / np.sqrt(d) if weights is None else np.asarray(weights)
    order = deletion_order(X, y, seed) if order is None else list(order)
    if sorted(order) != list(range(d)):
        raise ValueError("order must be a permutation of all attributes")
    features = list(range(d))
    history = []
    for feature in order:
        candidate = [a for a in features if a != feature]
        # Keep a nonempty subset so the predictive comparison is well-defined.
        if not candidate:
            continue
        cert = certificate(X, y, candidate, weights, eps, method=method)
        accepted = bool(cert["upper_bound"] + 1e-10 <= eta)
        history.append({"removed_candidate": int(feature), "accepted": accepted,
                        "upper_bound": float(cert["upper_bound"]),
                        "stable_fraction": float(cert["stable_fraction"])})
        if accepted:
            features = candidate
    final = certificate(X, y, features, weights, eps, method=method)
    assert final["upper_bound"] <= eta + 1e-9
    return {"features": features, "certificate": final, "history": history,
            "seconds": perf_counter() - start,
            "status": "certified_feasible_nonempty_subset",
            "minimality_proven": False}


def select_nominal_backward(X, y, *, eta=0.1, weights=None, order=None, seed=17):
    """Same-order nominal ablation, not a reproduction of an external method."""
    d = X.shape[1]
    weights = np.ones(d) / np.sqrt(d) if weights is None else weights
    target = gamma(X, y, weights=weights) - eta
    features = list(range(d))
    order = deletion_order(X, y, seed) if order is None else order
    for feature in order:
        candidate = [a for a in features if a != feature]
        if candidate and gamma(X, y, candidate, weights) >= target - 1e-12:
            features = candidate
    return features
