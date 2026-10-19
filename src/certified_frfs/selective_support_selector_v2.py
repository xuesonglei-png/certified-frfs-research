"""Nested support selection for certified selective kNN.

This module is an append-only experiment.  It keeps the original finite
training certificate and chooses among certified candidates using validation
accuracy, then certified-correct coverage.  Validation labels are used only
inside the nested selector; the outer test fold is never used for choice.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .knn_query import certify_knn_queries
from .relation_support_selector_v1 import (
    _candidate, _counts, _fit_relation, _ranking, weighted_knn,
)
from .validated_pool import preprocess_fit


EPS = 0.01
ETA = 0.1
RETENTION = 0.9


def _query_scores(A, y, B, truth, candidate):
    score = weighted_knn(A, y, B, truth, candidate["features"], candidate["weights"])
    fs = candidate["features"]
    embedded = np.zeros(A.shape[1], float)
    embedded[fs] = candidate["weights"]
    cert = certify_knn_queries(A, y, B, np.asarray(score["predictions"]), fs,
                               weights=embedded, eps=EPS, k=5,
                               allowed_enemies=2)
    correct = (np.asarray(score["predictions"]) == np.asarray(truth)) & np.asarray(cert["certified"])
    certified_ba = float(np.mean([
        np.mean(correct[np.asarray(truth) == c]) for c in np.unique(truth)
    ]))
    coverage = float(np.mean(cert["certified"]))
    score["query_certified_BA"] = certified_ba
    score["query_coverage"] = coverage
    return score


def _support_pool(A, y, weights, family, seed):
    """Prefix supports plus deterministic one-swap neighbours.

    The one-swap pool is deliberately small so that it can be audited exactly.
    It includes non-prefix supports without searching on outer labels.
    """
    ranking = _ranking(A, y, weights, family, seed)
    d = A.shape[1]
    supports = {tuple(sorted(ranking[:count])) for count in _counts(d)}
    for count in _counts(d):
        base = set(ranking[:count])
        outside = [a for a in ranking[count:]]
        for remove in sorted(base):
            for add in outside[:2]:
                supports.add(tuple(sorted((base - {remove}) | {add})))
    return ranking, sorted(supports, key=lambda fs: (len(fs), fs))


def fit_outer(X, y, *, seed=20261011, inner_splits=3, clean_tolerance=.005):
    X, y = np.asarray(X, float), np.asarray(y)
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2 or counts.min() < inner_splits:
        raise ValueError("insufficient class repetitions for nested selection")
    families = ("uniform_sqrt", "learned_majority")
    inner_records = []
    split = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
    for family in families:
        folds = []
        for fold, (fit_idx, val_idx) in enumerate(split.split(X, y)):
            A, B = preprocess_fit(X[fit_idx], X[val_idx])
            weights = _fit_relation(A, y[fit_idx], family)
            ranking, supports = _support_pool(A, y[fit_idx], weights, family, seed + fold)
            candidates = []
            for fs in supports:
                candidate = _candidate(A, y[fit_idx], weights, ranking, len(fs))
                if sorted(candidate["features"]) != list(fs):
                    # Rebuild the candidate for an exchanged support.  The
                    # relation and all certificate parameters remain fixed.
                    candidate = _candidate(A, y[fit_idx], weights,
                                           list(fs), len(fs))
                if candidate["joint_feasible"]:
                    score = _query_scores(A, y[fit_idx], B, y[val_idx], candidate)
                else:
                    score = None
                candidates.append({"features": list(fs), "candidate": candidate,
                                   "score": score})
            valid = [r for r in candidates if r["score"] is not None]
            if not valid:
                raise RuntimeError("no certified inner support candidate")
            folds.append({"fold": fold, "fit_indices": fit_idx.tolist(),
                          "validation_indices": val_idx.tolist(),
                          "reference_weights": weights.tolist(),
                          "ranking": ranking, "candidates": candidates})
        inner_records.append({"family": family, "folds": folds})
    valid = [r for block in inner_records for fold in block["folds"]
             for r in fold["candidates"] if r["score"] is not None]
    # Aggregate by family and support, requiring a candidate to be available
    # in every inner fold.  This avoids selecting a support from a lucky fold.
    grouped = []
    for family in families:
        block = next(b for b in inner_records if b["family"] == family)
        keys = sorted(set(tuple(r["features"]) for f in block["folds"] for r in f["candidates"]))
        for fs in keys:
            rows = [next((r for r in f["candidates"] if tuple(r["features"]) == fs), None)
                    for f in block["folds"]]
            if any(r is None or r["score"] is None for r in rows):
                continue
            grouped.append({"family": family, "features": list(fs),
                            "mean_BA": float(np.mean([r["score"]["clean_BA"] for r in rows])),
                            "mean_certified_BA": float(np.mean([r["score"]["query_certified_BA"] for r in rows])),
                            "mean_coverage": float(np.mean([r["score"]["query_coverage"] for r in rows]))})
    if not grouped:
        raise RuntimeError("no common certified support candidate")
    best = max(r["mean_BA"] for r in grouped)
    eligible = [r for r in grouped if r["mean_BA"] >= best - clean_tolerance - 1e-12]
    chosen = max(eligible, key=lambda r: (r["mean_certified_BA"], r["mean_coverage"],
                                           r["mean_BA"], -len(r["features"]),
                                           r["family"], tuple(r["features"])))
    A, _ = preprocess_fit(X, X)
    weights = _fit_relation(A, y, chosen["family"])
    ranking = list(chosen["features"])
    attempted = _candidate(A, y, weights, ranking, len(ranking))
    fallback = False
    if not attempted["joint_feasible"]:
        uniform = _fit_relation(A, y, "uniform_sqrt")
        attempted = _candidate(A, y, uniform, list(range(A.shape[1])), A.shape[1])
        chosen_family = "uniform_sqrt"
        fallback = True
    else:
        chosen_family = chosen["family"]
    attempted.update({"family": chosen_family, "selected_count": len(attempted["features"]),
                      "outer_refit_fallback": fallback})
    return {"selected": attempted, "chosen": chosen, "inner": inner_records,
            "refit_reference_weights": weights.tolist(), "refit_ranking": ranking,
            "seed": seed, "inner_splits": inner_splits,
            "certificate_parameters": {"eps": EPS, "eta": ETA, "retention": RETENTION},
            "clean_tolerance": clean_tolerance}
