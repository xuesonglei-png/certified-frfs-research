"""Nested relation--support co-design with the original finite certificate.

This module is a prospective experiment only.  The learned relation is used
both to generate candidate supports and by the deployed weighted 5-NN rule;
it is therefore not interchangeable with the older learned-reference pilot.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from .baselines import mi_scores, relieff_scores
from .capped_sparse import capped_sparse_projection
from .feasible_transport import weight_certificate
from .majority_metric import learn_majority_weights
from .redistributed_loss import transfer_weights
from .validated_pool import preprocess_fit


EPS = 0.01
ETA = 0.1
RETENTION = 0.9


def weighted_knn(A, y, B, truth, features, weights, *, k=5):
    """Evaluate a deterministic weighted Manhattan kNN predictor."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    fs = np.asarray(features, int)
    w = np.asarray(weights, float)
    if fs.ndim != 1 or len(fs) == 0 or w.shape != (len(fs),):
        raise ValueError("nonempty support and matching weights required")
    distance = np.zeros((len(B), len(A)))
    for a, weight in zip(fs, w):
        distance += weight * np.abs(B[:, a, None] - A[None, :, a])
    nn = np.argsort(distance, axis=1, kind="stable")[:, :min(k, len(A))]
    classes = np.unique(y)
    counts = np.column_stack([(y[nn] == c).sum(axis=1) for c in classes])
    pred = classes[counts.argmax(axis=1)]
    probabilities = (counts + 1.0) / (nn.shape[1] + len(classes))
    truth_index = np.searchsorted(classes, truth)
    return {
        "clean_BA": float(balanced_accuracy_score(truth, pred)),
        "macro_F1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "balanced_logloss": float(np.mean([
            log_loss(truth[truth == c], probabilities[truth == c], labels=classes)
            for c in np.unique(truth)
        ])),
        "predictions": pred.tolist(),
        "probabilities": probabilities.tolist(),
    }


def _fit_relation(A, y, family):
    d = A.shape[1]
    if family == "uniform_sqrt":
        return np.ones(d) / np.sqrt(d)
    if family == "learned_majority":
        learned = learn_majority_weights(A, y, eps=EPS, reg=0.01)
        return capped_sparse_projection(learned / learned.sum(), d, 2.0 / d) * np.sqrt(d)
    raise ValueError(f"unknown relation family: {family}")


def _ranking(A, y, weights, family, seed):
    # MI and Relief are computed on the fitted training geometry.  The third
    # score is an explicitly relation-aware class-contrast term; unlike a
    # columnwise rescaling followed by per-column normalization, it cannot
    # erase the learned weights.
    mi = mi_scores(A, y, random_state=seed)
    relief = relieff_scores(A, y, random_state=seed)
    same = y[:, None] == y[None, :]
    contrast = np.zeros(A.shape[1])
    for a in range(A.shape[1]):
        diffs = np.abs(A[:, a, None] - A[None, :, a])
        friends = diffs[same & ~np.eye(len(y), dtype=bool)]
        enemies = diffs[~same]
        contrast[a] = weights[a] * (float(np.mean(enemies)) - float(np.mean(friends)))
    scores = np.column_stack((mi, relief, contrast))
    ranks = np.argsort(np.argsort(-scores, axis=0, kind="stable"), axis=0)
    aggregate = ranks.mean(axis=1)
    return np.argsort(aggregate, kind="stable").astype(int).tolist()


def _candidate(A, y, weights, ranking, count):
    fs = sorted(ranking[:count])
    if len(fs) == A.shape[1]:
        reduced = weights.copy()
    else:
        mass = float(weights.sum())
        transferred = transfer_weights(A, fs, weights)
        cap = 2.0 * mass / len(weights)
        reduced = capped_sparse_projection(transferred / mass, len(fs), cap / mass) * mass
    uniform = np.ones(A.shape[1]) / np.sqrt(A.shape[1])
    cert = weight_certificate(A, y, fs, uniform, reduced, eps=EPS,
                             retention=RETENTION, dependency_eta=ETA)
    family_cert = weight_certificate(A, y, fs, weights, reduced, eps=EPS,
                                    retention=RETENTION, dependency_eta=ETA)
    def classwise(certificate):
        dep = certificate["dependency_certificate"]
        values = np.asarray(dep.get("redistributed_object_bounds", np.zeros(len(y))))
        bounds = [float(values[y == label].mean()) for label in np.unique(y)]
        return {"bounds": bounds, "feasible": all(b + (0 if np.array_equal(reduced, uniform) and len(fs) == len(uniform) else 1e-10*(1+abs(b))) <= ETA for b in bounds)}
    uniform_classwise, family_classwise = classwise(cert), classwise(family_cert)
    feasible = bool(cert["feasible"] and family_cert["feasible"] and
                    uniform_classwise["feasible"] and family_classwise["feasible"])
    return {"features": fs, "weights": reduced.tolist(), "certificate": cert,
            "family_certificate": family_cert, "classwise": uniform_classwise,
            "family_classwise": family_classwise, "joint_feasible": feasible}


def _counts(d):
    # The unchanged cap ``2 * mass / d`` makes supports below ceil(d/2)
    # infeasible; retaining that condition keeps this experiment aligned with
    # the certification protocol rather than silently relaxing the cap.
    minimum = int(np.ceil(d / 2))
    return sorted(set(max(minimum, int(np.ceil(d * f)))
                    for f in (0.25, 0.4, 0.55, 0.7, 0.85, 1.0)))


def fit_outer(X, y, *, seed=20261011, inner_splits=3):
    """Fit one outer training set using only its nested validation folds."""
    X, y = np.asarray(X, float), np.asarray(y)
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2 or counts.min() < inner_splits:
        raise ValueError("insufficient class repetitions for nested selection")
    split = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
    families = ("uniform_sqrt", "learned_majority")
    records = []
    for family in families:
        fold_rows = []
        for fold, (fit_idx, val_idx) in enumerate(split.split(X, y)):
            A, B = preprocess_fit(X[fit_idx], X[val_idx])
            weights = _fit_relation(A, y[fit_idx], family)
            ranking = _ranking(A, y[fit_idx], weights, family, seed + fold)
            candidates = []
            for count in _counts(A.shape[1]):
                candidate = _candidate(A, y[fit_idx], weights, ranking, count)
                if candidate["joint_feasible"]:
                    score = weighted_knn(A, y[fit_idx], B, y[val_idx],
                                         candidate["features"], candidate["weights"])
                else:
                    score = None
                candidates.append({"count": count, "candidate": candidate,
                                   "score": score})
            fold_rows.append({"fold": fold, "fit_indices": fit_idx.tolist(),
                              "validation_indices": val_idx.tolist(),
                              "reference_weights": weights.tolist(), "ranking": ranking,
                              "candidates": candidates})
        # Family/count are selected using only the inner validation folds.
        summary = []
        for count in _counts(X.shape[1]):
            rows = [c for r in fold_rows for c in r["candidates"]
                    if c["count"] == count and c["score"] is not None]
            if len(rows) != inner_splits:
                continue
            summary.append({"family": family, "count": count,
                            "mean_BA": float(np.mean([r["score"]["clean_BA"] for r in rows])),
                            "mean_logloss": float(np.mean([r["score"]["balanced_logloss"] for r in rows]))})
        records.append({"family": family, "folds": fold_rows, "summary": summary})
    choices = [r for record in records for r in record["summary"]]
    if not choices:
        raise RuntimeError("no nested candidate survived certification")
    # Fixed compression tolerance, not a confidence interval or risk bound.
    best_ba = max(c["mean_BA"] for c in choices)
    eligible = [c for c in choices if c["mean_BA"] >= best_ba - 0.005 - 1e-12]
    chosen = min(eligible, key=lambda c: (c["count"], c["mean_logloss"], c["family"]))
    A, _ = preprocess_fit(X, X)
    weights = _fit_relation(A, y, chosen["family"])
    ranking = _ranking(A, y, weights, chosen["family"], seed + 100)
    attempted = _candidate(A, y, weights, ranking, chosen["count"])
    candidate = attempted
    if not candidate["joint_feasible"]:
        uniform = _fit_relation(A, y, "uniform_sqrt")
        candidate = _candidate(A, y, uniform, list(range(A.shape[1])), A.shape[1])
        fallback = True
    else:
        fallback = False
    candidate.update({"family": "uniform_sqrt" if fallback else chosen["family"], "selected_count": chosen["count"],
                      "outer_refit_fallback": fallback})
    return {"selected": candidate, "chosen": chosen, "inner": records,
            "refit_attempt": attempted, "refit_reference_weights": weights.tolist(),
            "refit_ranking": ranking, "seed": seed, "inner_splits": inner_splits,
            "families": list(families), "counts": _counts(X.shape[1]),
            "certificate_parameters": {"eps": EPS, "eta": ETA, "retention": RETENTION}}

