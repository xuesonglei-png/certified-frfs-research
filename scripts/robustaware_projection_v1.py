"""Certificate-preserving robust projection for a development experiment.

The candidate target is produced by the existing hard-aware local search.  The
new step only chooses among a fixed interpolation grid whose original-full FRS
and classwise certificates are independently recomputed.  Its training score
is the mean balanced accuracy of leave-one-out 5NN over deterministic bounded
perturbations.  This is a finite training heuristic, not a test-risk bound.
"""
import numpy as np
from scipy.spatial.distance import cdist

from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate
from hardaware_guarded_soft5_descent import fit_hardaware_guarded_soft5
from simplex_soft_topk_targets import fit_simplex_target


def hard5_ba(A, y, weights):
    A = np.asarray(A, float); y = np.asarray(y); weights = np.asarray(weights, float)
    D = cdist(A * weights, A * weights, metric="cityblock")
    np.fill_diagonal(D, np.inf)
    k = min(5, len(A) - 1)
    neighbours = np.argsort(D, axis=1, kind="stable")[:, :k]
    classes = np.unique(y)
    votes = np.stack([(y[neighbours] == c).sum(axis=1) for c in classes], axis=1)
    pred = classes[votes.argmax(axis=1)]
    return float(np.mean([np.mean(pred[y == c] == c) for c in classes]))


def perturbed_hard5_ba(A, y, weights, *, draws=8, eps=.01, seed=20261011):
    """Deterministic bounded-noise LOO score used only for training selection."""
    A = np.asarray(A, float); y = np.asarray(y); values = []
    for draw in range(int(draws)):
        rng = np.random.default_rng(int(seed) + draw)
        noisy = np.clip(A + rng.uniform(-eps, eps, A.shape), 0., 1.)
        values.append(hard5_ba(noisy, y, weights))
    return float(np.mean(values))


def _certificate(A, y, features, weights, full):
    cert = weight_certificate(A, y, features, full, weights)
    cls = classwise_dependency_certificate(y, cert)
    return {"certificate": cert, "class_certificate": cls,
            "accepted": bool(cert["feasible"] and cls["feasible_all_classes"])}


def fit_robustaware_projection(A, y, features, anchor, *, draws=8, eps=.01,
                               max_iter=4, max_guard_checks=16):
    """Fit and select a certified robust projection around ``anchor``.

    Returns the exact selected weight vector and a complete train-only trace.
    A proposed update is never accepted unless the original full reference is
    feasible at that point.  If the anchor is outside that cell, the function
    returns it unchanged with an explicit status.
    """
    A, y = np.asarray(A, float), np.asarray(y)
    fs = list(features); anchor = np.asarray(anchor, float)
    d, mass = A.shape[1], float(np.sqrt(A.shape[1])); full = np.ones(d) / mass
    if (A.ndim != 2 or len(A) < 7 or y.shape != (len(A),) or not fs
            or fs != sorted(set(fs)) or any(int(j) != j or j < 0 or j >= d for j in fs)
            or anchor.shape != (len(fs),) or not np.isfinite(A).all()
            or A.min() < 0 or A.max() > 1 or not np.isfinite(anchor).all()
            or anchor.min() < 0 or abs(anchor.sum() - mass) > 1e-8
            or int(draws) != draws or draws < 1 or eps != .01
            or int(max_iter) != max_iter or max_iter < 1
            or int(max_guard_checks) != max_guard_checks or max_guard_checks < 1):
        raise ValueError("invalid robust projection input")
    anchor_guard = _certificate(A, y, fs, anchor, full)
    info = {"features": fs, "anchor": anchor.tolist(), "mass": mass,
            "draws": int(draws), "epsilon": float(eps), "max_iter": int(max_iter),
            "max_guard_checks": int(max_guard_checks), "guard_reference": "original_full",
            "anchor_guard": anchor_guard, "goal_achieved": False}
    if not anchor_guard["accepted"]:
        return anchor.copy(), {**info, "status": "anchor_outside_original_full_certificate",
                               "weights": anchor.tolist(), "candidate_targets": [],
                               "selected_interpolation": 0., "trace": []}

    # Generate two independent, already versioned soft targets.  Their details
    # are retained so the caller can audit the exact candidate source.
    unconstrained, unconstrained_fit = fit_simplex_target(A, y, fs, anchor, constrained=False)
    constrained, constrained_fit = fit_simplex_target(A, y, fs, anchor, constrained=True)
    target, hard_fit = fit_hardaware_guarded_soft5(
        A, y, fs, anchor, [unconstrained, constrained],
        max_iter=int(max_iter), max_guard_checks=int(max_guard_checks))
    candidates = [anchor, target]
    grid = (0., .05, .1, .25, .5, .75, 1.)
    trace = []
    for source, candidate in enumerate(candidates[1:], start=1):
        for t in grid:
            w = (1. - t) * anchor + t * candidate
            checked = _certificate(A, y, fs, w, full)
            clean = hard5_ba(A[:, fs], y, w)
            robust = perturbed_hard5_ba(A[:, fs], y, w, draws=draws, eps=eps)
            trace.append({"source": source, "interpolation": float(t),
                          "weights": w.tolist(), "certificate": checked,
                          "clean_BA": clean, "perturbed_BA": robust,
                          "accepted": bool(checked["accepted"])})
    accepted = [r for r in trace if r["accepted"]]
    # Robust BA is primary, clean BA breaks ties, then smallest movement.
    selected = min(accepted, key=lambda r: (-round(r["perturbed_BA"], 12),
                                              -round(r["clean_BA"], 12),
                                              round(float(np.abs(np.asarray(r["weights"]) - anchor).sum()), 12),
                                              r["source"], r["interpolation"]))
    chosen = np.asarray(selected["weights"], float)
    info.update({"status": "selected_certified_robust_projection",
                 "candidate_targets": [unconstrained.tolist(), constrained.tolist(), target.tolist()],
                 "candidate_fits": [unconstrained_fit, constrained_fit, hard_fit],
                 "selected_interpolation": selected["interpolation"],
                 "selected_source": selected["source"], "selected_clean_BA": selected["clean_BA"],
                 "selected_perturbed_BA": selected["perturbed_BA"], "trace": trace,
                 "weights": chosen.tolist(),
                 "scope": "finite_train_only_certificate_preserving_robust_projection"})
    return chosen, info

