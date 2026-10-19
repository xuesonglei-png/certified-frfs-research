"""Exploratory certificates for SUBSET-normalized fuzzy dependency.

Unlike the frozen additive model, every nonzero-weight subset divides its L1
distance by its own weight sum. Attribute inclusion is therefore NOT monotone.
The input table and the complete perturbation box must lie in [0,1]. Bounds
preserve the same perturbed table on both sides, and concern finite-table
dependency, not prediction. Floating-point safeguards are not formal proof.
"""
from __future__ import annotations

import numpy as np

from .core import (_inputs, _distance, _interval_distances,
                   _exact_stability_mask, interval_bounds, forest_maximum)
from .relative import graph_minimum_lower
from .star_extrema import star_extrema


def _validate_normalized(X, y, features, weights, eps, clip):
    X, y, features, weights = _inputs(X, y, features, weights)
    if np.any(X < 0) or np.any(X > 1):
        raise ValueError("subset-normalized model requires X in [0,1]")
    full_weight = float(weights.sum())
    retained_weight = float(weights[features].sum())
    if not np.isfinite(full_weight) or full_weight <= 0:
        raise ValueError("full-table weight sum must be finite and positive")
    if not np.isfinite(retained_weight) or retained_weight <= 0:
        raise ValueError("retained subset must have positive weight; empty subsets are invalid")
    lower, upper = interval_bounds(X, eps, clip)
    if np.any(lower < 0) or np.any(upper > 1):
        raise ValueError("the complete perturbation box must remain in [0,1]; use clip=True")
    return X, y, features, weights, lower, upper, full_weight, retained_weight


def normalized_relative_certificate(X, y, features=None, weights=None, eps=0.01,
                                    retention=0.9, scope="classwise",
                                    method="independent", clip=True,
                                    stability="interval"):
    """Upper-bound max_Z q*gamma_C,g(Z)-gamma_B,g(Z) with subset normalization.

    gamma_A averages nearest opposite-class distances normalized by sum(w_A).
    These distances are already <=1 on the specified box, so truncation is
    unnecessary. Single-class positive memberships follow the convention 1.
    Positive upper bounds are inconclusive, not proof of a violation.
    """
    (X, y, features, weights, lower, upper,
     full_weight, retained_weight) = _validate_normalized(X, y, features, weights, eps, clip)
    if not np.isscalar(retention) or not np.isfinite(retention) or not 0 < retention <= 1:
        raise ValueError("retention must be in (0,1]")
    if scope not in {"classwise", "global", "objectwise"}:
        raise ValueError("invalid scope")
    if method not in {"independent", "hybrid"} or stability not in {"interval", "exact"}:
        raise ValueError("invalid method/stability")
    q, rho = float(retention), 1.0 - float(retention)
    n, d = X.shape
    full = np.arange(d)
    omitted = np.setdiff1d(full, features)
    # Normalize the two metrics independently, first scaling by their own
    # largest positive weight. Avoid 1/W_B: a valid tiny retained mass can
    # have an overflowing reciprocal even though its normalized metric is
    # perfectly well defined. Stability comparisons use this same scale.
    full_scaled = weights / weights.max()
    full_weights = full_scaled / full_scaled.sum()
    retained_scaled = weights[features] / weights[features].max()
    retained_weights = np.zeros(d)
    retained_weights[features] = retained_scaled / retained_scaled.sum()
    retained_full_mass = float(np.clip(full_weights[features].sum(), 0.0, 1.0))
    mixed_weights = full_weights.copy()
    mixed_weights[features] = retained_weights[features]
    cross = y[:, None] != y[None, :]
    labels = np.unique(y)
    if scope == "global":
        groups = [("all", np.ones(n, bool))]
    elif scope == "objectwise":
        groups = [(int(i), np.arange(n) == i) for i in range(n)]
    else:
        groups = [(label.item() if hasattr(label, "item") else label, y == label)
                  for label in labels]
    if any(not mask.any() for _, mask in groups):
        raise ValueError("class labels must form nonempty groups (missing labels are invalid)")

    nominal_c = np.ones(n)
    nominal_b = np.ones(n)
    lc = np.ones(n)
    uc = np.ones(n)
    lb = np.ones(n)
    object_bounds = np.full(n, -rho)
    nominal_objects = np.full(n, -rho)
    signed_omitted = q
    # Equivalent to 1-q*alpha in real arithmetic, but avoids cancellation
    # when q is near one and the omitted normalized weight is very small.
    signed_retained = (1.0 - q) + q * float(full_weights[omitted].sum())
    singleton = bool(np.all(lower == upper))
    stable = np.zeros(n, bool)
    base_stable = stable.copy()
    nearest = np.full(n, -1, int)

    if len(labels) > 1:
        nominal_c = np.where(cross, _distance(X, full, full_weights), np.inf).min(axis=1)
        nominal_b = np.where(cross, _distance(X, features, retained_weights), np.inf).min(axis=1)
        nominal_objects = q * nominal_c - nominal_b
        lo_b, hi_b = _interval_distances(lower, upper, features, retained_weights)
        lo_c, hi_c = _interval_distances(lower, upper, full, full_weights)
        _, hi_d = _interval_distances(lower, upper, omitted, full_weights, need_lower=False)
        lb = np.where(cross, lo_b, np.inf).min(axis=1)
        lc = np.where(cross, lo_c, np.inf).min(axis=1)
        uc = np.where(cross, hi_c, np.inf).min(axis=1)
        min_hi_b = np.where(cross, hi_b, np.inf).min(axis=1)
        candidates = cross & (lo_b <= min_hi_b[:, None] + 1e-12)
        paired = signed_omitted * hi_d - signed_retained * lo_b
        candidate_upper = np.where(candidates, paired, -np.inf).max(axis=1)
        object_bounds = np.minimum(q * uc - lb, candidate_upper)
        if singleton:
            object_bounds = nominal_objects.copy()
        if method == "hybrid" and not singleton:
            nearest = np.argmin(np.where(cross, _distance(X, features, retained_weights), np.inf), axis=1)
            competitors = np.where(cross, lo_b, np.inf)
            competitors[np.arange(n), nearest] = np.inf
            base_stable = hi_b[np.arange(n), nearest] + 1e-12 < competitors.min(axis=1)
            stable = base_stable.copy()
            if stability == "exact":
                exact, _ = _exact_stability_mask(lower, upper, nearest, features, retained_weights, cross)
                stable |= exact

    output_groups = []
    lp_calls, star_calls = 0, 0
    for label, mask in groups:
        count = int(mask.sum())
        v = mask.astype(float) / count
        independent = float(v @ object_bounds)
        margin = float(v @ nominal_objects)
        signed_structure = independent
        deleted_max = retained_min = None
        stability_masks = [stable]
        if stability == "exact" and np.any(stable != base_stable):
            stability_masks.append(base_stable)
        for stability_mask in stability_masks:
            active = mask & stability_mask
            if method != "hybrid" or not active.any():
                continue
            edges = [(int(i), int(nearest[i]), float(v[i])) for i in np.flatnonzero(active)]
            if scope in {"classwise", "objectwise"}:
                extrema = star_extrema(lower, upper,
                                       [(i, j) for i, j, _ in edges],
                                       leaf_weights=[a for _, _, a in edges],
                                       attribute_weights=mixed_weights)
                maximum = float(extrema["maximum_by_attribute"][omitted].sum())
                raw_minimum = float(extrema["minimum_by_attribute"][features].sum())
                minimum = max(0.0, raw_minimum - 1e-10 * (1.0 + abs(raw_minimum)))
                star_calls += 1
            else:
                maximum = forest_maximum(lower[:, omitted], upper[:, omitted], edges, full_weights[omitted])
                minimum = 0.0
                if signed_retained:
                    for k in features:
                        if retained_weights[k] > 0:
                            solved = graph_minimum_lower(lower[:, k], upper[:, k], edges)
                            minimum += retained_weights[k] * solved["lower_bound"]
                            lp_calls += int(solved["lp_used"])
            coupled = signed_omitted * maximum - signed_retained * minimum
            candidate = min(coupled, float(v[active] @ object_bounds[active]))
            candidate += float(v[~active] @ object_bounds[~active])
            if candidate < signed_structure:
                signed_structure = float(candidate)
                deleted_max, retained_min = float(maximum), float(minimum)

        value = max(min(independent, signed_structure), margin) + 1e-10
        # Identical normalized metrics imply q*p_C-p_B=-(1-q)*p_C<=0.
        # This identity handles structural zero constraints without accepting
        # a positive tolerance as if it were a nonpositive mathematical bound.
        structural = len(omitted) == 0 or np.all(weights[omitted] == 0)
        if structural:
            value = min(value, 0.0)
        if singleton or len(labels) == 1:
            value = margin
        output_groups.append({
            "group": label, "n_objects": count, "nominal_margin": margin,
            "upper_bound": float(value), "independent_bound": independent,
            "signed_forest_bound": signed_structure,
            "signed_structure_bound": signed_structure,
            "separate_reference_bound": float(v @ (uc-lb) - rho*(v @ lc)),
            "full_dependency_nominal": float(v @ nominal_c),
            "full_dependency_lower": float(v @ lc),
            "retained_dependency_nominal": float(v @ nominal_b),
            "stable_count": int((mask & stable).sum()),
            "stable_uncapped_count": int((mask & stable).sum()),
            "deleted_graph_max": deleted_max,
            "retained_graph_min_lower": retained_min,
        })
    maximum_upper = max(group["upper_bound"] for group in output_groups)
    return {
        "model": "subset_weight_normalized_l1_on_unit_box",
        "attribute_inclusion_monotone": False,
        "retention": q, "scope": scope, "method": method,
        "features": features.tolist(), "groups": output_groups,
        "full_weight": full_weight, "retained_weight": retained_weight,
        "full_normalized_weights": full_weights.tolist(),
        "retained_normalized_weights": retained_weights.tolist(),
        "retained_full_mass_fraction": retained_full_mass,
        "max_upper_bound": float(maximum_upper),
        "nominal_max_margin": max(group["nominal_margin"] for group in output_groups),
        "certified": bool(maximum_upper <= 0),
        "stable_fraction": float(stable.mean()),
        "stable_uncapped_fraction": float(stable.mean()),
        "stability_evaluated": bool(method == "hybrid" and not singleton and len(labels) > 1),
        "lp_calls": lp_calls, "star_calls": star_calls,
        "numerical_guarantee": "floating_point_bounds_not_formal_verification",
    }


def select_normalized_relative(X, y, *, eps=0.01, retention=0.9, weights=None,
                               scope="classwise", method="independent", order=None,
                               seed=17, clip=True, stability="interval"):
    """Backward certified feasibility search; no monotonicity/minimality claim.

    Every accepted deletion is checked against the original FULL reference.
    A previously rejected feature need not remain infeasible after other
    deletions. This one-pass search makes no inclusion-minimality assertion.
    """
    from .selection import deletion_order
    (X, y, full, weights, _, _, _, _) = _validate_normalized(X, y, None, weights, eps, clip)
    if order is None:
        # The nearest-neighbor MI estimator has no repeated-class samples in
        # an all-singleton table. These degenerate tables still admit valid
        # certificates, so use a declared deterministic order there.
        _, class_counts = np.unique(y, return_counts=True)
        order = (full.tolist() if len(class_counts) < 2 or class_counts.max() < 2
                 else deletion_order(X, y, seed))
    else:
        order = list(order)
    if sorted(order) != full.tolist():
        raise ValueError("order must be a permutation of all attributes")
    features, history = full.tolist(), []
    for feature in order:
        candidate = [a for a in features if a != feature]
        if not candidate or weights[candidate].sum() <= 0:
            history.append({"candidate_removed": int(feature), "accepted": False,
                            "reason": "retained_weight_must_remain_positive"})
            continue
        cert = normalized_relative_certificate(X, y, candidate, weights, eps,
                                               retention, scope, method, clip, stability)
        accepted = cert["max_upper_bound"] <= -1e-10
        history.append({"candidate_removed": int(feature), "accepted": bool(accepted),
                        "max_upper_bound": cert["max_upper_bound"]})
        if accepted:
            features = candidate
    final = normalized_relative_certificate(X, y, features, weights, eps,
                                            retention, scope, method, clip, stability)
    if not final["certified"]:
        raise RuntimeError("final subset-normalized relative subset did not verify")
    return {"features": features, "certificate": final, "history": history,
            "status": "certified_subset_normalized_relative_feasible_subset",
            "model": "subset_weight_normalized_l1_on_unit_box",
            "attribute_inclusion_monotone": False, "minimality_proven": False}
