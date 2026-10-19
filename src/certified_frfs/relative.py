"""Exploratory uniform relative preservation of empirical fuzzy dependency.

These bounds concern the same perturbed table on both sides. They preserve
signed margins, do not divide by a possibly zero dependency, and do not
guarantee classifier accuracy. Ordinary floating-point safeguards are used,
not directed-rounding or formal verification. Original primary code is frozen.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .core import (_inputs, _distance, _interval_distances,
                   _exact_stability_mask, interval_bounds, forest_maximum)
from .star_extrema import star_extrema


def graph_minimum_lower(lower, upper, edges):
    """Lower bound for min sum_e a_e |z_i-z_j| on scalar intervals.

    A clipped feasible dual flow certifies a lower bound even if the LP is
    interrupted. An LP primal objective is NEVER used as that lower bound.
    This routine works on general nonnegative weighted graphs, not just trees.
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (lower.ndim != 1 or upper.shape != lower.shape or
            not np.isfinite(lower).all() or not np.isfinite(upper).all() or
            np.any(lower > upper)):
        raise ValueError("invalid scalar intervals")
    combined = {}
    for i, j, coefficient in edges:
        if (int(i) != i or int(j) != j or not 0 <= i < len(lower) or
                not 0 <= j < len(lower) or not np.isfinite(coefficient) or coefficient < 0):
            raise ValueError("invalid edge")
        if i != j and coefficient > 0:
            pair = tuple(sorted((int(i), int(j))))
            combined[pair] = combined.get(pair, 0.0) + coefficient
    if not combined:
        return {"lower_bound": 0.0, "primal_value": 0.0, "status": 0,
                "flow": [], "nodes": [], "lp_used": False}
    nodes = sorted({i for pair in combined for i in pair})
    index = {v: i for i, v in enumerate(nodes)}
    lo, hi = lower[nodes], upper[nodes]
    pairs = np.array([(index[i], index[j]) for i, j in combined], dtype=int)
    coefficients = np.array(list(combined.values()), float)
    n, m = len(nodes), len(pairs)
    independent = float(np.dot(coefficients, np.maximum(
        np.maximum(lo[pairs[:, 0]] - hi[pairs[:, 1]],
                   lo[pairs[:, 1]] - hi[pairs[:, 0]]), 0.0)))
    # A shared point is a feasible zero-cost assignment.
    if lo.max() <= hi.min():
        return {"lower_bound": 0.0, "primal_value": 0.0, "status": 0,
                "flow": [0.0] * m, "nodes": nodes, "lp_used": False}
    rows, cols, values = [], [], []
    for e, (i, j) in enumerate(pairs):
        rows.extend([2*e]*3 + [2*e+1]*3)
        cols.extend([i, j, n+e, i, j, n+e])
        values.extend([1., -1., -1., -1., 1., -1.])
    matrix = coo_matrix((values, (rows, cols)), shape=(2*m, n+m)).tocsr()
    objective = np.r_[np.zeros(n), coefficients]
    solved = linprog(objective, A_ub=matrix, b_ub=np.zeros(2*m),
                     bounds=list(zip(lo, hi)) + [(0., None)]*m,
                     method="highs", options={"time_limit": 2.0})
    flow = np.zeros(m)
    marginals = getattr(getattr(solved, "ineqlin", None), "marginals", None)
    if marginals is not None and len(marginals) == 2*m and np.isfinite(marginals).all():
        flow = np.clip(-marginals[::2] + marginals[1::2], -coefficients, coefficients)
    divergence = np.zeros(n)
    np.add.at(divergence, pairs[:, 0], flow)
    np.add.at(divergence, pairs[:, 1], -flow)
    dual = float(np.minimum(divergence * lo, divergence * hi).sum())
    # Algebraically feasible dual; small outward adjustment is numerical,
    # not a substitute for formal rounding verification.
    safeguard = 1e-10 * (1.0 + np.abs(divergence * lo).sum() + np.abs(divergence * hi).sum())
    bound = max(0.0, independent - safeguard, dual - safeguard)
    primal = None
    if solved.x is not None and np.isfinite(solved.x[:n]).all():
        z = np.clip(solved.x[:n], lo, hi)
        primal = float(np.dot(coefficients, np.abs(z[pairs[:, 0]] - z[pairs[:, 1]])))
    return {"lower_bound": float(bound), "primal_value": primal,
            "independent_lower": independent, "dual_value": dual,
            "status": int(solved.status), "flow": flow.tolist(),
            "nodes": nodes, "lp_used": True}


def relative_certificate(X, y, features=None, weights=None, eps=0.01,
                         retention=0.9, scope="classwise", method="independent",
                         clip=True, stability="interval"):
    """Upper bounds on max_Z q*gamma_C,g(Z)-gamma_B,g(Z), for each group.

    scope is 'classwise', 'global' or 'objectwise'. Only an upper bound <= 0
    verifies all group constraints. Positive bounds mean unresolved, not
    necessarily violated. The 'hybrid' method also uses a signed forest bound
    on B-stable objects whose retained witness distance is never above one.
    """
    X, y, features, weights = _inputs(X, y, features, weights)
    if not np.isscalar(retention) or not np.isfinite(retention) or not 0 < retention <= 1:
        raise ValueError("retention must be in (0,1]")
    if scope not in {"classwise", "global", "objectwise"}:
        raise ValueError("invalid scope")
    if method not in {"independent", "hybrid"} or stability not in {"interval", "exact"}:
        raise ValueError("invalid method/stability")
    q, rho = float(retention), 1.0-float(retention)
    lower, upper = interval_bounds(X, eps, clip)
    n, d = X.shape
    full = np.arange(d)
    omitted = np.setdiff1d(full, features)
    cross = y[:, None] != y[None, :]
    if scope == "global":
        groups = [("all", np.ones(n, bool))]
    elif scope == "objectwise":
        groups = [(int(i), np.arange(n) == i) for i in range(n)]
    else:
        groups = [(label.item() if hasattr(label, "item") else label, y == label)
                  for label in np.unique(y)]
    nominal_c = np.minimum(1.0, np.where(cross, _distance(X, full, weights), np.inf).min(axis=1))
    nominal_b = np.minimum(1.0, np.where(cross, _distance(X, features, weights), np.inf).min(axis=1))
    nominal_objects = q*nominal_c - nominal_b
    lo_b, hi_b = _interval_distances(lower, upper, features, weights)
    lo_c, hi_c = _interval_distances(lower, upper, full, weights)
    _, hi_d = _interval_distances(lower, upper, omitted, weights, need_lower=False)
    lb = np.minimum(1., np.where(cross, lo_b, np.inf).min(axis=1))
    lc = np.minimum(1., np.where(cross, lo_c, np.inf).min(axis=1))
    uc = np.minimum(1., np.where(cross, hi_c, np.inf).min(axis=1))
    min_hi_b = np.where(cross, hi_b, np.inf).min(axis=1)
    candidates = cross & (lo_b <= min_hi_b[:, None] + 1e-12)
    paired = q*np.minimum(1., lo_b+hi_d) - np.minimum(1., lo_b)
    candidate_upper = np.where(candidates, paired, -np.inf).max(axis=1)
    object_bounds = np.minimum(q*uc-lb, candidate_upper)
    loss_bounds = np.minimum(uc-lb, np.where(candidates, hi_d, 0.).max(axis=1))
    if len(np.unique(y)) == 1:
        object_bounds = np.full(n, -rho)
        loss_bounds = np.zeros(n)
    singleton = bool(np.all(lower == upper))
    if singleton:
        object_bounds = nominal_objects.copy()
        loss_bounds = nominal_c-nominal_b
    stable = np.zeros(n, bool)
    nearest = np.full(n, -1, int)
    base_stable = np.zeros(n, bool)
    if method == "hybrid" and not singleton and len(np.unique(y)) > 1:
        nearest = np.argmin(np.where(cross, _distance(X, features, weights), np.inf), axis=1)
        competitors = np.where(cross, lo_b, np.inf)
        competitors[np.arange(n), nearest] = np.inf
        base_stable = hi_b[np.arange(n), nearest]+1e-12 < competitors.min(axis=1)
        stable = base_stable.copy()
        if stability == "exact":
            exact, _ = _exact_stability_mask(lower, upper, nearest, features, weights, cross)
            stable |= exact
        # Ensures p_B=d_B on each selected stable witness.
        no_cap = hi_b[np.arange(n), nearest] <= 1.-1e-12
        base_stable &= no_cap
        stable &= no_cap
    output_groups = []
    lp_calls, star_calls = 0, 0
    for label, mask in groups:
        count = int(mask.sum())
        v = mask.astype(float)/count
        independent = float(v @ object_bounds)
        margin = float(v @ nominal_objects)
        signed_forest = independent
        deleted_max = retained_min = None
        masks = [stable]
        if stability == "exact" and np.any(stable != base_stable):
            masks.append(base_stable)
        for stability_mask in masks:
            active = mask & stability_mask
            if method != "hybrid" or not active.any():
                continue
            edges = [(int(i), int(nearest[i]), float(v[i])) for i in np.flatnonzero(active)]
            if scope in {"classwise", "objectwise"}:
                extrema = star_extrema(lower, upper,
                                       [(i, j) for i, j, _ in edges],
                                       leaf_weights=[a for _, _, a in edges],
                                       attribute_weights=weights)
                maximum = float(np.asarray(extrema["maximum_by_attribute"])[omitted].sum())
                minimum = max(0., float(np.asarray(extrema["minimum_by_attribute"])[features].sum())-1e-10)
                star_calls += 1
            else:
                maximum = forest_maximum(lower[:, omitted], upper[:, omitted], edges, weights[omitted])
                minimum = 0.0
            if rho and scope == "global":
                for k in features:
                    if weights[k] > 0:
                        solved = graph_minimum_lower(lower[:, k], upper[:, k], edges)
                        minimum += weights[k]*solved["lower_bound"]
                        lp_calls += int(solved["lp_used"])
            coupled = q*maximum-rho*minimum
            candidate = min(coupled, float(v[active] @ object_bounds[active]))
            candidate += float(v[~active] @ object_bounds[~active])
            if candidate < signed_forest:
                signed_forest, deleted_max, retained_min = float(candidate), float(maximum), float(minimum)
        value = min(independent, signed_forest)
        value = max(value, margin)+1e-10
        # An algebraic bound handles exact structural zero constraints.
        structural = len(omitted) == 0 or np.all(weights[omitted] == 0)
        if structural:
            value = min(value, 0.0)
        if singleton:
            value = margin
        output_groups.append({
            "group": label, "n_objects": count, "nominal_margin": margin,
            "upper_bound": float(value), "independent_bound": independent,
            "signed_forest_bound": signed_forest,
            "separate_reference_bound": float(v @ loss_bounds-rho*(v @ lc)),
            "full_dependency_nominal": float(v @ nominal_c),
            "full_dependency_lower": float(v @ lc),
            "retained_dependency_nominal": float(v @ nominal_b),
            "stable_uncapped_count": int((mask & stable).sum()),
            "deleted_graph_max": deleted_max, "retained_graph_min_lower": retained_min,
        })
    maximum_upper = max(g["upper_bound"] for g in output_groups)
    return {"retention": q, "scope": scope, "method": method,
            "features": features.tolist(), "groups": output_groups,
            "max_upper_bound": float(maximum_upper),
            "nominal_max_margin": max(g["nominal_margin"] for g in output_groups),
            "certified": bool(maximum_upper <= 0),
            "stable_uncapped_fraction": float(stable.mean()), "lp_calls": lp_calls,
            "star_calls": star_calls,
            "numerical_guarantee": "floating_point_bounds_not_formal_verification"}


def select_relative(X, y, *, eps=0.01, retention=0.9, weights=None,
                    scope="classwise", method="independent", order=None,
                    seed=17, clip=True, stability="interval"):
    """Same-order backward selection; feasibility only, no minimality claim."""
    from .selection import deletion_order
    X, y, full, weights = _inputs(X, y, None, weights)
    order = deletion_order(X, y, seed) if order is None else list(order)
    if sorted(order) != full.tolist():
        raise ValueError("order must be a permutation of all attributes")
    features, history = full.tolist(), []
    for feature in order:
        candidate = [a for a in features if a != feature]
        if not candidate:
            continue
        cert = relative_certificate(X, y, candidate, weights, eps, retention,
                                    scope, method, clip, stability)
        accepted = cert["max_upper_bound"] <= -1e-10
        history.append({"candidate_removed": int(feature), "accepted": bool(accepted),
                        "max_upper_bound": cert["max_upper_bound"]})
        if accepted:
            features = candidate
    final = relative_certificate(X, y, features, weights, eps, retention,
                                 scope, method, clip, stability)
    if not final["certified"]:
        raise RuntimeError("final relative subset did not verify")
    return {"features": features, "certificate": final, "history": history,
            "status": "certified_relative_feasible_nonempty_subset",
            "minimality_proven": False}
