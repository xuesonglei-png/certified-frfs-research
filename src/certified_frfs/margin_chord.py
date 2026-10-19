"""Chord-relaxation anchors with independent mixture bounds and witnesses.

This module never uses a chord-LP primal objective as a certificate lower
bound. It verifies one declared enemy against all declared same-class
friends; the remaining rows are interpreted as other enemy references.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .margin import mixture_enemy_bound


def chord_enemy_interval(lower, upper, anchor, enemy, friends, features,
                         weights, time_limit=2.0):
    """Return a numerical bracket for one enemy's robust margin.

    Let E_j(x) be distance from anchor x to the enemy box, and F_h(x)
    the farthest distance to friend h. The single-enemy target is
    M_ij=min_x [E_j(x)-min_h F_h(x)]. ``lower_bound`` comes from a feasible
    mixture. ``upper_bound`` is the actual pair margin of ``worst_case``;
    the chosen enemy is projected onto the anchor and each friend is placed
    coordinatewise at its farther endpoint in that SAME table.

    All rows except anchor and the complete declared friend set are treated
    as enemies when evaluating ``all_enemy_margin_at_witness``. This is an
    upper bound for the ALL-enemy minimum, whereas the returned lower bound
    is only for the selected enemy. Do not combine these different scopes.

    The chord primal objective is reported only for research diagnostics.
    A missing chord incumbent falls back to the box midpoint. ``delta``
    bounds M_ij-L_mix in real arithmetic for the OPTIMAL mixture L_mix; a
    returned suboptimal mixture or chord LP need not achieve that gap. The
    returned posterior gap is measured directly, not asserted <=delta.
    All solve statuses and allowances are numerical, not formal proof.
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    weights = np.asarray(weights, float)
    if (lower.ndim != 2 or lower.shape != upper.shape or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(lower > upper)):
        raise ValueError("invalid finite coordinate box")
    n, d = lower.shape

    def index(value, name):
        raw = np.asarray(value)
        if raw.ndim != 0 or raw.dtype.kind not in "iu" or not 0 <= int(raw) < n:
            raise ValueError(f"invalid {name} index")
        return int(raw)

    anchor, enemy = index(anchor, "anchor"), index(enemy, "enemy")
    if anchor == enemy:
        raise ValueError("anchor and enemy must be distinct")
    friends = np.array([index(v, "friend") for v in friends], dtype=int)
    if not len(friends) or len(set(friends)) != len(friends) or anchor in friends or enemy in friends:
        raise ValueError("friends must be distinct, nonempty, and exclude anchor and enemy")
    raw_features = np.asarray(list(features))
    if (raw_features.ndim != 1 or not len(raw_features) or raw_features.dtype.kind not in "iu"
            or len(np.unique(raw_features)) != len(raw_features)
            or np.any(raw_features < 0) or np.any(raw_features >= d)):
        raise ValueError("invalid attribute indices")
    features = raw_features.astype(int)
    if weights.shape != (d,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be finite nonnegative full-table vector")
    if time_limit is not None and (not np.isscalar(time_limit) or not np.isfinite(time_limit)
                                   or time_limit < 0):
        raise ValueError("time_limit must be nonnegative, finite or None")
    r, p = len(friends), len(features)
    lo, hi = lower[anchor, features], upper[anchor, features]
    flo, fhi = lower[np.ix_(friends, features)], upper[np.ix_(friends, features)]
    midpoints = flo/2 + fhi/2
    widths = hi-lo
    far_lo = np.maximum(np.abs(lo-flo), np.abs(lo-fhi))
    far_hi = np.maximum(np.abs(hi-flo), np.abs(hi-fhi))
    slope = np.zeros((r, p))
    np.divide(far_hi-far_lo, widths, out=slope, where=widths > 0)
    intercept = far_lo-slope*lo
    chord_errors = np.zeros((r, p))
    internal = (midpoints > lo) & (midpoints < hi) & (widths > 0)
    # Factor the product to avoid unnecessarily squaring the interval width.
    fraction = np.zeros((r, p))
    np.divide(midpoints-lo, widths, out=fraction, where=internal)
    chord_errors[internal] = (2*fraction*(hi-midpoints))[internal]
    per_friend_delta = chord_errors @ weights[features]
    delta = float(per_friend_delta.max())
    if not np.isfinite(delta) or not np.isfinite(intercept).all():
        raise ValueError("chord coefficients exceed finite floating-point range")

    # Variables are x[0:p], distance_to_enemy[0:p], epigraph t.
    objective = np.r_[np.zeros(2*p), 1.0]
    rows, rhs = [], []
    enemy_lo, enemy_hi = lower[enemy, features], upper[enemy, features]
    for k in range(p):
        row = np.zeros(2*p+1)
        row[k], row[p+k] = 1, -1
        rows.append(row)
        rhs.append(enemy_hi[k])
        row = np.zeros(2*p+1)
        row[k], row[p+k] = -1, -1
        rows.append(row)
        rhs.append(-enemy_lo[k])
    for h in range(r):
        # sum w*v - Ch_h(x) <= t.
        row = np.r_[-weights[features]*slope[h], weights[features], -1.0]
        rows.append(row)
        rhs.append(float(weights[features] @ intercept[h]))
    options = {} if time_limit is None else {"time_limit": float(time_limit)}
    solved = linprog(objective, A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
                     bounds=list(zip(lo, hi))+[(0, None)]*p+[(None, None)],
                     method="highs", options=options)
    x = lo/2+hi/2
    incumbent = getattr(solved, "x", None)
    has_incumbent = (incumbent is not None and np.shape(incumbent) == (2*p+1,)
                     and np.isfinite(np.asarray(incumbent)[:p]).all())
    projection_max = 0.0
    if has_incumbent:
        raw_anchor = np.asarray(incumbent)[:p]
        x = np.clip(raw_anchor, lo, hi)
        projection_max = float(np.max(np.abs(raw_anchor-x)))
    near = np.maximum(np.maximum(enemy_lo-x, x-enemy_hi), 0.0)
    chord_at_anchor = float(weights[features] @ near - np.min(
        (slope*x+intercept) @ weights[features]))
    mixture = mixture_enemy_bound(lower, upper, anchor, enemy, friends,
                                  features, weights, time_limit=time_limit)

    z = lower/2+upper/2
    z[anchor, features] = x
    z[enemy, features] = np.clip(x, enemy_lo, enemy_hi)
    z[np.ix_(friends, features)] = np.where(
        np.abs(x-flo) >= np.abs(x-fhi), flo, fhi)
    effective_weights = np.zeros(d)
    effective_weights[features] = weights[features]
    distances = np.abs(z-z[anchor]) @ effective_weights
    friend_min = float(distances[friends].min())
    pair_margin = float(distances[enemy]-friend_min)
    all_enemies = np.setdiff1d(np.arange(n), np.r_[anchor, friends])
    all_enemy_margin = float(distances[all_enemies].min()-friend_min)
    lower_bound = float(mixture["lower_bound"])
    gap = pair_margin-lower_bound
    separated = bool(not np.any(internal))
    solver_value = getattr(solved, "fun", None)
    solver_value = float(solver_value) if solver_value is not None and np.isfinite(solver_value) else None
    return {
        "target": "single_enemy_robust_margin", "anchor": anchor, "enemy": enemy,
        "friends": friends.tolist(), "all_enemy_indices": all_enemies.tolist(),
        "features": features.tolist(), "effective_weights": effective_weights.tolist(),
        "lower_bound": lower_bound, "upper_bound": pair_margin, "posterior_gap": float(gap),
        "pair_margin_at_witness": pair_margin, "all_enemy_margin_at_witness": all_enemy_margin,
        "same_distance_at_witness": friend_min, "enemy_distance_at_witness": float(distances[enemy]),
        "worst_case": z, "anchor_witness": x,
        "mixture_result": mixture, "delta": delta,
        "delta_by_friend": per_friend_delta.tolist(), "chord_gap_by_friend_attribute": chord_errors,
        "midpoint_separation": separated, "chord_objective_at_witness_anchor": chord_at_anchor,
        "numerical_opt_chord": solver_value, "chord_solver_status": int(solved.status),
        "chord_solver_success": bool(solved.success), "chord_solver_message": str(solved.message),
        "chord_has_incumbent": bool(has_incumbent), "anchor_projection_max": projection_max,
        "numerical_bound_conflict": bool(gap < -1e-7),
        "delta_guarantee_scope": "M_ij minus OPTIMAL mixture bound <= delta in real arithmetic; "
            "the measured posterior gap may be larger for interrupted/suboptimal solves",
        "numerical_guarantee": "floating_point_bounds_and_feasible_witness_not_formal_verification",
    }
