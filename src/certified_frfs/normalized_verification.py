"""Numerical MILP for subset-normalized fuzzy positive-region retention.

Each attribute set uses its own positive total weight. Consequently a
reduced dependency may exceed the full dependency; no monotonicity or
nonnegative-loss assumption is made. This exploratory model is separate
from the fixed-full-table-weight experiments.
"""
from __future__ import annotations

import numpy as np

from .core import _inputs, _interval_distances, interval_bounds, positive_region
from .verification import _MILP


def exact_normalized_margin_milp(X, y, features, weights=None, eps=0.01,
                                clip=True, object_weights=None, retention=0.9,
                                time_limit=10.0):
    """Bound max_Z sum_i v_i [q*pbar_C(i,Z)-pbar_B(i,Z)].

    Distances use w_k/sum_{a in A} w_a for attribute set A. Nominal data
    and the entire uncertainty box must lie in [0,1]; ``clip=False`` is
    allowed only if its resulting box also remains in that domain. B must
    be nonempty and both full and selected total weights must be positive.
    Nonnegative object weights are normalized to one, and may be supported
    on one class. The objective can range over [-1,q], including values
    below q-1. The min(1,distance) cap is retained but is redundant here.

    The lower bound is attained by the returned feasible ``worst_case``
    table. On solver timeout/missing data it retains the nominal witness.
    A separately normalized interval bound and a finite solver dual bound
    provide upper bounds. ``analytic_lower_bound`` additionally bounds the
    objective at every table; it is not claimed to be an attained attack.

    ``optimal`` refers to HiGHS numerical optimality with available primal
    and dual data. This is not a directed-rounding/formal certificate or a
    predictive-accuracy guarantee. All original experiment files are kept
    separate from this normalized variant.
    """
    X, y, features, weights = _inputs(X, y, features, weights)
    if np.any(X < 0) or np.any(X > 1):
        raise ValueError("normalized model requires nominal X in [0,1]")
    lower, upper = interval_bounds(X, eps, clip=clip)
    if np.any(lower < 0) or np.any(upper > 1):
        raise ValueError("normalized model requires the entire uncertainty box in [0,1]")
    full_mass, retained_mass = float(weights.sum()), float(weights[features].sum())
    if (not len(features) or not np.isfinite(full_mass) or full_mass <= 0
            or not np.isfinite(retained_mass) or retained_mass <= 0):
        raise ValueError("features must be nonempty and full/retained weight sums must be positive")
    retention = np.asarray(retention, dtype=float)
    if retention.ndim != 0 or not np.isfinite(retention) or not 0 < retention <= 1:
        raise ValueError("retention must be a finite scalar in (0,1]")
    q = float(retention)
    if time_limit is not None and (not np.isscalar(time_limit)
                                  or not np.isfinite(time_limit) or time_limit < 0):
        raise ValueError("time_limit must be nonnegative and finite, or None")
    n, d = X.shape
    if object_weights is None:
        v = np.full(n, 1.0 / n)
    else:
        v = np.asarray(object_weights, dtype=float)
        if (v.shape != (n,) or not np.isfinite(v).all() or np.any(v < 0)
                or not np.isfinite(v.sum()) or v.sum() <= 0):
            raise ValueError("object_weights must be finite, nonnegative and have positive sum")
        v = v / v.sum()
    full = np.arange(d)
    full_weights = weights / full_mass
    retained_weights = np.zeros(d)
    retained_weights[features] = weights[features] / retained_mass
    same_dependency = bool(np.all(weights[np.setdiff1d(full, features)] == 0))

    def margin(z):
        pc = positive_region(z, y, full, full_weights)
        pb = positive_region(z, y, features, retained_weights)
        return float(v @ (q * pc - pb))

    nominal = margin(X)
    common = dict(nominal_margin=nominal, retention=q, object_weights=v.tolist(),
                  features=features.tolist(), full_weight_mass=full_mass,
                  retained_weight_mass=retained_mass,
                  full_normalized_weights=full_weights.tolist(),
                  retained_normalized_weights=retained_weights.tolist(),
                  normalization="separate_total_attribute_weight")
    reason = None
    if len(np.unique(y)) < 2:
        reason = "Single class: every positive-region membership equals one"
    elif same_dependency and q == 1.0:
        reason = "Full and retained normalized dependencies coincide and retention equals one"
    elif np.all(lower == upper):
        reason = "Singleton uncertainty set"
    if reason is not None:
        return dict(common, lower_bound=nominal, upper_bound=nominal, gap=0.0,
                    optimal=True, status=0, message=reason,
                    worst_case=X.copy(), adversarial_X=X.copy(),
                    variables=0, constraints=0, analytic_lower_bound=nominal,
                    analytic_upper_bound=nominal, solver_upper_bound=None,
                    solver_has_incumbent=False, solver_has_dual_bound=False,
                    used_solver_dual_bound=False, numerical_bound_conflict=False,
                    witness_projection_max=0.0, verification_source="constant_objective")

    cross = y[:, None] != y[None, :]
    lo_c, hi_c = _interval_distances(lower, upper, full, full_weights)
    lo_b, hi_b = _interval_distances(lower, upper, features, retained_weights)

    def min_membership(distances):
        return np.minimum(1.0, np.where(cross, distances, np.inf).min(axis=1))

    lc, uc, lb, ub = map(min_membership, (lo_c, hi_c, lo_b, hi_b))
    if same_dependency:
        lower_objects, upper_objects = (q - 1.0) * uc, (q - 1.0) * lc
    else:
        lower_objects, upper_objects = q * lc - ub, q * uc - lb
    analytic_lower = max(-1.0, min(nominal, float(v @ lower_objects) - 1e-12))
    analytic_upper = min(q, max(nominal, float(v @ upper_objects) + 1e-12))

    model = _MILP()
    zvars = np.empty((n, d), dtype=int)
    for i in range(n):
        for k in range(d):
            zvars[i, k] = model.var(lower[i, k], upper[i, k])
    avars = {}
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] == y[j] or (v[i] == 0 and v[j] == 0):
                continue
            for k in np.flatnonzero(weights):
                qlo, qhi = lower[i, k] - upper[j, k], upper[i, k] - lower[j, k]
                a = model.var(0.0, max(abs(qlo), abs(qhi)))
                avars[i, j, k] = a
                zi, zj = zvars[i, k], zvars[j, k]
                if qlo >= 0:
                    model.constraint({a: 1, zi: -1, zj: 1}, 0.0, 0.0)
                elif qhi <= 0:
                    model.constraint({a: 1, zi: 1, zj: -1}, 0.0, 0.0)
                else:
                    sign = model.var(0.0, 1.0, integer=True)
                    model.constraint({a: 1, zi: -1, zj: 1}, lower=0.0)
                    model.constraint({a: 1, zi: 1, zj: -1}, lower=0.0)
                    model.constraint({a: 1, zi: -1, zj: 1, sign: -2 * qlo}, upper=-2 * qlo)
                    model.constraint({a: 1, zi: 1, zj: -1, sign: -2 * qhi}, upper=0.0)

    sets = (((full, full_weights, 1.0-q),) if same_dependency else
            ((full, full_weights, -q), (features, retained_weights, 1.0)))
    for attribute_set, local_weights, coefficient in sets:
        active_attributes = [k for k in attribute_set if local_weights[k] > 0]
        for i in np.flatnonzero(v):
            pos = model.var(0.0, 1.0, objective=coefficient * v[i])
            opposite = np.flatnonzero(y != y[i])
            selectors = [model.var(0.0, 1.0, integer=True)
                         for _ in range(len(opposite) + 1)]
            model.constraint({selector: 1.0 for selector in selectors}, 1.0, 1.0)
            model.constraint({pos: 1.0, selectors[0]: -1.0}, lower=0.0)
            for j, selector in zip(opposite, selectors[1:]):
                left, right = sorted((i, j))
                distance = {avars[left, right, k]: -local_weights[k] for k in active_attributes}
                distance[pos] = 1.0
                max_distance = sum(local_weights[k] * max(abs(lower[i, k] - upper[j, k]),
                                                          abs(upper[i, k] - lower[j, k]))
                                   for k in active_attributes)
                model.constraint(distance, upper=0.0)
                chosen_distance = dict(distance)
                chosen_distance[selector] = -max_distance
                model.constraint(chosen_distance, lower=-max_distance)

    solved = model.solve(time_limit, 0.0)
    best_z, lower_bound = X.copy(), nominal
    incumbent = getattr(solved, "x", None)
    has_incumbent = (incumbent is not None and np.shape(incumbent) == (len(model.lower),)
                     and np.isfinite(np.asarray(incumbent)).all())
    projection_max = 0.0
    if has_incumbent:
        raw_z = np.asarray(incumbent)[zvars]
        candidate_z = np.clip(raw_z, lower, upper)
        projection_max = float(np.max(np.abs(candidate_z - raw_z)))
        attack_value = margin(candidate_z)
        if attack_value > lower_bound:
            best_z, lower_bound = candidate_z, attack_value
    dual = getattr(solved, "mip_dual_bound", None)
    has_dual = bool(dual is not None and np.isfinite(dual))
    solver_upper = float(-dual) if has_dual else None
    conflict = bool(has_dual and solver_upper + 1e-8 < lower_bound)
    use_dual = has_dual and not conflict
    upper_bound = analytic_upper
    if use_dual:
        upper_bound = min(upper_bound, solver_upper + 1e-8)
    upper_bound = float(max(upper_bound, lower_bound))
    status, message = int(solved.status), str(solved.message)
    if not has_incumbent:
        message += "; no usable solver incumbent: returning the nominal witness"
    if not has_dual:
        message += "; no finite solver dual bound: using the analytic upper bound"
    if conflict:
        message += "; dual bound conflicts with a feasible witness: dual bound discarded"
    return dict(common, lower_bound=float(lower_bound), upper_bound=upper_bound,
                gap=float(upper_bound - lower_bound), status=status, message=message,
                optimal=bool(status == 0 and has_incumbent and has_dual
                             and not conflict and projection_max <= 1e-7),
                worst_case=best_z, adversarial_X=best_z.copy(),
                variables=len(model.lower), constraints=len(model.clower),
                analytic_lower_bound=float(analytic_lower), analytic_upper_bound=float(analytic_upper),
                solver_upper_bound=solver_upper, solver_has_incumbent=bool(has_incumbent),
                solver_has_dual_bound=has_dual, used_solver_dual_bound=bool(use_dual),
                numerical_bound_conflict=conflict, witness_projection_max=projection_max,
                verification_source="milp")
