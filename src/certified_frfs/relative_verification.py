"""Small-table MILP verification of signed relative-retention margins.

The objective is max_Z sum_i v_i (q p_C^Z(i) - p_B^Z(i)). A nonpositive
upper bound certifies retention on the finite training table. The objective
can be strictly negative; neither attacks nor bounds are truncated at zero.
This is numerical MILP verification, not an interval-arithmetic proof.
"""
from __future__ import annotations

import numpy as np

from .core import _inputs, _interval_distances, interval_bounds, positive_region
from .verification import _MILP


def exact_relative_margin_milp(X, y, features, weights=None, eps=0.01,
                               clip=True, object_weights=None, retention=0.9,
                               time_limit=10.0):
    """Bound the signed worst-case relative margin on a shared box.

    ``object_weights`` is a finite nonnegative length-n vector with positive
    sum; it is normalized to one. Zero entries are allowed, including a
    vector supported on just one class. ``retention`` must be in (0, 1].
    Feature weights always belong to the full table and are not rescaled.

    ``lower_bound`` is recomputed at the feasible ``worst_case`` table
    (also returned as ``adversarial_X``). On timeout or missing solver data,
    the nominal table remains an attained lower bound, and an independently
    valid signed interval bound remains the upper bound. A finite MILP dual
    bound can tighten that upper bound. ``optimal`` requires both a usable
    solver incumbent and a finite dual bound, and solver status zero.

    HiGHS tolerances plus a 1e-8 outward allowance apply; inspect ``gap`` and
    the diagnostic fields. A zero-information table can satisfy retention
    vacuously; this function does not assert that full-table information is
    positive, or give a classifier/population guarantee.
    """
    X, y, features, weights = _inputs(X, y, features, weights)
    lower, upper = interval_bounds(X, eps, clip=clip)
    retention = np.asarray(retention, dtype=float)
    if retention.ndim != 0 or not np.isfinite(retention) or not 0 < retention <= 1:
        raise ValueError("retention must be a finite scalar in (0, 1]")
    retention = float(retention)
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
    same_dependency = bool(np.all(weights[np.setdiff1d(full, features)] == 0))

    def margin(z):
        pc = positive_region(z, y, weights=weights)
        pb = positive_region(z, y, features, weights)
        return float(v @ (retention * pc - pb))

    nominal = margin(X)
    common = dict(nominal_margin=nominal, retention=retention,
                  object_weights=v.tolist(), features=features.tolist())
    constant_reason = None
    if len(np.unique(y)) < 2:
        constant_reason = "Single class: every positive-region membership equals one"
    elif not np.any(weights):
        constant_reason = "All attribute weights are zero"
    elif same_dependency and retention == 1.0:
        constant_reason = "Full and reduced dependencies coincide and retention equals one"
    elif np.all(lower == upper):
        constant_reason = "Singleton uncertainty set"
    if constant_reason is not None:
        return dict(common, lower_bound=nominal, upper_bound=nominal, gap=0.0,
                    optimal=True, status=0, message=constant_reason,
                    worst_case=X.copy(), adversarial_X=X.copy(),
                    variables=0, constraints=0, analytic_upper_bound=nominal,
                    solver_upper_bound=None, solver_has_incumbent=False,
                    solver_has_dual_bound=False, used_solver_dual_bound=False,
                    numerical_bound_conflict=False, witness_projection_max=0.0,
                    verification_source="constant_objective")

    # Independent signed interval bound; deliberately does not call the
    # nonnegative dependency-loss certificate, whose semantics differ.
    cross = y[:, None] != y[None, :]
    lo_c, hi_c = _interval_distances(lower, upper, full, weights)
    lo_b, _ = _interval_distances(lower, upper, features, weights)
    lower_pc = np.minimum(1.0, np.where(cross, lo_c, np.inf).min(axis=1))
    upper_pc = np.minimum(1.0, np.where(cross, hi_c, np.inf).min(axis=1))
    lower_pb = np.minimum(1.0, np.where(cross, lo_b, np.inf).min(axis=1))
    object_upper = ((retention - 1.0) * lower_pc if same_dependency
                    else retention * upper_pc - lower_pb)
    analytic_upper = min(retention, max(nominal, float(v @ object_upper) + 1e-12))

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
                qlo = lower[i, k] - upper[j, k]
                qhi = upper[i, k] - lower[j, k]
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

    # Combine identical full/reduced memberships before modeling. For q<1
    # their common signed coefficient remains negative in the maximization.
    sets = ((full, 1.0 - retention),) if same_dependency else (
        (full, -retention), (features, 1.0))
    for attribute_set, coefficient in sets:
        active_attributes = [k for k in attribute_set if weights[k] > 0]
        if not active_attributes:
            # There are at least two classes here, so p_empty is exactly 0.
            continue
        for i in np.flatnonzero(v):
            pos = model.var(0.0, 1.0, objective=coefficient * v[i])
            opposite = np.flatnonzero(y != y[i])
            selectors = [model.var(0.0, 1.0, integer=True)
                         for _ in range(len(opposite) + 1)]
            model.constraint({selector: 1.0 for selector in selectors}, 1.0, 1.0)
            model.constraint({pos: 1.0, selectors[0]: -1.0}, lower=0.0)
            for j, selector in zip(opposite, selectors[1:]):
                left, right = sorted((i, j))
                distance = {avars[left, right, k]: -weights[k] for k in active_attributes}
                distance[pos] = 1.0
                max_distance = sum(weights[k] * max(abs(lower[i, k] - upper[j, k]),
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
                     and np.isfinite(np.asarray(incumbent)[zvars]).all())
    projection_max = 0.0
    if has_incumbent:
        raw_z = np.asarray(incumbent)[zvars]
        candidate_z = np.clip(raw_z, lower, upper)
        projection_max = float(np.max(np.abs(candidate_z - raw_z)))
        # Solver p variables and objective are never used as attack values.
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
    status = int(solved.status)
    message = str(solved.message)
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
                analytic_upper_bound=float(analytic_upper), solver_upper_bound=solver_upper,
                solver_has_incumbent=bool(has_incumbent), solver_has_dual_bound=has_dual,
                used_solver_dual_bound=bool(use_dual), numerical_bound_conflict=conflict,
                witness_projection_max=projection_max, verification_source="milp")
