"""Small-table verification of a single object's joint-box LOO margin.

Distances are weighted L1 without truncation. The target and all references
share one perturbed table, so the target coordinate is never independently
chosen for its same-class and opposite-class comparisons.
"""
from __future__ import annotations

import numpy as np

from .core import _inputs, interval_bounds
from .verification import _MILP


def exact_negative_loo_margin_milp(X, y, target, features=None, weights=None,
                                   eps=0.01, clip=True, normalize=False,
                                   time_limit=10.0):
    """Bound max_Z [min_same d_B(target,h)-min_enemy d_B(target,j)].

    ``same`` excludes the target itself. ``eps`` broadcasts to the table,
    allowing independent query/reference coordinate boxes. ``normalize``
    divides selected attribute weights by their positive sum; otherwise the
    fixed attribute weights are used directly. Distances are never capped.
    A negative upper bound verifies a strictly nearer same-class reference
    throughout this finite-table box. A positive attained lower bound is a
    failure witness. Zero allows a distance tie and needs a tie policy.

    If no same-class reference or no enemy exists, ``applicable=False`` and
    ``status='not_applicable'`` are returned with no numeric bound. A target
    index error is rejected. Returned attack values are recomputed at the
    feasible ``worst_case`` table. On timeout, retain a nominal attained lower
    bound and an analytic interval upper bound unless a usable solver dual
    tightens it. HiGHS tolerances and numerical outward allowances do not
    provide an interval-arithmetic proof or a population guarantee.
    """
    X, y, features, weights = _inputs(X, y, features, weights)
    raw_target = np.asarray(target)
    if (raw_target.ndim != 0 or raw_target.dtype.kind not in "iu"
            or not 0 <= int(raw_target) < len(X)):
        raise ValueError("target must be a valid integer row index")
    target = int(raw_target)
    lower, upper = interval_bounds(X, eps, clip=clip)
    if not isinstance(normalize, (bool, np.bool_)):
        raise ValueError("normalize must be Boolean")
    if time_limit is not None and (not np.isscalar(time_limit)
                                  or not np.isfinite(time_limit) or time_limit < 0):
        raise ValueError("time_limit must be nonnegative and finite, or None")
    local_weights = np.zeros(X.shape[1])
    selected = weights[features]
    if normalize:
        if len(selected) == 0 or not np.any(selected > 0):
            raise ValueError("normalize=True requires positive selected weight")
        scaled = selected / selected.max()
        local_weights[features] = scaled / scaled.sum()
    else:
        local_weights[features] = selected
    same = np.flatnonzero((y == y[target]) & (np.arange(len(y)) != target))
    enemy = np.flatnonzero(y != y[target])
    common = dict(target=target, features=features.tolist(), normalize=bool(normalize),
                  effective_weights=local_weights.tolist(), same_reference_indices=same.tolist(),
                  enemy_reference_indices=enemy.tolist(), distance="uncapped_weighted_l1",
                  objective="max_same_min_distance_minus_enemy_min_distance")
    if len(same) == 0 or len(enemy) == 0:
        reason = ("Target class has no leave-one-out same-class reference"
                  if len(same) == 0 else "No opposite-class reference exists")
        return dict(common, applicable=False, status="not_applicable", message=reason,
                    lower_bound=None, upper_bound=None, gap=None, optimal=False,
                    worst_case=None, adversarial_X=None, variables=0, constraints=0)

    def evaluate(z):
        distances = np.abs(z - z[target]) @ local_weights
        nearest_same = int(same[np.argmin(distances[same])])
        nearest_enemy = int(enemy[np.argmin(distances[enemy])])
        same_distance, enemy_distance = float(distances[nearest_same]), float(distances[nearest_enemy])
        return same_distance - enemy_distance, nearest_same, nearest_enemy, same_distance, enemy_distance

    nominal, same_witness, enemy_witness, nominal_same, nominal_enemy = evaluate(X)
    if not np.isfinite(nominal):
        raise ValueError("weighted distances exceed the finite floating-point range")
    common.update(applicable=True, nominal_negative_margin=nominal,
                  nominal_same_distance=nominal_same, nominal_enemy_distance=nominal_enemy)
    if not np.any(local_weights) or np.all(lower == upper):
        return dict(common, lower_bound=nominal, upper_bound=nominal, gap=0.0,
                    optimal=True, status=0, message="Constant distances or singleton uncertainty set",
                    worst_case=X.copy(), adversarial_X=X.copy(),
                    nearest_same_at_witness=same_witness, nearest_enemy_at_witness=enemy_witness,
                    same_distance_at_witness=nominal_same, enemy_distance_at_witness=nominal_enemy,
                    analytic_lower_bound=nominal, analytic_upper_bound=nominal,
                    solver_upper_bound=None, solver_has_incumbent=False, solver_has_dual_bound=False,
                    used_solver_dual_bound=False, numerical_bound_conflict=False,
                    witness_projection_max=0.0, verification_source="constant_objective",
                    variables=0, constraints=0)

    qlo, qhi = lower[target] - upper, upper[target] - lower
    distance_lower = np.maximum(np.maximum(qlo, -qhi), 0.0) @ local_weights
    distance_upper = np.maximum(np.abs(qlo), np.abs(qhi)) @ local_weights
    if not np.isfinite(distance_lower).all() or not np.isfinite(distance_upper).all():
        raise ValueError("weighted box distances exceed the finite floating-point range")
    analytic_lower = float(distance_lower[same].min() - distance_upper[enemy].min())
    analytic_upper = float(distance_upper[same].min() - distance_lower[enemy].min())
    scale = max(1.0, float(distance_upper.max()))
    allowance = 1e-8 * scale
    analytic_lower = min(nominal, analytic_lower - 1e-12 * scale)
    analytic_upper = max(nominal, analytic_upper + 1e-12 * scale)

    model = _MILP()
    n, d = X.shape
    zvars = np.empty((n, d), dtype=int)
    for i in range(n):
        for k in range(d):
            zvars[i, k] = model.var(lower[i, k], upper[i, k])
    avars = {}
    active_attributes = np.flatnonzero(local_weights)
    for j in np.r_[same, enemy]:
        for k in active_attributes:
            a = model.var(0.0, max(abs(qlo[j, k]), abs(qhi[j, k])))
            avars[j, k] = a
            zi, zj = zvars[target, k], zvars[j, k]
            if qlo[j, k] >= 0:
                model.constraint({a: 1, zi: -1, zj: 1}, 0.0, 0.0)
            elif qhi[j, k] <= 0:
                model.constraint({a: 1, zi: 1, zj: -1}, 0.0, 0.0)
            else:
                sign = model.var(0.0, 1.0, integer=True)
                model.constraint({a: 1, zi: -1, zj: 1}, lower=0.0)
                model.constraint({a: 1, zi: 1, zj: -1}, lower=0.0)
                model.constraint({a: 1, zi: -1, zj: 1, sign: -2*qlo[j, k]}, upper=-2*qlo[j, k])
                model.constraint({a: 1, zi: 1, zj: -1, sign: -2*qhi[j, k]}, upper=0.0)
    # scipy minimizes: minus the same-class minimum, plus the enemy minimum.
    for references, coefficient in ((same, -1.0), (enemy, 1.0)):
        nearest = model.var(0.0, float(distance_upper[references].min()), objective=coefficient)
        selectors = [model.var(0.0, 1.0, integer=True) for _ in references]
        model.constraint({selector: 1.0 for selector in selectors}, 1.0, 1.0)
        for j, selector in zip(references, selectors):
            terms = {avars[j, k]: -local_weights[k] for k in active_attributes}
            terms[nearest] = 1.0
            model.constraint(terms, upper=0.0)
            chosen = dict(terms)
            chosen[selector] = -distance_upper[j]
            model.constraint(chosen, lower=-distance_upper[j])
    solved = model.solve(time_limit, 0.0)
    best_z, lower_bound = X.copy(), nominal
    incumbent = getattr(solved, "x", None)
    has_incumbent = (incumbent is not None and np.shape(incumbent) == (len(model.lower),)
                     and np.isfinite(np.asarray(incumbent)).all())
    projection_max = 0.0
    if has_incumbent:
        raw_z = np.asarray(incumbent)[zvars]
        candidate_z = np.clip(raw_z, lower, upper)
        projection_max = float(np.max(np.abs(candidate_z-raw_z)))
        attack, same_neighbor, enemy_neighbor, same_distance, enemy_distance = evaluate(candidate_z)
        if attack > lower_bound:
            best_z, lower_bound = candidate_z, attack
            same_witness, enemy_witness = same_neighbor, enemy_neighbor
    dual = getattr(solved, "mip_dual_bound", None)
    has_dual = bool(dual is not None and np.isfinite(dual))
    solver_upper = float(-dual) if has_dual else None
    conflict = bool(has_dual and solver_upper + allowance < lower_bound)
    use_dual = has_dual and not conflict
    upper_bound = analytic_upper
    if use_dual:
        upper_bound = min(upper_bound, solver_upper + allowance)
    upper_bound = float(max(upper_bound, lower_bound))
    _, same_witness, enemy_witness, same_distance, enemy_distance = evaluate(best_z)
    status, message = int(solved.status), str(solved.message)
    if not has_incumbent:
        message += "; no usable solver incumbent: returning nominal witness"
    if not has_dual:
        message += "; no finite solver dual bound: using analytic upper bound"
    if conflict:
        message += "; dual conflicts with feasible witness: dual discarded"
    return dict(common, lower_bound=float(lower_bound), upper_bound=upper_bound,
                gap=float(upper_bound-lower_bound), status=status, message=message,
                optimal=bool(status == 0 and has_incumbent and has_dual and not conflict
                             and projection_max <= 1e-7),
                worst_case=best_z, adversarial_X=best_z.copy(),
                nearest_same_at_witness=same_witness, nearest_enemy_at_witness=enemy_witness,
                same_distance_at_witness=same_distance, enemy_distance_at_witness=enemy_distance,
                analytic_lower_bound=analytic_lower, analytic_upper_bound=analytic_upper,
                solver_upper_bound=solver_upper, solver_has_incumbent=bool(has_incumbent),
                solver_has_dual_bound=has_dual, used_solver_dual_bound=bool(use_dual),
                numerical_bound_conflict=conflict, witness_projection_max=projection_max,
                numerical_outward_allowance=allowance, verification_source="milp",
                variables=len(model.lower), constraints=len(model.clower))
