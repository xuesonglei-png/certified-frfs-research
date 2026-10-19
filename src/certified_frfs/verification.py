"""Independent attacks and small-table mixed-integer verification.

The MILP models the interior of the perturbation box. Merely enumerating
box vertices is NOT an exact algorithm for the dependency-loss problem.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .core import _inputs, certificate, gamma, interval_bounds


def sampled_attack(X, y, features=None, weights=None, eps=0.0,
                   n_samples=100, seed=0, clip=True):
    """Return an empirical lower bound, never a robustness certificate."""
    X, y, features, weights = _inputs(X, y, features, weights)
    lower, upper = interval_bounds(X, eps, clip=clip)
    rng = np.random.default_rng(seed)
    best_z = X.copy()
    best = gamma(X, y, weights=weights) - gamma(X, y, features, weights)
    for sample in range(n_samples):
        if sample % 2:
            z = np.where(rng.integers(0, 2, size=X.shape), lower, upper)
        else:
            z = rng.uniform(lower, upper)
        value = gamma(z, y, weights=weights) - gamma(z, y, features, weights)
        if value > best:
            best, best_z = value, z
    return {"lower_bound": float(max(0.0, best)), "adversarial_X": best_z,
            "n_samples": int(n_samples), "seed": int(seed)}


class _MILP:
    def __init__(self):
        self.lower, self.upper, self.integer, self.objective = [], [], [], []
        self.rows, self.cols, self.values, self.clower, self.cupper = [], [], [], [], []

    def var(self, lower=0.0, upper=np.inf, integer=False, objective=0.0):
        index = len(self.lower)
        self.lower.append(lower)
        self.upper.append(upper)
        self.integer.append(int(integer))
        self.objective.append(objective)
        return index

    def constraint(self, coefficients, lower=-np.inf, upper=np.inf):
        row = len(self.clower)
        for col, value in coefficients.items():
            if value:
                self.rows.append(row)
                self.cols.append(col)
                self.values.append(value)
        self.clower.append(lower)
        self.cupper.append(upper)

    def solve(self, time_limit, mip_rel_gap):
        matrix = coo_matrix((self.values, (self.rows, self.cols)),
                            shape=(len(self.clower), len(self.lower))).tocsc()
        options = {"mip_rel_gap": mip_rel_gap, "presolve": True}
        if time_limit is not None:
            options["time_limit"] = float(time_limit)
        return milp(np.array(self.objective), integrality=np.array(self.integer),
                    bounds=Bounds(self.lower, self.upper),
                    constraints=LinearConstraint(matrix, self.clower, self.cupper),
                    options=options)


def exact_loss_milp(X, y, features=None, weights=None, eps=0.0,
                    clip=True, time_limit=60.0, mip_rel_gap=0.0):
    """Verify E(B) on small tables, returning attack and solver bounds.

    'optimal' means HiGHS reports optimality to its numerical tolerances;
    it does not mean a rational or interval-arithmetic proof. On timeout,
    lower_bound is an attained attack and upper_bound is the better of a
    valid analytic bound and the solver dual bound. Always inspect the gap.
    """
    X, y, features, weights = _inputs(X, y, features, weights)
    lower, upper = interval_bounds(X, eps, clip=clip)
    n, d = X.shape
    analytic = certificate(X, y, features, weights, eps, clip=clip)
    nominal = analytic["nominal_loss"]
    if analytic["upper_bound"] == 0.0 or np.all(lower == upper):
        return dict(lower_bound=nominal, upper_bound=nominal, gap=0.0,
                    optimal=True, status=0, message="Trivial or singleton uncertainty set",
                    adversarial_X=X.copy(), variables=0, constraints=0)
    model = _MILP()
    zvars = np.empty((n, d), dtype=int)
    for i in range(n):
        for k in range(d):
            zvars[i, k] = model.var(lower[i, k], upper[i, k])
    avars = {}
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] == y[j]:
                continue
            for k in range(d):
                if weights[k] == 0:
                    continue
                qlo, qhi = lower[i, k] - upper[j, k], upper[i, k] - lower[j, k]
                a = model.var(0.0, max(abs(qlo), abs(qhi)))
                avars[i, j, k] = a
                zi, zj = zvars[i, k], zvars[j, k]
                if qlo >= 0:
                    model.constraint({a: 1, zi: -1, zj: 1}, 0.0, 0.0)
                elif qhi <= 0:
                    model.constraint({a: 1, zi: 1, zj: -1}, 0.0, 0.0)
                else:
                    s = model.var(0.0, 1.0, integer=True)
                    model.constraint({a: 1, zi: -1, zj: 1}, lower=0.0)
                    model.constraint({a: 1, zi: 1, zj: -1}, lower=0.0)
                    model.constraint({a: 1, zi: -1, zj: 1, s: -2 * qlo}, upper=-2 * qlo)
                    model.constraint({a: 1, zi: 1, zj: -1, s: -2 * qhi}, upper=0.0)

    def distance_terms(i, j, attribute_set):
        left, right = sorted((i, j))
        return {avars[left, right, k]: weights[k] for k in attribute_set if weights[k] > 0}

    for attribute_set, sign in ((np.arange(d), -1.0), (features, 1.0)):
        for i in range(n):
            pos = model.var(0.0, 1.0, objective=sign / n)
            opposite = np.flatnonzero(y != y[i])
            selections = [model.var(0.0, 1.0, integer=True) for _ in range(len(opposite) + 1)]
            model.constraint({selector: 1.0 for selector in selections}, 1.0, 1.0)
            model.constraint({pos: 1.0, selections[0]: -1.0}, lower=0.0)
            for j, selector in zip(opposite, selections[1:]):
                terms = distance_terms(i, j, attribute_set)
                max_distance = sum(weights[k] * max(abs(lower[i, k] - upper[j, k]),
                                                    abs(upper[i, k] - lower[j, k]))
                                   for k in attribute_set)
                coefficients = {index: -value for index, value in terms.items()}
                coefficients[pos] = 1.0
                model.constraint(coefficients, upper=0.0)
                coefficients = dict(coefficients)
                coefficients[selector] = -max_distance
                model.constraint(coefficients, lower=-max_distance)
    solved = model.solve(time_limit, mip_rel_gap)
    best_z, lower_bound = X.copy(), nominal
    if solved.x is not None:
        candidate_z = np.clip(solved.x[zvars], lower, upper)
        attack_value = gamma(candidate_z, y, weights=weights) - gamma(candidate_z, y, features, weights)
        if attack_value > lower_bound:
            best_z, lower_bound = candidate_z, float(attack_value)
    dual = getattr(solved, "mip_dual_bound", None)
    solver_upper = float(-dual) if dual is not None and np.isfinite(dual) else 1.0
    upper_bound = min(analytic["upper_bound"], solver_upper + 1e-8)
    upper_bound = float(np.clip(max(upper_bound, lower_bound), 0.0, 1.0))
    return dict(lower_bound=float(lower_bound), upper_bound=upper_bound,
                gap=float(upper_bound - lower_bound), optimal=bool(solved.status == 0),
                status=int(solved.status), message=str(solved.message),
                adversarial_X=best_z, variables=len(model.lower),
                constraints=len(model.clower), solver_upper_bound=solver_upper,
                analytic_upper_bound=analytic["upper_bound"])
