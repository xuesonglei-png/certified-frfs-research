"""Numerical upper bounds for class FRS loss on one shared perturbed table.

The full min is a hypograph over a fixed witness subset. The reduced min is
a disjunction over every interval-possible enemy (and saturation if needed).
Each pair's absolute distance uses one shared convex-hull relaxation. Thus
every actual table embeds in the model, but a model incumbent need not be a
real attack. Only a finite solver DUAL bound is used as upper evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass
class Model:
    lower: list = field(default_factory=list)
    upper: list = field(default_factory=list)
    integer: list = field(default_factory=list)
    objective: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    cols: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rhs: list = field(default_factory=list)

    def var(self, lo, hi, cost=0., integer=False):
        i = len(self.lower)
        self.lower.append(float(lo)); self.upper.append(float(hi))
        self.integer.append(int(integer)); self.objective.append(float(cost))
        return i

    def le(self, terms, rhs):
        row = len(self.rhs)
        for col, value in terms.items():
            if value:
                self.rows.append(row); self.cols.append(col); self.values.append(float(value))
        self.rhs.append(float(rhs))

    def eq(self, terms, rhs):
        self.le(terms, rhs); self.le({k: -v for k, v in terms.items()}, -rhs)

    def arrays(self):
        matrix = coo_matrix((self.values, (self.rows, self.cols)),
                            shape=(len(self.rhs), len(self.lower))).tocsc()
        return matrix, np.asarray(self.rhs), np.asarray(self.objective)


def memberships(A, y, weights):
    """Direct distances, for attacks only, not a certificate primitive."""
    distance = np.zeros((len(A), len(A)))
    for a, w in enumerate(weights):
        distance += w * abs(A[:, a, None] - A[None, :, a])
    distance[np.asarray(y)[:, None] == np.asarray(y)[None, :]] = np.inf
    return np.minimum(1., distance.min(axis=1))


def build(A, y, full, reduced, target, *, eps=.01, witness_pool=8, exact_abs=False, object_upper=None):
    A, y = np.asarray(A, float), np.asarray(y)
    full, reduced = np.asarray(full, float), np.asarray(reduced, float)
    if (A.ndim != 2 or not len(A) or not A.shape[1] or y.shape != (len(A),)
            or full.shape != (A.shape[1],) or reduced.shape != full.shape
            or not np.isfinite(A).all() or A.min() < 0 or A.max() > 1
            or not np.isfinite(full).all() or not np.isfinite(reduced).all()
            or min(full.min(), reduced.min()) < 0 or not np.isfinite(eps) or eps < 0
            or not (y == target).any() or len(np.unique(y)) < 2
            or not isinstance(witness_pool, int) or witness_pool < 1):
        raise ValueError('invalid shared-table graph inputs')
    lo, hi = np.maximum(0., A-eps), np.minimum(1., A+eps)
    own = np.flatnonzero(y == target); n, d = A.shape
    if object_upper is not None:
        object_upper = np.asarray(object_upper, float)
        if object_upper.shape != (n,) or not np.isfinite(object_upper).all():
            raise ValueError('finite length-n certified object bounds required')
    m = Model(); z = np.empty((n, d), int)
    for i in range(n):
        for a in range(d):
            z[i, a] = m.var(lo[i, a], hi[i, a])
    pair_vars = {}; selections = []; absolutes_crossing = 0

    def distance(i, j, weights):
        nonlocal absolutes_crossing
        u, v = sorted((int(i), int(j))); terms = {}
        for a, weight in enumerate(weights):
            if not weight:
                continue
            key = u, v, a
            if key not in pair_vars:
                L, H = lo[u, a]-hi[v, a], hi[u, a]-lo[v, a]
                minimum = max(0., L, -H); maximum = max(abs(L), abs(H))
                q = m.var(minimum, maximum); pair_vars[key] = q
                zu, zv = int(z[u, a]), int(z[v, a])
                if L >= 0:
                    m.eq({q: 1., zu: -1., zv: 1.}, 0.)
                elif H <= 0:
                    m.eq({q: 1., zu: 1., zv: -1.}, 0.)
                else:
                    absolutes_crossing += 1
                    m.le({q: -1., zu: 1., zv: -1.}, 0.)
                    m.le({q: -1., zu: -1., zv: 1.}, 0.)
                    if exact_abs:
                        sign = m.var(0., 1., integer=True)
                        m.le({q: 1., zu: -1., zv: 1., sign: -2*L}, -2*L)
                        m.le({q: 1., zu: 1., zv: -1., sign: -2*H}, 0.)
                    else:
                        # |s| <= alpha*s+beta on s in [L,H].
                        alpha, beta = (H+L)/(H-L), -2*H*L/(H-L)
                        m.le({q: 1., zu: -alpha, zv: alpha}, beta)
            terms[pair_vars[key]] = float(weight)
        return terms

    for i in own:
        enemies = np.flatnonzero(y != y[i]); diffs = abs(A[i]-A[enemies])
        nominal = diffs @ full
        witnesses = enemies[np.lexsort((enemies, nominal))][:witness_pool]
        pair_lo = np.maximum(0., np.maximum(lo[i]-hi[enemies], lo[enemies]-hi[i])) @ reduced
        pair_hi = np.maximum(abs(lo[i]-hi[enemies]), abs(hi[i]-lo[enemies])) @ reduced
        cap = float(min(pair_hi)); possible_mask = pair_lo <= cap+1e-12
        possible = enemies[possible_mask]; possible_hi = pair_hi[possible_mask]
        red_lo, red_hi = min(1., float(min(pair_lo))), min(1., cap)
        t = m.var(0., 1., cost=-1./len(own))
        r = m.var(red_lo, red_hi, cost=1./len(own))
        if object_upper is not None:
            m.le({t: 1., r: -1.}, float(object_upper[i]))
        for k in witnesses:
            terms = {q: -v for q, v in distance(i, k, full).items()}; terms[t] = 1.
            m.le(terms, 0.)
        saturated = cap >= 1.-1e-12
        # Keeping an unnecessary saturation branch is conservative.
        count = len(possible)+int(saturated); selectors = []
        for j, upper in zip(possible, possible_hi, strict=True):
            terms = distance(i, j, reduced); terms[r] = -1.
            if count > 1:
                choice = m.var(0., 1., integer=True); selectors.append(choice)
                big_m = max(0., float(upper)-red_lo)
                terms[choice] = big_m; m.le(terms, big_m)
            else:
                m.le(terms, 0.)
        if saturated:
            choice = m.var(0., 1., integer=True); selectors.append(choice)
            m.le({r: -1., choice: 1.-red_lo}, -red_lo)
        if selectors:
            m.eq({q: 1. for q in selectors}, 1.)
        selections.append({'object': int(i), 'possible_enemies': possible.tolist(),
            'full_witnesses': witnesses.tolist(), 'saturation_branch': bool(saturated),
            'reduced_lower': red_lo, 'reduced_upper': red_hi})
    metadata = {'target_class': np.asarray(target).item(), 'class_size': len(own),
        'variables': len(m.lower), 'constraints': len(m.rhs),
        'integer_variables': sum(m.integer), 'shared_pair_coordinates': len(pair_vars),
        'crossing_pair_coordinates': absolutes_crossing, 'selections': selections,
        'eps': eps, 'witness_pool': witness_pool, 'exact_abs': bool(exact_abs),
        'object_cuts_used': object_upper is not None}
    return m, z, metadata


def class_upper(A, y, full, reduced, target, *, eps=.01, seconds=8., exact_abs=False, object_upper=None):
    start = time.perf_counter()
    A, y, full, reduced = np.asarray(A, float), np.asarray(y), np.asarray(full, float), np.asarray(reduced, float)
    # Validate before special cases too.
    model, z, meta = build(A, y, full, reduced, target, eps=eps, exact_abs=exact_abs, object_upper=object_upper)
    matrix, rhs, cost = model.arrays()
    nominal = float((memberships(A, y, full)-memberships(A, y, reduced))[y == target].mean())
    if np.array_equal(full, reduced):
        return {**meta, 'solver_status': 0, 'solver_upper_bound': 0., 'upper_bound': 0.,
                'nominal_signed_loss': nominal, 'observed_signed_loss_lower': nominal,
                'perturbed_table': A.tolist(), 'seconds': time.perf_counter()-start,
                'identity': True, 'solver_bound_available': True}
    fit = milp(cost, integrality=np.asarray(model.integer),
               bounds=Bounds(model.lower, model.upper),
               constraints=LinearConstraint(matrix, -np.inf, rhs),
               options={'time_limit': float(seconds), 'mip_rel_gap': 0., 'presolve': True})
    dual = getattr(fit, 'mip_dual_bound', None)
    # HiGHS may eliminate ALL integer variables and omit a MIP dual field.
    if dual is None and fit.status == 0 and fit.fun is not None:
        dual = fit.fun
    available = dual is not None and np.isfinite(dual)
    upper = float(-dual) if available else 1.
    allowance = 1e-7*(1.+abs(upper))
    Z = A.copy(); actual = nominal; violation = None; primal_objective = None
    if fit.x is not None and np.isfinite(fit.x).all():
        violation = max(0., float(max(matrix@fit.x-rhs)),
                        float(max(np.asarray(model.lower)-fit.x)),
                        float(max(fit.x-np.asarray(model.upper))))
        primal_objective = float(-cost@fit.x)
        candidate = np.clip(fit.x[z], np.maximum(0., A-eps), np.minimum(1., A+eps))
        candidate_loss = float((memberships(candidate, y, full)-memberships(candidate, y, reduced))[y == target].mean())
        if candidate_loss > actual:
            actual, Z = candidate_loss, candidate
    if available and actual > upper+allowance:
        raise ArithmeticError('solver upper contradicted by independently evaluated table')
    return {**meta, 'solver_status': int(fit.status), 'solver_message': str(fit.message),
            'solver_bound_available': bool(available), 'solver_upper_bound': upper,
            'upper_bound': float(np.clip(upper+allowance, 0., 1.)),
            'numerical_allowance': allowance, 'nominal_signed_loss': nominal,
            'observed_signed_loss_lower': actual, 'perturbed_table': Z.tolist(),
            'relaxation_primal_objective': primal_objective, 'primal_max_violation': violation,
            'mip_gap': float(fit.mip_gap) if getattr(fit, 'mip_gap', None) is not None else None,
            'mip_nodes': int(fit.mip_node_count) if getattr(fit, 'mip_node_count', None) is not None else None,
            'seconds': time.perf_counter()-start, 'identity': False,
            'scope': 'floating_solver_dual_shared_table_class_loss_upper_not_formal_proof_or_generalization'}
