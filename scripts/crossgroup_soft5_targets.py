"""Anisotropic fixed-mass soft5 targets; original FRS guards remain separate.

The search simplex allows a coordinate to carry all EXISTING metric mass.
It does not change the certified loss/retention/radius thresholds. No target
is admissible until independently checked by those same original guards.
"""
import numpy as np
from scipy.optimize import minimize

from soft_topk_predictive_metric import anchor_witness_polytope
from soft_topk_metric_objective import soft_topk_objective
from crossgroup_soft5_terms import crossgroup_terms
from certified_frfs.flexible_majority_repair import capped_simplex_minimum


def fit_crossgroup_target(A, y, features, anchor, *, constrained=True, temperature=.05, reg=.01, max_iter=200):
    A, y, anchor = np.asarray(A, float), np.asarray(y), np.asarray(anchor, float)
    fs = list(features)
    if (A.ndim != 2 or len(A) < 7 or not np.isfinite(A).all() or A.min() < 0 or A.max() > 1
            or y.shape != (len(A),) or not fs or fs != sorted(set(fs))
            or any(int(a) != a or not 0 <= a < A.shape[1] for a in fs)
            or anchor.shape != (len(fs),) or not np.isfinite(anchor).all() or anchor.min() < 0
            or not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(reg) or reg < 0
            or int(max_iter) != max_iter or max_iter < 1):
        raise ValueError('invalid training table, support, anchor or optimizer settings')
    d, p = A.shape[1], len(fs); mass, cap = np.sqrt(d), 1.
    if abs(anchor.sum()-mass) > 1e-8: raise ValueError('anchor must preserve original metric mass')
    info = {'features': fs, 'mass': float(mass), 'cap_normalized': cap,
            'temperature': temperature, 'reg': reg, 'entropy_delta': 1e-12,
            'max_iter': max_iter, 'neighborhood_k': 5, 'constrained': bool(constrained),
            'scope': 'crossgroup_soft5_search_target_not_original_FRS_admission_or_population_risk'}
    C, rhs, polytope = (anchor_witness_polytope(A, y, fs, anchor) if constrained
                        else (np.empty((0, p)), np.empty(0), None))
    info['polytope'] = polytope
    terms, group_info = crossgroup_terms(A[:, fs], y, eps=.01)
    info['crossgroup'] = group_info
    labels, sizes = np.unique(y, return_counts=True)
    if p == 1 or not len(terms[3]):
        return anchor.copy(), {**info, 'status': 'unique_simplex' if p == 1 else 'no_crossgroup_supervision',
                               'weights': anchor.tolist(), 'optimizer_success': True, 'optimizer_iterations': 0,
                               'training_objective_initial': None, 'training_objective_final': None,
                               'linear_constraints_verified': True}
    def objective(w): return soft_topk_objective(w, terms, temperature, reg)
    initial = anchor/mass; initial_value = objective(initial)[0]
    lower = capped_simplex_minimum(C, cap)*mass if len(C) else np.empty(0)
    active = lower-1e-10 < rhs
    solve_C, solve_rhs = C[active], rhs[active]
    scale = np.maximum(np.max(abs(solve_C), axis=1)*mass, 1e-12) if len(solve_C) else np.empty(0)
    J, b = solve_C*mass/scale[:, None], solve_rhs/scale
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum()-1, 'jac': lambda w: np.ones(p)}]
    if len(J): constraints.append({'type': 'ineq', 'fun': lambda w: J@w-b, 'jac': lambda w: J})
    fit = minimize(objective, initial, jac=True, method='SLSQP', bounds=[(0., cap)]*p,
                   constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-9})
    info.update({'optimizer_success': bool(fit.success), 'optimizer_status': int(fit.status),
                 'optimizer_message': str(fit.message), 'optimizer_iterations': int(fit.nit),
                 'training_objective_initial': initial_value, 'solver_constraint_rows': len(J)})
    if not fit.success or not np.isfinite(fit.x).all() or not np.isfinite(fit.fun):
        return anchor.copy(), {**info, 'status': 'optimizer_failed_anchor_fallback', 'weights': anchor.tolist(),
                              'training_objective_final': None, 'linear_constraints_verified': False}
    w = np.maximum(fit.x, 0); w *= mass/w.sum()
    candidate_slack, anchor_slack = C@w-rhs, C@anchor-rhs
    blend = 0.
    if len(C) and np.any(candidate_slack < 0):
        bad = candidate_slack < 0
        blend = float(min(1., 1.01*np.max(-candidate_slack[bad]/(anchor_slack[bad]-candidate_slack[bad]))+1e-10))
        w = (1-blend)*w+blend*anchor
    linear = not len(C) or bool(np.all(C@w >= rhs)); final_value = objective(w/mass)[0]
    info.update({'linear_constraints_verified': linear, 'training_objective_final': final_value,
                 'numerical_anchor_blend': blend, 'strict_positive_coordinates': int(np.sum(w > 0))})
    if not linear or w.max()/mass > cap+1e-10 or final_value > initial_value+1e-8:
        return anchor.copy(), {**info, 'status': 'recheck_failed_anchor_fallback', 'weights': anchor.tolist()}
    return w, {**info, 'status': 'local_crossgroup_soft5_target', 'weights': w.tolist()}
