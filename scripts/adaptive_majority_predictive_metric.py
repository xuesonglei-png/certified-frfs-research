"""Training-only local predictive optimization in an anchor witness polytope.

The polytope preserves fixed robust same-class witnesses. It neither controls
dependency loss nor guarantees the current checker's newly selected witnesses.
Every deployed candidate still needs the unchanged original/classwise guard.
"""
import hashlib

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

from capped_predictive_metric import fit_predictive_target
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.softvote_metric import softvote_terms
from certified_frfs.flexible_majority_repair import capped_simplex_minimum


def smooth_objective(w, terms, temperature=.05, reg=.01, delta=1e-12):
    C, same, mask, labels = terms
    classes = np.unique(labels)
    alpha = np.array([1/(len(classes)*sum(labels == c)) for c in labels])
    scores = -np.einsum('ijd,d->ij', C, w)/temperature
    all_scores, friend_scores = np.where(mask, scores, -np.inf), np.where(same, scores, -np.inf)
    normalizer, positive = logsumexp(all_scores, axis=1), logsumexp(friend_scores, axis=1)
    pa = np.exp(all_scores-normalizer[:, None]); ps = np.exp(friend_scores-positive[:, None])
    value = alpha@(normalizer-positive)
    grad = np.einsum('i,ij,ijd->d', alpha, ps-pa, C)/temperature
    log = np.log(len(w)*(w+delta))
    return float(value+reg*np.sum(w*log)), grad+reg*(log+w/(w+delta))


def anchor_witness_polytope(A, y, features, anchor, *, retention=.9, eps=.01):
    if not 0 < retention <= 1 or eps < 0:
        raise ValueError('invalid retention or radius')
    A, y = np.asarray(A, float), np.asarray(y)
    fs, anchor = list(features), np.asarray(anchor, float)
    d = A.shape[1]; full = np.ones(d)/np.sqrt(d)
    protected = MajorityTerms(A, y, weights=full, eps=eps).bounds(range(d), allowed_enemies=2) > 1e-10
    ws = MajorityTerms(A[:, fs], y, weights=anchor, eps=eps)
    bound = ws.bounds(range(len(fs)), allowed_enemies=2)
    original_ok = all(
        np.sum((y == c) & protected & (bound > 1e-10)) >= int(np.ceil(retention*np.sum((y == c) & protected)-1e-12))
        for c in np.unique(y))
    protection_reference = 'original_full' if original_ok else 'own_anchor'
    if not original_ok: protected = bound > 1e-10
    targets, witnesses, matrices = [], [], []
    for label in np.unique(y):
        ids = np.flatnonzero((y == label) & protected)
        required = int(np.ceil(retention*len(ids)-1e-12))
        eligible = ids[bound[ids] > 1e-10]
        if len(eligible) < required:
            raise ValueError('anchor does not retain the required original protected class')
        targets.extend(eligible[np.lexsort((eligible, -bound[eligible]))][:required].tolist())
    # ws.terms uses the three nearest same-class friends at the anchor metric.
    # They and the two enemy exceptions stay fixed throughout optimization.
    # Match the EXACT stored-coordinate cityblock convention used by ws;
    # multiplying a difference afterward can break near-equal friend ties.
    distance = cdist(A[:, fs]*anchor, A[:, fs]*anchor, 'cityblock')
    for i in targets:
        enemies = np.flatnonzero(y != y[i])
        C = ws.terms[:, i, enemies, :].transpose(1, 2, 0)
        enemy_bound = np.einsum('efp,p->ef', C, anchor).min(axis=1)
        order = np.lexsort((enemies, enemy_bound))
        kept = order[2:]
        matrices.append(C[kept].reshape(-1, len(fs)))
        ids = np.flatnonzero((y == y[i]) & (np.arange(len(y)) != i))
        friends = ids[np.lexsort((ids, distance[i, ids]))][:3]
        witnesses.append({'object': i, 'friends': friends.tolist(),
                          'excluded_enemies': enemies[order[:2]].tolist(),
                          'retained_enemies': enemies[kept].tolist()})
    C = np.concatenate(matrices) if matrices else np.empty((0, len(fs)))
    values = C@anchor
    if len(values) and np.min(values) <= 2e-10:
        raise ValueError('anchor fixed-witness margin failed numerical recheck')
    # Keep a positive witness margin while allowing slack from stronger rows.
    floors = np.minimum(1e-8, (values+2e-10)/2)
    info = {'targets': targets, 'witnesses': witnesses, 'constraint_rows': len(C),
            'matrix_sha256': hashlib.sha256(np.ascontiguousarray(C).tobytes()).hexdigest(),
            'floor_sha256': hashlib.sha256(floors.tobytes()).hexdigest(),
            'anchor_minimum_contrast': None if not len(values) else float(values.min()),
            'retention': retention, 'eps': eps, 'protection_reference': protection_reference,
            'scope': 'fixed_anchor_witness_sufficient_majority_only_not_dependency_or_query_risk'}
    return C, floors, info


def fit_majority_constrained_target(A, y, features, anchor, *, temperature=.05, reg=.01, max_iter=200):
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError('positive integer iteration limit required')
    # Share input validation and exact degenerate branches with the existing
    # capped learner. Do not use its unconstrained target as initialization.
    _, validation = fit_predictive_target(A, y, features, anchor, temperature=temperature, reg=reg, max_iter=1)
    validation = {**validation, 'max_iter': max_iter}
    A, y, anchor = np.asarray(A, float), np.asarray(y), np.asarray(anchor, float)
    fs = list(features); p, d = len(fs), A.shape[1]; mass, cap = np.sqrt(d), 2/d
    if validation['status'] in ('unique_capped_simplex', 'no_LOO_same_class_supervision'):
        return anchor.copy(), {**validation, 'status': 'constrained_'+validation['status'],
                               'polytope': None, 'linear_constraints_verified': True}
    C, floors, polytope = anchor_witness_polytope(A, y, fs, anchor)
    terms = softvote_terms(A[:, fs], y, eps=.01)
    # Smooth entropy is well defined even for a zero coordinate in the anchor.
    delta = 1e-12
    def objective(w):
        return smooth_objective(w, terms, temperature, reg, delta)
    initial = anchor/mass
    initial_value = objective(initial)[0]
    # Remove only rows true over the ENTIRE original capped simplex, using an
    # analytic linear minimum. Still recheck every original row after fitting.
    lower = capped_simplex_minimum(C, cap)*mass if len(C) else np.empty(0)
    active = lower-1e-10 < floors
    solve_C, solve_floor = C[active], floors[active]
    scale = np.maximum(np.max(abs(solve_C), axis=1)*mass, 1e-12) if len(solve_C) else np.empty(0)
    J = solve_C*mass/scale[:, None]; rhs = solve_floor/scale
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum()-1, 'jac': lambda w: np.ones(p)}]
    if len(solve_C):
        constraints.append({'type': 'ineq', 'fun': lambda w: J@w-rhs, 'jac': lambda w: J})
    fit = minimize(objective, initial, jac=True, method='SLSQP', bounds=[(0., cap)]*p,
                   constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-9})
    info = {'features': fs, 'temperature': temperature, 'reg': reg, 'entropy_delta': delta,
            'mass': float(mass), 'cap_normalized': cap, 'max_iter': max_iter,
            'optimizer_success': bool(fit.success), 'optimizer_status': int(fit.status),
            'optimizer_message': str(fit.message), 'optimizer_iterations': int(fit.nit),
            'training_objective_initial': initial_value, 'polytope': polytope,
            'solver_constraint_rows': len(solve_C), 'redundant_rows_removed': int((~active).sum()),
            'scope': 'local_fixed_witness_predictive_target_requires_separate_original_and_classwise_guard'}
    if not fit.success or not np.isfinite(fit.x).all() or not np.isfinite(fit.fun):
        return anchor.copy(), {**info, 'status': 'constrained_optimizer_failed_anchor_fallback',
                              'weights': anchor.tolist(), 'training_objective_final': None,
                              'linear_constraints_verified': False}
    v = np.maximum(fit.x, 0); v *= mass/v.sum()
    # Re-enter the polytope along the certified anchor when floating-point
    # solver output lies infinitesimally outside a face. Record that cleanup.
    candidate_slack = C@v-floors
    anchor_slack = C@anchor-floors
    blend = 0.
    if len(C) and np.any(candidate_slack < 0):
        bad = candidate_slack < 0
        blend = float(min(1., 1.01*np.max(-candidate_slack[bad]/(anchor_slack[bad]-candidate_slack[bad]))+1e-10))
        v = (1-blend)*v+blend*anchor
    contrast = C@v-floors
    # A small optimizer violation is not counted as a verified constraint.
    linear = not len(C) or bool(np.all(contrast >= 0))
    caps = v.max()/mass <= cap+1e-10
    final_value = objective(v/mass)[0]
    info.update({'training_objective_final': final_value, 'linear_constraints_verified': linear,
                 'numerical_anchor_blend': blend,
                 'minimum_constraint_slack': None if not len(C) else float(contrast.min())})
    if not linear or not caps or final_value > initial_value+1e-8:
        return anchor.copy(), {**info, 'status': 'constrained_recheck_failed_anchor_fallback', 'weights': anchor.tolist()}
    return v, {**info, 'status': 'local_majority_constrained_target', 'weights': v.tolist()}
