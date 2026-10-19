"""Prototype: accept predictive descent steps only under original FRS guards.

Pairwise mass transfers preserve a fixed anchor witness polytope. Every
accepted update ALSO passes fresh original global/classwise/majority checks.
This finite local search is neither an optimum nor a population risk bound.
"""
import numpy as np
from soft_topk_predictive_metric import anchor_witness_polytope
from soft_topk_metric_objective import soft_topk_objective
from certified_frfs.softvote_metric import softvote_terms
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate


def fit_guarded_soft5(A, y, features, anchor, *, max_iter=8, max_guard_checks=64):
    A, y, anchor = np.asarray(A, float), np.asarray(y), np.asarray(anchor, float)
    fs = list(features)
    if (A.ndim != 2 or len(A) < 7 or not np.isfinite(A).all() or A.min() < 0 or A.max() > 1
            or y.shape != (len(A),) or not fs or fs != sorted(set(fs))
            or any(int(a) != a or not 0 <= a < A.shape[1] for a in fs)
            or anchor.shape != (len(fs),) or not np.isfinite(anchor).all() or anchor.min() < 0
            or int(max_iter) != max_iter or max_iter < 1
            or int(max_guard_checks) != max_guard_checks or max_guard_checks < 1):
        raise ValueError('invalid training table, support, anchor or search budget')
    d, p = A.shape[1], len(fs); mass = np.sqrt(d); full = np.ones(d)/mass
    if abs(anchor.sum()-mass) > 1e-8: raise ValueError('anchor must preserve original metric mass')

    def guard(w):
        cert = weight_certificate(A, y, fs, full, w)
        classes = classwise_dependency_certificate(y, cert)
        return {'certificate': cert, 'class_certificate': classes,
                'accepted': bool(cert['feasible'] and classes['feasible_all_classes'])}

    initial_guard = guard(anchor)
    if not initial_guard['accepted']: raise ValueError('original-certified anchor required')
    C, rhs, polytope = anchor_witness_polytope(A, y, fs, anchor)
    assert polytope['protection_reference'] == 'original_full'
    terms = softvote_terms(A[:, fs], y, eps=.01)
    w = anchor/mass; value, gradient = soft_topk_objective(w, terms, .05, .01)
    initial_value = value; J = C*mass; records, accepted_steps = [], []
    status = 'iteration_budget_exhausted'; endpoint_guard = initial_guard
    for iteration in range(max_iter):
        if len(records) >= max_guard_checks:
            status = 'guard_budget_exhausted'; break
        receivers = np.argsort(gradient, kind='stable')[:min(3, p)]
        donors = np.array([j for j in np.argsort(-gradient, kind='stable') if w[j] > 1e-12])[:min(3, p)]
        best = None; exhausted = False
        for receiver in receivers:
            for donor in donors:
                if receiver == donor or gradient[receiver] >= gradient[donor]-1e-10: continue
                direction = np.zeros(p); direction[receiver] = 1; direction[donor] = -1
                limit = min(w[donor], 1-w[receiver])
                if len(J):
                    slack, slope = J@w-rhs, J@direction
                    bad = slope < 0
                    if np.any(bad): limit = min(limit, float(np.min(slack[bad]/(-slope[bad]))))
                limit = max(0., limit)*.99
                if limit <= 1e-12: continue
                for fraction in (1., .5, .25, .125, .0625, .03125):
                    proposal = w+limit*fraction*direction; actual = proposal*mass
                    if proposal.min() < 0 or (len(C) and not np.all(C@actual >= rhs)): continue
                    candidate_value, candidate_gradient = soft_topk_objective(proposal, terms, .05, .01)
                    if candidate_value >= value-1e-8: continue
                    if len(records) >= max_guard_checks:
                        exhausted = True; break
                    checked = guard(actual)
                    records.append({'iteration': iteration, 'receiver': int(fs[receiver]),
                                    'donor': int(fs[donor]), 'normalized_step': float(limit*fraction),
                                    'weights': actual.tolist(), 'training_objective': candidate_value,
                                    **checked})
                    if checked['accepted']:
                        if best is None or candidate_value < best[0]:
                            best = (candidate_value, proposal, candidate_gradient, len(records)-1, checked)
                        break
                if exhausted: break
            if exhausted: break
        if best is None:
            status = 'guard_budget_exhausted' if exhausted else 'no_improving_checked_pair'; break
        value, w, gradient, rid, endpoint_guard = best
        accepted_steps.append(rid)
        if exhausted:
            status = 'guard_budget_exhausted'; break
    result = w*mass
    assert abs(result.sum()-mass) < 1e-8 and np.min(result) >= 0
    assert endpoint_guard['accepted'] and value <= initial_value+1e-8
    return result, {'status': status, 'features': fs, 'mass': float(mass), 'cap_normalized': 1.,
                    'temperature': .05, 'reg': .01, 'max_iter': max_iter,
                    'max_guard_checks': max_guard_checks, 'polytope': polytope,
                    'initial_guard': initial_guard, 'endpoint_guard': endpoint_guard,
                    'checks': records, 'accepted_step_ids': accepted_steps,
                    'weights': result.tolist(), 'training_objective_initial': initial_value,
                    'training_objective_final': value,
                    'scope': 'finite_local_certified_pairwise_search_not_optimum_or_test_risk'}
