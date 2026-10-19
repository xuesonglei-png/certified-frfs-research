"""Seeded prototype: choose an admitted low-loss seed before guarded descent.

Pairwise mass transfers preserve a fixed anchor witness polytope. Every
accepted update ALSO passes fresh global/classwise/majority checks. Controls
whose original-full anchor is infeasible use their own reduced-table anchor
as reference. The proposed arm must still pass original-full deployment guards.
This finite local search is neither an optimum nor a population risk bound.
"""
import numpy as np
from soft_topk_predictive_metric import anchor_witness_polytope
from soft_topk_metric_objective import soft_topk_objective
from certified_frfs.softvote_metric import softvote_terms
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate


def fit_seeded_guarded_soft5(A, y, features, anchor, candidate_targets, *, max_iter=8, max_guard_checks=64):
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

    original_cert = weight_certificate(A, y, fs, full, anchor)
    original_classes = classwise_dependency_certificate(y, original_cert)
    original_initial = {'certificate': original_cert, 'class_certificate': original_classes,
                        'accepted': bool(original_cert['feasible'] and original_classes['feasible_all_classes'])}
    mode = 'original_full' if original_initial['accepted'] else 'own_anchor'

    def guard(w):
        if mode == 'original_full':
            cert = weight_certificate(A, y, fs, full, w)
        else:
            # Reduced-table full reference is EXACTLY the control's anchor;
            # no uniform or original-full certificate is asserted for it.
            cert = weight_certificate(A[:, fs], y, list(range(p)), anchor, w)
        classes = classwise_dependency_certificate(y, cert)
        return {'certificate': cert, 'class_certificate': classes,
                'accepted': bool(cert['feasible'] and classes['feasible_all_classes'])}

    candidates = np.asarray(candidate_targets, float)
    if (candidates.ndim != 2 or candidates.shape[1] != p or not 1 <= len(candidates) <= 2
            or not np.isfinite(candidates).all() or candidates.min() < 0
            or not np.allclose(candidates.sum(axis=1), mass, atol=1e-8, rtol=0)):
        raise ValueError('one or two valid fixed-mass candidate targets required')
    initial_guard = original_initial if mode == 'original_full' else guard(anchor)
    if not initial_guard['accepted']: raise RuntimeError('adaptive anchor failed its own guard')
    terms = softvote_terms(A[:, fs], y, eps=.01)
    initial_value = soft_topk_objective(anchor/mass, terms, .05, .01)[0]
    selected_seed, seed, seed_value, seed_guard = None, anchor.copy(), initial_value, initial_guard
    seed_records = []; seed_grid = (.05, .1, .25, .5, .75, 1.)
    for source, candidate in enumerate(candidates):
        for t in seed_grid:
            proposed = (1-t)*anchor+t*candidate
            proposed_value = soft_topk_objective(proposed/mass, terms, .05, .01)[0]
            checked = guard(proposed)
            seed_records.append({'source_target_index': source, 'interpolation': t,
                                 'weights': proposed.tolist(), 'training_objective': proposed_value,
                                 **checked})
            if checked['accepted'] and proposed_value < seed_value-1e-8:
                selected_seed, seed, seed_value, seed_guard = len(seed_records)-1, proposed, proposed_value, checked
    # Rebuild witnesses at the ACTUAL admitted seed. The loss/retention
    # guard reference remains the original anchor reference, never reset.
    C, rhs, polytope = anchor_witness_polytope(A, y, fs, seed)
    w = seed/mass; value, gradient = soft_topk_objective(w, terms, .05, .01)
    J = C*mass; records, accepted_steps = [], []
    status = 'iteration_budget_exhausted'; endpoint_guard = seed_guard
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
                    'original_initial_guard': original_initial, 'guard_reference': mode,
                    'initial_guard': initial_guard, 'endpoint_guard': endpoint_guard,
                    'candidate_targets': candidates.tolist(), 'seed_interpolations': list(seed_grid),
                    'seed_checks': seed_records, 'selected_seed_id': selected_seed,
                    'seed_guard': seed_guard, 'search_initial_weights': seed.tolist(),
                    'search_initial_objective': seed_value,
                    'checks': records, 'accepted_step_ids': accepted_steps,
                    'weights': result.tolist(), 'training_objective_initial': initial_value,
                    'training_objective_final': value,
                    'scope': 'finite_local_certified_pairwise_search_not_optimum_or_test_risk'}
