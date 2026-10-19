"""Training-only supervised metric target; no admission or query-risk claim."""
import numpy as np
from scipy.optimize import minimize

from run_recent_pool_certificate_admission import ROOT
from certified_frfs.softvote_metric import softvote_terms
from soft_topk_metric_objective import soft_topk_objective
from certified_frfs.capped_sparse import capped_sparse_projection


def fit_predictive_target(A, y, features, anchor, *, temperature=.05, reg=.01, max_iter=200):
    """Learn a capped-simplex soft-neighbor target on one fixed support.

    Coordinates outside the support never enter the predictive objective.
    The anchor gives initialization only. Both target and anchor conserve
    original full metric masssqrt(d) and cap2/d. This is a local nonconvex
    optimizer, not a certified model; original certificates must separately
    check EVERY proposed deployed mixture. Singleton classes have no same-
    class LOO friend and do not contribute to this soft-neighbor objective.
    """
    A, y = np.asarray(A, float), np.asarray(y)
    fs = list(features); anchor = np.asarray(anchor, float)
    if (A.ndim != 2 or not np.isfinite(A).all() or np.min(A) < 0 or np.max(A) > 1
            or y.shape != (len(A),) or not fs or fs != sorted(set(fs))
            or min(fs) < 0 or max(fs) >= A.shape[1] or any(int(a) != a for a in fs)
            or anchor.shape != (len(fs),) or not np.isfinite(anchor).all() or np.any(anchor < 0)
            or temperature <= 0 or reg < 0 or int(max_iter) != max_iter or max_iter < 1):
        raise ValueError('invalid normalized training table, support, anchor or optimizer parameters')
    d, p = A.shape[1], len(fs); mass, cap = np.sqrt(d), 2/d
    if p*cap < 1-1e-12: raise ValueError('support cannot hold unit mass at original cap')
    if abs(anchor.sum()-mass) > 1e-8: raise ValueError('anchor must conserve the original metric mass')
    if anchor.max()/mass > cap+1e-10: raise ValueError('anchor violates original normalized cap')
    labels, counts = np.unique(y, return_counts=True)
    info = {'features': fs, 'temperature': float(temperature), 'reg': float(reg), 'max_iter': int(max_iter),
            'mass': float(mass), 'cap_normalized': cap, 'singleton_training_classes': labels[counts == 1].tolist(),
            'scope': 'local_training_soft_neighbor_target_not_FRS_admission_or_test_accuracy'}
    if abs(p*cap-1) <= 1e-12:
        target = np.full(p, mass/p)
        return target, {**info, 'status': 'unique_capped_simplex', 'optimizer_success': True, 'optimizer_iterations': 0,
                        'weights': target.tolist(), 'training_objective_initial': None, 'training_objective_final': None}
    C, same, mask, positive_labels = softvote_terms(A[:, fs], y, eps=.01)
    if not len(positive_labels):
        return anchor.copy(), {**info, 'status': 'no_LOO_same_class_supervision', 'optimizer_success': True,
                              'optimizer_iterations': 0, 'weights': anchor.tolist(),
                              'training_objective_initial': None, 'training_objective_final': None}
    lower = 1e-9
    initial = (1-1e-5)*(anchor/mass)+1e-5*np.ones(p)/p
    def objective(w): return soft_topk_objective(w, (C, same, mask, positive_labels), temperature, reg)
    initial_value = objective(initial)[0]
    fit = minimize(objective, initial, jac=True, method='SLSQP', bounds=[(lower, cap)]*p,
                   constraints={'type': 'eq', 'fun': lambda w: w.sum()-1, 'jac': lambda w: np.ones(p)},
                   options={'maxiter': int(max_iter), 'ftol': 1e-9})
    info.update({'optimizer_success': bool(fit.success), 'optimizer_status': int(fit.status),
                 'optimizer_message': str(fit.message), 'optimizer_iterations': int(fit.nit),
                 'training_objective_initial': float(initial_value)})
    if not fit.success or not np.isfinite(fit.x).all():
        return anchor.copy(), {**info, 'status': 'optimizer_failed_explicit_anchor_fallback',
                              'weights': anchor.tolist(), 'training_objective_final': None}
    normalized = capped_sparse_projection(np.maximum(fit.x, 0)/np.maximum(fit.x, 0).sum(), p, cap)
    if normalized.min() <= 0:
        return anchor.copy(), {**info, 'status': 'nonpositive_entropy_coordinate_anchor_fallback',
                              'weights': anchor.tolist(), 'training_objective_final': None}
    final_value = float(objective(normalized)[0])
    if final_value > initial_value+1e-8:
        return anchor.copy(), {**info, 'status': 'objective_recheck_failed_anchor_fallback',
                              'weights': anchor.tolist(), 'training_objective_final': final_value}
    target = normalized*mass
    return target, {**info, 'status': 'local_predictive_target', 'weights': target.tolist(), 'training_objective_final': final_value}
