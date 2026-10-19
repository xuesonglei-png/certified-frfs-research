"""Crossfit probability descent through verified fixed-polytope directions."""
import numpy as np
from scipy.optimize import linprog
from .query_probability import QueryProbabilityObjective
from .joint_milp import joint_certified_metric, pairwise_loss_envelope
from .majority_metric import MajorityTerms
from .validated_pool import preprocess_fit


def unconstrained_oracle(full, costs, seconds, *, weight_cap_ratio=None):
    d = len(full); mass = float(full.sum())
    cap = mass if weight_cap_ratio is None else min(mass, weight_cap_ratio*mass/d)
    if not np.isfinite(cap) or cap < mass/d:
        raise ValueError('cap must permit the original uniform metric')
    upper = np.column_stack([-np.eye(d), -np.eye(d)])
    fit = linprog(np.r_[costs/mass, np.full(d, .001*costs.min()/mass)],
        A_ub=upper, b_ub=-full, A_eq=np.array([np.r_[np.ones(d), np.zeros(d)]]),
        b_eq=[mass], bounds=[(0., cap)]*d+[(0., mass)]*d, method='highs', options={'time_limit': seconds})
    v = full.copy()
    if fit.success and np.isfinite(fit.x).all():
        v = np.maximum(fit.x[:d], 0.)
        v *= mass/v.sum()
        if np.any(v > cap+1e-10): v = full.copy()
    return {'weights': v.tolist(), 'fit': {'solver_status': int(fit.status),
        'solver_success': bool(fit.success), 'exceptions': [], 'support_costs': costs.tolist(),
        'weight_cap_ratio': weight_cap_ratio, 'weight_cap': cap},
        'full_fallback': not fit.success, 'certificate': None}


def fit_feasible_query(X, y, *, steps=3, oracle_seconds=2., certified=True, objective=None, weight_cap_ratio=None, geometry='minmax'):
    if steps < 1 or oracle_seconds <= 0:
        raise ValueError('positive descent and oracle budgets required')
    if weight_cap_ratio is not None and (not np.isfinite(weight_cap_ratio) or weight_cap_ratio < 1):
        raise ValueError('finite weight-cap ratio at least one required')
    X, y = np.asarray(X, float), np.asarray(y)
    A, _ = preprocess_fit(X, X,geometry=geometry); d = A.shape[1]; full = np.ones(d)/np.sqrt(d); mass = full.sum()
    objective = objective if objective is not None else QueryProbabilityObjective(X, y, ridge=.01, temperature=.05, eps=.01,geometry=geometry)
    workspace = MajorityTerms(A, y, weights=full) if certified else None; current = full.copy()
    value, _ = objective(current/mass); trace = [{'iteration': 0, 'weights': current.tolist(), 'objective': value}]
    oracles = []; certificate = {'feasible': True, 'dependency_certificate': pairwise_loss_envelope(A, y, full, current)} if certified else None
    for iteration in range(steps):
        _, gradient = objective(current/mass)
        span = float(np.ptp(gradient))
        if span < 1e-12:
            break
        # Positive costs encode an affine rescaling of the linear loss gradient
        # at fixed mass. The oracle includes a small shift regularizer, so it is
        # approximate; only measured objective descent is accepted.
        costs = 1+(gradient-gradient.min())/span
        oracle = joint_certified_metric(A, y, continuous=True, support_costs=costs,
            time_limit=oracle_seconds, loss_slack=1e-8, support_cap_ratio=weight_cap_ratio) if certified else unconstrained_oracle(full, costs, oracle_seconds, weight_cap_ratio=weight_cap_ratio)
        oracles.append(oracle)
        direction = np.array(oracle['weights'])-current
        if np.max(np.abs(direction)) < 1e-12:
            break
        matrices = []
        for ex in oracle['fit']['exceptions']:
            i = ex['anchor']; enemies = np.setdiff1d(np.flatnonzero(y != y[i]), ex['excluded_enemies'])
            matrices.append(workspace.terms[:, i, enemies, :].transpose(1, 2, 0).reshape(-1, d))
        C = np.concatenate(matrices) if matrices else np.empty((0, d))
        accepted = None
        for exponent in range(12):
            alpha = 2.**(-exponent); candidate = current+alpha*direction
            if np.any(candidate < -1e-12):
                continue
            candidate = np.maximum(candidate, 0.)
            candidate *= mass/candidate.sum()
            if weight_cap_ratio is not None and np.any(candidate > weight_cap_ratio*mass/d+1e-10):
                continue
            if len(C) and np.any(C@candidate <= 1e-10):
                continue
            loss = pairwise_loss_envelope(A, y, full, candidate) if certified else None
            if certified and loss['upper_bound'] > .1:
                continue
            score, _ = objective(candidate/mass)
            if score < value-1e-9:
                accepted = (candidate, score, alpha, loss)
                break
        if accepted is None:
            break
        current, value, alpha, loss = accepted
        certificate = {'feasible': True, 'dependency_certificate': loss,
            'majority_rule': 'verified_fixed_original_witness_polytope',
            'targets': oracle['fit']['targets'], 'required': oracle['fit']['required'],
            'exceptions': oracle['fit']['exceptions']} if certified else None
        trace.append({'iteration': iteration+1, 'weights': current.tolist(), 'objective': value, 'step': alpha})
    return {'features': np.flatnonzero(current > 0).tolist(), 'weights': current.tolist(),
        'trace': trace, 'oracles': oracles, 'training_splits': objective.splits,
        'certificate': certificate, 'steps': steps, 'oracle_seconds': oracle_seconds,
        'certified': certified, 'weight_cap_ratio': weight_cap_ratio,
        'scope': 'training_crossfit_probability_local_descent_optional_empirical_certificate_not_test_risk'}
