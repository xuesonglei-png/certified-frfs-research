"""Budgeted direct hard-vote training search with independent certificates."""
import numpy as np
from .vote_aligned_refinement import VoteAlignedObjective
from .bound_direct import _project
from .feasible_transport import weight_certificate
from .validated_pool import preprocess_fit


def refine_hard_vote(X, y, initial, *, steps=5):
    X, y = np.asarray(X, float), np.asarray(y)
    d = X.shape[1]; mass = np.sqrt(d); full = np.ones(d)/mass
    wide = np.asarray(initial['weights'], float); fs = np.asarray(initial['features'], int)
    if wide.shape != (d,) or not len(fs) or len(set(fs)) != len(fs) or np.any(fs < 0) or np.any(fs >= d):
        raise ValueError('invalid support')
    if steps < 1 or not np.isfinite(wide).all() or np.any(wide < 0) or abs(wide.sum()-mass) > 1e-8 or np.max(wide) > 2*mass/d+1e-8:
        raise ValueError('invalid metric or budget')
    if set(fs) != set(np.flatnonzero(wide > 0)):
        raise ValueError('support mismatch')
    A, _ = preprocess_fit(X, X); obj = VoteAlignedObjective(X, y); w = wide/mass
    def cert(v):
        ids = np.flatnonzero(v > 0).tolist()
        return weight_certificate(A, y, ids, full, v[ids]*mass)
    certificate = cert(w)
    if not certificate['feasible']:
        raise ValueError('initial metric not certified')
    value, gradient = obj(w); utility = obj.hard_utility(w)
    trace = [{'iteration': 0, 'weights': wide.tolist(), 'objective': value,
              'training_utility': utility, 'certificate': certificate}]; attempts = []
    for iteration in range(1, steps+1):
        directions = [gradient, -gradient]
        rank = fs[np.argsort(gradient[fs], kind='stable')]
        for offset in range(min(3, len(fs)//2)):
            direction = np.zeros(d)
            direction[rank[offset]] = -1
            direction[rank[-1-offset]] = 1
            directions.append(direction)
        proposals = []
        for j, direction in enumerate(directions):
            for step in (.05, .01, .002):
                v = np.zeros(d); v[fs] = _project(w[fs]-step*direction[fs], 2/d)
                nu = obj.hard_utility(v)
                record = {'iteration': iteration, 'direction': j, 'step': step,
                          'training_utility': nu, 'accepted': False}
                attempts.append(record)
                if nu > utility+1e-12:
                    nv, ng = obj(v)
                    proposals.append((nu, nv, len(attempts)-1, v, ng))
        accepted = None
        for nu, nv, index, v, ng in sorted(proposals, key=lambda p: (-p[0], p[1], p[2])):
            nc = cert(v); attempts[index]['certificate'] = nc
            if nc['feasible']:
                attempts[index]['accepted'] = True
                accepted = v, nv, ng, nu, nc
                break
        if accepted is None:
            break
        w, value, gradient, utility, certificate = accepted
        trace.append({'iteration': iteration, 'weights': (w*mass).tolist(), 'objective': value,
                      'training_utility': utility, 'certificate': certificate})
    return {'features': np.flatnonzero(w > 0).tolist(), 'weights': (w*mass).tolist(),
            'trace': trace, 'attempts': attempts, 'certificate': certificate,
            'training_splits': obj.splits, 'steps': steps,
            'scope': 'direct_adaptive_training_hard_vote_search_not_test_risk'}
