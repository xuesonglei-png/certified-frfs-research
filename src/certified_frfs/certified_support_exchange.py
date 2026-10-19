"""Reference-only support exchange at unchanged acquisition cardinality."""
import numpy as np
from .vote_aligned_refinement import VoteAlignedObjective
from .hard_vote_search import refine_hard_vote
from .validated_pool import preprocess_fit
from .feasible_transport import weight_certificate


def refine_support_exchange(X, y, initial, *, steps=3):
    # Reuse validated initialization; one-step direct refinement is part of
    # this frozen recipe, not selected by external evaluation.
    warm = refine_hard_vote(X, y, initial, steps=1)
    X, y = np.asarray(X, float), np.asarray(y); d = X.shape[1]; mass = np.sqrt(d)
    A, _ = preprocess_fit(X, X); obj = VoteAlignedObjective(X, y)
    w = np.array(warm['weights'])/mass
    value, gradient = obj(w); utility = obj.hard_utility(w)
    trace = [{'iteration': 0, 'weights': (w*mass).tolist(), 'objective': value,
              'training_utility': utility, 'certificate': warm['certificate']}]; attempts = []
    for iteration in range(1, steps+1):
        present = np.flatnonzero(w > 0); absent = np.flatnonzero(w == 0)
        if not len(absent):
            break
        # Gradient directions propose exchanges; hard votes rank candidates.
        incoming = absent[np.argsort(gradient[absent], kind='stable')[:3]]
        outgoing = present[np.argsort(-gradient[present], kind='stable')[:3]]
        proposals = []
        for a in outgoing:
            for b in incoming:
                v = w.copy(); v[b] = v[a]; v[a] = 0
                nu = obj.hard_utility(v)
                record = {'iteration': iteration, 'outgoing': int(a), 'incoming': int(b),
                          'training_utility': nu, 'accepted': False}
                attempts.append(record)
                if nu > utility+1e-12:
                    nv, ng = obj(v)
                    proposals.append((nu, nv, len(attempts)-1, v, ng))
        accepted = None
        for nu, nv, index, v, ng in sorted(proposals, key=lambda p: (-p[0], p[1], p[2])):
            fs = np.flatnonzero(v > 0).tolist()
            nc = weight_certificate(A, y, fs, np.ones(d)/mass, v[fs]*mass)
            attempts[index]['certificate'] = nc
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
            'trace': trace, 'attempts': attempts, 'certificate': trace[-1]['certificate'],
            'training_splits': obj.splits, 'warm_start': warm, 'steps': steps,
            'scope': 'certified_training_support_exchange_not_test_risk'}
