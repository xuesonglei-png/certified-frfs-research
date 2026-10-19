"""Feasible exact-count controls; no changes to proposed certification.

Legacy cap 2/d is retained whenever feasible. Below half support, cap 1/k
is the smallest feasible extension and forces the unique uniform anchor.
"""
import numpy as np
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection

ORIGIN = 'uniform_minimal_feasible_exact_count_anchor_v2'


def exact_count_policies(A, block, count):
    d = A.shape[1]
    if isinstance(count, bool) or int(count) != count or not 1 <= count <= d:
        raise ValueError('count must be an integer in [1,d]')
    count = int(count)
    mass = np.sqrt(d)
    full = np.ones(d)/mass
    out = {}
    for prefix, key in (('fgs', 'fgs2023'), ('sfrip', 'sfrip2025')):
        weighted, uniform = [], []
        for path in block['paths'][key]:
            ranking = path['ranking']
            if sorted(ranking) != list(range(d)):
                raise ValueError('ranking must be a full feature permutation')
            fs = sorted(ranking[:count])
            if 2*count < d:
                w = np.full(count, mass/count)
                origin = ORIGIN
            else:
                w = full if count == d else capped_sparse_projection(
                    transfer_weights(A, fs, full)/mass, count, 2/d)*mass
                origin = 'ordinary_transferred_capped_metric'
            weighted.append({'features': fs, 'weights': w.tolist(), 'origin': origin})
            uniform.append({'features': fs, 'weights': np.ones(count).tolist(),
                            'origin': 'uniform_count_matched'})
        out[prefix+'_matched'] = weighted
        out[prefix+'_uniform_matched'] = uniform
    return out


def independent_exact_count_policies(A, block, count):
    """Legacy independent routine in old domain; direct formula in new domain."""
    d = A.shape[1]
    assert int(count) == count and 1 <= count <= d
    if 2*count >= d:
        from audit_multiscale_certified_metric_knn_pool import policies
        old = policies(A, block, count=count)
        return {k: old[k] for k in ('fgs_matched', 'sfrip_matched',
                                   'fgs_uniform_matched', 'sfrip_uniform_matched')}
    result = {}
    for prefix, method in [('fgs', 'fgs2023'), ('sfrip', 'sfrip2025')]:
        result[prefix+'_matched'] = []
        result[prefix+'_uniform_matched'] = []
        for path in block['paths'][method]:
            assert len(set(path['ranking'])) == d
            assert set(path['ranking']) == set(range(d))
            features = sorted(path['ranking'][0:count])
            result[prefix+'_matched'].append({'features': features,
                'weights': [float(d**.5/count) for _ in features], 'origin': ORIGIN})
            result[prefix+'_uniform_matched'].append({'features': features,
                'weights': [1. for _ in features], 'origin': 'uniform_count_matched'})
    return result
