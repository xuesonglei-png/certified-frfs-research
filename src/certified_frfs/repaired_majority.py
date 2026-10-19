"""Training-only support restoration and refitting for joint majority metrics."""
import numpy as np
from .sparse_majority import MajorityMarginObjective
from .sparse_neighbor import optimize_cardinality_objective
from .feasible_transport import weight_certificate


def repair_majority_metric(X, y, candidate, dense_weights, *, ridge, anchor_reg=0.):
    if not np.isfinite(anchor_reg) or anchor_reg < 0: raise ValueError('nonnegative anchor regularization required')
    X = np.asarray(X, float); d = X.shape[1]; full = np.ones(d)/np.sqrt(d); mass = full.sum()
    fs = list(candidate['features']); w = np.asarray(candidate['weights'], float).copy()
    cert = weight_certificate(X, y, fs, full, w[fs]); trace = []; anchor = w/mass
    if cert['feasible']:
        return {**candidate, 'certificate': cert, 'repair': {'trace': [], 'restored_budget': 0, 'original_full_fallback': False, 'anchor_reg': anchor_reg}}
    objective = MajorityMarginObjective(X, y, ridge=ridge)
    omitted = sorted(set(range(d))-set(fs), key=lambda a: (-dense_weights[a], a))
    original_size = len(fs); support = fs.copy(); final_fit = candidate['fit']
    for attribute in omitted:
        support = sorted(support+[attribute]); ids = np.asarray(support)
        start = w[ids]/mass
        start[ids == attribute] = max(float(dense_weights[attribute]), 1/d)
        start /= start.sum()
        def reduced_objective(v):
            wide = np.zeros(d); wide[ids] = v
            value, grad = objective(wide)
            value += anchor_reg*np.sum((wide-anchor)**2)
            grad += 2*anchor_reg*(wide-anchor)
            return value, grad[ids]
        v, final_fit = optimize_cardinality_objective(reduced_objective, start, len(ids), ridge=ridge)
        w = np.zeros(d); w[ids] = v*mass; fs = np.flatnonzero(w > 0).tolist()
        cert = weight_certificate(X, y, fs, full, w[fs])
        trace.append({'added_attribute': attribute, 'allowed_support': support.copy(),
            'features': fs, 'weights': w.tolist(), 'fit': final_fit, 'certificate': cert})
        if cert['feasible']:
            return {**candidate, 'features': fs, 'weights': w.tolist(), 'fit': final_fit,
                'certificate': cert, 'repair': {'trace': trace, 'restored_budget': len(support)-original_size,
                    'net_added_features': len(fs)-original_size, 'original_full_fallback': False, 'anchor_reg': anchor_reg}}
    fs = list(range(d)); cert = weight_certificate(X, y, fs, full, full)
    assert cert['feasible']
    return {**candidate, 'features': fs, 'weights': full.tolist(), 'certificate': cert,
        'fit': {'scope': 'original_uniform_full_fallback'}, 'repair': {'trace': trace,
        'restored_budget': d-original_size, 'net_added_features': d-original_size, 'original_full_fallback': True, 'anchor_reg': anchor_reg}}
