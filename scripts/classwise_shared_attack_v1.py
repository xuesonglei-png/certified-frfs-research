"""Class-specific shared-table attack; a feasible witness is only lower evidence."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def dependency(Z, y, w, target):
    Z, y, w = np.asarray(Z, float), np.asarray(y), np.asarray(w, float)
    if Z.ndim != 2 or y.shape != (len(Z),) or w.shape != (Z.shape[1],):
        raise ValueError('incompatible table, labels or weights')
    if not np.isfinite(Z).all() or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError('finite nonnegative metric required')
    own = y == target
    if not own.any():
        raise ValueError('target class absent')
    if len(np.unique(y)) < 2:
        return 1., np.zeros_like(Z)
    dist = cdist(Z*w, Z*w, metric='cityblock')
    dist[y[:, None] == y[None, :]] = np.inf
    nearest = dist.argmin(axis=1)
    values = dist[np.arange(len(Z)), nearest]
    ids = np.flatnonzero(own & (values < 1.))
    contributions = np.sign(Z[ids] - Z[nearest[ids]]) * w / own.sum()
    grad = np.zeros_like(Z)
    np.add.at(grad, ids, contributions)
    # Enemy coordinates remain shared even when outside the target class.
    np.add.at(grad, nearest[ids], -contributions)
    return float(np.minimum(1., values[own]).mean()), grad


def loss_gradient(Z, y, full, reduced, target):
    a, ga = dependency(Z, y, full, target)
    b, gb = dependency(Z, y, reduced, target)
    return a-b, ga-gb


def attack(A, y, full, reduced, target, *, eps=.01, starts=4, steps=16, seed=20261012):
    A = np.asarray(A, float)
    if not np.isfinite(A).all() or A.min() < 0 or A.max() > 1:
        raise ValueError('nominal table must lie in [0,1]')
    if not np.isfinite(eps) or eps < 0 or starts < 1 or steps < 1:
        raise ValueError('invalid radius or budget')
    low, high = np.maximum(0., A-eps), np.minimum(1., A+eps)
    rng = np.random.default_rng(seed)
    nominal, _ = loss_gradient(A, y, full, reduced, target)
    best, value_best = A.copy(), nominal
    runs = []
    for index in range(starts):
        Z = A.copy() if index == 0 else rng.uniform(low, high)
        value, grad = loss_gradient(Z, y, full, reduced, target)
        trace = [value]; scales = []
        for _ in range(steps):
            update = None
            for scale in (1., .5, .25, .125, .0625):
                trial = np.clip(Z+eps*scale*np.sign(grad), low, high)
                new, ng = loss_gradient(trial, y, full, reduced, target)
                if new > value+1e-12:
                    update = trial, new, ng, scale
                    break
            if update is None:
                break
            Z, value, grad, scale = update
            trace.append(value); scales.append(scale)
        if value > value_best:
            best, value_best = Z.copy(), value
        runs.append({'start': index, 'loss_trace': trace, 'accepted_scales': scales})
    return {'target_class': np.asarray(target).item(), 'nominal_signed_class_loss': nominal,
            'observed_signed_class_loss_lower': value_best, 'perturbed_table': best.tolist(),
            'runs': runs, 'eps': eps, 'starts': starts, 'steps': steps, 'seed': seed,
            'class_violation_witness': bool(value_best > .1+1e-10),
            'scope': 'local_shared_table_class_loss_lower_witness_never_safety_certificate'}
