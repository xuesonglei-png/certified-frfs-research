"""Exact real-arithmetic signed absolute-distance optimization on a forest.

Negative coefficients require interior states: binary endpoint DP is invalid.
All interval endpoints suffice as a common discrete alphabet. Prefix/suffix
maxima evaluate signed edge messages in linear alphabet time. Floating-point
implementation and reconstructed maximizer, not directed interval arithmetic.
"""
import numpy as np


def signed_forest_maximum(lower, upper, edges):
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.ndim != 1 or hi.shape != lo.shape or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
        raise ValueError('finite scalar intervals required')
    n = len(lo); aggregated = {}
    for i, j, c in edges:
        if int(i) != i or int(j) != j or not 0 <= i < n or not 0 <= j < n or not np.isfinite(c): raise ValueError('invalid signed edge')
        i, j = int(i), int(j)
        if i == j or c == 0: continue
        key = (min(i, j), max(i, j)); aggregated[key] = aggregated.get(key, 0.)+float(c)
    adjacency = [[] for _ in range(n)]
    for (i, j), c in aggregated.items():
        if c == 0: continue
        adjacency[i].append((j, c)); adjacency[j].append((i, c))
    parent = np.full(n, -2, int); incoming = np.zeros(n); order = []; roots = []
    for root in range(n):
        if parent[root] != -2: continue
        roots.append(root); parent[root] = -1; stack = [root]
        while stack:
            i = stack.pop(); order.append(i)
            for j, c in adjacency[i]:
                if j == parent[i]: continue
                if parent[j] != -2: raise ValueError('signed graph contains cycle')
                parent[j] = i; incoming[j] = c; stack.append(j)
    grid = np.unique(np.r_[lo, hi]); m = len(grid)
    dp = np.where((grid[None, :] >= lo[:, None]) & (grid[None, :] <= hi[:, None]), 0., -np.inf)
    choices = {}
    for i in reversed(order):
        p = parent[i]
        if p < 0: continue
        c = incoming[i]; left_values = dp[i]-c*grid; right_values = dp[i]+c*grid
        left = np.maximum.accumulate(left_values); right = np.maximum.accumulate(right_values[::-1])[::-1]
        # Recover deterministic maximizing state using running argmax indices.
        left_ids = np.empty(m, int); right_ids = np.empty(m, int); best = 0
        for t in range(m):
            if left_values[t] > left_values[best]: best = t
            left_ids[t] = best
        best = m-1
        for t in range(m-1, -1, -1):
            if right_values[t] >= right_values[best]: best = t
            right_ids[t] = best
        lhs = c*grid+left; rhs = -c*grid+right
        choose_left = lhs >= rhs
        dp[p] += np.maximum(lhs, rhs)
        choices[i] = np.where(choose_left, left_ids, right_ids)
    states = np.zeros(n, int)
    for root in roots: states[root] = int(np.argmax(dp[root]))
    for i in order:
        if parent[i] >= 0: states[i] = choices[i][states[parent[i]]]
    witness = grid[states] if n else np.array([])
    value = float(sum(dp[root].max() for root in roots))
    reconstructed = float(sum(c*abs(witness[i]-witness[j]) for (i, j), c in aggregated.items()))
    if abs(value-reconstructed) > 1e-8*(1+abs(value)): raise RuntimeError('forest witness/value mismatch')
    return {'maximum': value, 'witness': witness.tolist(), 'alphabet_size': m,
            'scope': 'signed_forest_piecewise_linear_maximum_float_not_rounding_proof'}
