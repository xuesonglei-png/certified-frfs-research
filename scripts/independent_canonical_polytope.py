"""Independent witness reconstruction using canonical stored-coordinate ties."""
import hashlib
import numpy as np
from independent_canonical_majority import friend_enemy_bounds


def independent_polytope(A, y, anchor, protected, info):
    lo, hi = np.maximum(0, A-.01), np.minimum(1, A+.01)
    bounds = np.full(len(A), -np.inf)
    for i in np.flatnonzero(protected):
        enemy = friend_enemy_bounds(A, y, anchor, i)
        assert enemy is not None
        raw = np.partition(enemy, 2)[2]; bounds[i] = raw-1e-10*(1+abs(raw))
    targets = info['targets']; assert len(set(targets)) == len(targets)
    assert all(protected[i] and bounds[i] > 1e-10 for i in targets)
    for c in np.unique(y):
        eligible = np.flatnonzero((y == c) & protected & (bounds > 1e-10))
        required = int(np.ceil(.9*sum((y == c) & protected)-1e-12))
        chosen = [i for i in targets if y[i] == c]
        assert len(chosen) == required
        unchosen = list(set(eligible)-set(chosen))
        if chosen and unchosen: assert min(bounds[chosen]) >= max(bounds[unchosen])-1e-9
    matrices = []
    assert [r['object'] for r in info['witnesses']] == targets
    for r in info['witnesses']:
        i = r['object']; friends = np.array(r['friends']); enemies = np.array(r['retained_enemies'])
        candidates = np.flatnonzero((y == y[i]) & (np.arange(len(y)) != i))
        distances = np.zeros(len(candidates))
        for a, w in enumerate(anchor): distances += abs(A[candidates, a]*w-A[i, a]*w)
        expected = candidates[np.lexsort((candidates, distances))][:3]
        assert friends.tolist() == expected.tolist()
        all_enemies = np.flatnonzero(y != y[i]); excluded = r['excluded_enemies']
        assert len(excluded) == 2 and len(set(excluded)) == 2
        assert set(excluded)|set(enemies) == set(all_enemies) and not set(excluded)&set(enemies)
        values = []
        for t in (lo[i], hi[i]):
            near = np.maximum(0, np.maximum(lo[all_enemies]-t, t-hi[all_enemies]))
            far = np.maximum(abs(t-lo[friends]), abs(t-hi[friends]))
            values.append(near[:, None, :]-far[None, :, :])
        all_C = np.minimum(*values)
        contrast = (all_C@anchor).min(axis=1)
        lookup = {int(e): j for j, e in enumerate(all_enemies)}
        assert max(contrast[[lookup[e] for e in excluded]]) <= min(contrast[[lookup[int(e)] for e in enemies]])+1e-9
        matrices.append(all_C[[lookup[int(e)] for e in enemies]].reshape(-1, len(anchor)))
    C = np.concatenate(matrices) if matrices else np.empty((0, len(anchor)))
    rhs = np.minimum(1e-8, (C@anchor+2e-10)/2)
    assert len(C) == info['constraint_rows']
    assert hashlib.sha256(np.ascontiguousarray(C).tobytes()).hexdigest() == info['matrix_sha256']
    assert hashlib.sha256(rhs.tobytes()).hexdigest() == info['floor_sha256']
    assert not len(C) or np.all(C@anchor > rhs)
    return C, rhs

