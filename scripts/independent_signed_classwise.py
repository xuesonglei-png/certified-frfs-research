"""Scalar knot geometry; independent signed witness upper calculations."""
import numpy as np


class SignedWitnessValidator:
    def __init__(self, A, y, full):
        self.A, self.y, self.full = np.asarray(A), np.asarray(y), np.asarray(full)
        self.lo, self.hi = np.maximum(0., A-.01), np.minimum(1., A+.01)
        self.cache = {}

    def upper(self, i, j, k, fs, wr):
        key = (i, j, k, tuple(fs))
        if key not in self.cache:
            retained, omitted = [], 0.
            for a, wf in enumerate(self.full):
                il, ih = self.lo[i, a], self.hi[i, a]
                if a not in fs:
                    omitted += wf*max(abs(il-self.hi[k, a]), abs(ih-self.lo[k, a])); continue
                if j == k:
                    near = max(0., il-self.hi[j, a], self.lo[j, a]-ih)
                    far = max(abs(il-self.hi[j, a]), abs(ih-self.lo[j, a]))
                    retained.append(('same', a, near, far))
                else:
                    knots = [il, ih]+[float(np.clip(t, il, ih)) for t in
                        (self.lo[j, a], self.hi[j, a], self.lo[k, a], self.hi[k, a])]
                    far = [wf*max(abs(t-self.lo[k, a]), abs(t-self.hi[k, a])) for t in knots]
                    near = [max(0., self.lo[j, a]-t, t-self.hi[j, a]) for t in knots]
                    retained.append(('different', a, np.asarray(near), np.asarray(far)))
            self.cache[key] = (retained, omitted)
        retained, value = self.cache[key]; lookup = {a: w for a, w in zip(fs, wr)}
        for kind, a, near, far in retained:
            if kind == 'same':
                coefficient = self.full[a]-lookup[a]; value += coefficient*(far if coefficient >= 0 else near)
            else:
                value += float(np.max(far-lookup[a]*near))
        return float(value)

    def object(self, i, fs, wr):
        enemies = [j for j in range(len(self.y)) if self.y[j] != self.y[i]]
        lb, ub = {}, {}
        for j in enemies:
            lb[j] = sum(w*max(0., self.lo[i, a]-self.hi[j, a], self.lo[j, a]-self.hi[i, a]) for a, w in zip(fs, wr))
            ub[j] = sum(w*max(abs(self.lo[i, a]-self.hi[j, a]), abs(self.hi[i, a]-self.lo[j, a])) for a, w in zip(fs, wr))
        cap = min(ub.values()); possible = [j for j in enemies if lb[j] <= cap+1e-12]
        nominal = lambda k: sum(abs(self.A[i, a]*w-self.A[k, a]*w) for a, w in enumerate(self.full))
        witnesses = sorted(enemies, key=lambda k: (nominal(k), k))[:8]
        candidate = [min(self.upper(i, j, k, fs, wr) for k in witnesses) for j in possible]
        raw = max(min(1.-lb[j], v) for j, v in zip(possible, candidate))
        unsaturated = cap <= 1-1e-12; value = raw if unsaturated else max(0., raw)
        return {'object': i, 'possible_enemies': possible, 'full_witnesses': witnesses,
                'reduced_uniform_upper_cap': float(cap), 'uniform_unsaturation': bool(unsaturated),
                'candidate_signed_upper_bounds': candidate, 'raw_signed_upper_bound': float(raw), 'upper_bound': float(value)}
