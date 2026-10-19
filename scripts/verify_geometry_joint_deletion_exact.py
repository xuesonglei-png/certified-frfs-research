"""Exact Q(sqrt(5)) certificate counterexample, independent of NumPy bounds.

The table is rational; squared correlations, simplex projection and triplet
bounds are rational. Physical weights introduce only a common sqrt(5).
Comparisons use rational squares, never decimal approximations.
"""
import hashlib
import json
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import numpy as np
import sympy as sp
from run_nested_geometry_forest import ROOT
from certified_frfs.geometry_only_forest_stage import GeometryStageContext


def sign(a, b):
    """Exact sign of a+b*sqrt(5), a,b rational."""
    if not b: return (a > 0)-(a < 0)
    if not a: return (b > 0)-(b < 0)
    if a > 0 and b > 0: return 1
    if a < 0 and b < 0: return -1
    delta = a*a-5*b*b
    if not delta: return 0
    return ((delta > 0)-(delta < 0)) if a > 0 else ((delta < 0)-(delta > 0))


def less(x, y): return sign(x[0]-y[0], x[1]-y[1]) < 0
def emin(x, y): return x if less(x, y) else y
def emax(x, y): return y if less(x, y) else x
def symbolic(x): return sp.Rational(x[0].numerator, x[0].denominator)+sp.sqrt(5)*sp.Rational(x[1].numerator, x[1].denominator)


def squared_correlation(X, a, b):
    n = len(X); sa, sb = sum(r[a] for r in X), sum(r[b] for r in X)
    cross = sum(r[a]*r[b] for r in X)-sa*sb/n
    va = sum(r[a]**2 for r in X)-sa**2/n; vb = sum(r[b]**2 for r in X)-sb**2/n
    assert va > 0 and vb > 0
    return cross**2/(va*vb)


def weights(X, fs):
    if len(fs) == 5: return [F(1,5)]*5
    z = [F(1,5)]*len(fs)
    for omitted in set(range(5))-set(fs):
        corr = [squared_correlation(X, a, omitted) for a in fs]
        assert sum(corr) > F(1,10**12)
        z = [u+F(1,5)*v/sum(corr) for u,v in zip(z,corr)]
    assert sum(z) == 1
    cap = F(2,5)
    for state in product((0,1,2), repeat=len(fs)):
        free = [a for a,s in enumerate(state) if s == 1]
        if not free: continue
        t = (sum(z[a] for a in free)+cap*sum(s == 2 for s in state)-1)/len(free)
        if all((z[a]-t <= 0 if s == 0 else 0 <= z[a]-t <= cap if s == 1 else z[a]-t >= cap) for a,s in enumerate(state)):
            result = [F(0) if s == 0 else cap if s == 2 else z[a]-t for a,s in enumerate(state)]
            assert sum(result) == 1
            return result
    raise AssertionError('exact capped simplex active set missing')


def boxes(X):
    eps = F(1,100)
    return [[max(F(0),v-eps) for v in row] for row in X], [[min(F(1),v+eps) for v in row] for row in X]


def majority(X, y, fs, w):
    lo, hi = boxes(X); bounds = []
    for i in range(len(X)):
        def distance(j): return sum(w[k]*abs(X[i][a]-X[j][a]) for k,a in enumerate(fs))
        friends = sorted([j for j in range(len(X)) if y[j] == y[i] and j != i], key=lambda j:(distance(j),j))[:3]
        assert len(friends) == 3
        enemies = []
        for j in range(len(X)):
            if y[j] == y[i]: continue
            values = []
            for h in friends:
                def scalar(a, t):
                    near = max(F(0),lo[j][a]-t,t-hi[j][a])
                    far = max(abs(t-lo[h][a]),abs(t-hi[h][a]))
                    return near-far
                values.append(sum(w[k]*min(scalar(a,lo[i][a]),scalar(a,hi[i][a])) for k,a in enumerate(fs)))
            enemies.append(min(values))
        bounds.append(sorted(enemies)[2])
    return bounds


def dependency(X, y, fs, w):
    lo, hi = boxes(X); omitted = set(range(5))-set(fs); full = F(1,5); object_bounds = []
    for i in range(len(X)):
        enemies = [j for j in range(len(X)) if y[j] != y[i]]
        lower = {j:sum(w[k]*max(F(0),lo[i][a]-hi[j][a],lo[j][a]-hi[i][a]) for k,a in enumerate(fs)) for j in enemies}
        upper = {j:sum(w[k]*max(abs(lo[i][a]-hi[j][a]),abs(hi[i][a]-lo[j][a])) for k,a in enumerate(fs)) for j in enemies}
        possible = [j for j in enemies if lower[j] <= min(upper.values())]
        # All six enemy witnesses are used; the implementation's pool8 also
        # includes all six. No approximate candidate/witness selection.
        worst = (F(0),F(0))
        for j in possible:
            witness_bounds = []
            for h in enemies:
                retained = F(0)
                for k,a in enumerate(fs):
                    if j == h:
                        coef = full-w[k]
                        dist = max(abs(lo[i][a]-hi[j][a]),abs(hi[i][a]-lo[j][a])) if coef >= 0 else max(F(0),lo[i][a]-hi[j][a],lo[j][a]-hi[i][a])
                        retained += coef*dist
                    else:
                        points = {lo[i][a],hi[i][a]}
                        points |= {max(lo[i][a],min(hi[i][a],t)) for t in (lo[j][a],hi[j][a],lo[h][a],hi[h][a])}
                        retained += max(full*max(abs(t-lo[h][a]),abs(t-hi[h][a]))-w[k]*max(F(0),lo[j][a]-t,t-hi[j][a]) for t in points)
                witness_bounds.append(retained+sum(full*max(abs(lo[i][a]-hi[h][a]),abs(hi[i][a]-lo[h][a])) for a in omitted))
            signed = min(witness_bounds)
            bound = emax((F(0),F(0)),emin((F(1),-lower[j]),(F(0),signed)))
            worst = emax(worst,bound)
        object_bounds.append(worst)
    return (sum(v[0] for v in object_bounds)/len(X),sum(v[1] for v in object_bounds)/len(X))


def main():
    source_path = ROOT/'results/geometry-joint-deletion-example.json'
    source = json.loads(source_path.read_text()); assert source['found']
    e = source['example']; y = e['y']; pair = e['removed_pair']; checked = None
    for digits in (3,4,6):
        X = [[F(f'{v:.{digits}f}') for v in row] for row in e['X']]
        original = majority(X,y,list(range(5)),[F(1,5)]*5)
        protected = [v > 0 for v in original]
        # Sufficient separation from the floating safeguard is also exact.
        assert all(v == 0 or abs(v) > F(1,10**8) for v in original)
        required = [int(sp.ceiling(F(9,10)*sum(p and z == c for p,z in zip(protected,y)))) for c in (0,1)]
        supports = [list(range(5))]+[[a for a in range(5) if a != j] for j in range(5)]+[[a for a in range(5) if a not in pair]]
        records = []
        for fs in supports:
            w = weights(X,fs); bounds = majority(X,y,fs,w)
            counts = [sum(p and v > 0 and z == c for p,v,z in zip(protected,bounds,y)) for c in (0,1)]
            loss = (F(0),F(0)) if len(fs) == 5 else dependency(X,y,fs,w)
            feasible = all(a >= b for a,b in zip(counts,required)) and not less((F(1,10),F(0)),loss)
            records.append({'features':fs,'normalized_weights_rational':[str(v) for v in w], 'physical_weights_scale':'sqrt(5)',
                'required':required,'retained_counts':counts,'majority_bounds_rational_coefficient':[str(v) for v in bounds],
                'dependency_exact':str(symbolic(loss)),'dependency_decimal':str(symbolic(loss).evalf(18)), 'feasible_exact':feasible})
        if records[0]['feasible_exact'] and all(not r['feasible_exact'] for r in records[1:6]) and records[-1]['feasible_exact']:
            checked = {'digits':digits,'X_rational':[[str(v) for v in row] for row in X],'y':y,'protected':protected,'records':records}; break
    assert checked is not None, 'rounded exact counterexample did not survive'
    # Numerical implementation is only an additional cross-check, not the
    # source of any rational/algebraic comparison above.
    context = GeometryStageContext(np.array([[float(v) for v in row] for row in X]), np.array(y))
    for record in checked['records']:
        candidate = context.candidate(record['features'])
        assert candidate['certificate']['feasible'] == record['feasible_exact']
        np.testing.assert_allclose(np.array(candidate['weights'])[record['features']],np.sqrt(5)*np.array([float(F(v)) for v in record['normalized_weights_rational']]),atol=1e-12)
        assert candidate['certificate']['retained_counts'] == record['retained_counts']
        if len(record['features']) < 5:
            np.testing.assert_allclose(candidate['certificate']['dependency_certificate']['raw_redistributed_bound'],float(sp.sympify(record['dependency_exact'])),atol=1e-12)
    result = {'verified_exact':True,'example':checked,'removed_pair':pair,'sympy_version':sp.__version__,
        'scope':'exact_algebraic_counterexample_single_deletion_inaccessible_certified_support_under_same_uniform_reference',
        'limits':'Algorithmic_certificate_feasibility_counterexample_not_novelty_check_or_classification_advantage',
        'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),source_path)}}
    (ROOT/'results/geometry-joint-deletion-exact.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('exact counterexample verified, decimals',checked['digits'],'protected',sum(protected),'required',required,flush=True)
    print([(r['features'],r['retained_counts'],r['dependency_decimal'],r['feasible_exact']) for r in records],flush=True)


if __name__ == '__main__': main()
