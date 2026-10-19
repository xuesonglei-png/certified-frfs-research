"""Independent scalar reconstruction of FRS cell geometry and bounds."""
import numpy as np
from independent_signed_classwise import SignedWitnessValidator
from independent_majority_hinge import coefficient
from audit_forest_compact_transfer import equivalent


def scalar_star_extrema(lo, hi, edges, a):
    upper, lower = 0., 0.
    for center in sorted({j for i, j in edges}):
        leaves = [i for i, j in edges if j == center]
        knots = {lo[center, a], hi[center, a]}
        knots.update(float(np.clip(t, lo[center, a], hi[center, a]))
                     for i in leaves for t in (lo[i, a], hi[i, a]))
        upper += max(sum(max(abs(t-lo[i, a]), abs(t-hi[i, a])) for i in leaves) for t in knots)
        lower += min(sum(max(0., lo[i, a]-t, t-hi[i, a]) for i in leaves) for t in knots)
    return upper, lower


def reconstruct_cell(A, y, info, counts):
    A, y = np.asarray(A), np.asarray(y); n, d = A.shape
    fs, anchor, full = info['features'], np.asarray(info['anchor']), np.asarray(info['full_weights'])
    assert info['eps'] == .01 and info['eta'] == .1
    assert fs == sorted(set(fs)) and anchor.shape == (len(fs),) and full.shape == (d,)
    lo, hi = np.maximum(0., A-.01), np.minimum(1., A+.01)
    wide = np.zeros(d); wide[fs] = anchor; identity = np.array_equal(wide, full)
    assert info['anchor_identity'] == identity
    signed = np.zeros(n); Q = np.zeros((n, len(fs))); eligible = {}; matrices, rhs = [], []
    validator = SignedWitnessValidator(A, y, full)
    for i in range(n):
        enemies = [j for j in range(n) if y[j] != y[i]]
        if not enemies:
            continue
        if not identity:
            signed[i] = validator.object(i, fs, anchor)['upper_bound']
            counts['independent_cell_signed_object_bounds'] += 1
        for q, a in enumerate(fs):
            Q[i, q] = max(max(abs(lo[i, a]-hi[j, a]), abs(hi[i, a]-lo[j, a])) for j in enemies)
        distance = lambda j: sum(abs(A[i, a]*w-A[j, a]*w) for a, w in zip(fs, anchor))
        j = min(enemies, key=lambda e: (distance(e), e)); others = [e for e in enemies if e != j]
        contrast = np.asarray([coefficient(A[:, fs], i, e, j, .01) for e in others]).reshape(-1, len(fs))
        values = contrast@anchor
        upper_coeff = np.asarray([max(abs(lo[i, a]-hi[j, a]), abs(hi[i, a]-lo[j, a])) for a in fs])
        upper = float(upper_coeff@anchor)
        if (not len(values) or values.min() > 4e-10) and upper < 1-4e-10:
            eligible[i] = {'object': i, 'nearest_enemy': j, 'other_enemies': others,
                'contrast': contrast.tolist(), 'floors': np.minimum(1e-8, (values+2e-10)/2).tolist(),
                'unsaturation_coefficients': upper_coeff.tolist(), 'unsaturation_ceiling': float(min(1-2e-10, (1+upper)/2))}
    equivalent(signed.tolist(), info['signed_anchor_object_bounds'])
    assert len(info['classes']) == len(np.unique(y)); chosen_all, rows = [], []
    recorded = {r['object']: r for r in info['selected_objects']}
    for c, r in zip(np.unique(y), info['classes'], strict=True):
        own = np.flatnonzero(y == c).tolist(); chosen = [i for i in own if i in eligible]
        edges = [(i, eligible[i]['nearest_enemy']) for i in chosen]
        extrema = [scalar_star_extrema(lo, hi, edges, a) for a in range(d)]
        H, L = np.asarray(extrema).T; omitted = sorted(set(range(d))-set(fs))
        coefficient_vector = full-wide; remaining = [i for i in own if i not in chosen]
        star = float(np.where(coefficient_vector >= 0, H, L)@coefficient_vector+signed[remaining].sum())
        plain = float(signed[own].sum()); use_star = bool(chosen and star <= plain+1e-12)
        remaining = remaining if use_star else own; chosen = chosen if use_star else []
        raw = star if use_star else plain
        expected = {'class': c.item(), 'count': len(own), 'mode': 'shared_star' if use_star else 'signed_Lipschitz',
            'selected_indices': chosen, 'edges': [[i, eligible[i]['nearest_enemy']] for i in chosen],
            'H': H.tolist() if use_star else [0.]*d, 'L': L.tolist() if use_star else [0.]*d,
            'remaining_indices': remaining, 'remaining_signed_anchor_sum': float(signed[remaining].sum()),
            'omitted_constant': float(full[omitted]@H[omitted]) if use_star else 0.,
            'remaining_lipschitz_coefficients': Q[remaining].sum(axis=0).tolist(),
            'anchor_raw_bound': raw/len(own), 'inflation': 1e-10*len(own)*(1+abs(raw/len(own)))}
        equivalent(expected, r); chosen_all.extend(chosen)
        for i in chosen:
            equivalent(eligible[i], recorded[i]); item = recorded[i]
            for row, floor in zip(item['contrast'], item['floors'], strict=True):
                matrices.append(row); rhs.append(floor); rows.append({'kind': 'stable_enemy', 'object': i})
            matrices.append((-np.asarray(item['unsaturation_coefficients'])).tolist())
            rhs.append(-item['unsaturation_ceiling']); rows.append({'kind': 'unsaturation', 'object': i})
        counts['independent_cell_class_reconstructions'] += 1
    assert set(chosen_all) == set(recorded) and len(chosen_all) == len(recorded)
    equivalent(rows, info['linear_cell_rows'])
    C, rhs = np.asarray(matrices, float).reshape(-1, len(fs)), np.asarray(rhs)
    anchor_report = scalar_cell_evaluation(info, anchor)
    equivalent(anchor_report, info['anchor_evaluation'])
    assert info['anchor_strictly_feasible'] == bool((not len(C) or np.all(C@anchor > rhs))
            and all(v < .1 for v in anchor_report['raw_upper_bounds_with_inflation']))
    return C, rhs


def scalar_cell_evaluation(info, weights):
    fs, full = info['features'], info['full_weights']; raw, stable = [], True
    for item in info['selected_objects']:
        for row, floor in zip(item['contrast'], item['floors']):
            stable &= sum(a*w for a, w in zip(row, weights)) >= floor
        stable &= sum(a*w for a, w in zip(item['unsaturation_coefficients'], weights)) <= item['unsaturation_ceiling']
    for r in info['classes']:
        value = r['remaining_signed_anchor_sum']+r['inflation']+r['omitted_constant']
        value += sum(a*max(0., v-w) for a, w, v in zip(r['remaining_lipschitz_coefficients'], weights, info['anchor']))
        if r['mode'] == 'shared_star':
            value += sum((full[a]-w)*(r['H'][a] if full[a]-w >= 0 else r['L'][a]) for a, w in zip(fs, weights))
        raw.append(float(value/r['count']))
    return {'raw_upper_bounds_with_inflation': raw, 'class_upper_bounds': np.clip(raw, 0, 1).tolist(),
        'cell_constraints_verified': bool(stable), 'feasible': bool(stable and all(v <= info['eta'] for v in raw)),
        'scope': 'shared_box_class_mean_dependency_bound_on_fixed_stable_cell'}


def independent_append_cell_lp(info, mass, p, n, append):
    total = 2*p+n; extras = []
    for r in info['classes']:
        Q = r['remaining_lipschitz_coefficients']
        cols = list(range(p+n, 2*p+n))+list(range(p))
        vals = [.5*mass*q for q in Q]+[-.5*mass*q for q in Q]
        constant = r['remaining_signed_anchor_sum']+r['omitted_constant']+r['inflation']+.5*sum(q*v for q,v in zip(Q,info['anchor']))
        if r['mode'] == 'shared_star':
            for j, a in enumerate(info['features']):
                z = total; total += 1; extras.append((None, None)); cols.append(z); vals.append(1.)
                for extreme in (r['H'][a], r['L'][a]):
                    append([j, z], [-mass*extreme, -1.], -info['full_weights'][a]*extreme)
        append(cols, vals, info['eta']*r['count']-constant)
    return total, extras
