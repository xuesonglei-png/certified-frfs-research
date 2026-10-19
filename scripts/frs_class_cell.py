"""Convex class-mean FRS constraints on a fixed stable-enemy cell.

Stable unsaturated objects use exact shared-star support functions. The
remaining objects use a signed anchor bound and a uniform metric Lipschitz
bound. Class bounds are valid on every shared table, not on a population.
"""
import numpy as np
from certified_frfs.core import _inputs, interval_bounds, _distance
from certified_frfs.margin import triplet_interval_min
from signed_classwise_dependency import signed_object_witness_bounds
from classwise_star_dependency import signed_star_maximum


def build_class_cell(A, y, features, anchor, full, *, eps=.01, eta=.1):
    A, y, fs, full = _inputs(A, y, features, full)
    anchor = np.asarray(anchor, float); fs = fs.tolist()
    if (not fs or anchor.shape != (len(fs),) or not np.isfinite(anchor).all()
        or anchor.min() < 0 or not np.isfinite(eta) or not 0 <= eta <= 1):
        raise ValueError('invalid class-cell support/anchor/tolerance')
    lo, hi = interval_bounds(A, eps); n, d = A.shape
    wide = np.zeros(d); wide[fs] = anchor
    identity = np.array_equal(wide, full)
    signed = np.zeros(n) if identity else np.asarray(signed_object_witness_bounds(A, y, fs, full, anchor, eps=eps)['signed_object_upper_bounds'])
    nominal = _distance(A, np.asarray(fs), wide); classes, sizes = np.unique(y, return_counts=True)
    object_data, selected, Q = {}, [], np.zeros((n, len(fs)))
    for i in range(n):
        enemies = np.flatnonzero(y != y[i])
        if not len(enemies):
            continue
        # Uniform Lipschitz coefficient for reduced min/saturated distance.
        # Taking max per coordinate is conservative and independent of w.
        far = np.maximum(abs(lo[i, fs]-hi[enemies][:, fs]), abs(hi[i, fs]-lo[enemies][:, fs]))
        Q[i] = far.max(axis=0)
        j = int(enemies[np.lexsort((enemies, nominal[i, enemies]))][0])
        other = enemies[enemies != j]
        contrast = triplet_interval_min(lo[i, fs][None, :], hi[i, fs][None, :],
                    lo[other][:, fs], hi[other][:, fs], lo[j, fs][None, :], hi[j, fs][None, :])
        values = contrast@anchor
        upper_coefficients = np.maximum(abs(lo[i, fs]-hi[j, fs]), abs(hi[i, fs]-lo[j, fs]))
        upper = float(upper_coefficients@anchor)
        stable = not len(values) or bool(np.min(values) > 4e-10)
        if stable and upper < 1-4e-10:
            floors = np.minimum(1e-8, (values+2e-10)/2)
            ceiling = min(1-2e-10, (1+upper)/2)
            object_data[i] = {'object': i, 'nearest_enemy': j, 'other_enemies': other.tolist(),
                'contrast': contrast.tolist(), 'floors': floors.tolist(),
                'unsaturation_coefficients': upper_coefficients.tolist(), 'unsaturation_ceiling': float(ceiling)}
            selected.append(i)
    reports, matrices, rhs, stability_rows = [], [], [], []
    for c, count in zip(classes, sizes):
        own = np.flatnonzero(y == c); eligible = [i for i in selected if y[i] == c]
        edges = [(i, object_data[i]['nearest_enemy']) for i in eligible]
        H, L = [], []
        for a in range(d):
            # Unit coefficients retrieve the exact positive/negative support
            # extrema from the already tested scalar shared-star primitive.
            H.append(signed_star_maximum(lo[:, a], hi[:, a], edges, 1.)['maximum'])
            L.append(-signed_star_maximum(lo[:, a], hi[:, a], edges, -1.)['maximum'])
        H, L = np.asarray(H), np.asarray(L); remainder = [i for i in own if i not in eligible]
        coefficient = full-wide
        star = float(np.where(coefficient >= 0, H, L)@coefficient+signed[remainder].sum())
        plain = float(signed[own].sum())
        use_star = bool(eligible and star <= plain+1e-12)
        chosen = eligible if use_star else []; remainder = remainder if use_star else own.tolist()
        raw = star if use_star else plain
        omitted = sorted(set(range(d))-set(fs))
        report = {'class': c.item(), 'count': int(count), 'mode': 'shared_star' if use_star else 'signed_Lipschitz',
            'selected_indices': chosen, 'edges': [[i, object_data[i]['nearest_enemy']] for i in chosen],
            'H': H.tolist() if use_star else [0.]*d, 'L': L.tolist() if use_star else [0.]*d,
            'remaining_indices': remainder, 'remaining_signed_anchor_sum': float(signed[remainder].sum()),
            'omitted_constant': float(full[omitted]@H[omitted]) if use_star else 0.,
            'remaining_lipschitz_coefficients': Q[remainder].sum(axis=0).tolist(),
            'anchor_raw_bound': raw/float(count), 'inflation': 1e-10*float(count)*(1+abs(raw/float(count)))}
        reports.append(report)
        for i in chosen:
            info = object_data[i]
            for row, floor in zip(info['contrast'], info['floors']):
                matrices.append(row); rhs.append(floor); stability_rows.append({'kind': 'stable_enemy', 'object': i})
            matrices.append((-np.asarray(info['unsaturation_coefficients'])).tolist())
            rhs.append(-info['unsaturation_ceiling']); stability_rows.append({'kind': 'unsaturation', 'object': i})
    C = np.asarray(matrices, float).reshape(-1, len(fs)); floors = np.asarray(rhs)
    info = {'features': fs, 'anchor': anchor.tolist(), 'full_weights': full.tolist(), 'eps': eps, 'eta': eta,
            'signed_anchor_object_bounds': signed.tolist(), 'anchor_identity': bool(identity),
            'classes': reports, 'selected_objects': [object_data[i] for r in reports for i in r['selected_indices']],
            'linear_cell_rows': stability_rows,
            'scope': 'convex_shared_star_plus_signed_Lipschitz_FRS_constraints_not_population_risk'}
    evaluation = evaluate_class_cell(info, anchor)
    info['anchor_evaluation'] = evaluation
    info['anchor_strictly_feasible'] = bool((not len(C) or np.all(C@anchor > floors))
                 and all(v < eta for v in evaluation['raw_upper_bounds_with_inflation']))
    return C, floors, info


def evaluate_class_cell(info, weights):
    fs = info['features']; weights = np.asarray(weights, float); anchor = np.asarray(info['anchor']); full = np.asarray(info['full_weights'])
    if weights.shape != anchor.shape or not np.isfinite(weights).all() or weights.min() < 0:
        raise ValueError('compatible finite nonnegative cell metric required')
    raw, stable = [], True
    for item in info['selected_objects']:
        contrast = np.asarray(item['contrast']).reshape(-1, len(fs))
        stable &= bool((not len(contrast) or np.all(contrast@weights >= np.asarray(item['floors'])))
                       and np.asarray(item['unsaturation_coefficients'])@weights <= item['unsaturation_ceiling'])
    for r in info['classes']:
        value = r['remaining_signed_anchor_sum']+np.asarray(r['remaining_lipschitz_coefficients'])@abs(weights-anchor)
        if r['mode'] == 'shared_star':
            coefficient = full[fs]-weights
            value += r['omitted_constant']+np.where(coefficient >= 0, np.asarray(r['H'])[fs], np.asarray(r['L'])[fs])@coefficient
        raw.append(float((value+r['inflation'])/r['count']))
    upper = np.clip(raw, 0, 1).tolist()
    return {'raw_upper_bounds_with_inflation': raw, 'class_upper_bounds': upper,
            'cell_constraints_verified': bool(stable), 'feasible': bool(stable and all(v <= info['eta'] for v in raw)),
            'scope': 'shared_box_class_mean_dependency_bound_on_fixed_stable_cell'}


def append_class_cell_lp(info, mass, p, n, append):
    """Append actual-weight cell constraints to normalized-weight hinge LP.

    Existing L1 variables start at p+n. Extra star epigraphs start at2p+n.
    Returns the total number of variables and free epigraph bounds.
    """
    offset = 2*p+n; extras = []
    full = np.asarray(info['full_weights']); fs = info['features']
    for r in info['classes']:
        cols, vals = list(range(p+n, 2*p+n)), (mass*np.asarray(r['remaining_lipschitz_coefficients'])).tolist()
        constant = r['remaining_signed_anchor_sum']+r['omitted_constant']+r['inflation']
        if r['mode'] == 'shared_star':
            for j, a in enumerate(fs):
                z = offset+len(extras); extras.append((None, None)); cols.append(z); vals.append(1.)
                for slope in (r['H'][a], r['L'][a]):
                    append([j, z], [-mass*slope, -1.], -full[a]*slope)
        append(cols, vals, info['eta']*r['count']-constant)
    return offset+len(extras), extras
