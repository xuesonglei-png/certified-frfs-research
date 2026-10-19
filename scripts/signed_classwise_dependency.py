"""Class-mean FRS bound with unstable-neighbor signed witness losses.

Negative object bounds are valid whenever reduced membership is uniformly
unsaturated, without a stable nearest enemy. Shared stars remain optional.
"""
import numpy as np
from certified_frfs.core import _inputs, interval_bounds, _distance, _interval_distances
from certified_frfs.redistributed_loss import redistributed_loss_bound
from classwise_star_dependency import classwise_star_dependency


def signed_object_witness_bounds(A, y, features, wf, wr, *, eps=.01, witness_pool=8, base_dependency=None):
    A, y, fs, wf = _inputs(A, y, features, wf); wr = np.asarray(wr, float)
    if wr.shape != (len(fs),) or not np.isfinite(wr).all() or np.any(wr < 0):
        raise ValueError('invalid reduced weights')
    if witness_pool != 8:
        raise ValueError('the fixed numerical witness pool must remain8')
    base = redistributed_loss_bound(A, y, fs, wf, wr, eps=eps) if base_dependency is None else base_dependency
    if 'redistributed_object_bounds' not in base:
        assert len(np.unique(y)) < 2 or (len(fs) == A.shape[1] and np.array_equal(wr, wf))
        return {'signed_object_upper_bounds': [0.]*len(A), 'objects': [], 'base_dependency': base,
                'scope': 'identity_or_single_class_zero_signed_loss'}
    assert base['features'] == fs.tolist()
    np.testing.assert_array_equal(base['full_weights'], wf); np.testing.assert_array_equal(base['reduced_weights'], wr)
    n, d = A.shape; lo, hi = interval_bounds(A, eps); wide = np.zeros(d); wide[fs] = wr
    lb, ub = _interval_distances(lo, hi, fs, wide); nominal = _distance(A, np.arange(d), wf)
    omitted = np.setdiff1d(np.arange(d), fs); values, objects = [], []
    for i in range(n):
        enemies = np.flatnonzero(y != y[i]); cap = float(np.min(ub[i, enemies]))
        possible = enemies[lb[i, enemies] <= cap+1e-12]
        witnesses = enemies[np.lexsort((enemies, nominal[i, enemies]))][:8]
        il, ih = lo[i, fs], hi[i, fs]
        jl, jh = lo[possible][:, fs][:, None, :], hi[possible][:, fs][:, None, :]
        kl, kh = lo[witnesses][:, fs][None, :, :], hi[witnesses][:, fs][None, :, :]
        def at(t):
            far = np.maximum(abs(t-kl), abs(t-kh)); near = np.maximum(0., np.maximum(jl-t, t-jh))
            return wf[fs]*far-wr*near
        terms = np.maximum(at(il), at(ih))
        for point in (jl, jh, kl, kh):
            terms = np.maximum(terms, at(np.clip(point, il, ih)))
        coefficient = wf[fs]-wr
        near = np.maximum(0., np.maximum(il-jh, jl-ih)); far = np.maximum(abs(il-jh), abs(ih-jl))
        terms = np.where((possible[:, None] == witnesses[None, :])[:, :, None],
                         coefficient*np.where(coefficient >= 0, far, near), terms)
        omitted_max = np.maximum(abs(lo[i, omitted]-hi[witnesses][:, omitted]),
                                 abs(hi[i, omitted]-lo[witnesses][:, omitted]))@wf[omitted]
        signed = (terms.sum(axis=2)+omitted_max[None, :]).min(axis=1)
        raw = float(np.max(np.minimum(1.-lb[i, possible], signed)))
        np.testing.assert_allclose(max(0., raw), base['redistributed_object_bounds'][i], atol=1e-10, rtol=1e-9)
        unsaturated = bool(cap <= 1-1e-12)
        value = raw if unsaturated else max(0., raw)
        values.append(value)
        objects.append({'object': i, 'possible_enemies': possible.tolist(), 'full_witnesses': witnesses.tolist(),
            'reduced_uniform_upper_cap': cap, 'uniform_unsaturation': unsaturated,
            'candidate_signed_upper_bounds': signed.tolist(), 'raw_signed_upper_bound': raw, 'upper_bound': value})
    return {'signed_object_upper_bounds': values, 'objects': objects, 'base_dependency': base,
            'scope': 'finite_same_table_signed_dependency_loss_before_class_positive_part_not_object_positive_losses'}


def signed_classwise_dependency(A, y, features, wf, wr, *, eps=.01, eta=.1, base_dependency=None, star_report=None):
    signed = signed_object_witness_bounds(A, y, features, wf, wr, eps=eps, base_dependency=base_dependency)
    star = classwise_star_dependency(A, y, features, wf, wr, eps=eps, eta=eta, base_dependency=signed['base_dependency']) if star_report is None else star_report
    labels = np.asarray(y); classes, counts = np.unique(labels, return_counts=True)
    object_values = np.asarray(signed['signed_object_upper_bounds']); reports, bounds = [], []
    for ci, c in enumerate(classes):
        own = labels == c; raw_independent = float(object_values[own].mean())
        raw_star = raw_independent
        if star['class_reports']:
            previous = star['class_reports'][ci]; selected = np.asarray(previous['selected_indices'], int)
            remaining = own.copy(); remaining[selected] = False
            raw_star = float((sum(r['maximum'] for r in previous['coordinates'])+object_values[remaining].sum())/counts[ci])
        padded = lambda value: float(np.clip(value+1e-10*(1+abs(value)), 0, 1))
        bound = min(star['class_certificate']['upper_bounds'][ci], padded(raw_independent), padded(raw_star))
        reports.append({'class': c.item(), 'count': int(counts[ci]), 'signed_objectwise_raw_mean': raw_independent,
                        'shared_star_plus_signed_remainder_raw': raw_star, 'upper_bound': bound,
                        'negative_object_bounds': int(sum(object_values[own] < -1e-12))})
        bounds.append(bound)
    cls = {**star['class_certificate'], 'upper_bounds': bounds, 'max_upper_bound': max(bounds),
           'feasible_all_classes': all(b <= eta for b in bounds),
           'scope': 'signed_same_table_class_mean_dependency_decrease_all_finite_reference_prior_mixtures'}
    return {'class_certificate': cls, 'original_class_certificate': star['original_class_certificate'],
            'star_class_certificate': star['class_certificate'], 'global_upper_bound': min(star['global_upper_bound'], float(np.asarray(counts)@bounds/len(labels))),
            'signed_object_report': signed, 'class_reports': reports,
            'scope': 'certification_refinement_only_no_classifier_or_population_risk_guarantee'}
