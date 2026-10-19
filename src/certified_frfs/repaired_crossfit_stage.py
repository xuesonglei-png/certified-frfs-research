"""OOF deletion with original-certified flexible-weight repair at blocked steps."""
import time
import numpy as np
from .adaptive_multiknot_stage import fit_adaptive_certified_stage
from .geometry_only_forest_stage import GeometryStageContext
from .validated_pool import preprocess_fit
from .heldout_projected_logloss import HeldoutProjectedLogloss, CrossfitProjectedLogloss
from .flexible_majority_repair import fit_flexible_majority_repair


def fit_repaired_crossfit_stage(raw, y, *, fraction=.8):
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('invalid support fraction')
    raw, y = np.asarray(raw, float), np.asarray(y)
    teacher_stage = fit_adaptive_certified_stage(raw, y, fraction=1.)
    context = GeometryStageContext(raw, y)
    full = list(range(context.d))
    selection = teacher_stage['teacher_selection']
    chosen = selection['chosen_index']
    objectives, fold_centers = [], []
    for block in selection['folds']:
        a, b = np.array(block['reference_indices']), np.array(block['validation_indices'])
        A, B = preprocess_fit(raw[a], raw[b])
        alpha = np.array([1/(len(np.unique(y[a]))*sum(y[a] == c)) for c in y[a]])
        center = alpha@A
        objectives.append(HeldoutProjectedLogloss(B, y[b], block['models'][chosen], center))
        fold_centers.append(center.tolist())
    objective = CrossfitProjectedLogloss(objectives)
    target = max(1, int(np.ceil(fraction*context.d)))
    initial = context.candidate(full)
    trace = [{**initial, 'crossfit_balanced_logloss': objective(full)}]
    active, attempts, rankings, repairs = full, [], [], []
    while len(active) > target:
        ranking = sorted([(objective([a for a in active if a != r]), r) for r in active],
                         key=lambda item: (round(item[0], 12), item[1]))
        rankings.append({'features': active.copy(), 'choices': [{'removed': int(r), 'loss': loss} for loss, r in ranking]})
        accepted = None
        for loss, removed in ranking:
            fs = [a for a in active if a != removed]
            candidate = context.candidate(fs)
            attempts.append({**candidate, 'removed': int(removed), 'crossfit_balanced_logloss': loss,
                             'accepted': bool(candidate['certificate']['feasible'])})
            if candidate['certificate']['feasible']:
                accepted = {**candidate, 'crossfit_balanced_logloss': loss}
                break
        if accepted is None:
            repair = {'ranking_index': len(rankings)-1, 'attempts': []}
            repairs.append(repair)
            for loss, removed in ranking:
                fs = [a for a in active if a != removed]
                start = time.perf_counter()
                weights, info = fit_flexible_majority_repair(context.A, y, fs, time_limit=10.)
                repair['attempts'].append({'removed': int(removed), 'loss': loss, 'fit': info,
                                           'wall_seconds': time.perf_counter()-start})
                if weights is not None:
                    embedded = np.zeros(context.d)
                    embedded[fs] = weights
                    accepted = {'features': fs, 'weights': embedded.tolist(), 'certificate': info['certificate'],
                                'crossfit_balanced_logloss': loss}
                    break
        if accepted is None:
            break
        trace.append(accepted)
        active = accepted['features']
    return {**teacher_stage, 'candidate': trace[-1], 'target_count': target, 'fraction': float(fraction),
            'actual_count': len(active), 'path': {'trace': trace, 'rankings': rankings, 'attempts': attempts,
            'repair_blocks': repairs, 'fold_centers': fold_centers, 'tie_decimals': 12},
            'scope': 'original_FRS_reference_certificate_OOF_flexible_metric_repair_separate_classifier_not_test_risk'}
