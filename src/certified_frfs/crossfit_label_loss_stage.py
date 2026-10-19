"""OOF teacher loss ranking under the unchanged original FRS contract."""
import numpy as np
from .adaptive_multiknot_stage import fit_adaptive_certified_stage
from .geometry_only_forest_stage import GeometryStageContext
from .validated_pool import preprocess_fit
from .heldout_projected_logloss import HeldoutProjectedLogloss, CrossfitProjectedLogloss


def fit_crossfit_label_loss_stage(raw, y, *, fraction=.8):
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
    np.testing.assert_allclose(objective(full), selection['options'][chosen]['mean_balanced_logloss'], atol=1e-12)
    target = max(1, int(np.ceil(fraction*context.d)))
    initial = context.candidate(full)
    trace = [{**initial, 'crossfit_balanced_logloss': objective(full)}]
    active, attempts, rankings = full, [], []
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
            break
        trace.append(accepted)
        active = accepted['features']
    return {**teacher_stage, 'candidate': trace[-1], 'target_count': target, 'fraction': float(fraction),
            'actual_count': len(active), 'path': {'trace': trace, 'rankings': rankings, 'attempts': attempts,
            'fold_centers': fold_centers, 'tie_decimals': 12},
            'scope': 'original_FRS_reference_certificate_crossfit_teacher_ranking_separate_classifier_not_test_risk'}
