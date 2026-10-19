"""Original FRS deletion ranked by fixed-teacher true-label nominal loss."""
import numpy as np
from .adaptive_multiknot_stage import fit_adaptive_certified_stage
from .geometry_only_forest_stage import GeometryStageContext
from .projected_forest_logloss import MultiknotProjectedLogloss


def fit_label_loss_certified_stage(raw, y, *, fraction=.8):
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('invalid support fraction')
    # Fraction1 selects/refits exactly the same internal teacher without
    # running the old MSE deletion. Both methods keep their original teacher
    # search; only the deletion loss changes in the new method.
    teacher_stage = fit_adaptive_certified_stage(raw, y, fraction=1.)
    context = GeometryStageContext(raw, y)
    full = list(range(context.d))
    objective = MultiknotProjectedLogloss(context.A, y, teacher_stage['teacher'])
    target = max(1, int(np.ceil(fraction*context.d)))
    initial = context.candidate(full)
    trace = [{**initial, 'projected_balanced_logloss': objective(full)}]
    active, attempts, rankings = full, [], []
    while len(active) > target:
        ranking = sorted([(objective([a for a in active if a != r]), r) for r in active],
                         key=lambda item: (round(item[0], 12), item[1]))
        rankings.append({'features': active.copy(), 'choices': [{'removed': int(r), 'loss': loss} for loss, r in ranking]})
        accepted = None
        for loss, removed in ranking:
            fs = [a for a in active if a != removed]
            candidate = context.candidate(fs)
            attempts.append({**candidate, 'removed': int(removed), 'projected_balanced_logloss': loss,
                             'accepted': bool(candidate['certificate']['feasible'])})
            if candidate['certificate']['feasible']:
                accepted = {**candidate, 'projected_balanced_logloss': loss}
                break
        if accepted is None:
            break
        trace.append(accepted)
        active = accepted['features']
    return {**teacher_stage, 'candidate': trace[-1], 'target_count': target, 'fraction': float(fraction),
            'actual_count': len(active), 'path': {'trace': trace, 'rankings': rankings, 'attempts': attempts,
            'center': objective.center.tolist(), 'tie_decimals': 12},
            'scope': 'original_FRS_reference_certificate_true_label_ranking_separate_classifier_not_test_risk'}
