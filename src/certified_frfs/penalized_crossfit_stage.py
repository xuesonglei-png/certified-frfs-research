"""Task-aligned OOF deletion with a complexity-adjusted teacher forest."""
import numpy as np
from .adaptive_multiknot_stage import MODEL_GRID, balanced_logloss
from .geometry_only_forest_stage import GeometryStageContext
from .validated_pool import preprocess_fit
from .nested_teacher_regularized import inner_splits
from .multiknot_forest import fit_multiknot_forest, scores
from .penalized_fuzzy_forest import penalized_forest
from .heldout_projected_logloss import HeldoutProjectedLogloss, CrossfitProjectedLogloss


def fit_penalized_crossfit_stage(raw, y, *, fraction=.8):
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('invalid support fraction')
    raw, y = np.asarray(raw, float), np.asarray(y)
    context = GeometryStageContext(raw, y)
    full = list(range(context.d))
    folds, caps = [], 0
    for a, b in inner_splits(y, seed=20261032):
        A, B = preprocess_fit(raw[a], raw[b])
        tree = penalized_forest(A, y[a])
        models, losses = [], []
        for config in MODEL_GRID:
            fit = fit_multiknot_forest(A, y[a], full, tree['edges'], **config)
            caps += not fit['optimizer_converged']
            models.append(fit)
            losses.append(balanced_logloss(scores(B, fit), np.array(fit['classes']), y[b]))
        folds.append({'reference_indices': a.tolist(), 'validation_indices': b.tolist(),
                      'tree': tree, 'models': models, 'balanced_logloss': losses})
    options = [{'config': config, 'mean_balanced_logloss': float(np.mean([f['balanced_logloss'][i] for f in folds]))}
               for i, config in enumerate(MODEL_GRID)]
    chosen = min(range(len(options)), key=lambda i: (round(options[i]['mean_balanced_logloss'], 12), i))
    tree = penalized_forest(context.A, y)
    teacher = fit_multiknot_forest(context.A, y, full, tree['edges'], **MODEL_GRID[chosen])
    caps += not teacher['optimizer_converged']
    objectives, fold_centers = [], []
    for block in folds:
        a, b = np.array(block['reference_indices']), np.array(block['validation_indices'])
        A, B = preprocess_fit(raw[a], raw[b])
        alpha = np.array([1/(len(np.unique(y[a]))*sum(y[a] == c)) for c in y[a]])
        center = alpha@A
        objectives.append(HeldoutProjectedLogloss(B, y[b], block['models'][chosen], center))
        fold_centers.append(center.tolist())
    objective = CrossfitProjectedLogloss(objectives)
    np.testing.assert_allclose(objective(full), options[chosen]['mean_balanced_logloss'], atol=1e-12)
    target = max(1, int(np.ceil(fraction*context.d)))
    initial = context.candidate(full)
    if not initial['certificate']['feasible']:
        raise RuntimeError('original full FRS certificate failed')
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
    return {'candidate': trace[-1], 'target_count': target, 'fraction': float(fraction),
            'actual_count': len(active), 'teacher': teacher, 'tree': tree,
            'teacher_selection': {'options': options, 'chosen_index': chosen, 'folds': folds, 'optimizer_caps': int(caps)},
            'path': {'trace': trace, 'rankings': rankings, 'attempts': attempts,
                     'fold_centers': fold_centers, 'tie_decimals': 12},
            'teacher_guards_enforced': False,
            'scope': 'original_FRS_reference_certificate_BIC_forest_crossfit_ranking_separate_classifier_not_test_risk'}
