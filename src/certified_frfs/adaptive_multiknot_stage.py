"""Training-only adaptive fuzzy teacher and original-FRS-certified deletion.

Teacher selection is nested inside the input reference. The separate deployed
classifier receives no teacher-margin guarantee. No validation/query API.
"""
import numpy as np
from scipy.special import logsumexp
from .validated_pool import preprocess_fit
from .nested_teacher_regularized import inner_splits
from .fuzzy_interaction_tree import interaction_tree
from .multiknot_forest import fit_multiknot_forest, scores
from .multiknot_projection import MultiknotProjectionLoss
from .geometry_only_forest_stage import GeometryStageContext

MODEL_GRID = [{'knot_count': m, 'C': C, 'pair_penalty': 1.}
              for m in (3, 5) for C in (.1, 1., 10.)]


def balanced_logloss(value, classes, truth):
    own = np.searchsorted(classes, truth)
    loss = logsumexp(value, axis=1)-value[np.arange(len(truth)), own]
    return float(np.mean([np.mean(loss[truth == c]) for c in np.unique(truth)]))


def fit_adaptive_certified_stage(raw, y, *, fraction=.8):
    raw, y = np.asarray(raw, float), np.asarray(y)
    if raw.ndim != 2 or y.shape != (len(raw),) or not 0 < fraction <= 1:
        raise ValueError('invalid reference or support fraction')
    folds, options, caps = [], [], 0
    for ca, cb in inner_splits(y, seed=20261032):
        A, B = preprocess_fit(raw[ca], raw[cb])
        tree = interaction_tree(A, y[ca])
        models, losses = [], []
        for config in MODEL_GRID:
            fit = fit_multiknot_forest(A, y[ca], list(range(A.shape[1])), tree['edges'], **config)
            caps += not fit['optimizer_converged']
            models.append(fit)
            losses.append(balanced_logloss(scores(B, fit), np.array(fit['classes']), y[cb]))
        folds.append({'reference_indices': ca.tolist(), 'validation_indices': cb.tolist(),
                      'tree': tree, 'models': models, 'balanced_logloss': losses})
    for i, config in enumerate(MODEL_GRID):
        options.append({'config': config, 'mean_balanced_logloss': float(np.mean([f['balanced_logloss'][i] for f in folds]))})
    chosen = min(range(len(options)), key=lambda i: (round(options[i]['mean_balanced_logloss'], 12), i))
    context = GeometryStageContext(raw, y)
    full = list(range(context.d))
    tree = interaction_tree(context.A, y)
    teacher = fit_multiknot_forest(context.A, y, full, tree['edges'], **MODEL_GRID[chosen])
    caps += not teacher['optimizer_converged']
    projection = MultiknotProjectionLoss(context.A, y, teacher)
    initial = context.candidate(full)
    if not initial['certificate']['feasible']:
        raise RuntimeError('original full FRS certificate failed')
    target = max(1, int(np.ceil(fraction*context.d)))
    trace = [{**initial, 'projection_loss': projection(full)}]
    rankings, attempts, active = [], [], full
    while len(active) > target:
        ranking = sorted([(projection([a for a in active if a != removed]), removed) for removed in active],
                         key=lambda item: (round(item[0], 12), item[1]))
        rankings.append({'features': active.copy(), 'choices': [{'removed': int(j), 'loss': cost} for cost, j in ranking]})
        accepted = None
        for cost, removed in ranking:
            fs = [a for a in active if a != removed]
            candidate = context.candidate(fs)
            attempts.append({**candidate, 'removed': int(removed), 'projection_loss': cost,
                             'accepted': bool(candidate['certificate']['feasible'])})
            if candidate['certificate']['feasible']:
                accepted = {**candidate, 'projection_loss': cost}
                break
        if accepted is None:
            break
        trace.append(accepted)
        active = accepted['features']
    return {'candidate': trace[-1], 'target_count': target, 'fraction': float(fraction),
            'actual_count': len(active), 'teacher_selection': {'options': options, 'chosen_index': chosen,
            'folds': folds, 'optimizer_caps': int(caps)}, 'teacher': teacher, 'tree': tree,
            'path': {'trace': trace, 'rankings': rankings, 'attempts': attempts,
                     'center': projection.center.tolist(), 'tie_decimals': 12},
            'teacher_guards_enforced': False,
            'scope': 'original_FRS_reference_certificate_separate_downstream_classifier_not_test_risk'}
