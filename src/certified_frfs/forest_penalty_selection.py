"""Training-reference-only selection of centered fuzzy interactions.

Queries are deliberately absent from the selector API. All preprocessing,
teachers, interaction trees and margin constraints are refitted inside CV.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .nested_teacher_regularized import inner_splits
from .validated_pool import preprocess_fit
from .linear_retention import linear_parameters
from .triangular_classifier import triangular_memberships
from .constrained_fuzzy_logistic import fit_constrained_logistic
from .centered_forest_fuzzy_logistic_2d import fit_forest_logistic
from .fuzzy_interaction_tree import interaction_tree
from .forest_fuzzy_classifier import forest_scores

OPTIONS = ('additive', '10.0', '3.0', '1.0')
SEED = 20261012


def balanced_accuracy(scores, classes, truth):
    truth = np.asarray(truth); classes = np.asarray(classes)
    prediction = classes[np.argmax(scores, axis=1)]
    return float(np.mean([np.mean(prediction[truth == c] == c) for c in np.unique(truth)]))


def additive_scores(X, fit):
    return np.einsum('ndk,qdk->nq', triangular_memberships(X), np.asarray(fit['coef'])) + fit['intercept']


def choose_option(rows):
    if not rows:
        raise ValueError('selection needs nonempty internal folds')
    means = {key: float(np.mean([row['scores'][key] for row in rows])) for key in OPTIONS}
    if not all(np.isfinite(list(means.values()))):
        raise ValueError('selection scores must be finite')
    # Stable ties prefer no interactions, then stronger regularization.
    selected = max(OPTIONS, key=lambda key: (round(means[key], 12), -OPTIONS.index(key)))
    return selected, means


def select_pair_penalty(raw, y):
    raw, y = np.asarray(raw, float), np.asarray(y)
    records = []
    for a, b in inner_splits(y, seed=SEED):
        A, B = preprocess_fit(raw[a], raw[b]); yt, truth = y[a], y[b]
        teacher = LogisticRegression(C=1., max_iter=2000, random_state=0).fit(
            triangular_memberships(A).reshape(len(A), -1), yt)
        W, bias, classes = linear_parameters(teacher)
        W = W.reshape(len(classes), A.shape[1], 3)
        additive = fit_constrained_logistic(A, yt, W, bias, classes,
            list(range(A.shape[1])), C=1., class_balanced=True)
        additive_failure = None
        if not additive['feasible']:
            additive_failure = additive
            additive = {'feasible': True, 'coef': W.tolist(), 'intercept': bias.tolist(),
                        'scope': 'explicit_original_teacher_fallback'}
        tree = interaction_tree(A, yt)
        fits = {}; ordinary = additive_scores(B, additive)
        scores = {'additive': balanced_accuracy(ordinary, classes, truth)}
        for key in OPTIONS[1:]:
            fit = fit_forest_logistic(A, yt, W, bias, classes, list(range(A.shape[1])),
                tree['edges'], C=1., pair_penalty=float(key))
            fits[key] = fit
            query_scores = forest_scores(B, fit['coef'], fit['intercept'], fit['edges'], fit['pair_coef']) if fit['feasible'] else ordinary
            scores[key] = balanced_accuracy(query_scores, classes, truth)
        records.append({'reference_indices': a.tolist(), 'validation_indices': b.tolist(),
            'teacher_coef': W.tolist(), 'teacher_intercept': bias.tolist(),
            'teacher_classes': classes.tolist(), 'additive_fit': additive,
            'additive_failure': additive_failure, 'tree': tree, 'forest_fits': fits, 'scores': scores})
    selected, means = choose_option(records)
    return {'selected': selected, 'mean_scores': means, 'folds': records, 'seed': SEED}
