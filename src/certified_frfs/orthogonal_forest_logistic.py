"""Common downstream classifier; pure centered interactions, no teacher guards.

An orthonormal contrast basis removes additive subspaces from pair terms.
Pair design scaling realizes an independent L2 penalty. sklearn optimization
is numerical; no certified optimum or population-accuracy guarantee is claimed.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .triangular_classifier import triangular_memberships
from .forest_fuzzy_classifier import forest_scores, forest_box_margins
from .linear_retention import linear_parameters

CONTRAST = np.array([[1/np.sqrt(2), 1/np.sqrt(6)],
                     [-1/np.sqrt(2), 1/np.sqrt(6)], [0., -2/np.sqrt(6)]])


def fit_orthogonal_forest(X, y, features, edges, *, C=1., pair_penalty=1.):
    X, y = np.asarray(X, float), np.asarray(y); fs = list(features)
    if not np.isfinite(C) or C <= 0 or not np.isfinite(pair_penalty) or pair_penalty <= 0:
        raise ValueError('positive C and pair penalty required')
    if X.ndim != 2 or y.shape != (len(X),) or len(set(fs)) != len(fs) or any(int(a) != a or not 0 <= a < X.shape[1] for a in fs):
        raise ValueError('invalid reference or support')
    classes = np.unique(y); q, d = len(classes), X.shape[1]
    if q < 2: raise ValueError('at least two training classes required')
    raw_edges = [list(e) for e in edges]
    forest_box_margins(X[:1], y[:1], np.zeros((q, d, 3)), np.zeros(q), classes,
                      raw_edges, np.zeros((q, len(raw_edges), 3, 3)), eps=0.)
    retained = [e for e in raw_edges if all(a in fs for a in e)]
    W = np.zeros((q, d, 3)); V = np.zeros((q, len(retained), 3, 3)); bias = np.zeros(q)
    if fs:
        Z = triangular_memberships(X); contrasts = Z@CONTRAST
        # Partition of unity lets unpenalized intercepts absorb each unary
        # constant. Minimum-norm main effects therefore lie in this 2D basis.
        blocks = [contrasts[:, fs].reshape(len(X), -1)]
        for u, v in retained:
            blocks.append((contrasts[:, u, :, None]*contrasts[:, v, None, :]).reshape(len(X), 4)/np.sqrt(pair_penalty))
        design = np.column_stack(blocks)
        model = LogisticRegression(C=C, class_weight='balanced', max_iter=4000, tol=1e-8,
                                   random_state=0).fit(design, y)
        coef, bias, classes = linear_parameters(model)
        main = coef[:, :2*len(fs)].reshape(q, len(fs), 2)
        W[:, fs] = np.einsum('qdi,ki->qdk', main, CONTRAST)
        for e in range(len(retained)):
            tensor = coef[:, 2*len(fs)+4*e:2*len(fs)+4*(e+1)].reshape(q, 2, 2)/np.sqrt(pair_penalty)
            V[:, e] = np.einsum('ki,qij,lj->qkl', CONTRAST, tensor, CONTRAST)
        # A common class score shift permits identical softmax probabilities.
        W -= W[:1].copy(); bias -= bias[:1].copy(); V -= V[:1].copy()
        iterations = int(model.n_iter_.max()); converged = iterations < model.max_iter
    else:
        iterations = 0; converged = True
    return {'coef': W.tolist(), 'pair_coef': V.tolist(), 'edges': retained, 'features': fs,
            'intercept': bias.tolist(), 'classes': classes.tolist(), 'C': float(C),
            'pair_penalty': float(pair_penalty), 'class_balanced': True,
            'teacher_guards_enforced': False, 'optimizer_converged': converged,
            'optimizer_iterations': iterations,
            'scope': 'common_pure_interaction_classifier_no_teacher_margin_or_population_guarantee'}


def orthogonal_scores(X, fit):
    q = len(fit['classes']); e = len(fit['edges'])
    return forest_scores(X, fit['coef'], fit['intercept'], fit['edges'], np.array(fit['pair_coef']).reshape(q, e, 3, 3))
