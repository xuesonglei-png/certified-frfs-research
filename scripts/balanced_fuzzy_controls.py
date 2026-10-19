"""Shared class-balanced fuzzy logistic controls, including empty supports."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from certified_frfs.linear_retention import linear_parameters
from certified_frfs.triangular_classifier import triangular_memberships
from run_common_fuzzy_classifier import score


def train(A, y, fs, C):
    if fs:
        expanded = triangular_memberships(A[:, fs]).reshape(len(A), -1)
        model = LogisticRegression(C=C, max_iter=2000, random_state=0,
                                   class_weight='balanced').fit(expanded, y)
        coef, b, classes = linear_parameters(model)
        W = np.zeros((len(classes), A.shape[1], 3))
        W[:, fs] = coef.reshape(len(classes), len(fs), 3)
    else:
        classes = np.unique(y)
        W = np.zeros((len(classes), A.shape[1], 3))
        b = np.zeros(len(classes))
    return {'coef': W.tolist(), 'intercept': b.tolist(), 'classes': classes.tolist(),
            'class_balanced': True}


def final_metrics(A, y, B, truth, fs, C, seed):
    fit = train(A, y, fs, C)
    rng = np.random.default_rng(seed)
    noise = [score(np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1), truth, fit)
             for _ in range(3)]
    return fit, {'clean': score(B, truth, fit), 'query_noise01': float(np.mean(noise))}
