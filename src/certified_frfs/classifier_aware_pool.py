"""Certified candidate-pool selection aligned with a downstream classifier."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from .feasible_query_descent import fit_feasible_query
from .trajectory_certificate import certify_capped_trajectory
from .capped_sparse import sparse_trajectory_extension
from .validated_pool import preprocess_fit
from .query_probability import QueryProbabilityObjective


def _score(A, y, B, truth, fs, C):
    if not fs:
        labels, counts = np.unique(y, return_counts=True)
        prediction = np.full(len(B), labels[counts.argmax()])
    else:
        prediction = LogisticRegression(C=C, max_iter=2000, random_state=0).fit(A[:, fs], y).predict(B[:, fs])
    return float(balanced_accuracy_score(truth, prediction))


def fit_classifier_aware_pool(X, y, *, C_grid=(.1, 1., 10.), folds=3):
    """Fit the fixed certified candidate pool and select by reference-only BA.

    The candidate generator is unchanged. Every candidate is certified on the
    complete reference table before selection; inner folds only choose the
    downstream logistic C and candidate. This is a training selector, not a
    test-risk certificate.
    """
    X, y = np.asarray(X, float), np.asarray(y)
    trajectory = fit_feasible_query(X, y, steps=10, certified=False, weight_cap_ratio=2)
    certified = certify_capped_trajectory(X, y, trajectory)
    pool = sparse_trajectory_extension(X, y, certified)
    candidates = [c for c in pool['candidates'] if c['certificate']['feasible']]
    if not candidates:
        raise ValueError('no feasible certified candidates')
    if folds != 3:
        raise ValueError('frozen recipe requires three folds, reduced for rare classes')
    splits = QueryProbabilityObjective(X,y,seed=1701).splits
    rows = []
    for index, C in enumerate(C_grid):
        for candidate_index, candidate in enumerate(candidates):
            scores = []
            for split in splits:
                ti,qi = np.array(split['reference']),np.array(split['query'])
                A, B = preprocess_fit(X[ti], X[qi]); fs = candidate['features']
                scores.append(_score(A, y[ti], B, y[qi], fs, C))
            rows.append({'candidate_index': candidate_index, 'C': float(C),
                         'mean_clean': float(np.mean(scores)), 'folds': scores,
                         'size': len(candidate['features'])})
    chosen = max(range(len(rows)), key=lambda i: (rows[i]['mean_clean'], -rows[i]['size'], -i))
    selected = candidates[rows[chosen]['candidate_index']]
    return {'selected': selected, 'candidates': candidates, 'trace': rows,
            'chosen_index': chosen, 'scope': 'classifier_aware_certified_pool_training_only'}
