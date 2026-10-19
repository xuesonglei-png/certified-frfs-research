import sys
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from certified_frfs.query_certificate import certify_queries

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_decision_iteration import query_aware_score, configs


def test_balanced_certified_correct_objective_handles_class_imbalance():
    A = np.array([[.1], [.15], [.85], [.9]])
    y = np.array([0, 0, 1, 1])
    B = np.array([[.11], [.12], [.14], [.88]])
    labels = np.array([0, 0, 1, 1])
    pred = KNeighborsClassifier(n_neighbors=1, metric='manhattan', algorithm='brute').fit(A, y).predict(B)
    cert = certify_queries(A, y, B, pred, [0])
    mask = np.array(cert['certified']) & (pred == labels)
    expected = np.mean([mask[labels == label].mean() for label in np.unique(labels)])
    value, protected = query_aware_score(A, y, B, labels, [0], 101, balanced=True)
    assert np.isclose(protected, expected)
    assert np.isclose(value, .75*balanced_accuracy_score(labels, pred)+.25*expected)


def test_all_primary_objectives_share_identical_candidates():
    methods = ['decision_robust_balanced_query', 'decision_nominal_balanced_query',
               'decision_accuracy_balanced_query']
    grids = [configs(m) for m in methods]
    assert len(grids[0]) == 12
    assert grids[0] == grids[1] == grids[2]
