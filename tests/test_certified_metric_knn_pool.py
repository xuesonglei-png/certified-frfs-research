"""Actual deployed metric, stable votes and certified fallback regression."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_certified_metric_knn_pool as runner
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def test_distance_weights_change_predictions_and_zero_metric_ties_stably():
    A = np.array([[0, 1], [.1, .9], [.2, .8], [.8, .2], [.9, .1], [1, 0]])
    y = np.array([0, 0, 0, 1, 1, 1]); B = np.array([[0, 0]])
    first = runner.evaluate(A, y, B, np.array([0]), [0, 1], np.array([1, 0]))
    second = runner.evaluate(A, y, B, np.array([0]), [0, 1], np.array([0, 1]))
    assert first['predictions'] == [0] and second['predictions'] == [1]
    tied = runner.evaluate(A, y, B, np.array([0]), [0, 1], np.zeros(2))
    assert tied['neighbor_indices'] == [[0, 1, 2, 3, 4]]


def test_real_vehicle_policies_use_certified_fallback_and_keep_raw_weights():
    saved = json.loads((runner.PARENT/'vehicle.json').read_text())
    X, y = load_dataset('vehicle'); ri = np.array(saved['reference_indices'])
    A, _ = preprocess_fit(X[ri], X[ri])
    policies = runner.policy_metrics(A, saved['outer'], count=15)
    assert all(len(m['features']) == 18 for m in policies['proposed'][:15])
    assert all(len(m['features']) == 15 for m in policies['unguarded_pool'][:15])
    for m in policies['fgs_matched']+policies['sfrip_matched']:
        assert len(m['features']) == len(m['weights']) == 15
        np.testing.assert_allclose(sum(m['weights']), np.sqrt(18), atol=1e-10)
        assert max(m['weights'])/np.sqrt(18) <= 2/18+1e-10


def test_vote_ties_use_smallest_class_and_probabilities_are_normalized():
    A = np.arange(6)[:, None]/6; y = np.array([2, 1, 2, 1, 3, 3]); B = np.array([[0]])
    result = runner.evaluate(A, y, B, np.array([1]), [0], np.ones(1))
    assert result['predictions'] == [1]
    np.testing.assert_allclose(np.sum(result['probabilities'], axis=1), 1)
