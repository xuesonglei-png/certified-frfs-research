"""Guards check requested metrics and never deploy rejected mixtures."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from predictive_target_knn_stage import PredictiveMetricStage, GRID
from run_predictive_target_knn_pilot import options
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def test_same_metric_learning_budget_for_three_pools_and_weighted_controls():
    assert len(GRID) == 7
    assert len(options('proposed')) == len(options('unguarded_pool')) == len(options('raw_repair_pool')) == 47*7
    assert len(options('fgs_matched')) == 9*7 and len(options('sfrip_matched')) == 6*7
    assert len(options('full')) == 7 and len(options('fgs_native')) == 9


def test_actual_rejected_target_falls_back_to_admitted_anchor():
    parent = json.loads((ROOT/'results/recent-multiscale-pool-admission/ecoli.json').read_text())
    diagnostic = json.loads((ROOT/'results/predictive-target-metric-admission/ecoli.json').read_text())
    record = diagnostic['records'][0]
    X, y = load_dataset('ecoli'); ri = np.array(parent['reference_indices']); A, _ = preprocess_fit(X[ri], X[ri])
    stage = PredictiveMetricStage(A, y[ri]); base = {'features': record['features'], 'weights': record['anchor'], 'origin': 'admitted_metric'}
    metric = stage.metric(base, 6, certified=True)
    assert metric['guard_fallback']
    np.testing.assert_allclose(metric['requested_weights'], record['target'], atol=1e-10)
    np.testing.assert_array_equal(metric['weights'], record['anchor'])
    exported = stage.export()['certificate_checks']
    assert not exported[metric['guard_id']]['accepted'] and exported[metric['deployed_guard_id']]['accepted']
    raw = stage.metric(base, 6, certified=False)
    np.testing.assert_allclose(raw['weights'], record['target'], atol=1e-10)
    assert raw['guard_id'] is None and not raw['guard_fallback']


def test_actual_feasible_target_is_kept_and_training_fit_reused():
    parent = json.loads((ROOT/'results/recent-multiscale-pool-admission/iris.json').read_text())
    diagnostic = json.loads((ROOT/'results/predictive-target-metric-admission/iris.json').read_text())
    record = next(r for r in diagnostic['records'] if len(r['features']) == 4)
    X, y = load_dataset('iris'); ri = np.array(parent['reference_indices']); A, _ = preprocess_fit(X[ri], X[ri])
    stage = PredictiveMetricStage(A, y[ri]); base = {'features': record['features'], 'weights': record['anchor'], 'origin': 'admitted_metric'}
    metric = stage.metric(base, 3, certified=True)
    assert not metric['guard_fallback'] and metric['guard_id'] == metric['deployed_guard_id']
    np.testing.assert_allclose(metric['weights'], .75*np.array(record['anchor'])+.25*np.array(record['target']), atol=1e-10)
    stage.metric(base, 4, certified=False)
    assert len(stage.learners) == 1


def test_empty_native_prior_requires_no_capped_metric_learner():
    A = np.zeros((8, 4)); y = np.repeat([0, 1], 4)
    stage = PredictiveMetricStage(A, y)
    metric = stage.metric({'features': [], 'weights': [], 'origin': 'uniform_native'}, 0, weighted=False)
    assert metric['features'] == [] and metric['learner_id'] is None and not stage.learners
