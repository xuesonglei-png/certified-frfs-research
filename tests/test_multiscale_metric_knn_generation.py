"""47recipe source generation and actual metric adaptation remain coherent."""
from pathlib import Path
import sys
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from prepare_multiscale_metric_knn_pilot import generated
import run_multiscale_certified_metric_knn_pool as runner
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def test_generated_sources_exact_and_compilable():
    for name, source in generated().items():
        assert (ROOT/name).read_text(encoding='utf-8') == source
        compile(source, name, 'exec')
    assert runner.COUNTS['proposed'] == runner.COUNTS['unguarded_pool'] == 47
    assert runner.COUNTS['raw_repair_pool'] == 47


def test_new_small_dimension_admission_actually_drives_metric_weights():
    saved = json.loads((ROOT/'results/recent-multiscale-pool-admission/banknote.json').read_text())
    old = json.loads((runner.PARENT/'banknote.json').read_text())
    X, y = load_dataset('banknote'); ri = np.array(saved['reference_indices'])
    A, _ = preprocess_fit(X[ri], X[ri])
    block = {**old['outer'], 'recipes': saved['recipes'], 'pool': saved['pool']}
    policies = runner.policy_metrics(A, block, count=3)
    assert len(policies['proposed']) == len(policies['unguarded_pool']) == 47
    assert len(policies['raw_repair_pool']) == 47
    assert any(m['features'] == [0, 1, 2] and m['origin'] == 'admitted_metric' for m in policies['proposed'])
    assert all(len(m['features']) == 3 for m in policies['fgs_matched']+policies['sfrip_matched'])
    for m in policies['proposed']:
        assert len(m['features']) == len(m['weights'])
        np.testing.assert_allclose(sum(m['weights']), 2., atol=1e-10)
