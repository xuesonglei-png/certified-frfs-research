"""Proper loss discriminates tied BA and immutable fit reuse is equivalent."""
from pathlib import Path
import sys
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_proper_loss_diverse_pool_pilot as runner
import audit_proper_loss_diverse_pool_pilot as auditor
from prepare_proper_loss_pool_pilot import generated
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def test_proper_loss_does_not_promote_bad_probabilities_on_BA_tie():
    rows = [{'selector_index': si, 'model_index': 0, 'folds': [
        {'balanced_logloss': loss, 'clean_BA': .8, 'actual_count': size}]*3} for si, (loss, size) in enumerate(((.7, 4), (.4, 5), (.4, 5)))]
    assert runner.choose(rows) == auditor.independent_choose(rows) == 1
    assert rows[1]['mean_balanced_logloss'] == np.mean([.4]*3)


def test_model_registry_reuse_preserves_predictions_and_full_lookup():
    saved = json.loads((runner.PREDICTION_PARENT/'banknote.json').read_text())
    X, y = load_dataset('banknote'); ri = np.array(saved['reference_indices']); block = saved['inner'][0]
    ca, cb = np.array(block['reference_indices']), np.array(block['validation_indices'])
    A, B = preprocess_fit(X[ri[ca]], X[ri[cb]])
    reused = runner.cached_registry(A, y[ri[ca]], block, B, y[ri[cb]])
    fresh = runner.Registry(A, y[ri[ca]], B, y[ri[cb]])
    for mi in range(6):
        a = reused.fit(list(range(4)), mi); b = fresh.fit(list(range(4)), mi)
        np.testing.assert_allclose(reused.evaluations[str(a)]['scores'], fresh.evaluations[str(b)]['scores'], atol=1e-12)
    assert len(reused.models) == len(block['models'])


def test_generated_sources_are_exact_and_compilable():
    for name, source in generated().items():
        assert (ROOT/name).read_text(encoding='utf-8') == source
        compile(source, name, 'exec')
