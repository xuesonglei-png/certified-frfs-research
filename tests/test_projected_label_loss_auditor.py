"""Independent auditor must check the actual new objective and cache identity."""
import importlib.util
from collections import Counter
from pathlib import Path
import sys
import numpy as np
from certified_frfs.label_loss_certified_stage import fit_label_loss_certified_stage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('projected_label_loss_test_auditor', ROOT/'scripts/audit_projected_label_loss_recent_pilot.py')
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def test_independent_auditor_reconstructs_teacher_true_label_ranking_and_FRS():
    rng = np.random.default_rng(377)
    signal = rng.uniform(size=26)
    X = np.column_stack([signal, signal, signal, signal, rng.uniform(size=26)])
    y = (signal > .5).astype(int)
    stage = fit_label_loss_certified_stage(X, y)
    assert stage['actual_count'] == 4
    assert stage['path']['rankings'] and stage['path']['attempts']
    counts = Counter()
    auditor.check_stage(X, y, stage, counts)
    assert counts['stages'] == 1 and counts['independent_training_fits'] > 6
    assert counts['FRS_certificate_replays'] >= 2
    # The full-support ranking value is nominal label loss, not zero MSE.
    assert stage['path']['trace'][0]['projected_balanced_logloss'] > 0.


def test_tree_cache_requires_identical_training_values_and_label_order(monkeypatch):
    calls = []
    def rebuild(X, y):
        calls.append((X.copy(), y.copy()))
        return {'reconstruction': len(calls)}
    monkeypatch.setattr(auditor, 'TREE_CACHE', {})
    monkeypatch.setattr(auditor, 'interaction_tree', rebuild)
    X, y = np.array([[0., .5], [.2, .8], [1., 1.]]), np.array([0, 1, 0])
    first = auditor.cached_tree(X, y)
    assert auditor.cached_tree(X.copy(), y.copy()) is first
    changed = X.copy()
    changed[0, 0] = .01
    assert auditor.cached_tree(changed, y) != first
    assert auditor.cached_tree(X, np.array([1, 0, 0])) != first
    assert len(calls) == 3
