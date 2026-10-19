"""A classwise safeguard can reject an originally feasible repaired metric."""
from pathlib import Path
from collections import Counter
import importlib.util
import json
import sys
import numpy as np
from certified_frfs.datasets import load_dataset
import certified_frfs.safeguarded_crossfit_stage as production

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('guarded_auditor_test', ROOT/'scripts/audit_safeguarded_crossfit_recent_pilot.py')
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def test_class_guard_rejects_rare_class_loss_without_changing_original_certificate(monkeypatch):
    saved = json.loads((ROOT/'results/classwise-repair-admission/ecoli.json').read_text())
    fits = {tuple(a['fit']['features']): a['fit'] for a in saved['attempts']}
    def oracle(A, y, fs, **kwargs):
        info = fits[tuple(fs)]
        weights = np.array(info['weights']) if info['status'] == 'original_certificate_passed' else None
        return weights, info
    monkeypatch.setattr(production, 'fit_flexible_majority_repair', oracle)
    X, y = load_dataset('ecoli')
    ri = np.array(saved['reference_indices'])
    stage = production.fit_safeguarded_crossfit_stage(X[ri], y[ri])
    assert stage['actual_count'] == 7
    rejected = [a['fit'] for r in stage['path']['repair_blocks'] for a in r['attempts'] if a['fit']['status'] == 'classwise_guard_failed']
    assert len(rejected) == 1 and rejected[0]['certificate']['feasible']
    assert not rejected[0]['classwise_dependency_certificate']['feasible_all_classes']
    counts = Counter()
    auditor.check_stage(X[ri], y[ri], stage, counts)
    assert counts['classwise_guard_rejections'] == 1


def test_class_guard_preserves_certified_safe_repair(monkeypatch):
    saved = json.loads((ROOT/'results/classwise-repair-admission/yeast.json').read_text())
    fit = saved['attempts'][0]['fit']
    def oracle(A, y, fs, **kwargs):
        assert fs == fit['features']
        return np.array(fit['weights']), fit
    monkeypatch.setattr(production, 'fit_flexible_majority_repair', oracle)
    X, y = load_dataset('yeast')
    ri = np.array(saved['reference_indices'])
    stage = production.fit_safeguarded_crossfit_stage(X[ri], y[ri])
    assert stage['actual_count'] == 7
    counts = Counter()
    auditor.check_stage(X[ri], y[ri], stage, counts)
    assert counts['accepted_metric_repairs'] == 1


def test_generated_guarded_sources_are_exact():
    from prepare_safeguarded_crossfit_recent_pilot import generated
    for name, source in generated().items():
        assert (ROOT/name).read_text(encoding='utf-8') == source
        compile(source, name, 'exec')
