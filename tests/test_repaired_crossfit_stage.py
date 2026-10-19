"""Audit both ordinary paths and genuine blocked-path repaired weights."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from certified_frfs.repaired_crossfit_stage import fit_repaired_crossfit_stage
import certified_frfs.repaired_crossfit_stage as production
from certified_frfs.datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('repaired_auditor_test', ROOT/'scripts/audit_repaired_crossfit_recent_pilot.py')
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def test_original_path_needs_no_repair_and_independent_audit_passes():
    x = np.r_[np.linspace(.05, .2, 8), np.linspace(.8, .95, 8)]
    X = np.column_stack([x]*5)
    y = np.repeat([0, 1], 8)
    stage = fit_repaired_crossfit_stage(X, y)
    assert stage['actual_count'] == 4 and not stage['path']['repair_blocks']
    counts = Counter()
    auditor.check_stage(X, y, stage, counts)
    assert counts['stages'] == 1


def test_real_blocked_path_with_audited_weights_has_independent_repair_replay(monkeypatch):
    # Deterministic test oracle uses independently audited training-only weights;
    # production experiments always solve afresh under their declared time budget.
    saved = json.loads((ROOT/'results/flexible-majority-repair-admission/ecoli.json').read_text())
    fits = {tuple(a['fit']['features']): a['fit'] for a in saved['attempts']}
    def oracle(A, y, fs, **kwargs):
        info = fits[tuple(fs)]
        return (np.array(info['weights']) if info['status'] == 'original_certificate_passed' else None), info
    monkeypatch.setattr(production, 'fit_flexible_majority_repair', oracle)
    X, y = load_dataset('ecoli')
    ri = np.array(saved['reference_indices'])
    stage = fit_repaired_crossfit_stage(X[ri], y[ri])
    assert stage['actual_count'] == 6 and stage['path']['repair_blocks']
    counts = Counter()
    auditor.check_stage(X[ri], y[ri], stage, counts)
    assert counts['accepted_metric_repairs'] == 1
    assert counts['independent_fixed_friend_targets'] > 0


def test_generated_sources_and_prototype_copy_are_exact():
    from prepare_repaired_crossfit_recent_pilot import generated
    for name, source in generated().items():
        assert (ROOT/'scripts'/name).read_text(encoding='utf-8') == source
        compile(source, name, 'exec')
    assert (ROOT/'src/certified_frfs/flexible_majority_repair.py').read_bytes() == (ROOT/'scripts/flexible_majority_repair.py').read_bytes()
