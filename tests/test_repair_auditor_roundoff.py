"""The independently calculated scalar loss is not bitwise reproducible."""
import importlib.util
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np
import pytest
from certified_frfs.datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/(name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_frozen_roundoff_failure_is_repaired_without_changing_results():
    diagnosis = json.loads((ROOT/'results/repair-audit-float-identity-diagnosis.json').read_text())
    row = next(r for r in diagnosis['rows'] if r['absolute_error'] > 0)
    saved = json.loads((ROOT/'results/repaired-crossfit-recent-pilot'/(row['dataset']+'.json')).read_text())
    X, y = load_dataset(row['dataset'])
    ri = np.array(saved['reference_indices'])
    if row['stage_index'] < len(saved['inner']):
        block = saved['inner'][row['stage_index']]
        a = np.array(block['reference_indices'])
        raw, train = X[ri[a]], y[ri[a]]
    else:
        block = saved['outer']
        raw, train = X[ri], y[ri]
    old = module('audit_repaired_crossfit_recent_pilot')
    with pytest.raises(AssertionError):
        old.check_stage(raw, train, block['stage'], Counter())
    fixed = module('audit_repaired_crossfit_recent_pilot_v2')
    counts = Counter()
    fixed.check_stage(raw, train, block['stage'], counts)
    assert counts['stages'] == 1


def test_v2_generation_preserves_every_other_check():
    from prepare_repaired_crossfit_auditor_v2 import generated
    actual = (ROOT/'scripts/audit_repaired_crossfit_recent_pilot_v2.py').read_text(encoding='utf-8')
    assert actual == generated()
    assert 'atol=1e-12' in actual
    assert 'gate = all(contrasts[m][\'mean_pp\'] >= .5' in actual
