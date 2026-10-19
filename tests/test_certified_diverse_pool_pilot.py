"""Regression checks for safety fallback and support-count matching."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_certified_diverse_pool_pilot as runner
import audit_certified_diverse_pool_pilot as auditor
from certified_frfs.datasets import load_dataset


def test_real_vehicle_rejection_preserves_baseline_origins_and_full_fallback():
    base = json.loads((runner.BASE/'vehicle.json').read_text())
    own = json.loads((runner.OOF/'vehicle.json').read_text())
    admission = json.loads((ROOT/'results/recent-pool-certificate-admission/vehicle.json').read_text())
    recipes = runner.candidate_origins(base['outer']['paths'], own['outer']['stage']['candidate']['features'], 18)
    selected = runner.supports(recipes, admission['pool'], base['outer']['paths'], 18, count=15)
    # All fifteen recent-selector supports failed admission. Their raw pool
    # controls remain present, while the certified recipes must use full.
    assert all(fs == list(range(18)) for fs in selected['proposed'][:15])
    assert all(len(fs) == 15 for fs in selected['unguarded_pool'][:15])
    assert len(selected['proposed'][15]) == 15
    assert all(len(fs) == 15 for fs in selected['fgs_matched']+selected['sfrip_matched'])
    assert recipes == auditor.expected_recipes(base['outer']['paths'], own['outer']['stage']['candidate']['features'], 18)


def test_fold_count_uses_selected_recipe_instead_of_local_best():
    d = 5
    paths = {'fgs2023': [{'ranking': list(range(d)), 'native': {'features': [0]}}]*9,
             'sfrip2025': [{'ranking': list(reversed(range(d))), 'native': {'features': []}}]*6}
    recipes = runner.candidate_origins(paths, [0, 1, 2, 3], d)
    fs4, full = list(range(4)), list(range(5))
    pool = [{'features': fs4, 'accepted': False}, {'features': [1, 2, 3, 4], 'accepted': True}, {'features': full, 'accepted': True}]
    chosen_recipe = 0
    first = runner.supports(recipes, pool, paths, d)
    count = len(first['proposed'][chosen_recipe])
    out = runner.supports(recipes, pool, paths, d, count=count)
    assert count == 5 and len(first['proposed'][9]) == 4
    assert all(len(fs) == 5 for fs in out['fgs_matched']+out['sfrip_matched'])


def test_real_full_only_admission_replays_original_and_class_contracts():
    saved = json.loads((runner.BASE/'banknote.json').read_text())
    X, y = load_dataset('banknote'); ri = np.array(saved['reference_indices'])
    recipes = [{'features': list(range(4)), 'origin': {'method': 'full_fallback'}}]
    pool = runner.admit(X[ri], y[ri], recipes)
    from collections import Counter
    from certified_frfs.validated_pool import preprocess_fit
    A, _ = preprocess_fit(X[ri], X[ri])
    accepted = auditor.check_pool(A, y[ri], recipes, pool, Counter())
    assert accepted[tuple(range(4))] and pool[0]['repair'] is None
    pool[0]['selected_class_certificate']['max_upper_bound'] = .9
    with pytest.raises(AssertionError):
        auditor.check_pool(A, y[ri], recipes, pool, Counter())


def test_selection_ignores_outer_outcomes_and_keeps_fixed_ties():
    rows = [{'selector_index': si, 'model_index': 0, 'folds': [
        {'clean_BA': .8, 'actual_count': size}, {'clean_BA': .7, 'actual_count': size}]} for si, size in enumerate((5, 4, 4))]
    assert runner.choose(rows) == auditor.independent_choose(rows) == 1
