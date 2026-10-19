"""Cap-compatible scales and provenance survive support deduplication."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_recent_multiscale_pool_admission as runner
from certified_frfs.datasets import load_dataset


def test_47_fixed_recipes_include_small_dataset_compression_and_all_origins():
    paths = {'fgs2023': [{'ranking': [0, 1, 2, 3]}]*9, 'sfrip2025': [{'ranking': [3, 2, 1, 0]}]*6}
    recipes = runner.recipes(paths, [0, 1, 2, 3], 4)
    assert len(recipes) == 47
    assert [len(r['features']) for r in recipes[:3]] == [2, 3, 4]
    assert {r['origin']['fraction'] for r in recipes[:45]} == {.5, .65, .8}
    assert all(len(r['features']) >= 2 for r in recipes)
    assert recipes[-1]['origin'] == {'method': 'full_fallback'}


def test_all_scales_are_mass_cap_compatible_on_dimensions_2_to_60():
    for d in range(2, 61):
        for fraction in runner.FRACTIONS:
            assert int(np.ceil(fraction*d))*(2/d) >= 1-1e-12


def test_identical_support_reuses_audited_record_but_keeps_new_origins():
    base = json.loads((runner.BASE/'banknote.json').read_text())
    cached = json.loads((runner.CACHE/'banknote.json').read_text())
    X, y = load_dataset('banknote'); ri = np.array(base['reference_indices'])
    candidates = [{'features': list(range(4)), 'origin': {'method': 'fgs2023', 'config_index': 0, 'fraction': .8}},
                  {'features': list(range(4)), 'origin': {'method': 'full_fallback'}}]
    records = runner.admission(X[ri], y[ri], candidates, cached['pool'])
    assert len(records) == 1 and records[0]['cache_reused'] and records[0]['accepted']
    assert records[0]['origins'] == [c['origin'] for c in candidates]
    assert records[0]['repair'] is None
