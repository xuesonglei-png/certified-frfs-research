"""Independent graph/fit replay including disconnected-forest box minima."""
from collections import Counter
from itertools import product
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from certified_frfs.penalized_fuzzy_forest import penalized_forest
from certified_frfs.penalized_crossfit_stage import fit_penalized_crossfit_stage
from certified_frfs.multiknot_forest import fit_multiknot_forest, box_margins, scores

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('penalized_auditor_tests', ROOT/'scripts/audit_penalized_forest_recent_pilot.py')
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def test_independent_contingency_MI_matches_forest_including_tie_bins():
    rng = np.random.default_rng(989)
    X = rng.uniform(size=(80, 6))
    X[:4, :2] = [[0, .25], [.25, .75], [.75, 1], [1, 0]]
    y = (X[:, 0]+X[:, 1] > 1).astype(int)
    actual = penalized_forest(X, y)
    expected = auditor.independent_penalized_forest(X, y)
    auditor.equivalent(actual, expected)
    assert len(actual['edges']) < X.shape[1]-1
    assert all(r['adjusted_gain'] > 0 for r in actual['selected'])


def test_independent_labels_do_not_force_spanning_tree():
    rng = np.random.default_rng(737)
    X = rng.uniform(size=(256, 12))
    y = rng.integers(0, 2, len(X))
    assert not penalized_forest(X, y)['edges']


def test_disconnected_pair_forest_exact_box_scores():
    rng = np.random.default_rng(48)
    X = rng.uniform(size=(60, 3))
    y = (X[:, 0]+X[:, 1] > 1).astype(int)
    fit = fit_multiknot_forest(X, y, [0, 1, 2], [[0, 1]], knot_count=3, C=1.)
    query, truth = np.array([[.5, .47, .6]]), np.array([1])
    bound = box_margins(query, truth, fit, eps=.1, witnesses=True)['margins'][0]
    points = [sorted(set([max(0., z-.1), min(1., z+.1)]+([.5] if abs(z-.5) <= .1 else []))) for z in query[0]]
    raw = np.array(list(product(*points)))
    values = scores(raw, fit)
    assert bound == pytest.approx(np.min(values[:, 1]-values[:, 0]), abs=1e-10)


def test_penalized_stage_independent_teacher_OOF_and_FRS_replay():
    rng = np.random.default_rng(377)
    signal = rng.uniform(size=26)
    X = np.column_stack([signal]*4+[rng.uniform(size=26)])
    y = (signal > .5).astype(int)
    stage = fit_penalized_crossfit_stage(X, y)
    counts = Counter()
    auditor.check_stage(X, y, stage, counts)
    assert stage['actual_count'] == 4 and counts['stages'] == 1


def test_generation_exact_and_no_production_graph_in_auditor():
    from prepare_penalized_forest_recent_pilot import generated
    for name, source in generated().items():
        assert (ROOT/'scripts'/name).read_text(encoding='utf-8') == source
        compile(source, name, 'exec')
    source = (ROOT/'scripts/audit_penalized_forest_recent_pilot.py').read_text(encoding='utf-8')
    assert 'certified_frfs.penalized_fuzzy_forest' not in source
    assert 'fuzzy_interaction_tree import' not in source
